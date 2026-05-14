"""Typed episode handle + in-memory dataset over a directory of episodes.

Built on top of the format-specific readers in :mod:`.episode_loader`.
Two layers:

  * :class:`Episode` — one episode's arrays plus its source path and
    optional sidecar metadata (``act_mode``, ``goal_xy``, …). Replaces
    the bare ``RawEpisode`` dict at call sites that want to thread the
    source identity through to logs / dumps.
  * :class:`EpisodeDataset` — the set of episodes living at a directory.
    Encapsulates two access modes: ``stream()`` for one-pass ingestion
    that never holds the full set in RAM (replay-buffer warmup), and
    ``load()`` for materializing the set into memory once for repeated
    re-iteration (the evaluator's cached set).

Discovery (``len(ds)`` / ``ds.paths``) is cheap and works without
loading. ``load()`` is idempotent so callers can call it from a
warmup hook without checking ``loaded`` first.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .episode_loader import (
  Progress,
  RawEpisode,
  _resolve_progress,
  load_hdf5_episode,
  load_rerun_episode,
)


SUPPORTED_FORMATS = ("hdf5", "rerun")

# Sidecar fields that live alongside the per-step arrays in the writer
# format but are properties of the whole episode. We surface them on
# ``Episode.metadata`` so call sites can read them without sniffing the
# raw arrays dict.
_METADATA_KEYS: tuple[str, ...] = ("act_mode", "goal_xy")


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Episode:
  """One episode loaded from disk: arrays + provenance.

  ``data`` carries the per-step arrays in the same flat-dict shape the
  processors in :mod:`.episode_processor` consume. ``metadata`` carries
  whole-episode sidecar fields the writers serialize separately.

  ``Episode`` instances are read-only — load once, share freely.
  """

  path: Path
  format: str
  data: RawEpisode
  metadata: dict[str, Any] = field(default_factory=dict)

  @property
  def stem(self) -> str:
    return self.path.stem

  @property
  def num_steps(self) -> int:
    """Number of recorded steps (length of the action / reward axis)."""
    if "reward" in self.data:
      return int(np.asarray(self.data["reward"]).shape[0])
    # Fall back to action — every episode has one or the other.
    for key in ("joint_action", "ee_action"):
      if key in self.data:
        return int(np.asarray(self.data[key]).shape[0])
    raise KeyError(f"episode {self.path.name} has neither reward nor action")

  def has(self, key: str) -> bool:
    return key in self.data

  def get(self, key: str, default: Any = None) -> Any:
    return self.data.get(key, default)


# ---------------------------------------------------------------------------
# EpisodeDataset
# ---------------------------------------------------------------------------


_FORMAT_SUFFIX = {"hdf5": ".hdf5", "rerun": ".rrd"}
_FORMAT_LOADER = {"hdf5": load_hdf5_episode, "rerun": load_rerun_episode}
_FORMAT_POOL: dict[str, type[ThreadPoolExecutor] | type[ProcessPoolExecutor]] = {
  # h5py reads release the GIL → threads suffice. The rerun reader runs
  # heavy per-step Python parsing → only a process pool actually
  # parallelizes it.
  "hdf5":  ThreadPoolExecutor,
  "rerun": ProcessPoolExecutor,
}


def _extract_metadata(raw: RawEpisode) -> dict[str, Any]:
  """Copy whole-episode sidecar fields out of a raw dict into a metadata dict.

  Non-destructive: the values stay in ``raw`` so processors that look
  them up there keep working. The metadata dict is for callers that
  want to read episode-level fields without sniffing ``data``.
  """
  return {k: raw[k] for k in _METADATA_KEYS if k in raw}


class EpisodeDataset:
  """The set of episodes recorded under one directory.

  Cheap to construct — ``__init__`` only globs paths, doesn't read
  any file. Call :meth:`load` (or :meth:`stream`) to actually decode.
  ``len(ds)`` and :attr:`paths` work without loading.
  """

  def __init__(self, directory: str | Path, *, format: str = "hdf5"):
    if format not in SUPPORTED_FORMATS:
      raise ValueError(f"unsupported format {format!r}; use one of {SUPPORTED_FORMATS}")
    self.directory = Path(directory)
    self.format    = format
    self._paths: list[Path] | None = None
    self._loaded: list[Episode] | None = None

  # ---------- discovery (cheap) ----------

  @property
  def paths(self) -> list[Path]:
    """Sorted list of episode file paths in the directory.

    Cached after the first scan — call sites can reuse without paying
    for repeated globs. Returns an empty list when the directory is
    missing.
    """
    if self._paths is None:
      if self.directory.exists():
        suffix = _FORMAT_SUFFIX[self.format]
        self._paths = sorted(self.directory.glob(f"episode-*{suffix}"))
      else:
        self._paths = []
    return self._paths

  def __len__(self) -> int:
    if self._loaded is not None:
      return len(self._loaded)
    return len(self.paths)

  @property
  def loaded(self) -> bool:
    return self._loaded is not None

  # ---------- loading ----------

  def load(
    self,
    *,
    num_workers: int | None = None,
    progress: Progress = True,
    num_episodes: int | None = None,
    rng: np.random.Generator | None = None,
  ) -> "EpisodeDataset":
    """Materialize every episode into memory. Idempotent.

    Returns ``self`` so callers can chain (``EpisodeDataset(...).load()``).
    On a missing directory the set stays empty (``len == 0``) — call
    sites that care should check ``len(dataset)``.

    ``num_workers=None`` picks ``max(1, cpu_count() // 2)`` — the same
    default the replay-buffer warmup uses.
    """
    if self._loaded is not None:
      return self
    self._loaded = list(self._stream(
      paths        = self._pick_paths(num_episodes, rng),
      num_workers  = self._resolve_workers(num_workers),
      progress     = progress,
    ))
    return self

  def stream(
    self,
    *,
    num_workers: int | None = None,
    progress: Progress = False,
    num_episodes: int | None = None,
    rng: np.random.Generator | None = None,
  ) -> Iterator[Episode]:
    """Yield :class:`Episode`\\ s without retaining them.

    Use this for one-pass ingestion (replay-buffer warmup) where the
    caller folds each episode into another structure and doesn't need
    to revisit. Does not populate :attr:`loaded`.
    """
    yield from self._stream(
      paths        = self._pick_paths(num_episodes, rng),
      num_workers  = self._resolve_workers(num_workers),
      progress     = progress,
    )

  # ---------- access after load() ----------

  def __iter__(self) -> Iterator[Episode]:
    if self._loaded is None:
      raise RuntimeError(
        "EpisodeDataset not loaded — call .load() first or use .stream()."
      )
    return iter(self._loaded)

  def __getitem__(self, key: int | str) -> Episode:
    """Index by integer position or by file stem (``"episode-00007"``)."""
    if self._loaded is None:
      raise RuntimeError(
        "EpisodeDataset not loaded — call .load() first or use .stream()."
      )
    if isinstance(key, int):
      return self._loaded[key]
    # By-stem lookup. Linear is fine: eval sets are O(100); building a
    # dict on every access would be wasted bookkeeping.
    for ep in self._loaded:
      if ep.stem == key:
        return ep
    raise KeyError(f"no episode with stem {key!r} in {self.directory}")

  # ---------- internals ----------

  def _resolve_workers(self, num_workers: int | None) -> int:
    if num_workers is None:
      import os
      return max(1, (os.cpu_count() or 2) // 2)
    return int(num_workers)

  def _pick_paths(
    self, num_episodes: int | None, rng: np.random.Generator | None,
  ) -> list[Path]:
    paths = list(self.paths)
    if num_episodes is not None and num_episodes < len(paths):
      rng = rng if rng is not None else np.random.default_rng()
      idx = rng.choice(len(paths), size=int(num_episodes), replace=False)
      paths = [paths[i] for i in sorted(idx.tolist())]
    return paths

  def _stream(
    self,
    *,
    paths: list[Path],
    num_workers: int,
    progress: Progress,
  ) -> Iterator[Episode]:
    """Shared driver for ``load`` and ``stream`` — yields Episode objects."""
    callback = _resolve_progress(progress)
    total    = len(paths)
    loader   = _FORMAT_LOADER[self.format]
    pool_cls = _FORMAT_POOL[self.format]

    def _emit(path: Path, raw: RawEpisode, i: int) -> Episode:
      if callback is not None:
        callback(i, total, path)
      metadata = _extract_metadata(raw)
      return Episode(path=path, format=self.format, data=raw, metadata=metadata)

    if num_workers <= 0 or total <= 1:
      for i, p in enumerate(paths, start=1):
        yield _emit(p, loader(p), i)
      return

    window = min(num_workers, total)
    with pool_cls(max_workers=num_workers) as pool:
      in_flight: deque = deque()
      next_submit = 0
      for _ in range(window):
        in_flight.append((next_submit, paths[next_submit], pool.submit(loader, paths[next_submit])))
        next_submit += 1

      yielded = 0
      while in_flight:
        _idx, path, fut = in_flight.popleft()
        raw = fut.result()
        yielded += 1
        yield _emit(path, raw, yielded)
        if next_submit < total:
          in_flight.append((next_submit, paths[next_submit], pool.submit(loader, paths[next_submit])))
          next_submit += 1
