"""Typed episode handle + in-memory dataset over a directory of episodes.

Two layers:

  * :class:`Episode` — one episode's arrays plus its source path and
    optional sidecar metadata (``act_mode``, ``goal_xy``, …). Replaces
    the bare raw-dict episodes that used to flow around the codebase.
    Use :meth:`Episode.from_file` to round-trip a single file (handy in
    one-off scripts and importer tests).
  * :class:`EpisodeDataset` — the set of episodes living at a directory.
    Encapsulates two access modes: ``stream()`` for one-pass ingestion
    that never holds the full set in RAM (replay-buffer warmup), and
    ``load()`` for materializing the set into memory once for repeated
    re-iteration (the evaluator's cached set).

Discovery (``len(ds)`` / ``ds.paths``) is cheap and works without
loading. ``load()`` is idempotent so callers can call it from a
warmup hook without checking ``loaded`` first.

This module owns the format-specific readers. The two reader
implementations (HDF5 / Rerun) are private; round-trip a single file
through :meth:`Episode.from_file`.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Union

import h5py  # type: ignore[import-untyped]
import numpy as np


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


ProgressCallback = Callable[[int, int, Path], None]
Progress = Union[bool, ProgressCallback]

RawEpisode = dict[str, Any]


SUPPORTED_FORMATS = ("hdf5", "rerun")

# Sidecar fields that live alongside the per-step arrays in the writer
# format but are properties of the whole episode. We surface them on
# ``Episode.metadata`` so call sites can read them without sniffing the
# raw arrays dict.
_METADATA_KEYS: tuple[str, ...] = ("act_mode", "goal_xy")


# ---------------------------------------------------------------------------
# Format-specific readers (private — go through Episode / EpisodeDataset)
# ---------------------------------------------------------------------------


# Maps RawEpisode key -> HDF5 dataset name. Action datasets are resolved
# dynamically since exactly one of joint_action / ee_action is present.
_HDF5_DATASET = {
  "image": "images",
  "reward": "rewards",
  "timestamp": "timestamps",
  "joint_qpos": "joint_qpos",
  "ee_pos": "ee_pos",
  "ee_quat": "ee_quat",
  "object_xy": "object_xy",
}


def _load_hdf5_episode(path: str | Path) -> RawEpisode:
  """Read one HDF5 episode written by ``HDF5EpisodeWriter``."""
  raw: RawEpisode = {}
  with h5py.File(path, "r") as f:
    for key, dataset in _HDF5_DATASET.items():
      raw[key] = np.asarray(f[dataset][()])

    if "joint_action" in f:
      raw["joint_action"] = np.asarray(f["joint_action"][()])
    elif "ee_action" in f:
      raw["ee_action"] = np.asarray(f["ee_action"][()])
    else:
      raise KeyError(f"{path}: missing action dataset (joint_action or ee_action)")

    if "metadata" in f:
      meta = f["metadata"]
      if "act_mode" in meta:
        am = meta["act_mode"][()]
        raw["act_mode"] = am.decode("utf-8") if isinstance(am, bytes) else str(am)
      if "goal_xy" in meta:
        raw["goal_xy"] = np.asarray(meta["goal_xy"][()], dtype=np.float32)
  return raw


def _collect_chunks_by_entity(recording) -> dict[str, list[dict]]:
  """Group a recording's chunks by entity path, skipping metadata."""
  by_entity: dict[str, list[dict]] = {}
  static: dict[str, dict] = {}
  for chunk in recording.chunks():
    entity = str(chunk.entity_path)
    if chunk.is_static:
      static[entity] = chunk.to_record_batch().to_pydict()
      continue
    if entity.startswith("/__"):
      continue
    by_entity.setdefault(entity, []).append(chunk.to_record_batch().to_pydict())
  by_entity["__static__"] = [static]
  return by_entity


def _ordered_scalar_column(chunk_dicts: list[dict], step_key: str = "step") -> np.ndarray:
  """Flatten per-chunk scalar rows into a single (N, d) array sorted by step."""
  rows: list[tuple[int, np.ndarray]] = []
  for d in chunk_dicts:
    scalars = d["Scalars:scalars"]
    steps = d[step_key]
    for s, v in zip(steps, scalars):
      rows.append((int(s), np.asarray(v, dtype=np.float32)))
  rows.sort(key=lambda x: x[0])
  return np.stack([v for _, v in rows], axis=0)


def _load_rerun_episode(path: str | Path) -> RawEpisode:
  """Read one Rerun ``.rrd`` episode written by ``RerunEpisodeWriter``."""
  from rerun.recording import load_recording

  rec = load_recording(str(path))
  by_entity = _collect_chunks_by_entity(rec)

  def _scalars(entity: str) -> np.ndarray:
    if entity not in by_entity:
      raise KeyError(f"entity {entity!r} not found in {path}")
    return _ordered_scalar_column(by_entity[entity])

  raw: RawEpisode = {}
  if "/joint_action" in by_entity:
    raw["joint_action"] = _scalars("/joint_action")
  elif "/ee_action" in by_entity:
    raw["ee_action"] = _scalars("/ee_action")
  else:
    raise KeyError(f"{path}: missing action entity (/joint_action or /ee_action)")

  raw["reward"] = _scalars("/reward").reshape(-1)
  raw["joint_qpos"] = _scalars("/joint_qpos")
  raw["ee_pos"] = _scalars("/ee_pos")
  raw["ee_quat"] = _scalars("/ee_quat")
  raw["object_xy"] = _scalars("/object_xy")

  image_rows: list[tuple[int, np.ndarray]] = []
  time_rows: list[tuple[int, float]] = []
  for d in by_entity["/camera/image"]:
    for s, buf, fmt, t in zip(
      d["step"], d["Image:buffer"], d["Image:format"], d["time"]
    ):
      f0 = fmt[0] if isinstance(fmt, list) else fmt
      w, h = int(f0["width"]), int(f0["height"])
      image_rows.append((int(s), np.asarray(buf, dtype=np.uint8).reshape(h, w, 3)))
      time_rows.append(
        (int(s), t.total_seconds() if hasattr(t, "total_seconds") else float(t))
      )
  image_rows.sort(key=lambda x: x[0])
  time_rows.sort(key=lambda x: x[0])
  raw["image"] = np.stack([img for _, img in image_rows], axis=0)
  raw["timestamp"] = np.asarray([t for _, t in time_rows], dtype=np.float32)

  return raw


_FORMAT_SUFFIX = {"hdf5": ".hdf5", "rerun": ".rrd"}
_FORMAT_LOADER = {"hdf5": _load_hdf5_episode, "rerun": _load_rerun_episode}
_FORMAT_POOL: dict[str, type[ThreadPoolExecutor] | type[ProcessPoolExecutor]] = {
  # h5py reads release the GIL → threads suffice. The rerun reader runs
  # heavy per-step Python parsing → only a process pool actually
  # parallelizes it.
  "hdf5":  ThreadPoolExecutor,
  "rerun": ProcessPoolExecutor,
}


# ---------------------------------------------------------------------------
# Progress helper (shared by load() and stream())
# ---------------------------------------------------------------------------


def _resolve_progress(progress: Progress) -> ProgressCallback | None:
  """Turn a ``Progress`` value into a per-file callback (or None)."""
  if progress is False:
    return None
  if callable(progress):
    return progress

  try:
    from tqdm import tqdm  # type: ignore[import-not-found]
  except ImportError:
    def _print_cb(i: int, total: int, path: Path) -> None:
      print(f"[{i}/{total}] {path.name}")

    return _print_cb

  bar: dict[str, Any] = {"tqdm": None}

  def _tqdm_cb(i: int, total: int, path: Path) -> None:
    if bar["tqdm"] is None:
      bar["tqdm"] = tqdm(total=total, unit="ep", desc="episodes")
    bar["tqdm"].set_postfix_str(path.name, refresh=False)
    bar["tqdm"].update(1)
    if i == total:
      bar["tqdm"].close()

  return _tqdm_cb


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


def _extract_metadata(raw: RawEpisode) -> dict[str, Any]:
  """Copy whole-episode sidecar fields out of a raw dict into a metadata dict.

  Non-destructive: the values stay in ``raw`` so processors that look
  them up there keep working. The metadata dict is for callers that
  want to read episode-level fields without sniffing ``data``.
  """
  return {k: raw[k] for k in _METADATA_KEYS if k in raw}


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

  @classmethod
  def from_file(cls, path: str | Path, *, format: str | None = None) -> "Episode":
    """Round-trip a single episode file into an :class:`Episode`.

    ``format`` is inferred from the file extension when omitted
    (``.hdf5`` → ``"hdf5"``, ``.rrd`` → ``"rerun"``); pass it explicitly
    when the extension doesn't match the format you want to read.
    """
    p = Path(path)
    fmt = format or _format_from_path(p)
    if fmt not in SUPPORTED_FORMATS:
      raise ValueError(f"unsupported format {fmt!r}; use one of {SUPPORTED_FORMATS}")
    raw = _FORMAT_LOADER[fmt](p)
    return cls(path=p, format=fmt, data=raw, metadata=_extract_metadata(raw))

  @property
  def stem(self) -> str:
    return self.path.stem

  @property
  def num_steps(self) -> int:
    """Number of recorded steps (length of the action / reward axis)."""
    if "reward" in self.data:
      return int(np.asarray(self.data["reward"]).shape[0])
    for key in ("joint_action", "ee_action"):
      if key in self.data:
        return int(np.asarray(self.data[key]).shape[0])
    raise KeyError(f"episode {self.path.name} has neither reward nor action")

  def has(self, key: str) -> bool:
    return key in self.data

  def get(self, key: str, default: Any = None) -> Any:
    return self.data.get(key, default)


def _format_from_path(path: Path) -> str:
  suffix = path.suffix.lower()
  for fmt, ext in _FORMAT_SUFFIX.items():
    if suffix == ext:
      return fmt
  raise ValueError(
    f"cannot infer episode format from suffix {suffix!r} on {path.name}; "
    f"pass format= explicitly"
  )


# ---------------------------------------------------------------------------
# EpisodeDataset
# ---------------------------------------------------------------------------


class EpisodeDataset:
  """The set of episodes recorded under one directory.

  Cheap to construct — ``__init__`` only records the path, doesn't read
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
