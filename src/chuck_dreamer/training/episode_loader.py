"""Read sim-recorded episodes from disk into raw ``RawEpisode`` dicts.

Episodes recorded by :class:`HDF5EpisodeWriter`/:class:`RerunEpisodeWriter`
carry one of ``joint_action`` / ``ee_action`` (selected by ``act_mode``
in metadata), the achieved arm qpos and EE pose per step, and the
object/goal state needed to recompute reward at training time.

This module reads those formats into a flat dict of stacked arrays.
Conversion to the replay buffer's ``(obs, action, reward, done)`` schema
is done by the processors in :mod:`.episode_processor`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator, Union

import h5py  # type: ignore[import-untyped]
import numpy as np


ProgressCallback = Callable[[int, int, Path], None]
Progress = Union[bool, ProgressCallback]


RawEpisode = dict[str, Any]


# ---------------------------------------------------------------------------
# Readers
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


def load_hdf5_episode(path: str | Path) -> RawEpisode:
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
  by_entity["__static__"] = [static]  # stash for later inspection
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


def load_rerun_episode(path: str | Path) -> RawEpisode:
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


# ---------------------------------------------------------------------------
# Directory iteration
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


def iter_episodes(
  directory: str | Path,
  format: str = "hdf5",
  progress: Progress = False,
  num_episodes: int | None = None,
  rng: np.random.Generator | None = None,
) -> Iterator[RawEpisode]:
  """Yield raw episodes from a directory of writer output.

  If ``num_episodes`` is given, a random subset of that many episodes is
  selected (without replacement) using ``rng`` (or a fresh default rng).
  When ``num_episodes`` exceeds the available files, all files are used.
  """
  directory = Path(directory)
  if format == "hdf5":
    paths = sorted(directory.glob("episode-*.hdf5"))
    loader = load_hdf5_episode
  elif format == "rerun":
    paths = sorted(directory.glob("episode-*.rrd"))
    loader = load_rerun_episode
  else:
    raise ValueError(f"unsupported format {format!r}; use 'hdf5' or 'rerun'")

  if num_episodes is not None and num_episodes < len(paths):
    rng = rng if rng is not None else np.random.default_rng()
    idx = rng.choice(len(paths), size=int(num_episodes), replace=False)
    paths = [paths[i] for i in sorted(idx.tolist())]

  callback = _resolve_progress(progress)
  total    = len(paths)
  for i, p in enumerate(paths, start=1):
    episode = loader(p)
    if callback is not None:
      callback(i, total, p)
    yield episode
