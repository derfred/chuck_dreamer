"""Load sim-collected episodes into the replay buffer.

The episode writers in ``world_model_pusher.sim.data_collection`` record
T steps of ``(image, action, reward, joint_qpos, ee_pos, ee_quat,
object_xy, timestamp)`` per file. The replay buffer expects a different
schema (``obs``, ``action``, ``reward``, ``done``) with one extra obs
entry (T+1 obs for T actions).

This module bridges the two: a small processor abstraction extracts the
observation from each raw step; the last step is dropped so the obs
count is T+1 for T-1 actions, preserving the buffer's invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol

import h5py  # type: ignore[import-untyped]
import numpy as np


RawEpisode = dict[str, np.ndarray]
Episode = dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Processor abstraction
# ---------------------------------------------------------------------------


class EpisodeProcessor(Protocol):
  """Converts a raw sim episode into replay-buffer schema.

  A processor must return a dict with keys ``obs`` (T+1, *obs_shape),
  ``action`` (T, action_dim), ``reward`` (T,), ``done`` (T,).
  """

  def __call__(self, raw: RawEpisode) -> Episode: ...


def _drop_last_and_pack(obs: np.ndarray, raw: RawEpisode) -> Episode:
  """Align a per-step obs array with the buffer's (T+1 obs, T action) layout.

  Sim episodes record N aligned steps of (obs_t, action_t, reward_t).
  We treat the N recorded obs as ``obs[0..N-1]`` and drop the last
  action/reward, yielding N obs and N-1 actions — i.e. T = N-1 with
  T+1 = N obs. The final ``done`` is set to True.
  """
  N = obs.shape[0]
  if N < 2:
    raise ValueError(f"episode too short: {N} steps (need >= 2)")

  action = np.asarray(raw["action"], dtype=np.float32)[: N - 1]
  reward = np.asarray(raw["reward"], dtype=np.float32)[: N - 1]
  done = np.zeros((N - 1,), dtype=bool)
  done[-1] = True

  return {
    "obs": obs.astype(np.float32, copy=False),
    "action": action,
    "reward": reward,
    "done": done,
  }


class StateVectorProcessor:
  """Default processor: concat ee_pos + ee_quat + object_xy + joint_qpos.

  Produces a flat float32 observation vector per step. Order is fixed
  so downstream consumers can unpack it consistently.
  """

  def __call__(self, raw: RawEpisode) -> Episode:
    ee_pos = np.asarray(raw["ee_pos"], dtype=np.float32)
    ee_quat = np.asarray(raw["ee_quat"], dtype=np.float32)
    object_xy = np.asarray(raw["object_xy"], dtype=np.float32)
    joint_qpos = np.asarray(raw["joint_qpos"], dtype=np.float32)
    obs = np.concatenate([ee_pos, ee_quat, object_xy, joint_qpos], axis=1)
    return _drop_last_and_pack(obs, raw)


class ImageProcessor:
  """Processor that uses the raw RGB image as the observation."""

  def __call__(self, raw: RawEpisode) -> Episode:
    obs = np.asarray(raw["image"], dtype=np.float32)
    return _drop_last_and_pack(obs, raw)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


_HDF5_DATASET = {
  "image": "images",
  "action": "actions",
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
  return raw


def _collect_chunks_by_entity(recording) -> dict[str, list[dict]]:
  """Group a recording's chunks by entity path, skipping metadata."""
  by_entity: dict[str, list[dict]] = {}
  for chunk in recording.chunks():
    entity = str(chunk.entity_path)
    if entity.startswith("/__") or chunk.is_static:
      continue
    by_entity.setdefault(entity, []).append(chunk.to_record_batch().to_pydict())
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
  """Read one Rerun ``.rrd`` episode written by ``RerunEpisodeWriter``.

  Uses the chunk iterator on a loaded recording. Each logged entity
  (``/action``, ``/reward``, ``/camera/image``, …) yields one or more
  chunks whose rows are indexed by the ``step`` timeline we set at
  write time.
  """
  from rerun.recording import load_recording

  rec = load_recording(str(path))
  by_entity = _collect_chunks_by_entity(rec)

  def _scalars(entity: str) -> np.ndarray:
    if entity not in by_entity:
      raise KeyError(f"entity {entity!r} not found in {path}")
    return _ordered_scalar_column(by_entity[entity])

  action = _scalars("/action")
  reward = _scalars("/reward").reshape(-1)
  joint_qpos = _scalars("/joint_qpos")
  ee_pos = _scalars("/ee_pos")
  ee_quat = _scalars("/ee_quat")
  object_xy = _scalars("/object_xy")

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
  images_arr = np.stack([img for _, img in image_rows], axis=0)
  timestamp = np.asarray([t for _, t in time_rows], dtype=np.float32)

  return {
    "image": images_arr,
    "action": action,
    "reward": reward,
    "joint_qpos": joint_qpos,
    "ee_pos": ee_pos,
    "ee_quat": ee_quat,
    "object_xy": object_xy,
    "timestamp": timestamp,
  }


# ---------------------------------------------------------------------------
# Directory iteration
# ---------------------------------------------------------------------------


def iter_episodes(
  directory: str | Path,
  format: str = "hdf5",
) -> Iterator[RawEpisode]:
  """Yield raw episodes from a directory of writer output.

  ``format`` is one of ``"hdf5"`` or ``"rerun"`` and must match the
  writer used to produce the files.
  """
  directory = Path(directory)
  if format == "hdf5":
    paths = sorted(directory.glob("episode_*.hdf5"))
    loader = load_hdf5_episode
  elif format == "rerun":
    paths = sorted(directory.glob("episode_*.rrd"))
    loader = load_rerun_episode
  else:
    raise ValueError(f"unsupported format {format!r}; use 'hdf5' or 'rerun'")

  for p in paths:
    yield loader(p)
