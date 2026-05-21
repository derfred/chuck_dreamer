"""Convert a LeRobot v3 HF dataset into the repo's HDF5/Rerun episode format.

LeRobot v3 stores per-frame proprio/action in a parquet file and camera
frames in an MP4 that concatenates all episodes. Episode boundaries and
the time window into the video are listed in ``meta/episodes/.../*.parquet``.

This module downloads those files for a given ``repo_id``, slices them
per episode, decodes the matching video range with PyAV, and hands the
result to :class:`HDF5EpisodeWriter` / :class:`RerunEpisodeWriter`.

Fields the repo's episode format expects but LeRobot teleop data does not
provide (``reward``, ``ee_pos``, ``ee_quat``, ``object_xy``) are zero-filled.
This is fine for image-mode training; state-mode training would need real
EE / object signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import av  # type: ignore[import-untyped]
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .episode_writer import EpisodeWriter

logger = logging.getLogger(__name__)


# A resolver maps a dataset-relative path (e.g. "meta/info.json") to a local
# file on disk. The HF backend downloads via ``hf_hub_download``; the local
# backend joins onto the dataset root. Both signatures return a ``str`` so
# the caller can hand it to ``cv2``/``pyarrow``/``av`` unchanged.
PathResolver = Callable[[str], str]


@dataclass
class _EpisodeSlice:
  episode_index: int
  data_from: int
  data_to: int                    # exclusive
  video_from_ts: float
  video_to_ts: float              # exclusive
  data_chunk: int
  data_file: int
  video_chunk: int
  video_file: int
  task: str
  length: int


def _data_path(chunk: int, file: int) -> str:
  return f"data/chunk-{chunk:03d}/file-{file:03d}.parquet"


def _video_path(video_key: str, chunk: int, file: int) -> str:
  return f"videos/{video_key}/chunk-{chunk:03d}/file-{file:03d}.mp4"


def _list_local_meta_episodes(root: Path) -> list[str]:
  """Return relative paths of ``meta/episodes/*.parquet`` under ``root``."""
  meta_dir = root / "meta" / "episodes"
  if not meta_dir.exists():
    return []
  return sorted(
    str(p.relative_to(root)).replace("\\", "/")
    for p in meta_dir.rglob("*.parquet")
  )


def _read_episodes_meta(
  dataset_id: str,
  video_key: str,
  resolver: PathResolver,
  list_meta_files: Callable[[str], list[str]],
) -> list[_EpisodeSlice]:
  """Read all episode rows from ``meta/episodes/chunk-*/file-*.parquet``.

  v3 stores episode metadata across one or more parquet files. ``dataset_id``
  is the HF repo id or a local path label (only used for error messages);
  ``resolver`` and ``list_meta_files`` decide where the data comes from.
  """
  files = list_meta_files(dataset_id)
  if not files:
    raise RuntimeError(f"{dataset_id}: no meta/episodes/*.parquet files found")

  vcol_chunk = f"videos/{video_key}/chunk_index"
  vcol_file = f"videos/{video_key}/file_index"
  vcol_from = f"videos/{video_key}/from_timestamp"
  vcol_to = f"videos/{video_key}/to_timestamp"

  slices: list[_EpisodeSlice] = []
  for rel in sorted(files):
    p = resolver(rel)
    table = pq.read_table(p, columns=[
      "episode_index", "tasks", "length",
      "data/chunk_index", "data/file_index",
      "dataset_from_index", "dataset_to_index",
      vcol_chunk, vcol_file, vcol_from, vcol_to,
    ])
    d = table.to_pydict()
    for i in range(table.num_rows):
      tasks = d["tasks"][i] or []
      slices.append(_EpisodeSlice(
        episode_index=int(d["episode_index"][i]),
        data_from=int(d["dataset_from_index"][i]),
        data_to=int(d["dataset_to_index"][i]),
        video_from_ts=float(d[vcol_from][i]),
        video_to_ts=float(d[vcol_to][i]),
        data_chunk=int(d["data/chunk_index"][i]),
        data_file=int(d["data/file_index"][i]),
        video_chunk=int(d[vcol_chunk][i]),
        video_file=int(d[vcol_file][i]),
        task=str(tasks[0]) if tasks else "",
        length=int(d["length"][i]),
      ))
  slices.sort(key=lambda s: s.episode_index)
  return slices


def _decode_video_range(
  video_path: str, from_ts: float, to_ts: float, expected: int,
) -> np.ndarray:
  """Decode frames in ``[from_ts, to_ts)``; return a ``(T, H, W, 3)`` uint8 stack.

  ``expected`` is the row count from the parquet — we trim or warn if the
  decoded count doesn't match exactly. PyAV's seek lands on the nearest
  keyframe ≤ ``from_ts``; we drop any frames whose pts is earlier than
  ``from_ts`` after the seek.
  """
  container = av.open(video_path)
  try:
    stream = container.streams.video[0]
    tb = stream.time_base
    assert tb is not None

    seek_pts = int(from_ts / float(tb))
    container.seek(seek_pts, any_frame=False, backward=True, stream=stream)

    frames: list[np.ndarray] = []
    eps = 1e-4  # 0.1 ms tolerance for float vs rational pts
    for frame in container.decode(stream):
      assert frame.pts is not None
      t = float(frame.pts * tb)
      if t + eps < from_ts:
        continue
      if t + eps >= to_ts:
        break
      frames.append(frame.to_ndarray(format="rgb24"))
      if len(frames) >= expected:
        break
  finally:
    container.close()

  if not frames:
    raise RuntimeError(f"decoded zero frames from {video_path} for [{from_ts}, {to_ts})")
  if len(frames) != expected:
    logger.warning(
      "frame count mismatch in %s [%.3f, %.3f): decoded %d, expected %d",
      Path(video_path).name, from_ts, to_ts, len(frames), expected,
    )
  return np.stack(frames[:expected], axis=0)


def _select_video_key(
  dataset_id: str,
  preferred: str | None,
  resolver: PathResolver,
) -> str:
  """Pick a video stream key from ``meta/info.json`` features."""
  import json

  info_path = resolver("meta/info.json")
  with open(info_path) as f:
    info = json.load(f)
  repo_id = dataset_id  # alias for error messages
  video_keys = [k for k, v in info.get("features", {}).items()
                if isinstance(v, dict) and v.get("dtype") == "video"]
  if not video_keys:
    raise RuntimeError(f"{repo_id}: no video features in meta/info.json")
  if preferred is None:
    return str(video_keys[0])
  if preferred not in video_keys:
    raise ValueError(
      f"{repo_id}: video key {preferred!r} not in dataset (available: {video_keys})")
  return preferred


def _hf_resolver(repo_id: str) -> PathResolver:
  return lambda rel: hf_hub_download(repo_id, rel, repo_type="dataset")


def _hf_list_meta(repo_id: str) -> list[str]:
  from huggingface_hub import HfApi
  api = HfApi()
  return [
    f for f in api.list_repo_files(repo_id, repo_type="dataset")
    if f.startswith("meta/episodes/") and f.endswith(".parquet")
  ]


def _local_resolver(root: Path) -> PathResolver:
  def resolve(rel: str) -> str:
    p = root / rel
    if not p.exists():
      raise FileNotFoundError(f"local dataset missing file: {p}")
    return str(p)
  return resolve


def import_dataset(
  repo_id: str,
  output_dir: str,
  *,
  format: str = "hdf5",
  video_key: str | None = None,
  max_episodes: int | None = None,
  tags: tuple[str, ...] = (),
  with_ee_pos: bool = True,
  with_object_pose: bool = True,
  name_prefix: str | None = None,
  episode_filter: set[int] | None = None,
  arm_calibration: dict | None = None,
) -> Iterator[tuple[int, Path]]:
  """Yield ``(episode_index, output_path)`` per converted episode.

  ``repo_id`` accepts either an HF dataset repo id (``"user/dataset"``) or
  the path to an on-disk LeRobot v3 dataset directory. The latter is
  detected by ``Path(repo_id).is_dir()``.

  Two post-import stages from :mod:`chuck_dreamer.sim.lerobot_pipeline`
  run in order on each assembled episode before it is written:

    * ``with_ee_pos`` (default True) rescales ``joint_qpos`` to radians
      and runs the FK MLP to fill ``ee_pos``, ``ee_quat`` and
      ``ee_action``. Trainers select between ``joint_action`` and
      ``ee_action`` based on ``cfg.env.act_mode``.
    * ``with_object_pose`` (default True) runs SAM2 + per-frame pose
      fit + RTS smoothing to fill ``object_xy`` and
      ``object_gap_too_long``. Requires per-dataset calibration and a
      cached frame-0 prompt; missing either raises.

  ``tags`` are stamped onto each written episode's metadata. The
  importer is the canonical way to mark recordings as e.g. ``"real"``
  so the replay buffer's tag-protection and tag-weighting can pick
  them up later (see :class:`ReplayBuffer`).

  Generator so callers can wrap in ``tqdm`` and show progress. The
  returned path points at the file produced by :class:`HDF5EpisodeWriter`
  or :class:`RerunEpisodeWriter`.
  """
  from .lerobot_pipeline import apply_ee_pos, apply_object_pose

  local_root: Path | None = None
  if Path(repo_id).is_dir():
    local_root = Path(repo_id)
    resolver = _local_resolver(local_root)
    list_meta = lambda _id: _list_local_meta_episodes(local_root)  # noqa: E731
  else:
    resolver = _hf_resolver(repo_id)
    list_meta = _hf_list_meta

  resolved_video_key = _select_video_key(repo_id, video_key, resolver)
  slices             = _read_episodes_meta(
    repo_id, resolved_video_key, resolver, list_meta)
  if episode_filter is not None:
    slices = [s for s in slices if s.episode_index in episode_filter]
  if max_episodes is not None:
    slices = slices[:max_episodes]
  if not slices:
    raise RuntimeError(f"{repo_id}: no episodes to import")

  writer = EpisodeWriter(output_dir, format=format)

  # Cache the parquet + video downloads keyed by (chunk, file) so we don't
  # re-download or re-read the parquet on every episode.
  parquet_cache: dict[tuple[int, int], dict] = {}
  video_cache: dict[tuple[int, int], str] = {}

  for sl in slices:
    pkey = (sl.data_chunk, sl.data_file)
    if pkey not in parquet_cache:
      pq_path = resolver(_data_path(*pkey))
      tbl = pq.read_table(pq_path, columns=[
        "action", "observation.state", "timestamp",
        "frame_index", "episode_index",
      ])
      parquet_cache[pkey] = tbl.to_pydict()
    cols = parquet_cache[pkey]

    vkey = (sl.video_chunk, sl.video_file)
    if vkey not in video_cache:
      video_cache[vkey] = resolver(_video_path(resolved_video_key, *vkey))
    video_local = video_cache[vkey]

    s, e = sl.data_from, sl.data_to
    action = np.asarray(cols["action"][s:e], dtype=np.float32)
    state = np.asarray(cols["observation.state"][s:e], dtype=np.float32)
    timestamp = np.asarray(cols["timestamp"][s:e], dtype=np.float32)
    T = action.shape[0]
    if T != sl.length:
      logger.warning(
        "episode %d: parquet length %d != meta length %d", sl.episode_index, T, sl.length)

    images = _decode_video_range(
      video_local, sl.video_from_ts, sl.video_to_ts, expected=T)
    if images.shape[0] != T:
      T = min(T, images.shape[0])
      action = action[:T]
      state = state[:T]
      timestamp = timestamp[:T]
      images = images[:T]

    n_joints = state.shape[1]
    episode = {
      "image":        images,
      "joint_action": action,
      "reward":       np.zeros((T,),          dtype=np.float32),
      "timestamp":    timestamp,
      "joint_qpos":   state,
      "ee_pos":       np.zeros((T, 3),        dtype=np.float32),
      "ee_quat":      np.zeros((T, 4),        dtype=np.float32),
      "object_xy":    np.zeros((T, 2),        dtype=np.float32),
    }
    metadata = {
      "config": {
        "source_repo":   repo_id,
        "video_key":     resolved_video_key,
        "episode_index": sl.episode_index,
        "task":          sl.task,
        "n_joints":      n_joints,
      },
      "seed":    sl.episode_index,
      "source":  f"lerobot:{repo_id}",
      "outcome": "imported",
    }
    if tags:
      metadata["tags"] = tuple(tags)
    if arm_calibration is not None:
      # Same shape as the live calibration writes — see
      # `docs/calibrate_live_arm_brief.md`.
      metadata["T_world_arm"]    = arm_calibration["T_world_arm"]
      metadata["arm_diagnostics"] = arm_calibration["diagnostics"]
      metadata["arm_metadata"]    = arm_calibration["metadata"]

    if with_ee_pos:
      apply_ee_pos(episode, metadata)
    if with_object_pose:
      apply_object_pose(episode, metadata)
    if name_prefix:
      suffix = f"{name_prefix}-{sl.episode_index:05d}"
    else:
      suffix = f"{sl.episode_index:05d}"
    out_path = writer.write_episode(
      episode, metadata=metadata, name_suffix=suffix)
    yield sl.episode_index, out_path
