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
from pathlib import Path
from typing import Callable, Iterator

import av  # type: ignore[import-untyped]
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from chuck_dreamer.common.episode_spec import EpisodeSpec
from chuck_dreamer.sim.episode_writer import EpisodeWriter

logger = logging.getLogger(__name__)


# A resolver maps a dataset-relative path (e.g. "meta/info.json") to a local
# file on disk. The HF backend downloads via ``hf_hub_download``; the local
# backend joins onto the dataset root. Both signatures return a ``str`` so
# the caller can hand it to ``cv2``/``pyarrow``/``av`` unchanged.
PathResolver = Callable[[str], str]


def _data_path(chunk: int, file: int) -> str:
  return f"data/chunk-{chunk:03d}/file-{file:03d}.parquet"


def _video_path(video_key: str, chunk: int, file: int) -> str:
  return f"videos/{video_key}/chunk-{chunk:03d}/file-{file:03d}.mp4"


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


def _hf_resolver(repo_id: str) -> PathResolver:
  return lambda rel: hf_hub_download(repo_id, rel, repo_type="dataset")


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
  tags: tuple[str, ...] = (),
  with_ee_pos: bool = True,
  with_object_pose: bool = True,
  name_prefix: str | None = None,
  source: EpisodeSpec | None = None,
  arm_calibration: dict | None = None,
) -> Iterator[tuple[int, Path]]:
  """Yield ``(episode_index, output_path)`` per converted episode.

  ``repo_id`` accepts either an HF dataset repo id (``"user/dataset"``) or
  the path to an on-disk LeRobot v3 dataset directory. The latter is
  detected by ``Path(repo_id).is_dir()``.

  Two post-import stages from :mod:`chuck_dreamer.lerobot.pipeline`
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
  from .stages import StageContext, enabled_from_flags, resolve_stages

  enabled = enabled_from_flags(
    with_ee_pos=with_ee_pos, with_object_pose=with_object_pose)
  stages = resolve_stages(enabled)
  ctx = StageContext(source_repo=repo_id)

  local_root: Path | None = None
  if Path(repo_id).is_dir():
    local_root = Path(repo_id)
    resolver = _local_resolver(local_root)
  else:
    resolver = _hf_resolver(repo_id)

  spec = source if source is not None else EpisodeSpec(dataset_id=repo_id)
  slices, resolved_video_key = spec.read_episodes(video_key=video_key, root=local_root)
  if not slices:
    raise RuntimeError(f"{repo_id}: no episodes to import")

  writer = EpisodeWriter(output_dir, format=format)

  for sl in slices:
    pq_path = resolver(_data_path(sl.data_chunk, sl.data_file))
    cols = pq.read_table(pq_path, columns=[
      "action", "observation.state", "timestamp",
      "frame_index", "episode_index",
    ]).to_pydict()

    video_local = resolver(_video_path(resolved_video_key, sl.video_chunk, sl.video_file))

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

    ctx.masks.clear()
    for stage in stages:
      stage.apply(episode, metadata, ctx)
    if name_prefix:
      suffix = f"{name_prefix}-{sl.episode_index:05d}"
    else:
      suffix = f"{sl.episode_index:05d}"
    out_path = writer.write_episode(
      episode, metadata=metadata, name_suffix=suffix)
    yield sl.episode_index, out_path
