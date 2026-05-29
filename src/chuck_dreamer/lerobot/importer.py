"""Convert a LeRobot v3 HF dataset into the repo's HDF5/Rerun episode format.

LeRobot v3 stores per-frame proprio/action in parquet and camera frames in
MP4s. Rather than parsing those sidecars by hand, this module loads the
dataset through :class:`lerobot.datasets.lerobot_dataset.LeRobotDataset`,
which decodes video frames and exposes every feature as a per-frame dict.
We group those frames by episode, stack them, and hand the result to
:class:`HDF5EpisodeWriter` / :class:`RerunEpisodeWriter`.

Fields the repo's episode format expects but LeRobot teleop data does not
provide (``reward``, ``ee_pos``, ``ee_quat``, ``object_xy``) are zero-filled.
This is fine for image-mode training; state-mode training would need real
EE / object signals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

from chuck_dreamer.sim.episode_writer import EpisodeWriter

if TYPE_CHECKING:
  from .stages import Run

logger = logging.getLogger(__name__)


def _frame_to_image(frame) -> np.ndarray:
  """LeRobot returns video frames as CHW float32 in [0, 1]; the repo's
  episode format wants HWC uint8. Convert one frame."""
  arr = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
  arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
  return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def import_dataset(
  run: Run,
  output_dir: str,
  *,
  format: str = "hdf5",
  video_key: str | None = None,
  tags: tuple[str, ...] = (),
  name_prefix: str | None = None,
) -> Iterator[tuple[int, Path]]:
  """Yield ``(episode_index, output_path)`` per converted episode.

  ``run`` carries the :class:`EpisodeSpec` and the enabled stage flags (the
  caller builds it once and shares it with the doctor). ``run.spec.dataset_id``
  accepts either an HF dataset repo id (``"user/dataset"``) or the path to an
  on-disk LeRobot v3 dataset directory; ``run`` detects the latter and exposes
  it as ``run.local_root``.

  Two post-import stages, resolved per episode by
  :class:`chuck_dreamer.lerobot.stages.Pipeline`, run in order on each
  assembled episode before it is written:

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
  from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

  spec = run.spec

  # read_slices resolves the video key and applies this spec's episode
  # filter, returning slices in episode-index order. It only touches the
  # meta/ sidecars, so it stays offline for a local dataset root.
  slices, resolved_video_key = run.read_slices(video_key=video_key)
  if not slices:
    raise RuntimeError(f"{spec.dataset_id}: no episodes to import")
  selected = [sl.episode_index for sl in slices]

  ds = LeRobotDataset(
    spec.dataset_id, root=run.local_root, episodes=selected, download_videos=True)

  # LeRobotDataset flattens the selected episodes into a single frame
  # sequence; group consecutive frames by their episode_index.
  frames_by_episode: dict[int, list[dict]] = {idx: [] for idx in selected}
  for i in range(len(ds)):
    item = ds[i]
    frames_by_episode[int(item["episode_index"])].append(item)

  writer = EpisodeWriter(output_dir, format=format)

  for sl in slices:
    frames = frames_by_episode[sl.episode_index]
    if not frames:
      logger.warning("episode %d: no frames decoded, skipping", sl.episode_index)
      continue

    images = np.stack(
      [_frame_to_image(f[resolved_video_key]) for f in frames], axis=0)
    action = np.stack(
      [np.asarray(f["action"], dtype=np.float32) for f in frames], axis=0)
    state = np.stack(
      [np.asarray(f["observation.state"], dtype=np.float32) for f in frames], axis=0)
    timestamp = np.asarray(
      [float(f["timestamp"]) for f in frames], dtype=np.float32)
    T = action.shape[0]
    if T != sl.length:
      logger.warning(
        "episode %d: decoded length %d != meta length %d", sl.episode_index, T, sl.length)

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
        "source_repo":   spec.dataset_id,
        "video_key":     resolved_video_key,
        "episode_index": sl.episode_index,
        "task":          sl.task,
        "n_joints":      n_joints,
      },
      "seed":    sl.episode_index,
      "source":  f"lerobot:{spec.dataset_id}",
      "outcome": "imported",
    }
    if tags:
      metadata["tags"] = tuple(tags)

    for stage in run.pipeline(sl.episode_index):
      stage.apply(episode, metadata)
    if name_prefix:
      suffix = f"{name_prefix}-{sl.episode_index:05d}"
    else:
      suffix = f"{sl.episode_index:05d}"
    out_path = writer.write_episode(
      episode, metadata=metadata, name_suffix=suffix)
    yield sl.episode_index, out_path
