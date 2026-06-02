"""Convert a LeRobot v3 HF dataset into the repo's HDF5/Rerun episode format.

LeRobot v3 stores per-frame proprio/action in parquet and camera frames in
MP4s. Rather than parsing those sidecars by hand, the :class:`Run` loads the
dataset through ``LeRobotDataset`` and hands back the decoded, stacked arrays
for one episode as an :class:`EpisodeFrames`. This module groups those into
the repo's episode dict, runs the per-episode stages, and writes the result
via :class:`HDF5EpisodeWriter` / :class:`RerunEpisodeWriter`. It never touches
``lerobot`` or a raw frame dict — all dataset access is behind the run.

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
  from .pipeline import Run

logger = logging.getLogger(__name__)


def import_dataset(
  run: Run,
  output_dir: str,
  *,
  format: str = "hdf5",
  tags: tuple[str, ...] = (),
  name_prefix: str | None = None,
) -> Iterator[tuple[int, Path]]:
  """Yield ``(episode_index, output_path)`` per converted episode.

  ``run`` carries the :class:`EpisodeSpec` and the enabled stage flags (the
  caller builds it once and shares it with the doctor). ``run.dataset_id``
  accepts either an HF dataset repo id (``"user/dataset"``) or the path to an
  on-disk LeRobot v3 dataset directory; ``run`` detects the latter and exposes
  it as ``run.local_root``. The run also owns all dataset access: it yields the
  selected ``run.slices`` and the decoded ``run.episode_frames(idx)`` arrays.

  Two post-import stages, resolved per episode by :class:`Run`, run in order
  on each assembled episode before it is written:

    * ``with_ee_pos`` (default True) rescales ``joint_qpos`` to radians
      and runs the FK MLP to fill ``ee_pos``, ``ee_quat`` and
      ``ee_action``. Trainers select between ``joint_action`` and
      ``ee_action`` based on ``cfg.env.act_mode``.
    * ``with_object_pose`` (default True) runs SAM2 + a per-frame
      analysis-by-synthesis pose fit to fill ``object_xy`` and
      ``object_gap_too_long`` (and a ``camera/mesh_overlay`` track in the
      Rerun output). Requires per-dataset calibration and a cached frame-0
      prompt; missing either raises.

  ``tags`` are stamped onto each written episode's metadata. The
  importer is the canonical way to mark recordings as e.g. ``"real"``
  so the replay buffer's tag-protection and tag-weighting can pick
  them up later (see :class:`ReplayBuffer`).

  Generator so callers can wrap in ``tqdm`` and show progress. The
  returned path points at the file produced by :class:`HDF5EpisodeWriter`
  or :class:`RerunEpisodeWriter`.
  """
  slices = run.slices
  resolved_video_key = run.video_key
  if not slices:
    raise RuntimeError(f"{run.dataset_id}: no episodes to import")

  writer = EpisodeWriter(output_dir, format=format)

  for sl in slices:
    frames = run.episode_frames(sl.episode_index)
    if frames is None:
      logger.warning("episode %d: no frames decoded, skipping", sl.episode_index)
      continue

    T = frames.length
    if T != sl.length:
      logger.warning(
        "episode %d: decoded length %d != meta length %d", sl.episode_index, T, sl.length)

    episode = {
      "image":        frames.images,
      "joint_action": frames.action,
      "reward":       np.zeros((T,),          dtype=np.float32),
      "timestamp":    frames.timestamp,
      "joint_qpos":   frames.state,
      "ee_pos":       np.zeros((T, 3),        dtype=np.float32),
      "ee_quat":      np.zeros((T, 4),        dtype=np.float32),
      "object_xy":    np.zeros((T, 2),        dtype=np.float32),
    }
    metadata = {
      "config": {
        "source_repo":   run.dataset_id,
        "video_key":     resolved_video_key,
        "episode_index": sl.episode_index,
        "task":          sl.task,
        "n_joints":      frames.n_joints,
      },
      "seed":    sl.episode_index,
      "source":  f"lerobot:{run.dataset_id}",
      "outcome": "imported",
      "number_of_frames": T,
    }
    # Write-time hint (not persisted): lets the Rerun writer losslessly
    # stream-copy this episode's window out of the source MP4 instead of
    # re-encoding the decoded frames. Only present on the import path.
    if sl.video_path is not None:
      window = float(sl.video_to_ts) - float(sl.video_from_ts)
      metadata["source_video"] = {
        "path":    str(sl.video_path),
        "from_ts": float(sl.video_from_ts),
        "to_ts":   float(sl.video_to_ts),
        "fps":     (sl.length / window) if window > 0 else None,
      }
    if tags:
      metadata["tags"] = tuple(tags)

    for stage in run.pipeline(sl.episode_index):
      stage.apply(episode, metadata)

    if name_prefix:
      suffix = f"{name_prefix}-{sl.episode_index:05d}"
    else:
      suffix = f"{sl.episode_index:05d}"

    out_path = writer.write_episode(episode, metadata=metadata, name_suffix=suffix)
    yield sl.episode_index, out_path
