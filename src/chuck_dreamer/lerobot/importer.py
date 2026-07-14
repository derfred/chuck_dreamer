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
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from chuck_dreamer.common.episode import Episode
from chuck_dreamer.common.episode_writer import EpisodeWriter

if TYPE_CHECKING:
  from chuck_dreamer.lerobot.episode_spec import EpisodeSlice
  from .pipeline import Run

logger = logging.getLogger(__name__)


def assemble_episode(
  run: Run,
  sl: EpisodeSlice,
  *,
  tags: tuple[str, ...] = (),
  name_prefix: str | None = None,
) -> tuple[Episode, dict[str, Any], str] | None:
  """Decode one episode and build its ``(episode, metadata, name_suffix)``,
  *before* any analysis stage runs. Returns ``None`` if no frames decoded.

  This is the shared assembler for both the serial loop and the parallel
  producer, so the :class:`Episode` / metadata construction lives in
  exactly one place. The caller is responsible for running the stages (serially
  in :func:`import_dataset`, split across producer/worker lanes in
  :mod:`chuck_dreamer.lerobot.parallel`) and writing the result."""
  frames = run.episode_frames(sl.episode_index)
  if frames is None:
    logger.warning("episode %d: no frames decoded, skipping", sl.episode_index)
    return None

  T = frames.length
  if T != sl.length:
    logger.warning(
      "episode %d: decoded length %d != meta length %d",
      sl.episode_index, T, sl.length)

  episode = Episode.from_arrays({
    "image":        frames.images,
    "joint_action": frames.action,
    "reward":       np.zeros((T,),          dtype=np.float32),
    "timestamp":    frames.timestamp,
    "joint_qpos":   frames.state,
    "ee_pos":       np.zeros((T, 3),        dtype=np.float32),
    "ee_quat":      np.zeros((T, 4),        dtype=np.float32),
    "object_xy":    np.zeros((T, 2),        dtype=np.float32),
  })
  metadata = {
    "config": {
      "source_repo":   run.dataset_id,
      "video_key":     run.video_key,
      "episode_index": sl.episode_index,
      "task":          sl.task,
      "n_joints":      frames.n_joints,
    },
    "seed":    sl.episode_index,
    "source":  f"lerobot:{run.dataset_id}",
    "outcome": "imported",
    "number_of_frames": T,
  }
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

  if name_prefix:
    name_suffix = f"{name_prefix}-{sl.episode_index:05d}"
  else:
    name_suffix = f"{sl.episode_index:05d}"

  return episode, metadata, name_suffix


def import_dataset(
  run: Run,
  output_dir: str,
  *,
  format: str = "hdf5",
  tags: tuple[str, ...] = (),
  name_prefix: str | None = None,
  jobs: int = 1,
  devices: list[str] | None = None,
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

  ``jobs`` > 1 runs the parallel pipeline (see
  :mod:`chuck_dreamer.lerobot.parallel`): GPU producer(s) stream episodes
  through decode + ``ee_pos`` + SAM2 segmentation while a pool of ``jobs`` CPU
  worker processes runs the ``object_pose`` fit and writes. ``devices`` pins one
  producer per GPU (e.g. ``["cuda:0", "cuda:1"]``). ``jobs == 1`` (the default)
  is the serial path below, byte-for-byte unchanged — the correctness anchor.
  Per-episode results are identical across ``jobs`` because the only stage with
  cross-frame state (``object_pose``'s warm-start) runs *within* one episode,
  and parallelism is only *across* episodes.

  Generator so callers can wrap in ``tqdm`` and show progress. The
  returned path points at the file produced by :class:`HDF5EpisodeWriter`
  or :class:`RerunEpisodeWriter`.
  """
  slices = run.slices
  if not slices:
    raise RuntimeError(f"{run.dataset_id}: no episodes to import")

  if jobs != 1:
    from .parallel import import_dataset_parallel
    yield from import_dataset_parallel(
      run, output_dir, format=format, tags=tags, name_prefix=name_prefix,
      jobs=jobs, devices=devices)
    return

  writer = EpisodeWriter(output_dir, format=format)

  for sl in slices:
    assembled = assemble_episode(
      run, sl, tags=tags, name_prefix=name_prefix)
    if assembled is None:
      continue
    episode, metadata, name_suffix = assembled

    for stage in run.pipeline(sl.episode_index):
      stage.apply(episode, metadata)

    out_path = writer.write_episode(
      episode, metadata=metadata, name_suffix=name_suffix)
    yield sl.episode_index, out_path
