"""Convert a LeRobot v3 HF dataset into the repo's HDF5/Rerun episode format.

LeRobot v3 stores per-frame proprio/action in parquet and camera frames in
MP4s. Rather than parsing those sidecars by hand, the :class:`Run` loads the
dataset through ``LeRobotDataset`` and hands back the decoded, stacked arrays
for one episode as an :class:`EpisodeFrames`. This module groups those into
the repo's episode dict, runs the per-episode stages, and writes the result
via :class:`HDF5EpisodeWriter` / :class:`RerunEpisodeWriter`. It never touches
``lerobot`` or a raw frame dict — all dataset access is behind the run.

``reward`` (absent from teleop data) is zero-filled; the EE / object
coordinate tracks exist only when the pipeline nodes that produce them are
enabled — consumers treat them as optional.
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
  frames = run.episode_frames(sl.episode_index)
  if frames is None:
    logger.warning("episode %d: no frames decoded, skipping", sl.episode_index)
    return None

  T = frames.length
  if T != sl.length:
    logger.warning(
      "episode %d: decoded length %d != meta length %d",
      sl.episode_index, T, sl.length)

  arrays: dict[str, Any] = {
    "joint_action": frames.action,
    "reward":       np.zeros((T,),          dtype=np.float32),
    "timestamp":    frames.timestamp,
    "joint_qpos":   frames.state,
  }
  if frames.images is not None:
    arrays["image"] = frames.images
  episode = Episode.from_arrays(arrays)
  metadata = {
    "config": {
      "source_repo":   run.dataset_id,
      "episode_index": sl.episode_index,
      "task":          sl.task,
      "n_joints":      frames.n_joints,
    },
    "seed":    sl.episode_index,
    "source":  f"lerobot:{run.dataset_id}",
    "outcome": "imported",
    "number_of_frames": T,
  }
  if sl.video_path is not None and frames.images is not None:
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


def drop_video_field(episode: Episode) -> None:
  """Suppress the RGB stack from an episode's *written* output (``--no-video``)."""
  if "image" in episode:
    episode.set("image", episode["image"], persist=False)


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
      and runs the FK to fill ``ee_pos_arm``, ``ee_quat_arm`` and
      ``ee_action_arm`` (mm). Trainers select between ``joint_action`` and
      the EE action based on ``cfg.env.act_mode``.
    * ``with_object_pose`` (default True) runs SAM2 + a per-frame
      analysis-by-synthesis pose fit to fill ``obj_pos_table`` (with its
      ``obj_pos_table.valid`` sibling, and a ``camera/mesh_overlay`` track
      in the Rerun output). Requires the calibration artifacts and a cached
      frame-0 prompt in the store; missing either raises.
    * ``with_table_frame`` adds the frame-crossing nodes (``ee_pos_table``
      / ``ee_quat_table`` / ``ee_action_table`` / ``obj_pos_arm``), which
      need the dataset's ``table_to_arm`` transform in the store.

  ``--no-video`` (``run.drop_video``) omits the RGB stack from the written
  files, which carry the derived tracks at a small fraction of the size (and
  are not usable for image-observation training). When a stage still consumes
  the pixels the frames are decoded as usual and only their persistence is
  switched off, right before the write; when nothing reads them the decode is
  skipped outright — see :attr:`Run.decodes_video`.

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

  from .trackset import EpisodeScope, TrackSet, run_episode

  writer = EpisodeWriter(output_dir, format=format)

  for sl in slices:
    assembled = assemble_episode(run, sl, tags=tags, name_prefix=name_prefix)
    if assembled is not None:
      episode, metadata, name_suffix = assembled

      ts = TrackSet(EpisodeScope(run.dataset_id, sl.episode_index), episode, metadata)
      run_episode(run.pipeline(sl.episode_index), ts)

      if run.drop_video:
        drop_video_field(episode)

      out_path = writer.write_episode(episode, metadata=metadata, name_suffix=name_suffix)
      yield sl.episode_index, out_path
