"""Session storage backends.

Both backends share the same ``scene/`` subtree layout for phase A; they
differ only in how phase B's lerobot output is preserved:

  * :class:`LeRobotBackend` keeps the lerobot dataset directory in place
    under ``root/dataset/``.
  * :class:`EpisodesBackend` lets phase B write into a scratch
    ``root/_lerobot_tmp/`` directory and, on :meth:`finalize`, runs
    :func:`chuck_dreamer.sim.lerobot_import.import_dataset` against the
    local path to produce ``root/episodes/episode-NNNNN.{hdf5,rrd}``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .scene_types import (
  CalibrationCapture,
  CameraIntrinsics,
  MarkerClip,
  SceneRegistration,
)

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
  return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _write_intrinsics(dir_: Path, name: str, intr: CameraIntrinsics) -> None:
  """Write one camera's intrinsics + per-capture images to ``dir_``."""
  cam_dir = dir_ / name
  captures_dir = cam_dir / "captures"
  captures_dir.mkdir(parents=True, exist_ok=True)
  import cv2

  descriptions: list[dict[str, Any]] = []
  for i, cap in enumerate(intr.captures):
    if cap.image_path.exists() and cap.image_path.parent != captures_dir:
      shutil.copyfile(cap.image_path, captures_dir / cap.image_path.name)
    descriptions.append({
      "index": i,
      "image": cap.image_path.name,
      "description": cap.description,
      "n_corners": int(cap.corners_px.shape[0]),
    })
  (cam_dir / "captures" / "descriptions.json").write_text(
    json.dumps(descriptions, indent=2))

  (dir_ / f"{name}.json").write_text(json.dumps({
    "K":            intr.K.tolist(),
    "dist":         intr.dist.tolist(),
    "reproj_error": float(intr.reproj_error),
    "image_size":   list(intr.image_size),
    "n_captures":   len(intr.captures),
  }, indent=2))
  _ = cv2  # silence "unused" lint


def _write_marker_clip(
  dir_: Path,
  clip: MarkerClip,
  *,
  save_raw_frames: bool,
) -> None:
  """Persist one marker clip as mp4 + .npz blob summary (+ optional raw frames)."""
  import cv2

  loc_dir = dir_ / clip.loc_name
  loc_dir.mkdir(parents=True, exist_ok=True)

  T, H, W, _ = clip.frames.shape
  mp4_path = loc_dir / "frames.mp4"
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = cv2.VideoWriter(str(mp4_path), fourcc, float(clip.fps), (W, H))
  if not writer.isOpened():
    logger.warning("marker %s: VideoWriter failed to open; skipping mp4", clip.loc_name)
  else:
    for t in range(T):
      writer.write(cv2.cvtColor(clip.frames[t], cv2.COLOR_RGB2BGR))
    writer.release()

  np.savez(
    loc_dir / "blob.npz",
    centroid_xy=clip.blob_centroids_px,
    radius_px=clip.blob_radii_px,
    detected=clip.blob_detected,
  )
  if save_raw_frames:
    np.savez_compressed(loc_dir / "frames.npz", frames=clip.frames)
  (loc_dir / "meta.json").write_text(json.dumps({
    "loc_name":          clip.loc_name,
    "duration_s":        clip.duration_s,
    "fps":               clip.fps,
    "n_frames":          int(T),
    "image_size":        [W, H],
    "detected_fraction": clip.detected_fraction,
    "save_raw_frames":   save_raw_frames,
  }, indent=2))


class SessionStorage(ABC):
  """Common interface for both phase-B storage backends."""

  def __init__(self, root: Path, *, save_marker_raw_frames: bool = True) -> None:
    self.root = Path(root)
    self.root.mkdir(parents=True, exist_ok=True)
    self.scene_dir = self.root / "scene"
    self.save_marker_raw_frames = save_marker_raw_frames
    self._session_started_at = _utc_now_iso()

  # --- phase A ---

  def write_scene(self, scene: SceneRegistration) -> None:
    intr_dir = self.scene_dir / "intrinsics"
    intr_dir.mkdir(parents=True, exist_ok=True)
    for name, intr in scene.intrinsics.items():
      _write_intrinsics(intr_dir, name, intr)

    markers_dir = self.scene_dir / "markers"
    markers_dir.mkdir(parents=True, exist_ok=True)
    for clip in scene.markers:
      _write_marker_clip(markers_dir, clip, save_raw_frames=self.save_marker_raw_frames)

  # --- phase B ---

  @abstractmethod
  def episodes_root(self) -> Path:
    """Directory passed to upstream lerobot record as ``dataset.root``."""

  @abstractmethod
  def kind(self) -> str:
    """Short identifier (e.g. ``"lerobot"`` / ``"episodes"``) for the manifest."""

  def finalize(self, *, recording_cfg_snapshot: dict[str, Any] | None = None) -> None:
    """Write the top-level ``session.json`` manifest. Subclasses extend."""
    manifest = {
      "started_at":  self._session_started_at,
      "finished_at": _utc_now_iso(),
      "storage":     self.kind(),
      "recording":   recording_cfg_snapshot or {},
    }
    (self.root / "session.json").write_text(json.dumps(manifest, indent=2))


class LeRobotBackend(SessionStorage):
  """Phase B writes a LeRobotDataset under ``root/dataset/`` and we keep it."""

  def episodes_root(self) -> Path:
    p = self.root / "dataset"
    p.mkdir(parents=True, exist_ok=True)
    return p

  def kind(self) -> str:
    return "lerobot"


class EpisodesBackend(SessionStorage):
  """Phase B writes to ``root/_lerobot_tmp/``; finalize re-imports into ``root/episodes/``."""

  def __init__(
    self,
    root: Path,
    *,
    save_marker_raw_frames: bool = True,
    episodes_format: str = "rerun",
  ) -> None:
    super().__init__(root, save_marker_raw_frames=save_marker_raw_frames)
    self.episodes_format = episodes_format

  def episodes_root(self) -> Path:
    p = self.root / "_lerobot_tmp"
    p.mkdir(parents=True, exist_ok=True)
    return p

  def kind(self) -> str:
    return "episodes"

  def finalize(self, *, recording_cfg_snapshot: dict[str, Any] | None = None) -> None:
    # Import the scratch dataset into the canonical episodes layout.
    from chuck_dreamer.sim.lerobot_import import import_dataset
    out_dir = self.root / "episodes"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = self.episodes_root()
    n = 0
    try:
      for _ep_idx, _path in import_dataset(
        str(scratch), str(out_dir),
        format=self.episodes_format, max_episodes=None,
      ):
        n += 1
    except Exception as exc:
      logger.error("episodes import failed: %s", exc)
      raise
    else:
      shutil.rmtree(scratch, ignore_errors=True)
    finally:
      super().finalize(recording_cfg_snapshot=recording_cfg_snapshot)
    logger.info("imported %d episodes into %s", n, out_dir)


def build_storage(
  kind: str,
  root: Path,
  *,
  save_marker_raw_frames: bool = True,
  episodes_format: str = "rerun",
) -> SessionStorage:
  """Factory used by the CLI / recorder."""
  if kind == "lerobot":
    return LeRobotBackend(root, save_marker_raw_frames=save_marker_raw_frames)
  if kind == "episodes":
    return EpisodesBackend(
      root,
      save_marker_raw_frames=save_marker_raw_frames,
      episodes_format=episodes_format,
    )
  raise ValueError(f"unknown storage kind {kind!r} (expected 'lerobot' or 'episodes')")


__all__ = [
  "CalibrationCapture",
  "CameraIntrinsics",
  "MarkerClip",
  "SceneRegistration",
  "SessionStorage",
  "LeRobotBackend",
  "EpisodesBackend",
  "build_storage",
]
