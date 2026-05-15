"""Top-level orchestrator for ``chuck-dreamer record``.

Sequence:

  1. Resolve cameras (auto-detect width/height/fps where needed).
  2. Open cameras as :class:`OpenCVFrameSource` adapters.
  3. Run phase A (calibration + marker clips) and write its results
     through the storage backend.
  4. Disconnect phase-A cameras (lerobot will re-open them).
  5. Build a :class:`lerobot.scripts.lerobot_record.RecordConfig` from the
     same resolved camera specs and call upstream ``record(cfg)``.
  6. Storage backend finalises (writes manifest; for ``EpisodesBackend``
     this also runs the local-path importer).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from .config_builders import (
  build_camera_configs,
  build_record_config,
  resolve_camera_specs,
  snapshot_recording_config,
)
from .scene_registration import (
  CheckerboardSpec,
  MarkerCaptureSpec,
  OpenCVFrameSource,
  default_ui,
  register_scene,
)
from .storage import build_storage

logger = logging.getLogger(__name__)


def _open_phase_a_cameras(resolved: dict, cam_cfgs: dict) -> dict[str, OpenCVFrameSource]:
  """Build :class:`OpenCVFrameSource` for each resolved camera."""
  sources: dict[str, OpenCVFrameSource] = {}
  try:
    for name in resolved:
      sources[name] = OpenCVFrameSource(name=name, camera_cfg=cam_cfgs[name])
  except Exception:
    for s in sources.values():
      s.disconnect()
    raise
  return sources


def run(
  cfg: DictConfig,
  output: Path,
  *,
  storage: str = "lerobot",
  episodes_format: str = "rerun",
  skip_calibration: bool = False,
  skip_markers: bool = False,
  ui: Any = None,
  record_fn: Any = None,
) -> Path:
  """Run a full record session and return the session output directory.

  ``record_fn`` defaults to :func:`lerobot.scripts.lerobot_record.record`
  but is overridable so tests can short-circuit phase B.
  """
  output = Path(output)
  scene_cfg = cfg.recording.scene

  storage_backend = build_storage(
    storage,
    output,
    save_marker_raw_frames=bool(scene_cfg.get("save_marker_raw_frames", True)),
    episodes_format=episodes_format,
  )

  resolved = resolve_camera_specs(cfg)
  cam_cfgs = build_camera_configs(resolved)
  sources = _open_phase_a_cameras(resolved, cam_cfgs)

  try:
    primary_name = str(scene_cfg.primary_camera)
    checker_cfg = scene_cfg.checkerboard
    checker = CheckerboardSpec(
      cols=int(checker_cfg.cols),
      rows=int(checker_cfg.rows),
      square_size_m=float(checker_cfg.square_size_m),
    )
    markers = MarkerCaptureSpec(
      locations=tuple(str(s) for s in scene_cfg.marker_locations),
      clip_duration_s=float(scene_cfg.marker_clip_duration_s),
      min_blob_fraction=float(scene_cfg.marker_clip_min_blob_frames),
    )
    scene = register_scene(
      sources, primary_name, checker, markers,
      scene_dir=storage_backend.scene_dir,
      min_calibration_captures=int(scene_cfg.get("min_calibration_captures", 5)),
      skip_calibration=skip_calibration,
      skip_markers=skip_markers,
      ui=ui or default_ui(),
    )
    storage_backend.write_scene(scene)
  finally:
    for s in sources.values():
      s.disconnect()

  # Phase B — hand off to upstream lerobot. The cameras are reopened by
  # the upstream factory using the same OpenCVCameraConfig instances.
  rec_cfg = build_record_config(
    cfg,
    dataset_root=storage_backend.episodes_root(),
    resolved_cameras=resolved,
  )
  if record_fn is None:
    from lerobot.scripts.lerobot_record import record as record_fn  # type: ignore[no-redef]
  record_fn(rec_cfg)

  storage_backend.finalize(recording_cfg_snapshot=snapshot_recording_config(cfg))
  return output
