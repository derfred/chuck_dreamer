"""Build lerobot config dataclasses from the project's OmegaConf ``cfg``.

Two responsibilities:

  * Resolve ``"auto"`` width/height/fps on camera configs by probing the
    device once with ``cv2.VideoCapture``. The resolved ints are written
    back into ``cfg.recording.cameras`` so downstream code sees concrete
    values.
  * Translate the ``cfg.recording.*`` subtree into typed lerobot
    dataclasses (``RecordConfig``, ``DatasetRecordConfig``,
    ``RobotConfig`` subclass, ``TeleoperatorConfig`` subclass,
    ``CameraConfig`` dict) for upstream
    :func:`lerobot.scripts.lerobot_record.record`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedCameraSpec:
  """Concrete camera config + the field values we need outside lerobot."""

  name: str
  index_or_path: int | str
  width: int
  height: int
  fps: int
  cfg_type: str   # currently always "opencv"


_AUTO_TOKENS = ("auto", None)


def _probe_opencv_device(index_or_path: int | str) -> tuple[int, int, float]:
  """Open the device with cv2, read native (width, height, fps), close.

  Returns floats from OpenCV; caller coerces to int. fps can legitimately
  be 0 on some webcams, in which case the caller substitutes a fallback.
  """
  import cv2  # local import — keeps test isolation cheap

  cap = cv2.VideoCapture(index_or_path) if isinstance(index_or_path, int) \
        else cv2.VideoCapture(str(index_or_path))
  if not cap.isOpened():
    raise RuntimeError(f"camera probe: cannot open {index_or_path!r}")
  try:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
  finally:
    cap.release()
  if w <= 0 or h <= 0:
    raise RuntimeError(
      f"camera probe: device {index_or_path!r} reported invalid frame size ({w}x{h})")
  return w, h, fps


def resolve_camera_specs(cfg: DictConfig) -> dict[str, ResolvedCameraSpec]:
  """Resolve every camera in ``cfg.recording.cameras``.

  Auto-detect runs once per camera and the resolved ints are written back
  into ``cfg`` so subsequent reads (e.g. the session manifest) see the
  same numbers.
  """
  fallback_fps = int(cfg.recording.get("cameras_defaults", {}).get("fallback_fps", 30))
  resolved: dict[str, ResolvedCameraSpec] = {}

  for name, raw in cfg.recording.cameras.items():
    cam_type = str(raw.get("type", "opencv"))
    if cam_type != "opencv":
      raise NotImplementedError(
        f"camera {name!r}: type {cam_type!r} not yet supported (only 'opencv')")

    index_or_path: Any = raw["index_or_path"]
    need_probe = any(raw.get(k) in _AUTO_TOKENS for k in ("width", "height", "fps"))
    w_in, h_in, fps_in = raw.get("width"), raw.get("height"), raw.get("fps")

    if need_probe:
      pw, ph, pfps = _probe_opencv_device(index_or_path)
      w  = int(w_in)   if w_in   not in _AUTO_TOKENS else pw
      h  = int(h_in)   if h_in   not in _AUTO_TOKENS else ph
      if fps_in in _AUTO_TOKENS:
        if pfps <= 0:
          logger.warning(
            "camera %s: device reported fps=%s; falling back to %d",
            name, pfps, fallback_fps)
          fps = fallback_fps
        else:
          fps = int(round(pfps))
      else:
        fps = int(fps_in)
    else:
      w, h, fps = int(w_in), int(h_in), int(fps_in)

    # Persist the resolved values back into cfg.
    raw["width"], raw["height"], raw["fps"] = w, h, fps

    resolved[name] = ResolvedCameraSpec(
      name=name, index_or_path=index_or_path,
      width=w, height=h, fps=fps, cfg_type=cam_type)
  return resolved


def build_camera_configs(resolved: dict[str, ResolvedCameraSpec]) -> dict[str, Any]:
  """Construct ``OpenCVCameraConfig`` dataclasses from resolved specs."""
  from lerobot.cameras.opencv import OpenCVCameraConfig

  out: dict[str, Any] = {}
  for name, spec in resolved.items():
    out[name] = OpenCVCameraConfig(
      index_or_path=spec.index_or_path,
      width=spec.width,
      height=spec.height,
      fps=spec.fps,
    )
  return out


def build_robot_config(cfg: DictConfig, cameras: dict[str, Any]) -> Any:
  """Return a draccus-registered ``RobotConfig`` subclass instance."""
  rcfg = cfg.recording.robot
  robot_type = str(rcfg.type)
  if robot_type in ("so101_follower", "so100_follower"):
    from lerobot.robots.so_follower import SOFollowerRobotConfig
    return SOFollowerRobotConfig(
      port=str(rcfg.port),
      cameras=cameras,
      id=rcfg.get("id"),
    )
  raise NotImplementedError(f"robot type {robot_type!r} not wired up yet")


def build_teleop_config(cfg: DictConfig) -> Any | None:
  """Return a draccus-registered ``TeleoperatorConfig`` subclass, or None."""
  tcfg = cfg.recording.get("teleop")
  if tcfg is None:
    return None
  teleop_type = str(tcfg.type)
  if teleop_type in ("so101_leader", "so100_leader"):
    from lerobot.teleoperators.so_leader import SOLeaderTeleopConfig
    return SOLeaderTeleopConfig(port=str(tcfg.port), id=tcfg.get("id"))
  raise NotImplementedError(f"teleop type {teleop_type!r} not wired up yet")


def build_record_config(
  cfg: DictConfig,
  *,
  dataset_root: Path,
  resolved_cameras: dict[str, ResolvedCameraSpec] | None = None,
) -> Any:
  """Build the lerobot ``RecordConfig`` from ``cfg`` + a target dataset path.

  If ``resolved_cameras`` is None, the cameras are resolved here. Pass a
  pre-resolved dict when phase A has already probed them so we don't open
  the cameras twice.
  """
  from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig

  if resolved_cameras is None:
    resolved_cameras = resolve_camera_specs(cfg)
  cam_cfgs = build_camera_configs(resolved_cameras)
  robot_cfg = build_robot_config(cfg, cam_cfgs)
  teleop_cfg = build_teleop_config(cfg)

  d = cfg.recording.dataset
  ds_cfg = DatasetRecordConfig(
    repo_id=str(d.repo_id),
    single_task=str(d.single_task),
    root=str(dataset_root),
    fps=int(d.fps),
    episode_time_s=float(d.episode_time_s),
    reset_time_s=float(d.reset_time_s),
    num_episodes=int(d.num_episodes),
    push_to_hub=bool(d.get("push_to_hub", False)),
    streaming_encoding=bool(d.get("streaming_encoding", False)),
    encoder_threads=d.get("encoder_threads"),
  )
  return RecordConfig(
    robot=robot_cfg,
    dataset=ds_cfg,
    teleop=teleop_cfg,
  )


def snapshot_recording_config(cfg: DictConfig) -> dict[str, Any]:
  """Return a JSON-serialisable copy of ``cfg.recording`` for the session manifest."""
  return OmegaConf.to_container(cfg.recording, resolve=True)  # type: ignore[return-value]
