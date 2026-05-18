"""Tests for :mod:`chuck_dreamer.real.config_builders`.

Camera auto-detect is exercised via a monkeypatched ``cv2.VideoCapture``.
``build_record_config`` is asserted to map cfg fields onto the lerobot
dataclasses correctly without touching real hardware.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from chuck_dreamer.real import config_builders as cb


@pytest.fixture
def base_cfg():
  return OmegaConf.create({
    "recording": {
      "robot":  {"type": "so101_follower", "port": "/dev/null", "id": None},
      "teleop": {"type": "so101_leader",   "port": "/dev/null", "id": None},
      "cameras": {
        "top": {
          "type": "opencv",
          "index_or_path": 0,
          "width": "auto",
          "height": "auto",
          "fps": "auto",
        },
      },
      "cameras_defaults": {"fallback_fps": 30},
      "dataset": {
        "repo_id":            "local/test",
        "single_task":        "push",
        "fps":                30,
        "episode_time_s":     5.0,
        "reset_time_s":       2.0,
        "num_episodes":       1,
        "push_to_hub":        False,
        "streaming_encoding": False,
        "encoder_threads":    None,
      },
      "scene": {
        "primary_camera": "top",
        "checkerboard": {"cols": 9, "rows": 6, "square_size_m": 0.025},
        "min_calibration_captures":   5,
        "marker_locations":           ["off_screen", "start"],
        "marker_clip_duration_s":     1.0,
        "marker_clip_min_blob_frames": 0.8,
        "save_marker_raw_frames":     True,
      },
    },
  })


class _FakeVideoCapture:
  """Stand-in for ``cv2.VideoCapture`` returning fixed device properties."""

  def __init__(self, _id, w, h, fps):
    self._w, self._h, self._fps = w, h, fps

  def isOpened(self): return True

  def get(self, prop):
    import cv2
    return {
      cv2.CAP_PROP_FRAME_WIDTH:  self._w,
      cv2.CAP_PROP_FRAME_HEIGHT: self._h,
      cv2.CAP_PROP_FPS:          self._fps,
    }[prop]

  def release(self): pass


def _patch_capture(monkeypatch, *, w=640, h=480, fps=30.0):
  import cv2
  monkeypatch.setattr(cv2, "VideoCapture", lambda idx: _FakeVideoCapture(idx, w, h, fps))


# ---------------------------------------------------------------------------
# resolve_camera_specs
# ---------------------------------------------------------------------------


def test_resolve_camera_specs_auto_probes_native_values(monkeypatch, base_cfg):
  _patch_capture(monkeypatch, w=1280, h=720, fps=60.0)
  resolved = cb.resolve_camera_specs(base_cfg)
  spec = resolved["top"]
  assert spec.width == 1280
  assert spec.height == 720
  assert spec.fps == 60
  # cfg should be updated in-place so downstream code sees concrete ints.
  assert base_cfg.recording.cameras.top.width == 1280
  assert base_cfg.recording.cameras.top.fps == 60


def test_resolve_camera_specs_falls_back_when_fps_zero(monkeypatch, base_cfg):
  _patch_capture(monkeypatch, w=640, h=480, fps=0.0)
  resolved = cb.resolve_camera_specs(base_cfg)
  assert resolved["top"].fps == 30  # from cameras_defaults.fallback_fps


def test_resolve_camera_specs_honours_explicit_overrides(monkeypatch, base_cfg):
  _patch_capture(monkeypatch, w=1280, h=720, fps=60.0)
  base_cfg.recording.cameras.top.width = 320
  base_cfg.recording.cameras.top.height = 240
  resolved = cb.resolve_camera_specs(base_cfg)
  assert resolved["top"].width == 320
  assert resolved["top"].height == 240
  # fps was still auto.
  assert resolved["top"].fps == 60


def test_resolve_camera_specs_skip_probe_when_all_concrete(monkeypatch, base_cfg):
  # Hard-fail if cv2.VideoCapture is touched.
  import cv2

  def _bomb(_):
    raise AssertionError("VideoCapture must not be called when no 'auto' fields")
  monkeypatch.setattr(cv2, "VideoCapture", _bomb)

  base_cfg.recording.cameras.top.width = 320
  base_cfg.recording.cameras.top.height = 240
  base_cfg.recording.cameras.top.fps = 25
  resolved = cb.resolve_camera_specs(base_cfg)
  assert resolved["top"].fps == 25


def test_resolve_camera_specs_rejects_unknown_type(monkeypatch, base_cfg):
  base_cfg.recording.cameras.top.type = "realsense"
  with pytest.raises(NotImplementedError):
    cb.resolve_camera_specs(base_cfg)


# ---------------------------------------------------------------------------
# build_record_config end-to-end
# ---------------------------------------------------------------------------


def test_build_record_config_wires_fields(monkeypatch, base_cfg, tmp_path):
  _patch_capture(monkeypatch, w=640, h=480, fps=30.0)
  rec = cb.build_record_config(base_cfg, dataset_root=tmp_path / "ds")
  # Robot — so101_follower and so100_follower share one class; either name OK.
  assert rec.robot.type in ("so101_follower", "so100_follower")
  assert rec.robot.port == "/dev/null"
  assert "top" in rec.robot.cameras
  assert rec.robot.cameras["top"].width == 640
  # Dataset
  assert rec.dataset.repo_id == "local/test"
  assert rec.dataset.single_task == "push"
  assert rec.dataset.fps == 30
  assert rec.dataset.num_episodes == 1
  assert Path(rec.dataset.root).name == "ds"
  # Teleop — same shared-class quirk applies.
  assert rec.teleop is not None
  assert rec.teleop.type in ("so101_leader", "so100_leader")
