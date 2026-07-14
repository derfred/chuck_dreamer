"""Unit tests for the M3 camera sensors.

``SimCameraSensor`` is tested against a real headless ``MujocoBackend`` (skipped
when MuJoCo is absent); ``WebcamSensor`` is tested with a monkeypatched
``cv2.VideoCapture`` so it needs no real camera and no OpenCV at import time.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from chuck_dreamer.runtime.modalities import IMAGE
from chuck_dreamer.runtime.sensors import SimCameraSensor, WebcamSensor


# -- SimCameraSensor ----------------------------------------------------------


def test_sim_camera_renders_configured_size():
  pytest.importorskip("mujoco")
  from chuck_dreamer.config import load_config
  from chuck_dreamer.runtime.mujoco_backend import MujocoBackend

  backend = MujocoBackend.from_config(load_config(), viewer=False, realtime=False)
  sensor = SimCameraSensor(backend, height=48, width=64, name="camera/front")
  assert sensor.produces == (IMAGE,)
  assert sensor.latest() is None       # nothing before start

  sensor.start()
  try:
    snap = sensor.latest()
    assert snap is not None
    img = snap[IMAGE]
    assert img.shape == (48, 64, 3)
    assert img.dtype == np.uint8
  finally:
    sensor.stop()
    backend.stop()
  assert sensor.latest() is None       # nothing after stop


def test_sim_camera_from_config_parses_render_size():
  pytest.importorskip("mujoco")
  from chuck_dreamer.config import load_config
  from chuck_dreamer.runtime.mujoco_backend import MujocoBackend

  backend = MujocoBackend.from_config(load_config(), viewer=False, realtime=False)
  sensor = SimCameraSensor.from_config(
    load_config(), backend=backend, render_size="32x40", name="camera/front")
  sensor.start()
  try:
    assert sensor.latest()[IMAGE].shape == (32, 40, 3)
  finally:
    sensor.stop()
    backend.stop()


def test_sim_camera_rejects_non_mujoco_backend():
  class _Bare:
    pass

  with pytest.raises(TypeError):
    SimCameraSensor(_Bare())


# -- WebcamSensor -------------------------------------------------------------


class _FakeCapture:
  def __init__(self, index):
    self.index = index
    self.released = False

  def set(self, *_a):
    return True

  def read(self):
    # BGR frame; the sensor converts to RGB.
    return True, np.full((4, 4, 3), [10, 20, 30], dtype=np.uint8)

  def release(self):
    self.released = True


def _install_fake_cv2(monkeypatch):
  fake = types.ModuleType("cv2")
  fake.VideoCapture = _FakeCapture
  fake.CAP_PROP_FRAME_HEIGHT = 4
  fake.CAP_PROP_FRAME_WIDTH = 3
  fake.COLOR_BGR2RGB = 4

  def cvtColor(frame, code):  # noqa: N802
    return frame[..., ::-1].copy()   # BGR -> RGB

  fake.cvtColor = cvtColor
  monkeypatch.setitem(sys.modules, "cv2", fake)
  return fake


def test_webcam_latest_non_blocking_and_converts_bgr(monkeypatch):
  _install_fake_cv2(monkeypatch)
  sensor = WebcamSensor(device_index=0, height=4, width=4, poll_rate_hz=200.0)
  assert sensor.latest() is None       # before first frame
  sensor.start()
  try:
    deadline_met = False
    import time
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
      snap = sensor.latest()
      if snap is not None:
        deadline_met = True
        break
      time.sleep(0.005)
    assert deadline_met
    img = snap[IMAGE]
    assert img.shape == (4, 4, 3)
    # BGR [10,20,30] -> RGB [30,20,10]
    np.testing.assert_array_equal(img[0, 0], [30, 20, 10])
  finally:
    sensor.stop()
