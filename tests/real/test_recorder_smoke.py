"""End-to-end smoke test for :func:`chuck_dreamer.real.recorder.run`.

We can't actually run the lerobot record loop here (no SO101, no
LeRobotDataset), so phase B is mocked out. Phase A is driven with the
stub camera + UI, and the storage backend layout is verified at the
end.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf

from chuck_dreamer.real import recorder
from chuck_dreamer.real.scene_registration import OpenCVFrameSource

from ._stubs import StubFrameSource, StubUi


@pytest.fixture
def cfg(tmp_path):
  return OmegaConf.create({
    "recording": {
      "robot":  {"type": "so101_follower", "port": "/dev/null", "id": None},
      "teleop": {"type": "so101_leader",   "port": "/dev/null", "id": None},
      "cameras": {
        "top": {
          "type": "opencv", "index_or_path": 0,
          "width": 320, "height": 240, "fps": 10,
        },
      },
      "cameras_defaults": {"fallback_fps": 30},
      "dataset": {
        "repo_id": "local/test", "single_task": "push", "fps": 10,
        "episode_time_s": 1.0, "reset_time_s": 0.5, "num_episodes": 1,
        "push_to_hub": False, "streaming_encoding": False,
        "encoder_threads": None,
      },
      "scene": {
        "primary_camera": "top",
        "checkerboard": {"cols": 9, "rows": 6, "square_size_m": 0.02},
        "min_calibration_captures": 1,
        "marker_locations": ["off_screen"],
        "marker_clip_duration_s": 0.2,
        "marker_clip_min_blob_frames": 0.8,
        "save_marker_raw_frames": False,
      },
    },
  })


def _blob_frame(*, H=240, W=320, blob=False) -> np.ndarray:
  img = np.full((H, W, 3), 30, dtype=np.uint8)
  if blob:
    cv2.circle(img, (W // 2, H // 2), 25, (240, 240, 240), -1)
  return img


def test_recorder_run_writes_session_layout(cfg, tmp_path, monkeypatch):
  # Replace OpenCVFrameSource with our stub so no real camera is opened.
  stub_src = StubFrameSource(
    name="top", fps=10, image_size=(320, 240),
    frame_factory=lambda _i: _blob_frame(blob=False))

  def fake_ctor(name, camera_cfg):
    return stub_src
  monkeypatch.setattr(
    "chuck_dreamer.real.recorder.OpenCVFrameSource",
    fake_ctor,
  )

  # Skip phase A.1 (intrinsics need a checkerboard render — covered
  # elsewhere) and drive only phase A.2 with a single off_screen slot.
  ui = StubUi(key_script=["space"])

  recorded: dict = {}

  def fake_record(rec_cfg):
    recorded["called"] = True
    recorded["repo_id"] = rec_cfg.dataset.repo_id
    Path(rec_cfg.dataset.root).mkdir(parents=True, exist_ok=True)
    # Drop a marker so we can assert dataset/ exists with content.
    (Path(rec_cfg.dataset.root) / "marker").write_text("ok")

  out = recorder.run(
    cfg, tmp_path / "session",
    storage="lerobot",
    skip_calibration=True,
    skip_markers=False,
    ui=ui,
    record_fn=fake_record,
  )
  assert recorded["called"] is True
  assert recorded["repo_id"] == "local/test"

  # Storage layout
  assert (out / "session.json").exists()
  manifest = json.loads((out / "session.json").read_text())
  assert manifest["storage"] == "lerobot"
  assert (out / "scene" / "markers" / "off_screen" / "meta.json").exists()
  assert (out / "dataset" / "marker").read_text() == "ok"
  assert stub_src.disconnected, "phase-A camera should be disconnected before phase B"


def test_recorder_run_episodes_backend(cfg, tmp_path, monkeypatch):
  stub_src = StubFrameSource(
    name="top", fps=10, image_size=(320, 240),
    frame_factory=lambda _i: _blob_frame(blob=False))
  monkeypatch.setattr(
    "chuck_dreamer.real.recorder.OpenCVFrameSource",
    lambda name, camera_cfg: stub_src,
  )

  # Stub both phase B and the finalize-time importer so we don't need a
  # real LeRobot dataset on disk.
  def fake_record(rec_cfg):
    Path(rec_cfg.dataset.root).mkdir(parents=True, exist_ok=True)

  def fake_import(src, dst, *, format, max_episodes):
    Path(dst).mkdir(parents=True, exist_ok=True)
    (Path(dst) / "episode-00000.hdf5").write_bytes(b"fake")
    yield 0, Path(dst) / "episode-00000.hdf5"

  import chuck_dreamer.sim.lerobot_import as li
  monkeypatch.setattr(li, "import_dataset", fake_import)

  ui = StubUi(key_script=["space"])
  out = recorder.run(
    cfg, tmp_path / "session",
    storage="episodes",
    episodes_format="hdf5",
    skip_calibration=True,
    skip_markers=False,
    ui=ui,
    record_fn=fake_record,
  )
  # Scratch dir was cleaned up; canonical episode dir exists.
  assert not (out / "_lerobot_tmp").exists()
  assert (out / "episodes" / "episode-00000.hdf5").exists()
  manifest = json.loads((out / "session.json").read_text())
  assert manifest["storage"] == "episodes"
