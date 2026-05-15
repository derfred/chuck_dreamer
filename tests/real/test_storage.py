"""Tests for :mod:`chuck_dreamer.real.storage`.

Backend layout is verified end-to-end: write a small ``SceneRegistration``,
check the on-disk file tree, and (for ``EpisodesBackend``) make sure
``finalize`` invokes the importer with the scratch dataset directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from chuck_dreamer.real import storage as st
from chuck_dreamer.real.scene_types import (
  CalibrationCapture,
  CameraIntrinsics,
  MarkerClip,
  SceneRegistration,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_intrinsics(tmp_path: Path) -> CameraIntrinsics:
  img_path = tmp_path / "src_capture-000.png"
  cv2.imwrite(str(img_path), np.full((24, 32, 3), 200, dtype=np.uint8))
  return CameraIntrinsics(
    K=np.eye(3),
    dist=np.zeros(5),
    reproj_error=0.3,
    image_size=(32, 24),
    captures=[CalibrationCapture(
      image_path=img_path,
      description="left view",
      corners_px=np.zeros((54, 2), dtype=np.float32),
    )],
  )


def _make_clip(loc: str, *, T: int = 4, blob: bool = True) -> MarkerClip:
  H, W = 24, 32
  frames = np.full((T, H, W, 3), 30, dtype=np.uint8)
  centroids = np.full((T, 2), np.nan, dtype=np.float32)
  radii = np.full((T,), np.nan, dtype=np.float32)
  detected = np.zeros((T,), dtype=bool)
  if blob:
    for t in range(T):
      cv2.circle(frames[t], (16, 12), 6, (240, 240, 240), -1)
      centroids[t] = (16.0, 12.0)
      radii[t] = 6.0
      detected[t] = True
  return MarkerClip(
    loc_name=loc, frames=frames,
    blob_centroids_px=centroids, blob_radii_px=radii, blob_detected=detected,
    fps=10, duration_s=T / 10.0,
  )


@pytest.fixture
def scene(tmp_path):
  return SceneRegistration(
    intrinsics={"top": _make_intrinsics(tmp_path)},
    markers=[
      _make_clip("off_screen", blob=False),
      _make_clip("start", blob=True),
    ],
  )


# ---------------------------------------------------------------------------
# write_scene layout
# ---------------------------------------------------------------------------


def test_lerobot_backend_writes_scene_layout(tmp_path, scene):
  be = st.LeRobotBackend(tmp_path / "session", save_marker_raw_frames=False)
  be.write_scene(scene)
  root = be.root
  # Intrinsics
  intr_json = json.loads((root / "scene" / "intrinsics" / "top.json").read_text())
  assert intr_json["n_captures"] == 1
  assert intr_json["image_size"] == [32, 24]
  # Per-capture
  desc = json.loads(
    (root / "scene" / "intrinsics" / "top" / "captures" / "descriptions.json").read_text())
  assert desc[0]["description"] == "left view"
  # Markers
  for loc in ("off_screen", "start"):
    meta = json.loads((root / "scene" / "markers" / loc / "meta.json").read_text())
    assert meta["loc_name"] == loc
    assert meta["n_frames"] == 4
    assert (root / "scene" / "markers" / loc / "blob.npz").exists()
    # save_marker_raw_frames=False → no raw frames .npz
    assert not (root / "scene" / "markers" / loc / "frames.npz").exists()


def test_lerobot_backend_saves_raw_frames_when_enabled(tmp_path, scene):
  be = st.LeRobotBackend(tmp_path / "session", save_marker_raw_frames=True)
  be.write_scene(scene)
  for loc in ("off_screen", "start"):
    npz_path = be.root / "scene" / "markers" / loc / "frames.npz"
    assert npz_path.exists()
    z = np.load(npz_path)
    assert z["frames"].shape == (4, 24, 32, 3)


def test_lerobot_backend_finalize_writes_manifest(tmp_path, scene):
  be = st.LeRobotBackend(tmp_path / "session")
  be.write_scene(scene)
  # episodes_root() should also exist.
  ep_root = be.episodes_root()
  assert ep_root.exists() and ep_root.name == "dataset"
  be.finalize(recording_cfg_snapshot={"task": "push"})
  manifest = json.loads((be.root / "session.json").read_text())
  assert manifest["storage"] == "lerobot"
  assert manifest["recording"]["task"] == "push"
  assert "started_at" in manifest and "finished_at" in manifest


# ---------------------------------------------------------------------------
# EpisodesBackend.finalize
# ---------------------------------------------------------------------------


def test_episodes_backend_finalize_runs_importer(tmp_path, scene, monkeypatch):
  be = st.EpisodesBackend(
    tmp_path / "session",
    save_marker_raw_frames=False,
    episodes_format="hdf5",
  )
  be.write_scene(scene)
  scratch = be.episodes_root()
  assert scratch.name == "_lerobot_tmp"

  # Stub the importer so we don't need a real LeRobot dataset.
  imported_args: dict = {}

  def fake_import(src, dst, *, format, max_episodes):
    imported_args["src"] = src
    imported_args["dst"] = dst
    imported_args["format"] = format
    # Produce one fake episode file the test can inspect.
    Path(dst).mkdir(parents=True, exist_ok=True)
    (Path(dst) / "episode-00000.hdf5").write_bytes(b"fake")
    yield 0, Path(dst) / "episode-00000.hdf5"

  import chuck_dreamer.sim.lerobot_import as li
  monkeypatch.setattr(li, "import_dataset", fake_import)

  be.finalize(recording_cfg_snapshot={"task": "push"})
  assert imported_args["src"] == str(scratch)
  assert imported_args["dst"] == str(be.root / "episodes")
  assert imported_args["format"] == "hdf5"
  # Scratch dir cleaned up.
  assert not scratch.exists()
  # Manifest written.
  manifest = json.loads((be.root / "session.json").read_text())
  assert manifest["storage"] == "episodes"


# ---------------------------------------------------------------------------
# build_storage factory
# ---------------------------------------------------------------------------


def test_build_storage_factory(tmp_path):
  assert isinstance(st.build_storage("lerobot", tmp_path / "a"), st.LeRobotBackend)
  assert isinstance(st.build_storage("episodes", tmp_path / "b"), st.EpisodesBackend)
  with pytest.raises(ValueError):
    st.build_storage("nope", tmp_path / "c")
