"""Tests for the LeRobot v3 → repo-episode importer.

The importer talks to Hugging Face via :func:`hf_hub_download`. To keep
tests offline we build a tiny synthetic dataset (info.json + episodes
parquet + data parquet + MP4) inside ``tmp_path`` and monkeypatch
``hf_hub_download`` so it resolves repo-relative paths to those local
files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

av = pytest.importorskip("av")

from chuck_dreamer.sim import lerobot_import as li  # noqa: E402
from chuck_dreamer.training.episode_loader import load_hdf5_episode  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic dataset construction
# ---------------------------------------------------------------------------


VIDEO_KEY = "observation.images.wrist"
FPS = 10
H, W = 32, 32


def _write_mp4(path: Path, n_frames: int) -> None:
  """Write an MP4 whose i-th frame is the constant greyscale value ``i * 8``.

  ``mpeg4`` + ``yuv420p`` keeps R==G==B, so per-frame brightness survives
  the chroma-subsampling round-trip closely enough for ``np.isclose``
  with a small tolerance.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  container = av.open(str(path), mode="w")
  stream = container.add_stream("mpeg4", rate=FPS)
  stream.width = W
  stream.height = H
  stream.pix_fmt = "yuv420p"
  for i in range(n_frames):
    img = np.full((H, W, 3), i * 8, dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    for pkt in stream.encode(frame):
      container.mux(pkt)
  for pkt in stream.encode():
    container.mux(pkt)
  container.close()


def _write_data_parquet(path: Path, n_frames: int) -> None:
  """One parquet row per frame: action/state as 6-vec, t increasing by 1/FPS."""
  actions = [[float(i + 0.1 * j) for j in range(6)] for i in range(n_frames)]
  states  = [[float(i + 0.01 * j) for j in range(6)] for i in range(n_frames)]
  ts      = [float(i) / FPS for i in range(n_frames)]
  tbl = pa.table({
    "action":            pa.array(actions, type=pa.list_(pa.float32(), 6)),
    "observation.state": pa.array(states,  type=pa.list_(pa.float32(), 6)),
    "timestamp":         pa.array(ts,      type=pa.float32()),
    "frame_index":       pa.array(list(range(n_frames)), type=pa.int64()),
    "episode_index":     pa.array([0] * n_frames, type=pa.int64()),
  })
  path.parent.mkdir(parents=True, exist_ok=True)
  pq.write_table(tbl, path)


def _write_episodes_parquet(path: Path, episodes: list[dict]) -> None:
  """``episodes`` is a list of ``dict`` with keys matching v3 column names."""
  cols: dict[str, list] = {
    "episode_index":                       [e["episode_index"] for e in episodes],
    "tasks":                               [e["tasks"]         for e in episodes],
    "length":                              [e["length"]        for e in episodes],
    "data/chunk_index":                    [0] * len(episodes),
    "data/file_index":                     [0] * len(episodes),
    "dataset_from_index":                  [e["data_from"]     for e in episodes],
    "dataset_to_index":                    [e["data_to"]       for e in episodes],
    f"videos/{VIDEO_KEY}/chunk_index":     [0] * len(episodes),
    f"videos/{VIDEO_KEY}/file_index":      [0] * len(episodes),
    f"videos/{VIDEO_KEY}/from_timestamp":  [e["v_from"]        for e in episodes],
    f"videos/{VIDEO_KEY}/to_timestamp":    [e["v_to"]          for e in episodes],
  }
  schema = pa.schema([
    ("episode_index",                       pa.int64()),
    ("tasks",                               pa.list_(pa.string())),
    ("length",                              pa.int64()),
    ("data/chunk_index",                    pa.int64()),
    ("data/file_index",                     pa.int64()),
    ("dataset_from_index",                  pa.int64()),
    ("dataset_to_index",                    pa.int64()),
    (f"videos/{VIDEO_KEY}/chunk_index",     pa.int64()),
    (f"videos/{VIDEO_KEY}/file_index",      pa.int64()),
    (f"videos/{VIDEO_KEY}/from_timestamp",  pa.float32()),
    (f"videos/{VIDEO_KEY}/to_timestamp",    pa.float32()),
  ])
  path.parent.mkdir(parents=True, exist_ok=True)
  pq.write_table(pa.table(cols, schema=schema), path)


def _write_info(path: Path, with_video: bool = True) -> None:
  info = {
    "codebase_version": "v3.0",
    "fps":              FPS,
    "features": {
      "action":            {"dtype": "float32", "shape": [6]},
      "observation.state": {"dtype": "float32", "shape": [6]},
      "timestamp":         {"dtype": "float32", "shape": [1]},
    },
  }
  if with_video:
    info["features"][VIDEO_KEY] = {
      "dtype": "video",
      "shape": [H, W, 3],
      "info":  {"video.height": H, "video.width": W, "video.fps": FPS},
    }
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as f:
    json.dump(info, f)


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
  """Build a 2-episode synthetic LeRobot v3 dataset under ``tmp_path/repo``.

  Episode 0: frames [0, 6),  video ts [0.0, 0.6)
  Episode 1: frames [6, 10), video ts [0.6, 1.0)

  Patches ``hf_hub_download`` and ``HfApi.list_repo_files`` to resolve
  ``REPO_ID`` paths against this directory.
  """
  root = tmp_path / "repo"
  root.mkdir()

  n_frames = 10
  _write_mp4(root / f"videos/{VIDEO_KEY}/chunk-000/file-000.mp4", n_frames)
  _write_data_parquet(root / "data/chunk-000/file-000.parquet", n_frames)
  _write_episodes_parquet(
    root / "meta/episodes/chunk-000/file-000.parquet",
    [
      {"episode_index": 0, "tasks": ["pick"], "length": 6,
       "data_from": 0, "data_to": 6, "v_from": 0.0, "v_to": 6 / FPS},
      {"episode_index": 1, "tasks": ["place"], "length": 4,
       "data_from": 6, "data_to": 10, "v_from": 6 / FPS, "v_to": 10 / FPS},
    ],
  )
  _write_info(root / "meta/info.json")

  def fake_download(repo_id, filename, repo_type=None, **kwargs):
    p = root / filename
    if not p.exists():
      raise FileNotFoundError(f"fake_download: missing {filename!r} under {root}")
    return str(p)

  class FakeApi:
    def list_repo_files(self, repo_id, repo_type=None):
      return [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]

  monkeypatch.setattr(li, "hf_hub_download", fake_download)
  monkeypatch.setattr("huggingface_hub.HfApi", lambda: FakeApi())
  return root


# ---------------------------------------------------------------------------
# _decode_video_range
# ---------------------------------------------------------------------------


def test_decode_video_range_returns_expected_frames(tmp_path):
  mp4 = tmp_path / "v.mp4"
  _write_mp4(mp4, 10)
  frames = li._decode_video_range(str(mp4), 0.0, 1.0, expected=10)

  assert frames.shape == (10, H, W, 3)
  assert frames.dtype == np.uint8
  # i-th frame was encoded as constant i*8. mpeg4+yuv420p drifts a few
  # units; mean within a single unit is plenty of headroom.
  for i in range(10):
    assert abs(float(frames[i].mean()) - i * 8) < 4, f"frame {i}"


def test_decode_video_range_skips_frames_before_from_ts(tmp_path):
  mp4 = tmp_path / "v.mp4"
  _write_mp4(mp4, 10)
  # Frames at indices 4..7 → ts 0.4..0.7 in a 10 fps stream.
  frames = li._decode_video_range(str(mp4), 0.4, 0.8, expected=4)

  assert frames.shape == (4, H, W, 3)
  for offset, i in enumerate(range(4, 8)):
    assert abs(float(frames[offset].mean()) - i * 8) < 4, f"offset {offset}"


def test_decode_video_range_caps_at_expected(tmp_path):
  mp4 = tmp_path / "v.mp4"
  _write_mp4(mp4, 10)
  # Window is wider than the cap; decoder should stop after ``expected``.
  frames = li._decode_video_range(str(mp4), 0.0, 1.0, expected=3)
  assert frames.shape[0] == 3


def test_decode_video_range_raises_on_empty_window(tmp_path):
  mp4 = tmp_path / "v.mp4"
  _write_mp4(mp4, 10)
  with pytest.raises(RuntimeError, match="zero frames"):
    li._decode_video_range(str(mp4), 5.0, 6.0, expected=10)


# ---------------------------------------------------------------------------
# _select_video_key
# ---------------------------------------------------------------------------


def test_select_video_key_picks_first_when_unspecified(tmp_path, monkeypatch):
  info_path = tmp_path / "meta/info.json"
  _write_info(info_path)
  monkeypatch.setattr(li, "hf_hub_download", lambda *a, **k: str(info_path))

  assert li._select_video_key("repo", None) == VIDEO_KEY


def test_select_video_key_honors_preferred(tmp_path, monkeypatch):
  info_path = tmp_path / "meta/info.json"
  _write_info(info_path)
  monkeypatch.setattr(li, "hf_hub_download", lambda *a, **k: str(info_path))

  assert li._select_video_key("repo", VIDEO_KEY) == VIDEO_KEY


def test_select_video_key_rejects_unknown_key(tmp_path, monkeypatch):
  info_path = tmp_path / "meta/info.json"
  _write_info(info_path)
  monkeypatch.setattr(li, "hf_hub_download", lambda *a, **k: str(info_path))

  with pytest.raises(ValueError, match="not in dataset"):
    li._select_video_key("repo", "observation.images.nope")


def test_select_video_key_raises_when_no_video_features(tmp_path, monkeypatch):
  info_path = tmp_path / "meta/info.json"
  _write_info(info_path, with_video=False)
  monkeypatch.setattr(li, "hf_hub_download", lambda *a, **k: str(info_path))

  with pytest.raises(RuntimeError, match="no video features"):
    li._select_video_key("repo", None)


# ---------------------------------------------------------------------------
# import_dataset — end-to-end against the fake repo
# ---------------------------------------------------------------------------


def test_import_dataset_writes_one_file_per_episode(tmp_path, fake_repo):
  out = tmp_path / "out"
  results = list(li.import_dataset("any/repo", str(out), format="hdf5"))

  assert [idx for idx, _ in results] == [0, 1]
  for _, path in results:
    assert Path(path).exists()
    assert Path(path).suffix == ".hdf5"


def test_import_dataset_per_episode_shapes_and_metadata(tmp_path, fake_repo):
  out = tmp_path / "out"
  list(li.import_dataset("any/repo", str(out), format="hdf5"))

  ep0 = load_hdf5_episode(out / "episode-00000.hdf5")
  ep1 = load_hdf5_episode(out / "episode-00001.hdf5")

  # Episode 0 has 6 frames, episode 1 has 4 — matching the meta we wrote.
  assert ep0["image"].shape == (6, H, W, 3)
  assert ep0["joint_qpos"].shape == (6, 6)
  assert ep0["joint_action"].shape == (6, 6)
  assert ep0["act_mode"] == "joint"
  assert ep1["image"].shape == (4, H, W, 3)
  assert ep1["joint_action"].shape == (4, 6)


def test_import_dataset_zero_fills_missing_signals(tmp_path, fake_repo):
  out = tmp_path / "out"
  list(li.import_dataset("any/repo", str(out), format="hdf5"))

  ep0 = load_hdf5_episode(out / "episode-00000.hdf5")
  # Fields LeRobot teleop doesn't carry must come back as zeros.
  assert np.all(ep0["reward"]    == 0)
  assert np.all(ep0["ee_pos"]    == 0)
  assert np.all(ep0["ee_quat"]   == 0)
  assert np.all(ep0["object_xy"] == 0)


def test_import_dataset_carries_action_state_and_timestamps(tmp_path, fake_repo):
  out = tmp_path / "out"
  list(li.import_dataset("any/repo", str(out), format="hdf5"))

  ep0 = load_hdf5_episode(out / "episode-00000.hdf5")
  ep1 = load_hdf5_episode(out / "episode-00001.hdf5")

  # We wrote action[i, j] = i + 0.1 * j and state[i, j] = i + 0.01 * j.
  np.testing.assert_allclose(
    ep0["joint_action"][0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-5)
  np.testing.assert_allclose(
    ep0["joint_qpos"][3],
    [3.0, 3.01, 3.02, 3.03, 3.04, 3.05], atol=1e-5)
  # Episode 1 starts at row 6 of the parquet.
  np.testing.assert_allclose(
    ep1["joint_action"][0], [6.0, 6.1, 6.2, 6.3, 6.4, 6.5], atol=1e-5)
  # Timestamps are taken verbatim from the parquet, not re-zeroed per episode.
  np.testing.assert_allclose(ep0["timestamp"], np.arange(6) / FPS, atol=1e-5)
  np.testing.assert_allclose(ep1["timestamp"], np.arange(6, 10) / FPS, atol=1e-5)


def test_import_dataset_image_content_per_episode(tmp_path, fake_repo):
  out = tmp_path / "out"
  list(li.import_dataset("any/repo", str(out), format="hdf5"))

  ep0 = load_hdf5_episode(out / "episode-00000.hdf5")
  ep1 = load_hdf5_episode(out / "episode-00001.hdf5")

  # Episode 0 → frames 0..5; episode 1 → frames 6..9. Synth frames have
  # mean ≈ i*8, drift bounded by chroma subsampling.
  for i in range(6):
    assert abs(float(ep0["image"][i].mean()) - i * 8) < 4, f"ep0 frame {i}"
  for offset, i in enumerate(range(6, 10)):
    assert abs(float(ep1["image"][offset].mean()) - i * 8) < 4, f"ep1 frame {offset}"


def test_import_dataset_respects_max_episodes(tmp_path, fake_repo):
  out = tmp_path / "out"
  results = list(li.import_dataset(
    "any/repo", str(out), format="hdf5", max_episodes=1))

  assert [idx for idx, _ in results] == [0]
  assert (out / "episode-00000.hdf5").exists()
  assert not (out / "episode-00001.hdf5").exists()


def test_import_dataset_rejects_unknown_video_key(tmp_path, fake_repo):
  out = tmp_path / "out"
  with pytest.raises(ValueError, match="not in dataset"):
    list(li.import_dataset(
      "any/repo", str(out), format="hdf5", video_key="observation.images.nope"))
