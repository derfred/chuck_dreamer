"""Tests for the LeRobot v3 → repo-episode importer.

Two collaborators are stubbed to keep tests offline:

* Episode metadata is read through ``EpisodeSpec.read_episodes``, which
  constructs a ``LeRobotDatasetMetadata``. We monkeypatch that class with
  a fake exposing ``.video_keys`` and an ``.episodes`` table built from
  in-test rows.
* Per-episode data parquet + video MP4 are still fetched via the importer's
  resolver (``hf_hub_download``), which we patch to resolve repo-relative
  paths against a synthetic dataset dir under ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

av = pytest.importorskip("av")

from chuck_dreamer.lerobot import importer as li  # noqa: E402
from chuck_dreamer.common import episode_spec as es  # noqa: E402
from chuck_dreamer.training.episode_dataset import Episode  # noqa: E402


def load_hdf5_episode(path):
  return Episode.from_file(path, format="hdf5").data


# ---------------------------------------------------------------------------
# Synthetic dataset construction
# ---------------------------------------------------------------------------


VIDEO_KEY = "observation.images.wrist"
FPS = 10
H, W = 32, 32


def _write_mp4(path: Path, n_frames: int) -> None:
  """Write an MP4 whose i-th frame is the constant greyscale value ``i * 8``."""
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


# ---------------------------------------------------------------------------
# Fake LeRobotDatasetMetadata
# ---------------------------------------------------------------------------


class _FakeMeta:
  """Stand-in for ``LeRobotDatasetMetadata``.

  ``read_episodes`` only touches ``.video_keys`` and iterates
  ``.episodes`` reading the v3 column keys off each row, so a list of
  plain dicts is enough.
  """

  def __init__(self, repo_id, root=None, **kwargs):
    self.video_keys = [VIDEO_KEY]
    self.episodes = [
      {
        "episode_index": 0, "tasks": ["pick"], "length": 6,
        "dataset_from_index": 0, "dataset_to_index": 6,
        "data/chunk_index": 0, "data/file_index": 0,
        f"videos/{VIDEO_KEY}/chunk_index": 0,
        f"videos/{VIDEO_KEY}/file_index": 0,
        f"videos/{VIDEO_KEY}/from_timestamp": 0.0,
        f"videos/{VIDEO_KEY}/to_timestamp": 6 / FPS,
      },
      {
        "episode_index": 1, "tasks": ["place"], "length": 4,
        "dataset_from_index": 6, "dataset_to_index": 10,
        "data/chunk_index": 0, "data/file_index": 0,
        f"videos/{VIDEO_KEY}/chunk_index": 0,
        f"videos/{VIDEO_KEY}/file_index": 0,
        f"videos/{VIDEO_KEY}/from_timestamp": 6 / FPS,
        f"videos/{VIDEO_KEY}/to_timestamp": 10 / FPS,
      },
    ]


class _FakeMetaNoVideo(_FakeMeta):
  def __init__(self, repo_id, root=None, **kwargs):
    super().__init__(repo_id, root=root, **kwargs)
    self.video_keys = []


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
  """Build a 2-episode synthetic dataset + stub its metadata reader.

  Episode 0: frames [0, 6),  video ts [0.0, 0.6)
  Episode 1: frames [6, 10), video ts [0.6, 1.0)

  Patches ``LeRobotDatasetMetadata`` (used by ``read_episodes``) with
  ``_FakeMeta`` and ``hf_hub_download`` (used by the importer's resolver)
  to resolve repo-relative paths against ``tmp_path/repo``.
  """
  root = tmp_path / "repo"
  root.mkdir()

  n_frames = 10
  _write_mp4(root / f"videos/{VIDEO_KEY}/chunk-000/file-000.mp4", n_frames)
  _write_data_parquet(root / "data/chunk-000/file-000.parquet", n_frames)

  def fake_download(repo_id, filename, repo_type=None, **kwargs):
    p = root / filename
    if not p.exists():
      raise FileNotFoundError(f"fake_download: missing {filename!r} under {root}")
    return str(p)

  monkeypatch.setattr(li, "hf_hub_download", fake_download)
  monkeypatch.setattr(
    "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", _FakeMeta)
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
  for i in range(10):
    assert abs(float(frames[i].mean()) - i * 8) < 4, f"frame {i}"


def test_decode_video_range_skips_frames_before_from_ts(tmp_path):
  mp4 = tmp_path / "v.mp4"
  _write_mp4(mp4, 10)
  frames = li._decode_video_range(str(mp4), 0.4, 0.8, expected=4)

  assert frames.shape == (4, H, W, 3)
  for offset, i in enumerate(range(4, 8)):
    assert abs(float(frames[offset].mean()) - i * 8) < 4, f"offset {offset}"


def test_decode_video_range_caps_at_expected(tmp_path):
  mp4 = tmp_path / "v.mp4"
  _write_mp4(mp4, 10)
  frames = li._decode_video_range(str(mp4), 0.0, 1.0, expected=3)
  assert frames.shape[0] == 3


def test_decode_video_range_raises_on_empty_window(tmp_path):
  mp4 = tmp_path / "v.mp4"
  _write_mp4(mp4, 10)
  with pytest.raises(RuntimeError, match="zero frames"):
    li._decode_video_range(str(mp4), 5.0, 6.0, expected=10)


# ---------------------------------------------------------------------------
# read_episodes — video-key resolution off the (faked) metadata
# ---------------------------------------------------------------------------


def test_read_episodes_picks_first_video_key_when_unspecified(monkeypatch):
  monkeypatch.setattr(
    "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", _FakeMeta)
  slices, vkey = es.EpisodeSpec("any/repo").read_episodes()
  assert vkey == VIDEO_KEY
  assert [s.episode_index for s in slices] == [0, 1]


def test_read_episodes_honors_preferred_video_key(monkeypatch):
  monkeypatch.setattr(
    "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", _FakeMeta)
  _, vkey = es.EpisodeSpec("any/repo").read_episodes(video_key=VIDEO_KEY)
  assert vkey == VIDEO_KEY


def test_read_episodes_rejects_unknown_video_key(monkeypatch):
  monkeypatch.setattr(
    "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", _FakeMeta)
  with pytest.raises(ValueError, match="not among"):
    es.EpisodeSpec("any/repo").read_episodes(video_key="observation.images.nope")


def test_read_episodes_raises_when_no_video_features(monkeypatch):
  monkeypatch.setattr(
    "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", _FakeMetaNoVideo)
  with pytest.raises(ValueError, match="no video features"):
    es.EpisodeSpec("any/repo").read_episodes()


def test_read_episodes_applies_spec_filter(monkeypatch):
  monkeypatch.setattr(
    "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", _FakeMeta)
  slices, _ = es.EpisodeSpec.parse("any/repo#1").read_episodes()
  assert [s.episode_index for s in slices] == [1]


# ---------------------------------------------------------------------------
# import_dataset — end-to-end against the fake repo
# ---------------------------------------------------------------------------


def test_import_dataset_writes_one_file_per_episode(tmp_path, fake_repo):
  out = tmp_path / "out"
  results = list(li.import_dataset(
    "any/repo", str(out), format="hdf5",
    with_ee_pos=False, with_object_pose=False))

  assert [idx for idx, _ in results] == [0, 1]
  for _, path in results:
    assert Path(path).exists()
    assert Path(path).suffix == ".hdf5"


def test_import_dataset_per_episode_shapes_and_metadata(tmp_path, fake_repo):
  out = tmp_path / "out"
  list(li.import_dataset(
    "any/repo", str(out), format="hdf5",
    with_ee_pos=False, with_object_pose=False))

  ep0 = load_hdf5_episode(out / "episode-00000.hdf5")
  ep1 = load_hdf5_episode(out / "episode-00001.hdf5")

  assert ep0["image"].shape == (6, H, W, 3)
  assert ep0["joint_qpos"].shape == (6, 6)
  assert ep0["joint_action"].shape == (6, 6)
  assert "act_mode" not in ep0
  assert "ee_action" not in ep0
  assert ep1["image"].shape == (4, H, W, 3)
  assert ep1["joint_action"].shape == (4, 6)


def test_import_dataset_zero_fills_missing_signals(tmp_path, fake_repo):
  out = tmp_path / "out"
  list(li.import_dataset(
    "any/repo", str(out), format="hdf5",
    with_ee_pos=False, with_object_pose=False))

  ep0 = load_hdf5_episode(out / "episode-00000.hdf5")
  assert np.all(ep0["reward"]    == 0)
  assert np.all(ep0["ee_pos"]    == 0)
  assert np.all(ep0["ee_quat"]   == 0)
  assert np.all(ep0["object_xy"] == 0)


def test_import_dataset_carries_action_state_and_timestamps(tmp_path, fake_repo):
  out = tmp_path / "out"
  list(li.import_dataset(
    "any/repo", str(out), format="hdf5",
    with_ee_pos=False, with_object_pose=False))

  ep0 = load_hdf5_episode(out / "episode-00000.hdf5")
  ep1 = load_hdf5_episode(out / "episode-00001.hdf5")

  np.testing.assert_allclose(
    ep0["joint_action"][0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-5)
  np.testing.assert_allclose(
    ep0["joint_qpos"][3],
    [3.0, 3.01, 3.02, 3.03, 3.04, 3.05], atol=1e-5)
  np.testing.assert_allclose(
    ep1["joint_action"][0], [6.0, 6.1, 6.2, 6.3, 6.4, 6.5], atol=1e-5)
  np.testing.assert_allclose(ep0["timestamp"], np.arange(6) / FPS, atol=1e-5)
  np.testing.assert_allclose(ep1["timestamp"], np.arange(6, 10) / FPS, atol=1e-5)


def test_import_dataset_image_content_per_episode(tmp_path, fake_repo):
  out = tmp_path / "out"
  list(li.import_dataset(
    "any/repo", str(out), format="hdf5",
    with_ee_pos=False, with_object_pose=False))

  ep0 = load_hdf5_episode(out / "episode-00000.hdf5")
  ep1 = load_hdf5_episode(out / "episode-00001.hdf5")

  for i in range(6):
    assert abs(float(ep0["image"][i].mean()) - i * 8) < 4, f"ep0 frame {i}"
  for offset, i in enumerate(range(6, 10)):
    assert abs(float(ep1["image"][offset].mean()) - i * 8) < 4, f"ep1 frame {offset}"


def test_import_dataset_respects_max_episodes(tmp_path, fake_repo):
  out = tmp_path / "out"
  results = list(li.import_dataset(
    "any/repo", str(out), format="hdf5",
    with_ee_pos=False, with_object_pose=False, max_episodes=1))

  assert [idx for idx, _ in results] == [0]
  assert (out / "episode-00000.hdf5").exists()
  assert not (out / "episode-00001.hdf5").exists()


def test_import_dataset_respects_source_episode_filter(tmp_path, fake_repo):
  out = tmp_path / "out"
  spec = es.EpisodeSpec.parse("any/repo#1")
  results = list(li.import_dataset(
    "any/repo", str(out), format="hdf5",
    with_ee_pos=False, with_object_pose=False, source=spec))

  assert [idx for idx, _ in results] == [1]
  assert (out / "episode-00001.hdf5").exists()
  assert not (out / "episode-00000.hdf5").exists()


def test_import_dataset_rejects_unknown_video_key(tmp_path, fake_repo):
  out = tmp_path / "out"
  with pytest.raises(ValueError, match="not among"):
    list(li.import_dataset(
      "any/repo", str(out), format="hdf5",
      with_ee_pos=False, with_object_pose=False,
      video_key="observation.images.nope"))


def test_import_dataset_stamps_tags_on_each_episode(tmp_path, fake_repo):
  out = tmp_path / "out"
  list(li.import_dataset(
    "any/repo", str(out), format="hdf5",
    with_ee_pos=False, with_object_pose=False, tags=("real", "demo")))

  for ep_idx in (0, 1):
    ep = Episode.from_file(out / f"episode-{ep_idx:05d}.hdf5")
    assert ep.metadata.get("tags") == ("real", "demo")


def test_import_dataset_no_tags_by_default(tmp_path, fake_repo):
  out = tmp_path / "out"
  list(li.import_dataset(
    "any/repo", str(out), format="hdf5",
    with_ee_pos=False, with_object_pose=False))
  ep = Episode.from_file(out / "episode-00000.hdf5")
  assert "tags" not in ep.metadata


def test_import_dataset_rerun_round_trip_carries_tags(tmp_path, fake_repo):
  out = tmp_path / "out"
  list(li.import_dataset(
    "any/repo", str(out), format="rerun",
    with_ee_pos=False, with_object_pose=False, tags=("real",)))
  ep = Episode.from_file(out / "episode-00000.rrd")
  assert ep.metadata.get("tags") == ("real",)
