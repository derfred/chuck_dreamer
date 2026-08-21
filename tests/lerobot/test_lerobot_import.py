"""Tests for the LeRobot v3 -> repo-episode importer.

The importer loads source data through ``LeRobotDataset`` rather than
parsing parquet/MP4 by hand, so the end-to-end tests build a real tiny
two-episode v3 dataset on disk (``LeRobotDataset.create`` + ``add_frame``
+ ``finalize``) under ``tmp_path`` and import from that directory. A
complete on-disk dataset loads fully offline, so no network stubbing is
needed for ``import_dataset``.

The ``read_episodes`` tests still stub ``LeRobotDatasetMetadata`` directly,
since they exercise the (unchanged) ``EpisodeSpec`` selection layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")

from chuck_dreamer.lerobot import importer as li  # noqa: E402
from chuck_dreamer.lerobot import episode_spec as es  # noqa: E402
from chuck_dreamer.training.episode_dataset import Episode  # noqa: E402


def load_hdf5_episode(path):
  return Episode.from_file(path, format="hdf5").data


# ---------------------------------------------------------------------------
# Synthetic real dataset construction
# ---------------------------------------------------------------------------


VIDEO_KEY = "observation.images.wrist"
FPS = 10
H, W = 32, 32

# Episode 0: 6 frames, episode 1: 4 frames. Frame i has greyscale value
# i * 8 and action/state vectors keyed off the global frame index so the
# tests can assert which frame landed where.
EP_LENS = [(6, "pick"), (4, "place")]


@pytest.fixture
def fake_repo(tmp_path):
  """Build a real 2-episode LeRobot v3 dataset on disk and return its root.

  The frame at global index ``g`` has constant greyscale value ``g * 8``
  (AV1-lossy on decode), action ``[g+0.1*j]`` and state ``[g+0.01*j]``.
  """
  from lerobot.datasets.lerobot_dataset import LeRobotDataset

  root = tmp_path / "repo"
  features = {
    VIDEO_KEY: {"dtype": "video", "shape": (H, W, 3),
                "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
  }
  ds = LeRobotDataset.create(
    repo_id="test/tiny", fps=FPS, features=features,
    root=str(root), use_videos=True)

  g = 0
  for ep_len, task in EP_LENS:
    for _ in range(ep_len):
      ds.add_frame({
        VIDEO_KEY: np.full((H, W, 3), g * 8, dtype=np.uint8),
        "observation.state": np.array([g + 0.01 * j for j in range(6)],
                                      dtype=np.float32),
        "action": np.array([g + 0.1 * j for j in range(6)], dtype=np.float32),
        "task": task,
      })
      g += 1
    ds.save_episode()
  ds.finalize()
  return root


# ---------------------------------------------------------------------------
# Fake LeRobotDatasetMetadata — for the read_episodes (selection) tests
# ---------------------------------------------------------------------------


class _FakeMeta:
  """Stand-in for ``LeRobotDatasetMetadata`` used by ``read_episodes``.

  ``read_episodes`` touches ``.video_keys``, ``.root`` and ``.video_path``
  (the formattable MP4 template) and iterates ``.episodes`` reading the v3
  column keys off each row, so a list of plain dicts plus those attrs is
  enough.
  """

  def __init__(self, repo_id, root=None, **kwargs):
    self.video_keys = [VIDEO_KEY]
    self.root = root or f"/fake/{repo_id}"
    self.video_path = (
      "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
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


# ---------------------------------------------------------------------------
# read_episodes — video-key resolution + selection off the (faked) metadata
# ---------------------------------------------------------------------------


def test_read_episodes_picks_first_video_key_when_unspecified(monkeypatch):
  monkeypatch.setattr(
    "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", _FakeMeta)
  slices, vkey = es.EpisodeSpec("any/repo").read_episodes()
  assert vkey == VIDEO_KEY
  assert [s.episode_index for s in slices] == [0, 1]


def test_read_episodes_resolves_absolute_video_path(monkeypatch):
  # Each slice carries the absolute MP4 path (root + chunk/file template),
  # so a downstream stage can decode the video without re-opening metadata.
  monkeypatch.setattr(
    "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", _FakeMeta)
  slices, _ = es.EpisodeSpec("any/repo").read_episodes(root="/data/repo")
  assert str(slices[0].video_path) == (
    f"/data/repo/videos/{VIDEO_KEY}/chunk-000/file-000.mp4")


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
# import_dataset — end-to-end against the real fake_repo dataset
# ---------------------------------------------------------------------------


def _import(out, root, **kw):
  from chuck_dreamer.lerobot.pipeline import Run
  spec = es.EpisodeSpec.parse(str(root), **kw.pop("parse_kw", {}))
  params = {"with_ee_pos": False, "with_object_pose": False,
            "no_video": kw.pop("no_video", False)}
  run = Run(None, spec, params)
  return list(li.import_dataset(
    run, str(out), format=kw.pop("format", "hdf5"), **kw))


def test_run_pipeline_exposes_on_disk_slice_to_context(fake_repo):
  # Run.pipeline populates ctx.slice_for(idx) with the episode's resolved,
  # on-disk MP4 path so the segmentation stage decodes the video without
  # re-opening LeRobotDatasetMetadata.
  from chuck_dreamer.lerobot.pipeline import Run
  run = Run(None, es.EpisodeSpec.parse(str(fake_repo)),
            {"with_ee_pos": False, "with_object_pose": False})
  run.pipeline(0)
  sl = run.ctx.slice_for(0)
  assert sl.episode_index == 0
  assert sl.video_path is not None and Path(sl.video_path).exists()
  assert sl.video_to_ts > sl.video_from_ts


def test_import_dataset_writes_one_file_per_episode(tmp_path, fake_repo):
  out = tmp_path / "out"
  results = _import(out, fake_repo)

  assert [idx for idx, _ in results] == [0, 1]
  for _, path in results:
    assert Path(path).exists()
    assert Path(path).suffix == ".hdf5"


def test_import_dataset_per_episode_shapes_and_metadata(tmp_path, fake_repo):
  out = tmp_path / "out"
  _import(out, fake_repo)

  ep0 = load_hdf5_episode(out / "episode-00000.hdf5")
  ep1 = load_hdf5_episode(out / "episode-00001.hdf5")

  assert ep0["image"].shape == (6, H, W, 3)
  assert ep0["image"].dtype == np.uint8
  assert ep0["joint_qpos"].shape == (6, 6)
  assert ep0["joint_action"].shape == (6, 6)
  assert "act_mode" not in ep0
  assert "ee_action" not in ep0
  assert ep1["image"].shape == (4, H, W, 3)
  assert ep1["joint_action"].shape == (4, 6)


def test_import_dataset_zero_fills_missing_signals(tmp_path, fake_repo):
  out = tmp_path / "out"
  _import(out, fake_repo)

  ep0 = load_hdf5_episode(out / "episode-00000.hdf5")
  assert np.all(ep0["reward"]    == 0)
  assert np.all(ep0["ee_pos"]    == 0)
  assert np.all(ep0["ee_quat"]   == 0)
  assert np.all(ep0["object_xy"] == 0)


def test_no_video_omits_images_but_keeps_other_tracks(tmp_path, fake_repo):
  # --no-video drops only the RGB stack; every other per-step track is
  # written exactly as it would be otherwise.
  import h5py

  out = tmp_path / "out"
  _import(out, fake_repo, no_video=True)

  with h5py.File(out / "episode-00000.hdf5", "r") as f:
    assert "images" not in f
    assert f["joint_action"].shape == (6, 6)
    assert f["joint_qpos"].shape == (6, 6)
    assert f["timestamps"].shape == (6,)


def test_no_video_file_is_much_smaller(tmp_path, fake_repo):
  with_video = tmp_path / "with"
  without    = tmp_path / "without"
  _import(with_video, fake_repo)
  _import(without, fake_repo, no_video=True)

  big   = (with_video / "episode-00000.hdf5").stat().st_size
  small = (without / "episode-00000.hdf5").stat().st_size
  assert small < big


def test_no_video_without_frame_consumers_skips_decode(tmp_path, fake_repo,
                                                       monkeypatch):
  # With no node declaring ``image``, --no-video must not decode the video at
  # all: the run reports decodes_video False and the frames come back with
  # images=None.
  from chuck_dreamer.lerobot.pipeline import Run

  seen: list[bool] = []
  original = Run.episode_frames

  def spy(self, episode_index):
    seen.append(self.decodes_video)
    frames = original(self, episode_index)
    assert frames is not None and frames.images is None
    return frames

  monkeypatch.setattr(Run, "episode_frames", spy)

  out = tmp_path / "out"
  _import(out, fake_repo, no_video=True)

  assert seen and not any(seen), seen
  assert (out / "episode-00000.hdf5").exists()


def test_no_video_with_frame_consumer_still_decodes(tmp_path, fake_repo,
                                                    monkeypatch):
  # A node that declares ``image`` forces the decode even under --no-video;
  # the frames are used and then simply left out of the written file.
  import h5py

  from chuck_dreamer.common.tracks import Frame
  from chuck_dreamer.lerobot.harness import InputDecl
  from chuck_dreamer.lerobot.pipeline import Run

  class _NeedsFrames:
    name = "needs_frames"
    lane = "producer"
    inputs = (InputDecl("image", frame=Frame.IMAGE),)
    outputs = ()

    def run(self, view):
      view.track("image")  # actually consume the pixels
      return []

  monkeypatch.setattr(Run, "pipeline",
                      lambda self, episode_index: [_NeedsFrames()])

  seen: list[bool] = []
  original = Run.episode_frames

  def spy(self, episode_index):
    seen.append(self.decodes_video)
    return original(self, episode_index)

  monkeypatch.setattr(Run, "episode_frames", spy)

  out = tmp_path / "out"
  _import(out, fake_repo, no_video=True)

  assert seen and all(seen), seen
  with h5py.File(out / "episode-00000.hdf5", "r") as f:
    assert "images" not in f  # decoded, used, but not persisted


def test_import_dataset_carries_action_state_and_timestamps(tmp_path, fake_repo):
  out = tmp_path / "out"
  _import(out, fake_repo)

  ep0 = load_hdf5_episode(out / "episode-00000.hdf5")
  ep1 = load_hdf5_episode(out / "episode-00001.hdf5")

  np.testing.assert_allclose(
    ep0["joint_action"][0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-4)
  np.testing.assert_allclose(
    ep0["joint_qpos"][3],
    [3.0, 3.01, 3.02, 3.03, 3.04, 3.05], atol=1e-4)
  np.testing.assert_allclose(
    ep1["joint_action"][0], [6.0, 6.1, 6.2, 6.3, 6.4, 6.5], atol=1e-4)
  # timestamps are episode-relative in LeRobot
  np.testing.assert_allclose(ep0["timestamp"], np.arange(6) / FPS, atol=1e-4)
  np.testing.assert_allclose(ep1["timestamp"], np.arange(4) / FPS, atol=1e-4)


def test_import_dataset_image_content_per_episode(tmp_path, fake_repo):
  out = tmp_path / "out"
  _import(out, fake_repo)

  ep0 = load_hdf5_episode(out / "episode-00000.hdf5")
  ep1 = load_hdf5_episode(out / "episode-00001.hdf5")

  # AV1 encode/decode is lossy; allow a generous tolerance on the mean.
  for i in range(6):
    assert abs(float(ep0["image"][i].mean()) - i * 8) < 12, f"ep0 frame {i}"
  for offset, g in enumerate(range(6, 10)):
    assert abs(float(ep1["image"][offset].mean()) - g * 8) < 12, f"ep1 frame {offset}"


def test_import_dataset_respects_source_episode_filter(tmp_path, fake_repo):
  out = tmp_path / "out"
  results = _import(out, fake_repo, parse_kw={"allow_frames": False})
  # sanity: no filter -> both
  assert [idx for idx, _ in results] == [0, 1]

  from chuck_dreamer.lerobot.pipeline import Run
  out2 = tmp_path / "out2"
  spec = es.EpisodeSpec.parse(f"{fake_repo}#1", allow_frames=False)
  run = Run(None, spec, {"with_ee_pos": False, "with_object_pose": False})
  results2 = list(li.import_dataset(run, str(out2), format="hdf5"))

  assert [idx for idx, _ in results2] == [1]
  assert (out2 / "episode-00001.hdf5").exists()
  assert not (out2 / "episode-00000.hdf5").exists()


def test_import_dataset_rejects_unknown_video_key(tmp_path, fake_repo):
  from chuck_dreamer.lerobot.pipeline import Run
  out = tmp_path / "out"
  spec = es.EpisodeSpec.parse(str(fake_repo))
  run = Run(None, spec, {"with_ee_pos": False, "with_object_pose": False},
            video_key="observation.images.nope")
  with pytest.raises(ValueError, match="not among"):
    list(li.import_dataset(run, str(out), format="hdf5"))


def test_import_dataset_stamps_tags_on_each_episode(tmp_path, fake_repo):
  out = tmp_path / "out"
  _import(out, fake_repo, tags=("real", "demo"))

  for ep_idx in (0, 1):
    ep = Episode.from_file(out / f"episode-{ep_idx:05d}.hdf5")
    assert ep.metadata.get("tags") == ("real", "demo")


def test_import_dataset_no_tags_by_default(tmp_path, fake_repo):
  out = tmp_path / "out"
  _import(out, fake_repo)
  ep = Episode.from_file(out / "episode-00000.hdf5")
  assert "tags" not in ep.metadata


def test_import_dataset_rerun_round_trip_carries_tags(tmp_path, fake_repo):
  out = tmp_path / "out"
  _import(out, fake_repo, format="rerun", tags=("real",))
  ep = Episode.from_file(out / "episode-00000.rrd")
  assert ep.metadata.get("tags") == ("real",)
