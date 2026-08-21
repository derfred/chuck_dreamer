"""Runner-level store caching of the CACHED ``object_masks`` output.

Drives the segmentation node through the harness with a fake segmenter +
fake video decode: the first run computes and the *runner* puts the masks
into the artifact store; a second run with unchanged inputs loads them (the
node body never runs); changing any declared input (prompts, video window)
recomputes. The cache key is the ``input_versions`` map the NodeView
assembles from every declared input.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from chuck_dreamer.common.episode import Episode
from chuck_dreamer.lerobot.context import RunContext
from chuck_dreamer.lerobot.nodes import ObjUvExtractionNode, SegmentationNode
from chuck_dreamer.lerobot.trackset import EpisodeScope, TrackSet, run_episode

T, H, W = 4, 6, 8
DS = "user/ds"
EP = 2


@pytest.fixture
def video(tmp_path):
  p = tmp_path / "chunk-000.mp4"
  p.write_bytes(b"not a real mp4, identity comes from name+size")
  return p


def _make_ctx(tmp_path, video, *, with_store: bool) -> RunContext:
  cfg = OmegaConf.create(
    {"store": {"root": str(tmp_path / "store")}} if with_store else {})
  c = RunContext(config=cfg, source_repo=DS)
  c.episode_index = EP
  c.slices_by_index = {EP: SimpleNamespace(
    episode_index=EP, video_path=video, video_from_ts=0.0, video_to_ts=2.0)}
  return c


@pytest.fixture
def ctx(tmp_path, video):
  return _make_ctx(tmp_path, video, with_store=True)


def _write_prompts(store, prompts: dict[int, list[float]]) -> None:
  from chuck_dreamer.store import annotation_record, for_episode
  scope = for_episode(DS, EP)
  store.put("object_prompts", scope,
            {str(fi): pr for fi, pr in prompts.items()},
            annotation_record("object_prompts", scope, "test"))


def _masks():
  masks = [np.zeros((H, W), dtype=bool) for _ in range(T)]
  for m in masks[:3]:
    m[2:4, 3:6] = True
  masks[3] = None  # one drop-out frame
  return masks


def _fake_nodes(ctx, tmp_path, monkeypatch):
  """The segmentation + obj_uv nodes with SAM2 and the video decode faked."""
  seg = SegmentationNode(ctx)
  calls = {"n": 0}

  def fake_segment(jpg_dir, keyframes, hf_repo):
    calls["n"] += 1
    assert hf_repo  # pinned identity from assets/manifest.json
    return _masks()

  @contextmanager
  def fake_video_as_jpgs(sl, episode_index):
    yield tmp_path / "jpgs"

  monkeypatch.setattr(seg, "_segment_jpgs", fake_segment)
  monkeypatch.setattr(seg, "_video_as_jpgs", fake_video_as_jpgs)
  return [seg, ObjUvExtractionNode(ctx)], calls


def _run(nodes):
  episode = Episode.from_arrays({
    "timestamp": np.arange(T, dtype=np.float32),
    "image":     np.zeros((T, H, W, 3), dtype=np.uint8),
  })
  ts = TrackSet(EpisodeScope(DS, EP), episode, {})
  run_episode(nodes, ts)
  return ts


@pytest.fixture
def nodes_and_calls(ctx, tmp_path, monkeypatch):
  _write_prompts(ctx.store, {0: [4.0, 3.0]})
  return _fake_nodes(ctx, tmp_path, monkeypatch)


def test_first_run_computes_and_caches(nodes_and_calls, ctx):
  nodes, calls = nodes_and_calls
  ts = _run(nodes)
  assert calls["n"] == 1
  from chuck_dreamer.store import for_episode
  payload, record = ctx.store.get("object_masks", for_episode(DS, EP))
  assert record.node == "segmentation"
  assert set(record.input_versions) == {
    "timestamp", "object_prompts", "dataset.episodes", "sam2_weights"}
  np.testing.assert_array_equal(payload["valid"], [True, True, True, False])
  # Trajectory outputs derived from the masks.
  assert np.asarray(ts.episode.get("segmentation_target")).shape == (T, H, W)


def test_second_run_skips_sam2_with_identical_result(nodes_and_calls):
  nodes, calls = nodes_and_calls
  ts1 = _run(nodes)
  ts2 = _run(nodes)
  assert calls["n"] == 1  # second run: cache hit
  np.testing.assert_array_equal(
    np.asarray(ts1.episode.get("segmentation_target")),
    np.asarray(ts2.episode.get("segmentation_target")))
  a = np.asarray(ts1.episode.get("obj_uv"))
  b = np.asarray(ts2.episode.get("obj_uv"))
  np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
  np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
  # The masks track round-trips including validity (drop-out frame 3).
  masks = ts2.get_track("object_masks")
  assert masks.valid is not None
  np.testing.assert_array_equal(masks.valid, [True, True, True, False])
  # obj_uv carries the same validity.
  uv = ts2.get_track("obj_uv")
  np.testing.assert_array_equal(uv.valid, masks.valid)


def test_prompt_change_invalidates(nodes_and_calls, ctx):
  nodes, calls = nodes_and_calls
  _run(nodes)
  _write_prompts(ctx.store, {0: [5.0, 3.0]})  # moved click
  _run(nodes)
  assert calls["n"] == 2


def test_video_change_invalidates(nodes_and_calls, video):
  nodes, calls = nodes_and_calls
  _run(nodes)
  video.write_bytes(b"re-recorded video with a different size........")
  _run(nodes)
  assert calls["n"] == 2


def test_no_store_config_disables_caching(tmp_path, video, monkeypatch):
  ctx = _make_ctx(tmp_path, video, with_store=False)
  # Without a store there is no prompt artifact to resolve; hand the node
  # an unidentifiable prompt map directly (version=None also disables
  # caching on its own).
  monkeypatch.setattr(
    RunContext, "resolve_keyframe_prompts",
    lambda self, ep: ({0: (4, 3)}, None))
  nodes, calls = _fake_nodes(ctx, tmp_path, monkeypatch)

  _run(nodes)
  _run(nodes)
  assert calls["n"] == 2          # no store -> no caching, recomputes
  assert ctx.store is None
