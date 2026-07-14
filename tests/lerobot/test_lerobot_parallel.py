"""Tests for the parallel LeRobot importer (GPU producer → CPU worker pool).

The headline property is **equivalence**: ``import_dataset(..., jobs=N)`` must
produce byte-identical episode files to the serial ``jobs=1`` path, because the
only stage with cross-frame state (``object_pose``'s warm-start) runs *within*
one episode and parallelism is only *across* episodes.

SAM2 / calibration aren't available in CI, so these drive the orchestrator with
the producer-only stages that *are* exercisable here (raw import + the MuJoCo
``ee_pos`` FK) and, for the worker-pool path, a tiny injected worker-lane stage
that needs no GPU. They reuse the real on-disk ``fake_repo`` dataset from
``test_lerobot_import``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")

from chuck_dreamer.lerobot import importer as li  # noqa: E402
from chuck_dreamer.lerobot import episode_spec as es  # noqa: E402
from chuck_dreamer.training.episode_dataset import Episode  # noqa: E402

# Reuse the real 2-episode on-disk dataset fixture and its frame dims. The
# F811s from using ``fake_repo`` as a test arg are silenced via per-file-ignores
# in .flake8 (the standard imported-pytest-fixture pattern).
from tests.lerobot.test_lerobot_import import fake_repo, H, W  # noqa: E402,F401


def _run(root, params, **kw):
  from chuck_dreamer.lerobot.pipeline import Run
  spec = es.EpisodeSpec.parse(str(root))
  return Run(None, spec, params, **kw)


def _import(out, root, params, *, jobs, devices=None, **kw):
  run = _run(root, params)
  return list(li.import_dataset(
    run, str(out), format=kw.pop("format", "hdf5"), jobs=jobs,
    devices=devices, **kw))


def _episode_arrays(path: Path) -> dict:
  return Episode.from_file(path, format="hdf5").data


def _assert_episodes_equal(dir_a: Path, dir_b: Path, names: list[str]) -> None:
  for name in names:
    a = _episode_arrays(dir_a / name)
    b = _episode_arrays(dir_b / name)
    assert a.keys() == b.keys(), name
    for k in a:
      np.testing.assert_array_equal(a[k], b[k], err_msg=f"{name}:{k}")


# ---------------------------------------------------------------------------
# Producer-only path (no worker-lane stage): producers write directly.
# Both stages off → exercises orchestration + queues + shutdown only.
# ---------------------------------------------------------------------------


def test_parallel_raw_import_matches_serial(tmp_path, fake_repo):
  params = {"with_ee_pos": False, "with_object_pose": False}
  serial = tmp_path / "serial"
  par = tmp_path / "par"

  r_serial = _import(serial, fake_repo, params, jobs=1)
  r_par = _import(par, fake_repo, params, jobs=2)

  assert sorted(i for i, _ in r_serial) == [0, 1]
  assert sorted(i for i, _ in r_par) == [0, 1]
  _assert_episodes_equal(serial, par, ["episode-00000.hdf5", "episode-00001.hdf5"])


def test_parallel_ee_pos_matches_serial(tmp_path, fake_repo):
  # ee_pos is a real producer-lane stage (MuJoCo FK) — its output must be
  # identical whether it runs serially or on a producer thread.
  pytest.importorskip("mujoco")
  from chuck_dreamer.common import FK_MODEL_PATH
  if not FK_MODEL_PATH.exists():
    pytest.skip("FK MuJoCo model asset missing")

  params = {"with_ee_pos": True, "with_object_pose": False}
  serial = tmp_path / "serial"
  par = tmp_path / "par"

  _import(serial, fake_repo, params, jobs=1)
  _import(par, fake_repo, params, jobs=3)

  _assert_episodes_equal(serial, par, ["episode-00000.hdf5", "episode-00001.hdf5"])
  # Sanity: ee_pos actually populated (not all zeros).
  ep0 = _episode_arrays(par / "episode-00000.hdf5")
  assert np.any(ep0["ee_pos"] != 0.0)


# ---------------------------------------------------------------------------
# Worker-pool path: inject a trivial worker-lane stage so the full
# producer → queue → spawn-worker → write pipeline runs without a GPU.
# ---------------------------------------------------------------------------


def test_worker_main_reads_masks_off_episode_and_writes(tmp_path, fake_repo, monkeypatch):
  # The worker subprocess re-imports stage modules under ``spawn`` and so can't
  # be monkeypatched across the boundary; instead drive ``_worker_main``
  # directly in-process with the real ``ObjectPoseStage`` replaced (in the
  # registry it builds from) by a stamp stage. This covers the worker loop's
  # contract: build worker-lane stages → apply → pop _name_suffix → write → put
  # _Result, and that the masks ride to the stage *on the episode* (no ctx).
  import queue as _queue

  import chuck_dreamer.lerobot.stages.registry as registry
  from chuck_dreamer.lerobot import parallel
  from chuck_dreamer.lerobot.pipeline import Run

  seen_masks = {}

  class _StampStage:
    name = "object_pose"
    produces = ("object_xy",)
    requires = ()
    lane = "worker"

    def __init__(self, ctx):
      self.ctx = ctx

    def apply(self, episode, metadata):
      idx = int(metadata["config"]["episode_index"])
      seen_masks[idx] = episode.get("object_masks")
      T = len(episode["image"])
      episode["object_xy"] = np.full((T, 2), float(idx + 1), dtype=np.float32)

  # build_registry references ObjectPoseStage from its own module namespace.
  monkeypatch.setattr(registry, "ObjectPoseStage", _StampStage)

  # Assemble a real episode the way a producer would, then attach the masks as
  # a non-persisted scratch field (what segment:object does in the producer).
  run = Run(None, es.EpisodeSpec.parse(str(fake_repo)),
            {"with_ee_pos": False, "with_object_pose": False})
  sl = run.slices[0]
  episode, metadata, name_suffix = li.assemble_episode(run, sl)
  metadata["_name_suffix"] = name_suffix
  masks = [np.ones((H, W), dtype=bool)] * len(episode["image"])
  episode.set("object_masks", masks, persist=False)

  out = tmp_path / "out"
  task_q: _queue.Queue = _queue.Queue()
  result_q: _queue.Queue = _queue.Queue()
  task_q.put(parallel._Task(sl.episode_index, episode, metadata))
  task_q.put(parallel._STOP)

  parallel._worker_main(None, run.dataset_id, str(out), "hdf5",
                        ("object_pose",), task_q, result_q)

  res = result_q.get_nowait()
  assert res.error is None and res.out_path is not None
  assert Path(res.out_path).exists()
  # masks rode in on the episode and the stamp ran.
  assert seen_masks[sl.episode_index] is masks
  ep = _episode_arrays(out / "episode-00000.hdf5")
  assert np.all(ep["object_xy"] == sl.episode_index + 1)
  # The scratch masks field was NOT persisted to disk.
  import h5py
  with h5py.File(out / "episode-00000.hdf5", "r") as f:
    assert "object_masks" not in f
  # _name_suffix was popped, not persisted into metadata.
  assert "_name_suffix" not in metadata


def test_worker_main_reports_error_without_crashing(tmp_path, fake_repo, monkeypatch):
  import queue as _queue

  import chuck_dreamer.lerobot.stages.registry as registry
  from chuck_dreamer.lerobot import parallel
  from chuck_dreamer.lerobot.pipeline import Run

  class _BoomStage:
    name = "object_pose"
    produces = ()
    requires = ()
    lane = "worker"

    def __init__(self, ctx):
      self.ctx = ctx

    def apply(self, episode, metadata):
      raise ValueError("boom")

  monkeypatch.setattr(registry, "ObjectPoseStage", _BoomStage)

  run = Run(None, es.EpisodeSpec.parse(str(fake_repo)),
            {"with_ee_pos": False, "with_object_pose": False})
  sl = run.slices[0]
  episode, metadata, name_suffix = li.assemble_episode(run, sl)
  metadata["_name_suffix"] = name_suffix

  task_q: _queue.Queue = _queue.Queue()
  result_q: _queue.Queue = _queue.Queue()
  task_q.put(parallel._Task(sl.episode_index, episode, metadata))
  task_q.put(parallel._STOP)

  parallel._worker_main(None, run.dataset_id, str(tmp_path),
                        "hdf5", ("object_pose",), task_q, result_q)

  res = result_q.get_nowait()
  assert res.out_path is None
  assert res.error is not None and "boom" in res.error
