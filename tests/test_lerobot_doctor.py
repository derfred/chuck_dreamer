"""Tests for ``import-lerobot --doctor`` (``_doctor_import_lerobot``).

The doctor's whole point is to be *generic*: it drives the same
:class:`Run` the importer would use, iterating each selected episode's
pipeline and reporting every stage's ``requirements()`` without hard-coding
any artifact. We verify that with a fake ``Run`` whose pipeline yields fake
stages whose requirements we control, then assert the doctor's pass/fail
result and its emitted lines.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from chuck_dreamer.lerobot import cli as lcli
from chuck_dreamer.lerobot.stages import Requirement


class _FakeStage:
  def __init__(self, name, reqs):
    self.name = name
    self._reqs = reqs
    self.requires = ()
    self.produces = ()

  def requirements(self):
    return self._reqs


class _FakeRun:
  """Stand-in for Run: the doctor reads .spec/.with_* and calls
  read_slices() + pipeline(ep_idx). We yield the same fake stages for every
  episode and expose a configurable selected-episode list."""

  def __init__(self, stages, *, episodes=(0,), with_ee_pos=True,
               with_object_pose=True, dataset_id="user/ds"):
    self.spec = SimpleNamespace(dataset_id=dataset_id)
    self.with_ee_pos = with_ee_pos
    self.with_object_pose = with_object_pose
    self._stages = stages
    self._episodes = episodes

  def read_slices(self):
    slices = [SimpleNamespace(episode_index=i) for i in self._episodes]
    return slices, "observation.images.wrist"

  def pipeline(self, episode_index):
    return list(self._stages)


def _doctor(run, episode_config_path=None):
  return lcli._doctor_import_lerobot(run, episode_config_path=episode_config_path)


def _existing(tmp_path, name="ok.json") -> Path:
  p = tmp_path / name
  p.write_text("{}")
  return p


def _missing(tmp_path, name="missing.json") -> Path:
  return tmp_path / name


# ---------------------------------------------------------------------------
# all present / some missing
# ---------------------------------------------------------------------------


def test_doctor_passes_when_all_requirements_satisfied(tmp_path):
  reqs = [Requirement("artifact A", _existing(tmp_path, "a"), "make a")]
  assert _doctor(_FakeRun([_FakeStage("s", reqs)])) is True


def test_doctor_fails_when_a_requirement_missing(tmp_path):
  reqs = [
    Requirement("present", _existing(tmp_path, "a"), "make a"),
    Requirement("absent", _missing(tmp_path, "b"), "make b"),
  ]
  assert _doctor(_FakeRun([_FakeStage("s", reqs)])) is False


def test_doctor_reports_remediation_for_missing(tmp_path, capsys):
  reqs = [Requirement("absent", _missing(tmp_path), "run the fix command")]
  _doctor(_FakeRun([_FakeStage("s", reqs)]))
  out = capsys.readouterr().out
  assert "absent" in out
  assert "run the fix command" in out


def test_doctor_is_generic_over_whatever_stages_declare(tmp_path, capsys):
  # The doctor must surface requirements from arbitrary stages — proving it
  # doesn't hard-code the known artifact set.
  reqs = [Requirement("totally novel artifact", _missing(tmp_path), "do it")]
  ok = _doctor(_FakeRun([_FakeStage("brand_new_stage", reqs)]))
  assert ok is False
  assert "totally novel artifact" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# dedup by (label, path)
# ---------------------------------------------------------------------------


def test_doctor_dedups_requirements_sharing_a_path(tmp_path, capsys):
  # Two stages naming the same artifact (e.g. segmentation + pose both want
  # the mesh) should print it once — even though it recurs across every
  # episode's pipeline.
  shared = _missing(tmp_path, "mesh.obj")
  s1 = _FakeStage("a", [Requirement("object mesh", shared, "set mesh")])
  s2 = _FakeStage("b", [Requirement("object mesh", shared, "set mesh")])
  _doctor(_FakeRun([s1, s2], episodes=(0, 1, 2)))
  out = capsys.readouterr().out
  assert out.count("object mesh") == 1


# ---------------------------------------------------------------------------
# per-episode requirements (distinct label) are checked once each
# ---------------------------------------------------------------------------


def test_doctor_checks_per_episode_requirements_separately(tmp_path, capsys):
  # A stage whose requirement label varies per episode must be reported for
  # each episode, not deduped away.
  class _PerEpStage:
    name = "seg"
    requires = ()
    produces = ()

    def __init__(self, run):
      self._run = run

    def requirements(self):
      ep = self._run._current_ep
      return [Requirement(f"prompt (ep {ep})", _missing(tmp_path), "prompt it")]

  run = _FakeRun([], episodes=(0, 1))

  # Track which episode pipeline() was asked for so the stage can label by it.
  run._current_ep = 0
  stage = _PerEpStage(run)

  def pipeline(episode_index):
    run._current_ep = episode_index
    return [stage]

  run.pipeline = pipeline
  _doctor(run)
  out = capsys.readouterr().out
  assert "prompt (ep 0)" in out
  assert "prompt (ep 1)" in out


# ---------------------------------------------------------------------------
# episode-config / T_world_arm branch (importer-level, not a stage)
# ---------------------------------------------------------------------------


def test_doctor_checks_episode_config_and_fk_when_path_given(tmp_path, capsys):
  # No stage requirements; the episode-config branch must still check the
  # config file AND the FK model (FK runs for T_world_arm regardless of
  # --with-ee-pos).
  ep_cfg = _existing(tmp_path, "fk_episode_config.json")
  _doctor(
    _FakeRun([], with_ee_pos=False, with_object_pose=False),
    episode_config_path=ep_cfg)
  out = capsys.readouterr().out
  assert "fk_episode_config.json" in out
  assert "FK MuJoCo model" in out


def test_doctor_no_episode_config_skips_that_branch(capsys):
  ok = _doctor(_FakeRun([], with_ee_pos=False, with_object_pose=False))
  out = capsys.readouterr().out
  # Nothing to check → passes, and the episode-config artifact isn't named.
  assert ok is True
  assert "fk_episode_config.json" not in out
