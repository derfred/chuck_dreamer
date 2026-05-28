"""Tests for ``import-lerobot --doctor`` (``_doctor_import_lerobot``).

The doctor's whole point post-refactor is to be *generic*: it resolves the
same stages the importer would run and reports each stage's
``requirements()`` without hard-coding any artifact. We verify that by
stubbing ``resolve_stages`` with fake stages whose requirements we control,
then asserting the doctor's pass/fail result and its emitted lines.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chuck_dreamer.lerobot import cli as lcli
from chuck_dreamer.lerobot.stages import Requirement


class _FakeStage:
  def __init__(self, name, reqs):
    self.name = name
    self._reqs = reqs
    self.requires = ()
    self.produces = ()

  def requirements(self, ctx):
    return self._reqs


def _patch_resolve(monkeypatch, stages):
  """Make the doctor's resolve_stages return ``stages`` regardless of flags.

  The doctor imports ``resolve_stages`` lazily from ``.stages`` inside the
  function body, so it must be patched at the source module, not on ``cli``.
  """
  monkeypatch.setattr(
    "chuck_dreamer.lerobot.stages.resolve_stages", lambda enabled: stages)


@pytest.fixture
def fake_ctx():
  """A click context object exposing the obj['config_path'] the doctor reads."""
  return SimpleNamespace(obj={"config_path": []})


def _existing(tmp_path, name="ok.json") -> Path:
  p = tmp_path / name
  p.write_text("{}")
  return p


def _missing(tmp_path, name="missing.json") -> Path:
  return tmp_path / name


# ---------------------------------------------------------------------------
# all present / some missing
# ---------------------------------------------------------------------------


def test_doctor_passes_when_all_requirements_satisfied(
    monkeypatch, fake_ctx, tmp_path):
  reqs = [Requirement("artifact A", _existing(tmp_path, "a"), "make a")]
  _patch_resolve(monkeypatch, [_FakeStage("s", reqs)])

  ok = lcli._doctor_import_lerobot(
    fake_ctx, "user/ds",
    with_ee_pos=True, with_object_pose=True, episode_config_path=None)
  assert ok is True


def test_doctor_fails_when_a_requirement_missing(
    monkeypatch, fake_ctx, tmp_path):
  reqs = [
    Requirement("present", _existing(tmp_path, "a"), "make a"),
    Requirement("absent", _missing(tmp_path, "b"), "make b"),
  ]
  _patch_resolve(monkeypatch, [_FakeStage("s", reqs)])

  ok = lcli._doctor_import_lerobot(
    fake_ctx, "user/ds",
    with_ee_pos=True, with_object_pose=True, episode_config_path=None)
  assert ok is False


def test_doctor_reports_remediation_for_missing(
    monkeypatch, fake_ctx, tmp_path, capsys):
  reqs = [Requirement("absent", _missing(tmp_path), "run the fix command")]
  _patch_resolve(monkeypatch, [_FakeStage("s", reqs)])

  lcli._doctor_import_lerobot(
    fake_ctx, "user/ds",
    with_ee_pos=True, with_object_pose=True, episode_config_path=None)
  out = capsys.readouterr().out
  assert "absent" in out
  assert "run the fix command" in out


def test_doctor_is_generic_over_whatever_stages_declare(
    monkeypatch, fake_ctx, tmp_path, capsys):
  # The doctor must surface requirements from arbitrary stages — proving it
  # doesn't hard-code the known artifact set.
  reqs = [Requirement("totally novel artifact", _missing(tmp_path), "do it")]
  _patch_resolve(monkeypatch, [_FakeStage("brand_new_stage", reqs)])

  ok = lcli._doctor_import_lerobot(
    fake_ctx, "user/ds",
    with_ee_pos=True, with_object_pose=True, episode_config_path=None)
  assert ok is False
  assert "totally novel artifact" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# dedup by path
# ---------------------------------------------------------------------------


def test_doctor_dedups_requirements_sharing_a_path(
    monkeypatch, fake_ctx, tmp_path, capsys):
  # Two stages naming the same artifact (e.g. segmentation + pose both want
  # the mesh) should print it once.
  shared = _missing(tmp_path, "mesh.obj")
  s1 = _FakeStage("a", [Requirement("object mesh", shared, "set mesh")])
  s2 = _FakeStage("b", [Requirement("object mesh", shared, "set mesh")])
  _patch_resolve(monkeypatch, [s1, s2])

  lcli._doctor_import_lerobot(
    fake_ctx, "user/ds",
    with_ee_pos=True, with_object_pose=True, episode_config_path=None)
  out = capsys.readouterr().out
  assert out.count("object mesh") == 1


# ---------------------------------------------------------------------------
# episode-config / T_world_arm branch (importer-level, not a stage)
# ---------------------------------------------------------------------------


def test_doctor_checks_episode_config_and_fk_when_path_given(
    monkeypatch, fake_ctx, tmp_path, capsys):
  # No stage requirements; the episode-config branch must still check the
  # config file AND the FK model (FK runs for T_world_arm regardless of
  # --with-ee-pos).
  _patch_resolve(monkeypatch, [])
  ep_cfg = _existing(tmp_path, "fk_episode_config.json")

  lcli._doctor_import_lerobot(
    fake_ctx, "user/ds",
    with_ee_pos=False, with_object_pose=False, episode_config_path=ep_cfg)
  out = capsys.readouterr().out
  assert "fk_episode_config.json" in out
  assert "FK MuJoCo model" in out


def test_doctor_no_episode_config_skips_that_branch(
    monkeypatch, fake_ctx, capsys):
  _patch_resolve(monkeypatch, [])
  ok = lcli._doctor_import_lerobot(
    fake_ctx, "user/ds",
    with_ee_pos=False, with_object_pose=False, episode_config_path=None)
  out = capsys.readouterr().out
  # Nothing to check → passes, and the episode-config artifact isn't named.
  assert ok is True
  assert "fk_episode_config.json" not in out
