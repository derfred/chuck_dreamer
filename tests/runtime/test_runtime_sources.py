"""Tests for :mod:`chuck_dreamer.runtime.sources` (scripted policies).

The scripted producers are :class:`~chuck_dreamer.policy.Policy`
implementations. Their time -> target math (``target_at``) is pure and
asserted directly; ``act(obs)`` is checked to read ``obs.t``; and ``reset``
is checked to accept both a joint vector and a SceneConfig.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from chuck_dreamer.runtime.modalities import RuntimeObservation
from chuck_dreamer.policy import Action
from chuck_dreamer.runtime.sources import GoToPose, ManualPolicy, SineSweep


def _obs(t: float, n: int = 1) -> RuntimeObservation:
  return RuntimeObservation(t=t, q_meas=np.zeros(n))


# -- GoToPose ----------------------------------------------------------------


def test_go_to_pose_endpoints_and_midpoint():
  p = GoToPose(q_goal=np.array([2.0, -4.0]), duration_s=2.0)
  p.reset(np.array([0.0, 0.0]))
  np.testing.assert_allclose(p.target_at(0.0), [0.0, 0.0])
  np.testing.assert_allclose(p.target_at(1.0), [1.0, -2.0])
  np.testing.assert_allclose(p.target_at(2.0), [2.0, -4.0])


def test_go_to_pose_holds_after_duration():
  p = GoToPose(q_goal=np.array([1.0]), duration_s=1.0)
  p.reset(np.array([0.0]))
  np.testing.assert_allclose(p.target_at(5.0), [1.0])


def test_go_to_pose_respects_nonzero_start():
  p = GoToPose(q_goal=np.array([4.0]), duration_s=4.0)
  p.reset(np.array([2.0]))
  np.testing.assert_allclose(p.target_at(2.0), [3.0])  # halfway 2 -> 4


def test_go_to_pose_act_reads_obs_time():
  p = GoToPose(q_goal=np.array([2.0]), duration_s=2.0)
  p.reset(np.array([0.0]))
  np.testing.assert_allclose(p.act(_obs(1.0)).q, [1.0])


def test_go_to_pose_none_goal_holds_the_reset_pose():
  # `q_goal: null` in config means "go to wherever reset anchors us" -- the
  # runtime resets against the backend's home pose, so this replaces the old
  # "home" sentinel without the harness resolving anything.
  p = GoToPose(q_goal=None, duration_s=2.0)
  p.reset(np.array([0.7, -0.2]))
  np.testing.assert_allclose(p.target_at(0.0), [0.7, -0.2])
  np.testing.assert_allclose(p.target_at(5.0), [0.7, -0.2])


def test_go_to_pose_none_goal_before_reset_raises():
  with pytest.raises(RuntimeError, match="reset"):
    GoToPose(q_goal=None, duration_s=1.0).target_at(0.0)


def test_go_to_pose_rejects_bad_duration():
  with pytest.raises(ValueError, match="duration_s"):
    GoToPose(q_goal=np.array([1.0]), duration_s=0.0)


# -- SineSweep ---------------------------------------------------------------


def test_sine_sweep_at_t0_is_center():
  p = SineSweep(amplitude=1.0, freq_hz=0.5, phase=0.0)
  p.reset(np.array([0.3, -0.2]))
  np.testing.assert_allclose(p.target_at(0.0), [0.3, -0.2], atol=1e-12)


def test_sine_sweep_quarter_period_peak():
  p = SineSweep(amplitude=2.0, freq_hz=0.5, phase=0.0)  # period 2s
  p.reset(np.array([0.0]))
  np.testing.assert_allclose(p.target_at(0.5), [2.0], atol=1e-12)


def test_sine_sweep_per_joint_broadcast():
  p = SineSweep(
    amplitude=np.array([1.0, 2.0]), freq_hz=0.5,
    phase=np.array([0.0, np.pi / 2]),
  )
  p.reset(np.array([0.0, 0.0]))
  np.testing.assert_allclose(p.target_at(0.0), [0.0, 2.0], atol=1e-12)


def test_sine_sweep_explicit_center_overrides_reset():
  p = SineSweep(amplitude=1.0, freq_hz=0.5, center=np.array([5.0]))
  p.reset(np.array([0.0]))
  np.testing.assert_allclose(p.target_at(0.0), [5.0], atol=1e-12)


def test_sine_sweep_act_reads_obs_time():
  p = SineSweep(amplitude=2.0, freq_hz=0.5, phase=0.0)
  p.reset(np.array([0.0]))
  np.testing.assert_allclose(p.act(_obs(0.5)).q, [2.0], atol=1e-12)


def test_sine_sweep_target_before_reset_raises():
  with pytest.raises(RuntimeError, match="reset"):
    SineSweep(amplitude=1.0, freq_hz=0.5).target_at(0.0)


# -- reset accepts a SceneConfig-like object ---------------------------------


def test_reset_accepts_scene_with_joint_initial_qpos():
  scene = SimpleNamespace(joint_initial_qpos=[1.0, 2.0])
  p = GoToPose(q_goal=np.array([3.0, 4.0]), duration_s=2.0)
  p.reset(scene)
  np.testing.assert_allclose(p.target_at(0.0), [1.0, 2.0])


def test_reset_rejects_scene_without_qpos():
  scene = SimpleNamespace(joint_initial_qpos=None)
  p = SineSweep(amplitude=1.0, freq_hz=0.5)
  with pytest.raises(ValueError, match="joint_initial_qpos"):
    p.reset(scene)

# -- constructed straight from config ----------------------------------------


def test_params_accept_omegaconf_lists():
  """Constructors are fed ``runtime.policy.params`` verbatim, so a per-joint
  YAML list arrives as a ListConfig, not a sequence numpy understands."""
  params = OmegaConf.create({"amplitude": [1.0, 2.0], "freq_hz": 0.5,
                             "phase": [0.0, np.pi / 2]})
  p = SineSweep(amplitude=params.amplitude, freq_hz=params.freq_hz,
                phase=params.phase)
  p.reset(np.zeros(2))
  # joint 0 at t=0 sits at center; joint 1 is a quarter period ahead -> at peak.
  np.testing.assert_allclose(p.target_at(0.0), [0.0, 2.0], atol=1e-12)


def test_go_to_pose_accepts_omegaconf_goal():
  cfg = OmegaConf.create({"q_goal": [1.0, 2.0]})
  p = GoToPose(q_goal=cfg.q_goal, duration_s=1.0)
  p.reset(np.zeros(2))
  np.testing.assert_allclose(p.target_at(1.0), [1.0, 2.0])


@pytest.mark.parametrize("factory", [GoToPose, SineSweep, ManualPolicy])
def test_constructs_with_no_params(factory):
  """`target:` with no `params:` must construct -- every parameter has a default."""
  policy = factory()
  policy.reset(np.zeros(3))
  assert policy.act(_obs(0.1, n=3)).q.shape == (3,)


# -- the Action contract -----------------------------------------------------


@pytest.mark.parametrize("make", [
  lambda: GoToPose(q_goal=np.zeros(6), duration_s=1.0),
  lambda: SineSweep(amplitude=0.1, freq_hz=1.0),
  lambda: ManualPolicy(),
])
def test_scripted_policies_return_an_action(make):
  """The runtime channel takes Actions only -- there is no array fallback."""
  policy = make()
  policy.reset(np.zeros(6))
  obs = _obs(0.5, n=6)

  action = policy.act(obs)
  assert isinstance(action, Action)
  assert action.obs is obs          # tied to the observation it answers
  assert action.is_joint_space
