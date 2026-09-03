from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from chuck_dreamer.common.tracks import Frame
from chuck_dreamer.policy import Action, ActionSpaceError


class _Obs:
  def __init__(self, t=0.0):
    self.t = t


# -- target spaces -----------------------------------------------------------


def test_joint_space_action():
  obs = _Obs(1.5)
  a = Action(obs, q=[0.1, 0.2, 0.3])
  assert a.is_joint_space
  assert a.frame is Frame.NONE
  assert a.pos is None
  assert a.obs is obs
  np.testing.assert_array_equal(a.q, [0.1, 0.2, 0.3])


@pytest.mark.parametrize("kwarg,frame", [
  ("arm_pos", Frame.ARM),
  ("world_pos", Frame.TABLE),   # the mat/world frame is TABLE in tracks.py
])
def test_cartesian_action_carries_its_frame(kwarg, frame):
  a = Action(_Obs(), **{kwarg: [0.2, 0.0, 0.15]})
  assert not a.is_joint_space
  assert a.frame is frame
  np.testing.assert_array_equal(a.pos, [0.2, 0.0, 0.15])


def test_cartesian_action_refuses_to_guess_joints():
  """The loop commands joints; a half-specified target must never pass silently."""
  a = Action(_Obs(), arm_pos=[0.2, 0.0, 0.15])
  with pytest.raises(ActionSpaceError, match="resolve it through IK"):
    _ = a.q


def test_exactly_one_target_space_required():
  with pytest.raises(ActionSpaceError, match="needs one of"):
    Action(_Obs())
  with pytest.raises(ActionSpaceError, match="exactly one"):
    Action(_Obs(), q=[0.0], arm_pos=[0.0, 0.0, 0.0])


def test_cartesian_targets_must_be_3d():
  with pytest.raises(ActionSpaceError, match="3-D point"):
    Action(_Obs(), arm_pos=[0.2, 0.0])


def test_targets_must_be_one_dimensional():
  with pytest.raises(ActionSpaceError, match="1-D vector"):
    Action(_Obs(), q=np.zeros((2, 3)))


# -- identity ----------------------------------------------------------------


def test_seq_is_unique_and_increasing():
  a, b, c = (Action(_Obs(), q=[0.0]) for _ in range(3))
  assert a.seq < b.seq < c.seq


def test_equal_setpoints_are_still_distinct_commands():
  """A policy re-publishing the same pose is issuing it again."""
  obs = _Obs()
  a = Action(obs, q=[1.0, 2.0])
  b = Action(obs, q=[1.0, 2.0])
  assert a != b
  assert a == a


def test_comparison_never_raises_on_array_ambiguity():
  """`if action != current` must work; elementwise != would be ambiguous."""
  a = Action(_Obs(), q=[1.0, 2.0, 3.0])
  b = Action(_Obs(), q=[1.0, 2.0, 3.0])
  assert bool(a != b) is True          # would raise if compared elementwise
  assert (a != None) is True           # noqa: E711 - and against a non-Action


def test_hashable_by_seq():
  a = Action(_Obs(), q=[0.0])
  assert len({a, a}) == 1


def test_seq_is_unique_across_threads():
  seqs: list[int] = []
  lock = threading.Lock()

  def mint():
    local = [Action(_Obs(), q=[0.0]).seq for _ in range(200)]
    with lock:
      seqs.extend(local)

  threads = [threading.Thread(target=mint) for _ in range(8)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  assert len(seqs) == len(set(seqs)) == 8 * 200


# -- immutability + resolve --------------------------------------------------


def test_target_is_isolated_from_the_caller():
  q = np.array([1.0, 2.0])
  a = Action(_Obs(), q=q)
  q[0] = 99.0
  np.testing.assert_array_equal(a.q, [1.0, 2.0])
  with pytest.raises(ValueError):
    a.q[0] = 99.0


def test_resolve_keeps_obs_and_seq():
  """Resolving through IK does not make it a different command."""
  obs = _Obs(2.0)
  cartesian = Action(obs, arm_pos=[0.2, 0.0, 0.15])
  resolved = cartesian.resolve(np.full(6, 0.5))

  assert resolved.seq == cartesian.seq       # same command, now in joint space
  assert resolved.obs is obs
  assert resolved.is_joint_space
  np.testing.assert_array_equal(resolved.q, np.full(6, 0.5))


# -- creation time / age -----------------------------------------------------


def test_t_created_is_an_absolute_monotonic_stamp():
  """Raw ``time.monotonic()``, so a consumer keeping a different clock origin
  can still subtract it — the property ``obs.t`` lacks, since that one is
  relative to the *producing* loop's start."""
  before = time.monotonic()
  a      = Action(_Obs(), q=np.zeros(3))
  after  = time.monotonic()
  assert before <= a.t_created <= after


def test_age_at_measures_from_creation_to_the_given_instant():
  a   = Action(_Obs(), q=np.zeros(3))
  age = a.age_at(a.t_created + 0.25)
  assert age == pytest.approx(0.25)
  # And against a real later clock reading it is positive but small.
  assert 0.0 <= a.age_at(time.monotonic()) < 1.0


def test_age_at_is_negative_for_an_instant_before_creation():
  """No clamping: a caller passing a stale timestamp gets an honest negative
  rather than a zero that would hide the mistake."""
  a = Action(_Obs(), q=np.zeros(3))
  assert a.age_at(a.t_created - 0.1) == pytest.approx(-0.1)


def test_resolve_keeps_the_original_creation_time():
  """Resolving does not make it a different command (it keeps ``seq``), so it
  must not reset the clock either — otherwise IK would erase the latency the
  action accumulated before it."""
  a = Action(_Obs(), arm_pos=[0.1, 0.2, 0.3])
  r = a.resolve(np.zeros(6))
  assert r.t_created == a.t_created
