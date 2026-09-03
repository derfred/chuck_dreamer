"""Tests for :mod:`chuck_dreamer.runtime.control_channel`."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from chuck_dreamer.policy import Action
from chuck_dreamer.runtime.control_channel import ControlChannel
from chuck_dreamer.runtime.control_state import ControlState


class _Obs:
  def __init__(self, t=0.0):
    self.t = t


def _action(*q, t=0.0):
  return Action(_Obs(t), q=np.asarray(q, dtype=float))


def test_none_before_first_publish():
  ch = ControlChannel()
  assert ch.get() is None
  assert ch.seq == 0


def test_latest_wins():
  ch = ControlChannel()
  ch.publish(_action(1.0, 2.0))
  ch.publish(_action(3.0, 4.0))
  np.testing.assert_array_equal(ch.get().q, [3.0, 4.0])


def test_seq_tracks_the_pending_action():
  """The channel keeps no counter of its own -- identity rides on the Action."""
  ch = ControlChannel()
  a, b = _action(0.0), _action(0.0)
  ch.publish(a)
  assert ch.seq == a.seq
  ch.publish(b)
  assert ch.seq == b.seq
  assert b.seq > a.seq


def test_actions_are_immutable_so_the_slot_needs_no_copy():
  ch = ControlChannel()
  q = np.array([1.0, 2.0])
  ch.publish(Action(_Obs(), q=q))
  q[0] = 99.0                                   # mutate the caller's array
  np.testing.assert_array_equal(ch.get().q, [1.0, 2.0])
  with pytest.raises(ValueError):               # and the stored one is read-only
    ch.get().q[0] = 99.0


def test_repeated_setpoint_is_still_a_new_action():
  """Identical joint values must not read as "no new command"."""
  ch = ControlChannel()
  first = _action(1.0, 2.0)
  ch.publish(first)
  same_value = _action(1.0, 2.0)
  ch.publish(same_value)
  assert ch.get() != first
  assert ch.get() == same_value


def test_concurrent_writers_smoke():
  ch = ControlChannel()
  published = {float(i) for i in range(8)}

  def writer(i):
    for _ in range(200):
      ch.publish(_action(float(i)))

  threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()

  final = ch.get()
  assert final is not None
  assert float(final.q[0]) in published
  assert final.seq == ch.seq


# -- control -> policy direction ----------------------------------------------


def _state(*q, now=0.0, tick=0):
  return ControlState(now=now, q=np.asarray(q, dtype=float), tick=tick)


def test_no_state_before_the_first_tick():
  ch = ControlChannel()
  assert ch.state() is None


def test_state_is_latest_wins():
  ch = ControlChannel()
  ch.publish_state(_state(1.0, 2.0, tick=1))
  ch.publish_state(_state(3.0, 4.0, tick=2))
  np.testing.assert_array_equal(ch.state().q, [3.0, 4.0])


def test_identity_rides_on_the_state_not_the_channel():
  """The channel keeps no counter: `tick` on the state is what says which one."""
  ch = ControlChannel()
  ch.publish_state(_state(0.0, tick=7))
  assert ch.state().tick == 7
  ch.publish_state(_state(0.0, tick=8))
  assert ch.state().tick == 8


def test_the_two_directions_are_independent():
  """A publish in one direction must not disturb the other slot."""
  ch = ControlChannel()
  ch.publish(_action(1.0))
  ch.publish_state(_state(9.0))
  assert ch.get() is not None and ch.state() is not None
  np.testing.assert_array_equal(ch.get().q, [1.0])
  np.testing.assert_array_equal(ch.state().q, [9.0])


def test_concurrent_state_and_action_writers_smoke():
  """Both directions under contention: no lost slot, no torn seq."""
  ch = ControlChannel()

  def action_writer(i):
    for _ in range(200):
      ch.publish(_action(float(i)))

  def state_writer(i):
    for _ in range(200):
      ch.publish_state(_state(float(i)))

  threads = ([threading.Thread(target=action_writer, args=(i,)) for i in range(4)]
             + [threading.Thread(target=state_writer, args=(i,)) for i in range(4)])
  for t in threads:
    t.start()
  for t in threads:
    t.join()

  assert ch.get() is not None
  assert ch.state() is not None
