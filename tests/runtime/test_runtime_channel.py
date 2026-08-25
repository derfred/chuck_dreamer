"""Tests for :mod:`chuck_dreamer.runtime.setpoint_channel`."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from chuck_dreamer.policy import Action
from chuck_dreamer.runtime.setpoint_channel import SetpointChannel


class _Obs:
  def __init__(self, t=0.0):
    self.t = t


def _action(*q, t=0.0):
  return Action(_Obs(t), q=np.asarray(q, dtype=float))


def test_none_before_first_publish():
  ch = SetpointChannel()
  assert ch.get() is None
  assert ch.seq == 0


def test_latest_wins():
  ch = SetpointChannel()
  ch.publish(_action(1.0, 2.0))
  ch.publish(_action(3.0, 4.0))
  np.testing.assert_array_equal(ch.get().q, [3.0, 4.0])


def test_seq_tracks_the_pending_action():
  """The channel keeps no counter of its own -- identity rides on the Action."""
  ch = SetpointChannel()
  a, b = _action(0.0), _action(0.0)
  ch.publish(a)
  assert ch.seq == a.seq
  ch.publish(b)
  assert ch.seq == b.seq
  assert b.seq > a.seq


def test_publish_rejects_a_bare_array():
  """A policy returning a raw vector is a programming error at this seam."""
  ch = SetpointChannel()
  with pytest.raises(TypeError, match="Action"):
    ch.publish(np.zeros(2))


def test_actions_are_immutable_so_the_slot_needs_no_copy():
  ch = SetpointChannel()
  q = np.array([1.0, 2.0])
  ch.publish(Action(_Obs(), q=q))
  q[0] = 99.0                                   # mutate the caller's array
  np.testing.assert_array_equal(ch.get().q, [1.0, 2.0])
  with pytest.raises(ValueError):               # and the stored one is read-only
    ch.get().q[0] = 99.0


def test_repeated_setpoint_is_still_a_new_action():
  """Identical joint values must not read as "no new command"."""
  ch = SetpointChannel()
  first = _action(1.0, 2.0)
  ch.publish(first)
  same_value = _action(1.0, 2.0)
  ch.publish(same_value)
  assert ch.get() != first
  assert ch.get() == same_value


def test_concurrent_writers_smoke():
  ch = SetpointChannel()
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
