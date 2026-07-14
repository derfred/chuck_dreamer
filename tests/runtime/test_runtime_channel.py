"""Tests for :mod:`chuck_dreamer.runtime.setpoint_channel`."""

from __future__ import annotations

import threading

import numpy as np

from chuck_dreamer.runtime.setpoint_channel import SetpointChannel


def test_none_before_first_publish():
  ch = SetpointChannel()
  assert ch.get() is None
  assert ch.seq == 0


def test_latest_wins():
  ch = SetpointChannel()
  ch.publish(np.array([1.0, 2.0]))
  ch.publish(np.array([3.0, 4.0]))
  np.testing.assert_array_equal(ch.get(), [3.0, 4.0])


def test_seq_increments_per_publish():
  ch = SetpointChannel()
  ch.publish(np.zeros(2))
  ch.publish(np.zeros(2))
  ch.publish(np.zeros(2))
  assert ch.seq == 3


def test_copy_isolation_on_publish():
  ch = SetpointChannel()
  q = np.array([1.0, 2.0])
  ch.publish(q)
  q[0] = 99.0  # mutate caller's array after publish
  np.testing.assert_array_equal(ch.get(), [1.0, 2.0])


def test_copy_isolation_on_get():
  ch = SetpointChannel()
  ch.publish(np.array([1.0, 2.0]))
  got = ch.get()
  got[0] = 99.0  # mutate the returned array
  np.testing.assert_array_equal(ch.get(), [1.0, 2.0])


def test_get_with_seq_atomic():
  ch = SetpointChannel()
  ch.publish(np.array([5.0]))
  value, seq = ch.get_with_seq()
  np.testing.assert_array_equal(value, [5.0])
  assert seq == 1


def test_concurrent_writers_smoke():
  ch = SetpointChannel()
  published = {(float(i),) for i in range(8)}

  def writer(i):
    for _ in range(200):
      ch.publish(np.array([float(i)]))

  threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()

  final = ch.get()
  assert final is not None
  assert (float(final[0]),) in published
  assert ch.seq == 8 * 200
