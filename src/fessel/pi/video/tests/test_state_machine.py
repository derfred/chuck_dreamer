"""State-machine tests — no GStreamer needed (fake PipelineHandle).

These verify the production property the spike lacked: serialised
transitions, exactly one pipeline at a time, no leaked pipeline, and
correct off/starting/running/stopping flow incl. connect timeout and
disconnect-with-backoff.
"""

import time

from fessel_schemas import LiveStateValue, ModeTriplet

from video.state_machine import LiveStateMachine, PipelineHandle

MODE = ModeTriplet(resolution="640x480", fps=30, bitrate_bps=1_000_000)


class FakePipeline(PipelineHandle):
  def __init__(self, registry: list, sm_box: list) -> None:
    self.started = False
    self.stopped = False
    self.idr_count = 0
    self._registry = registry
    self._sm_box = sm_box
    registry.append(self)

  def start(self) -> None:
    self.started = True

  def stop(self) -> None:
    self.stopped = True
    # Graceful teardown reports EXITED back to the SM.
    self._sm_box[0].on_exited()

  def force_idr(self) -> None:
    self.idr_count += 1


def make_sm():
  live: list[FakePipeline] = []
  sm_box: list = [None]
  states: list[LiveStateValue] = []

  def factory(mode, path):
    return FakePipeline(live, sm_box)

  def publish(state, path, mode):
    states.append(state)

  sm = LiveStateMachine(
    pipeline_factory=factory,
    publish_state=publish,
    connect_timeout_s=0.3,
    reconnect_backoff_s=0.1,
  )
  sm_box[0] = sm
  return sm, live, states


def wait_until(predicate, timeout=2.0):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if predicate():
      return True
    time.sleep(0.01)
  return False


def test_activate_to_running_then_deactivate_to_off():
  sm, live, _ = make_sm()
  sm.start()
  sm.request_activate(MODE, "pi")
  assert wait_until(lambda: sm.state is LiveStateValue.starting)
  sm.on_connected()
  assert wait_until(lambda: sm.state is LiveStateValue.running)
  assert len(live) == 1 and live[0].started

  sm.request_deactivate("pi")
  assert wait_until(lambda: sm.state is LiveStateValue.off)
  assert live[0].stopped
  sm.shutdown()


def test_connect_timeout_returns_to_off():
  sm, live, _ = make_sm()
  sm.start()
  sm.request_activate(MODE, "pi")
  assert wait_until(lambda: sm.state is LiveStateValue.starting)
  # Never call on_connected -> connect_timeout (0.3s) fires -> off.
  assert wait_until(lambda: sm.state is LiveStateValue.off, timeout=2.0)
  sm.shutdown()


def test_no_overlapping_pipelines_under_rapid_cycles():
  sm, live, _ = make_sm()
  sm.start()
  for _ in range(5):
    sm.request_activate(MODE, "pi")
    sm.request_deactivate("pi")
  # Settle.
  assert wait_until(lambda: sm.state in (LiveStateValue.off, LiveStateValue.starting))
  time.sleep(0.2)
  # No two pipelines ever started without the prior one being stopped:
  # at most one pipeline is un-stopped at any settle point.
  unstopped = [p for p in live if p.started and not p.stopped]
  assert len(unstopped) <= 1
  sm.shutdown()


def test_force_idr_fired_once_per_connect():
  # V2.2: IDR is forced exactly once when the publisher connects, and an
  # idempotent re-activate in running does not re-fire it on the same
  # pipeline (over-triggering can cause bitrate spikes).
  sm, live, _ = make_sm()
  sm.start()
  sm.request_activate(MODE, "pi")
  assert wait_until(lambda: sm.state is LiveStateValue.starting)
  sm.on_connected()
  assert wait_until(lambda: sm.state is LiveStateValue.running)
  assert wait_until(lambda: live[0].idr_count == 1)
  # Idempotent activate while running must not re-force IDR.
  sm.request_activate(MODE, "pi")
  time.sleep(0.1)
  assert live[0].idr_count == 1
  sm.shutdown()


def test_disconnect_triggers_reconnect_to_starting():
  sm, live, _ = make_sm()
  sm.start()
  sm.request_activate(MODE, "pi")
  assert wait_until(lambda: sm.state is LiveStateValue.starting)
  sm.on_connected()
  assert wait_until(lambda: sm.state is LiveStateValue.running)
  n_before = len(live)
  sm.on_disconnected()
  # Goes back to starting and (after backoff) respawns a new pipeline.
  assert wait_until(lambda: sm.state is LiveStateValue.starting)
  assert wait_until(lambda: len(live) > n_before, timeout=2.0)
  sm.shutdown()
