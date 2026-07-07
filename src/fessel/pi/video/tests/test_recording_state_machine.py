"""Recording state-machine tests — no GStreamer (fake RecordingPipelineHandle).

Verifies the V4.3 properties: serialised transitions, one recording at a time
(a start while recording OR stopping is rejected synchronously), finalisation
writes metadata only on a clean stop, a start timeout returns to idle WITHOUT
finalising (partial output stays on disk), and that a start retried once idle
runs (the plan's `stopping|start->queue` was simplified to reject).
"""

import time

from fessel_schemas import ModeTriplet, RecordingStateValue

from video.recording_state_machine import RecordingPipelineHandle, RecordingStateMachine

MODE = ModeTriplet(resolution="1920x1080", fps=30, bitrate_bps=8_000_000)


class FakeRec(RecordingPipelineHandle):
  def __init__(self, registry, sm_box, defer_exit=False):
    self.started = False
    self.stopped = False
    self._sm_box = sm_box
    # When defer_exit is set, stop() does NOT emit EXITED — the test drives it
    # via emit_exit(), so the `stopping` state is deterministically observable.
    self._defer_exit = defer_exit
    registry.append(self)

  def start(self):
    self.started = True

  def stop(self):
    self.stopped = True
    if not self._defer_exit:
      self._sm_box[0].on_exited()

  def emit_exit(self):
    self._sm_box[0].on_exited()


def make_sm(start_timeout_s=0.3, defer_exit=False):
  live = []
  sm_box = [None]
  states = []
  finals = []
  # Records (rid, lookback_seconds) each factory call saw, so look-back
  # propagation start -> factory can be asserted.
  spawns = []

  def factory(rid, lookback_seconds):
    spawns.append((rid, lookback_seconds))
    return FakeRec(live, sm_box, defer_exit=defer_exit)

  def publish(state, rid, started):
    states.append((state, rid))

  def finalise(meta, upload_when_done):
    finals.append((meta, upload_when_done))

  sm = RecordingStateMachine(
    pipeline_factory=factory,
    publish_state=publish,
    finalise=finalise,
    mode_provider=lambda: MODE,
    start_timeout_s=start_timeout_s,
    count_segments=lambda _id: 4,
  )
  sm_box[0] = sm
  return sm, live, states, finals, spawns


def wait_until(predicate, timeout=2.0):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if predicate():
      return True
    time.sleep(0.01)
  return False


def test_start_to_recording_then_stop_finalises():
  sm, live, _, finals, spawns = make_sm()
  sm.start()
  assert sm.request_start("r1", operator="octocat") is True
  assert wait_until(lambda: sm.state is RecordingStateValue.starting)
  sm.on_segment()
  assert wait_until(lambda: sm.state is RecordingStateValue.recording)
  assert sm.active_recording_id == "r1"
  assert len(live) == 1 and live[0].started

  sm.request_stop()
  assert wait_until(lambda: sm.state is RecordingStateValue.idle)
  assert live[0].stopped
  assert len(finals) == 1
  meta, upload = finals[0]
  assert meta.id == "r1" and meta.operator == "octocat" and meta.segments == 4
  # Mode comes from the mode_provider (a deploy setting), not the request.
  assert meta.mode.resolution == "1920x1080" and meta.ended_at is not None
  assert upload is False  # not requested
  sm.shutdown()


def test_lookback_and_upload_propagate():
  sm, live, _, finals, spawns = make_sm()
  sm.start()
  assert sm.request_start("r1", lookback_seconds=45.0, upload_when_done=True) is True
  assert wait_until(lambda: sm.state is RecordingStateValue.starting)
  # The look-back reached the factory (-> the recording handle).
  assert spawns == [("r1", 45.0)]
  sm.on_segment()
  assert wait_until(lambda: sm.state is RecordingStateValue.recording)
  sm.request_stop()
  assert wait_until(lambda: len(finals) == 1)
  _meta, upload = finals[0]
  assert upload is True  # flagged for upload on finalise
  sm.shutdown()


def test_only_one_recording_at_a_time():
  sm, _, _, _, _ = make_sm()
  sm.start()
  assert sm.request_start("r1") is True
  assert wait_until(
    lambda: sm.state is RecordingStateValue.recording or sm.state is RecordingStateValue.starting
  )
  sm.on_segment()
  assert wait_until(lambda: sm.state is RecordingStateValue.recording)
  # A start while recording is rejected SYNCHRONOUSLY (so the HTTP layer 409s).
  assert sm.request_start("r2") is False
  assert sm.active_recording_id == "r1"
  sm.shutdown()


def test_start_timeout_returns_idle_without_finalise():
  sm, _, _, finals, _ = make_sm(start_timeout_s=0.2)
  sm.start()
  assert sm.request_start("r1") is True
  assert wait_until(lambda: sm.state is RecordingStateValue.starting)
  # Never call on_segment -> timeout fires -> idle. Partial output stays on disk
  # but NO metadata is written (the recording never finalised, V4.3).
  assert wait_until(lambda: sm.state is RecordingStateValue.idle, timeout=2.0)
  assert finals == []
  sm.shutdown()


def test_failed_start_returns_idle_without_finalise():
  sm, _, _, finals, _ = make_sm()
  sm.start()
  assert sm.request_start("r1") is True
  assert wait_until(lambda: sm.state is RecordingStateValue.starting)
  sm.on_failed()
  assert wait_until(lambda: sm.state is RecordingStateValue.idle)
  assert finals == []
  sm.shutdown()


def test_start_during_stopping_is_rejected_then_runs_once_idle():
  # V4.3 (simplified): a start in ANY non-idle state — including `stopping` —
  # is rejected synchronously (the plan's `stopping|start->queue` was dropped,
  # see recording_state_machine docstring). Once the prior recording finalises
  # to idle, a fresh start proceeds. So the operator gesture is "retry after
  # stop", not "queued across teardown".
  # defer_exit keeps the SM in `stopping` until we drive emit_exit(), so the
  # reject-during-stopping window is deterministically observable.
  sm, live, _, finals, _ = make_sm(defer_exit=True)
  sm.start()
  sm.request_start("r1")
  assert wait_until(lambda: sm.state is RecordingStateValue.starting)
  sm.on_segment()
  assert wait_until(lambda: sm.state is RecordingStateValue.recording)
  sm.request_stop()
  assert wait_until(lambda: sm.state is RecordingStateValue.stopping)
  # A start arriving WHILE stopping is rejected (not queued).
  assert sm.request_start("r2") is False
  # Now let the prior recording finalise cleanly.
  live[0].emit_exit()
  assert wait_until(lambda: len(finals) == 1)  # r1 finalised
  assert wait_until(lambda: sm.state is RecordingStateValue.idle)
  # Once idle, the retried start proceeds (reaching `recording` needs only the
  # first segment; the deferred fake's stop-without-exit is harmless on
  # shutdown's bounded join).
  assert sm.request_start("r2") is True
  assert wait_until(lambda: sm.state is RecordingStateValue.starting)
  sm.on_segment()
  assert wait_until(lambda: sm.state is RecordingStateValue.recording)
  assert sm.active_recording_id == "r2"
  sm.shutdown()
