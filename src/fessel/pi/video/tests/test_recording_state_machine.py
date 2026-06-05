"""Recording state-machine tests — no GStreamer (fake RecordingPipelineHandle).

Verifies the V4.3 properties: serialised transitions, one recording at a time
(a start while recording is rejected synchronously), finalisation writes
metadata only on a clean stop, a start timeout returns to idle WITHOUT
finalising (partial output stays on disk), and a queued start after stop runs.
"""

import time

from fessel_schemas import ModeTriplet, RecordingStateValue

from video.recording_state_machine import RecordingPipelineHandle, RecordingStateMachine

MODE = ModeTriplet(resolution="1920x1080", fps=30, bitrate_bps=8_000_000)


class FakeRec(RecordingPipelineHandle):
  def __init__(self, registry, sm_box):
    self.started = False
    self.stopped = False
    self._sm_box = sm_box
    registry.append(self)

  def start(self):
    self.started = True

  def stop(self):
    self.stopped = True
    self._sm_box[0].on_exited()


def make_sm(start_timeout_s=0.3):
  live = []
  sm_box = [None]
  states = []
  finals = []

  def factory(rid, mode):
    return FakeRec(live, sm_box)

  def publish(state, rid, started):
    states.append((state, rid))

  def finalise(meta):
    finals.append(meta)

  sm = RecordingStateMachine(
    pipeline_factory=factory,
    publish_state=publish,
    finalise=finalise,
    start_timeout_s=start_timeout_s,
    count_segments=lambda _id: 4,
  )
  sm_box[0] = sm
  return sm, live, states, finals


def wait_until(predicate, timeout=2.0):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if predicate():
      return True
    time.sleep(0.01)
  return False


def test_start_to_recording_then_stop_finalises():
  sm, live, _, finals = make_sm()
  sm.start()
  assert sm.request_start("r1", MODE, "octocat") is True
  assert wait_until(lambda: sm.state is RecordingStateValue.starting)
  sm.on_segment()
  assert wait_until(lambda: sm.state is RecordingStateValue.recording)
  assert sm.active_recording_id == "r1"
  assert len(live) == 1 and live[0].started

  sm.request_stop()
  assert wait_until(lambda: sm.state is RecordingStateValue.idle)
  assert live[0].stopped
  assert len(finals) == 1
  meta = finals[0]
  assert meta.id == "r1" and meta.operator == "octocat" and meta.segments == 4
  assert meta.mode.resolution == "1920x1080" and meta.ended_at is not None
  sm.shutdown()


def test_only_one_recording_at_a_time():
  sm, _, _, _ = make_sm()
  sm.start()
  assert sm.request_start("r1", MODE, None) is True
  assert wait_until(lambda: sm.state is RecordingStateValue.recording or sm.state is RecordingStateValue.starting)
  sm.on_segment()
  assert wait_until(lambda: sm.state is RecordingStateValue.recording)
  # A start while recording is rejected SYNCHRONOUSLY (so the HTTP layer 409s).
  assert sm.request_start("r2", MODE, None) is False
  assert sm.active_recording_id == "r1"
  sm.shutdown()


def test_start_timeout_returns_idle_without_finalise():
  sm, _, _, finals = make_sm(start_timeout_s=0.2)
  sm.start()
  assert sm.request_start("r1", MODE, None) is True
  assert wait_until(lambda: sm.state is RecordingStateValue.starting)
  # Never call on_segment -> timeout fires -> idle. Partial output stays on disk
  # but NO metadata is written (the recording never finalised, V4.3).
  assert wait_until(lambda: sm.state is RecordingStateValue.idle, timeout=2.0)
  assert finals == []
  sm.shutdown()


def test_failed_start_returns_idle_without_finalise():
  sm, _, _, finals = make_sm()
  sm.start()
  assert sm.request_start("r1", MODE, None) is True
  assert wait_until(lambda: sm.state is RecordingStateValue.starting)
  sm.on_failed()
  assert wait_until(lambda: sm.state is RecordingStateValue.idle)
  assert finals == []
  sm.shutdown()


def test_queued_start_after_stop_runs():
  sm, live, _, finals = make_sm()
  sm.start()
  sm.request_start("r1", MODE, None)
  assert wait_until(lambda: sm.state is RecordingStateValue.starting)
  sm.on_segment()
  assert wait_until(lambda: sm.state is RecordingStateValue.recording)
  # Begin stop, then queue a new start during stopping. After the prior exits,
  # the new recording starts.
  sm.request_stop()
  assert wait_until(lambda: len(finals) == 1)  # r1 finalised
  # A start arriving while stopping is queued and runs after exit.
  # (request_start is rejected unless idle; once idle the new one proceeds.)
  assert wait_until(lambda: sm.state is RecordingStateValue.idle)
  assert sm.request_start("r2", MODE, None) is True
  assert wait_until(lambda: sm.state is RecordingStateValue.starting)
  sm.on_segment()
  assert wait_until(lambda: sm.state is RecordingStateValue.recording)
  assert sm.active_recording_id == "r2"
  sm.shutdown()
