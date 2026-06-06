"""AnalysisCoordinator bridge tests (V5.2/V5.4/V5.6/V5.7 wiring).

Drives the coordinator's on_vision_*/on_audio_* callbacks directly (the role the
GStreamer handles play) with a fake publish + fake recorder, and asserts the
right MQTT topics, the rate-limiting, the violation->recording trigger, and the
rest-period state machine. No GStreamer.
"""

from fessel_schemas import AnomalyType
from fessel_shared import topics

from video.analysis import AnalysisCoordinator
from video.vision import FrameResult, VisionAnalyzer, VisionConfig
from video.audio import AudioAnalyzer, AudioConfig


class FakeClock:
  def __init__(self):
    self.t = 0.0

  def __call__(self):
    return self.t


class FakeRecorder:
  def __init__(self):
    self.triggers: list[tuple] = []

  def trigger(self, etype, ts, details):
    self.triggers.append((etype, ts, details))
    return f"rec-{len(self.triggers)}"


def _coord(clock=None, recorder=None, published=None):
  clock = clock or FakeClock()
  published = published if published is not None else []
  va = VisionAnalyzer(VisionConfig())
  aa = AudioAnalyzer(AudioConfig())
  coord = AnalysisCoordinator(
    vision=va,
    audio=aa,
    recorder=recorder,
    publish=lambda topic, payload, qos, retain: published.append((topic, payload, qos, retain)),
    summary_hz=1.0,
    level_hz=2.0,
    now=clock,
  )
  return coord, va, aa, published


def _frame(ema=0.04, score=0.04):
  return FrameResult(activity_score=score, activity_score_ema=ema, events=[])


def test_vision_summary_rate_limited_to_1hz():
  clock = FakeClock()
  coord, _, _, published = _coord(clock=clock)
  # First frame at t=0 publishes (last_summary_at starts 0, due immediately).
  coord.on_vision_summary(_frame())
  # A second frame 0.5s later does NOT (under the 1s period).
  clock.t = 0.5
  coord.on_vision_summary(_frame())
  # 1.0s after the first -> publishes again.
  clock.t = 1.0
  coord.on_vision_summary(_frame())
  summaries = [p for p in published if p[0] == topics.VISION_SUMMARY]
  assert len(summaries) == 2
  # max_since_last is carried in the payload.
  assert "activity_score_max_since_last" in summaries[0][1]


def test_audio_level_retained_and_rate_limited():
  clock = FakeClock()
  coord, _, _, published = _coord(clock=clock)
  coord.on_audio_level(-30.0, -28.0)  # t=0 publishes
  clock.t = 0.2
  coord.on_audio_level(-31.0, -29.0)  # under 0.5s period -> no publish
  clock.t = 0.5
  coord.on_audio_level(-32.0, -30.0)  # publishes
  levels = [p for p in published if p[0] == topics.AUDIO_LEVEL]
  assert len(levels) == 2
  # Retained per the topic map (the live meter on connect).
  assert levels[0][3] is True


def test_safe_box_violation_publishes_and_triggers_recording():
  rec = FakeRecorder()
  coord, _, _, published = _coord(recorder=rec)
  coord.on_vision_events([(AnomalyType.safe_box_violation, {"outside_fraction": 0.18})])
  # Published on the safe-box event topic.
  ev = [p for p in published if p[0] == topics.VISION_EVENT_SAFE_BOX]
  assert len(ev) == 1
  assert ev[0][1]["type"] == "safe_box_violation"
  # Triggered exactly one anomaly recording.
  assert len(rec.triggers) == 1
  assert rec.triggers[0][0] is AnomalyType.safe_box_violation


def test_cleared_event_publishes_but_does_not_trigger_recording():
  rec = FakeRecorder()
  coord, _, _, published = _coord(recorder=rec)
  coord.on_vision_events([(AnomalyType.safe_box_cleared, {"outside_fraction": 0.0})])
  ev = [p for p in published if p[0] == topics.VISION_EVENT_SAFE_BOX_CLEARED]
  assert len(ev) == 1
  # A *_cleared twin does not start a recording.
  assert rec.triggers == []


def test_audio_spike_uses_audio_topic_and_triggers():
  rec = FakeRecorder()
  coord, _, _, published = _coord(recorder=rec)
  coord.on_audio_events(
    [(AnomalyType.audio_spike, {"peak_db": -3.0, "baseline_db": -42.0, "duration_ms": 180})]
  )
  ev = [p for p in published if p[0] == topics.AUDIO_EVENT_SPIKE]
  assert len(ev) == 1 and ev[0][2] == topics.QOS_AUDIO_EVENT
  assert len(rec.triggers) == 1


def test_rest_from_supervisor_state_drives_vision_rest():
  coord, va, _, _ = _coord()
  assert va.in_rest is False
  coord.note_supervisor_state("PAUSED")
  assert va.in_rest is True
  coord.note_supervisor_state("RUNNING")
  assert va.in_rest is False


def test_rest_from_episode_boundaries():
  coord, va, _, _ = _coord()
  # Between episode_end and episode_start the arm is at rest.
  coord.note_jetson_event("episode_end")
  assert va.in_rest is True
  coord.note_jetson_event("episode_start")
  assert va.in_rest is False


def test_supervisor_rest_overrides_episode_active():
  coord, va, _, _ = _coord()
  coord.note_jetson_event("episode_start")  # active by episodes
  coord.note_supervisor_state("STOPPED")  # but operator stopped -> rest
  assert va.in_rest is True
