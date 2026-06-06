"""AnomalyTracker unit tests (S5.1–S5.3): live values, health, and the log."""

from fessel_schemas import AnomalyType

from supervisor.anomalies import AnomalyTracker


class Clock:
  def __init__(self):
    self.t = 100.0

  def __call__(self):
    return self.t


def _tracker(**kw):
  clock = kw.pop("clock", Clock())
  return AnomalyTracker(now=clock, **kw), clock


def test_vision_summary_and_heartbeat_health():
  t, clock = _tracker(staleness_s=6.0)
  # No heartbeat yet -> not healthy.
  assert t.vision_state().healthy is False
  t.note_vision_summary({"activity_score_ema": 0.07})
  t.note_vision_heartbeat({"last_frame_at": "2026-06-05T00:00:00Z"})
  vs = t.vision_state()
  assert vs.activity_score_ema == 0.07
  assert vs.last_frame_at == "2026-06-05T00:00:00Z"
  assert vs.healthy is True
  # Let the heartbeat go stale -> unhealthy.
  clock.t += 10.0
  assert t.vision_state().healthy is False


def test_audio_level_and_health():
  t, clock = _tracker(staleness_s=6.0)
  assert t.audio_state().healthy is False
  t.note_audio_level({"rms_db": -38.5})
  a = t.audio_state()
  assert a.level_db == -38.5
  assert a.healthy is True
  clock.t += 10.0
  assert t.audio_state().healthy is False


def test_violation_opens_entry_clear_flips_it():
  t, _ = _tracker()
  t.note_event(AnomalyType.safe_box_violation, {"ts": "t0", "outside_fraction": 0.18})
  log = t.full_log()
  assert len(log) == 1
  assert log[0].type is AnomalyType.safe_box_violation
  assert log[0].cleared is False
  assert log[0].payload["outside_fraction"] == 0.18
  # The clear twin flips the open entry.
  t.note_event(AnomalyType.safe_box_cleared, {"ts": "t1"})
  assert t.full_log()[0].cleared is True
  # The clear did NOT open a new entry.
  assert len(t.full_log()) == 1


def test_anomaly_recording_links_to_entry():
  t, _ = _tracker()
  t.note_event(AnomalyType.audio_spike, {"ts": "t0", "peak_db": -3})
  t.note_anomaly_recording(
    {
      "anomaly_recording_id": "rec-9",
      "anomaly_event_type": "audio_spike",
      "trigger_ts": "t0",
    }
  )
  assert t.full_log()[0].anomaly_recording_id == "rec-9"


def test_log_is_bounded_newest_first():
  t, _ = _tracker(log_cap=3)
  for i in range(5):
    t.note_event(AnomalyType.rest_violation, {"ts": f"t{i}"})
  log = t.full_log()
  assert len(log) == 3  # bounded
  # Newest first.
  assert [e.ts for e in log] == ["t4", "t3", "t2"]


def test_recent_capped_below_full_log():
  t, _ = _tracker(log_cap=100, recent_cap=2)
  for i in range(5):
    t.note_event(AnomalyType.safe_box_violation, {"ts": f"t{i}"})
  assert len(t.recent()) == 2
  assert len(t.full_log()) == 5
