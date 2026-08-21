"""AnomalyTracker unit tests (S5.1–S5.3): live values, health, and the log."""

from datetime import datetime, timezone

from fessel_schemas import AnomalyType

from supervisor.anomalies import AnomalyTracker


class Clock:
  def __init__(self):
    self.t = 100.0

  def __call__(self):
    return self.t


class WallClock:
  """Wall-clock stand-in for the frame-staleness check. Starts at the epoch
  seconds of FRESH_FRAME so a heartbeat carrying that stamp reads as current."""

  def __init__(self):
    self.t = datetime(2026, 6, 5, 0, 0, 0, tzinfo=timezone.utc).timestamp()

  def __call__(self):
    return self.t


# A frame stamp matching WallClock's start: "frames are flowing right now".
FRESH_FRAME = "2026-06-05T00:00:00Z"


def _tracker(**kw):
  clock = kw.pop("clock", Clock())
  wall = kw.pop("wall", WallClock())
  return AnomalyTracker(now=clock, wall_now=wall, **kw), clock, wall


def test_vision_summary_and_heartbeat_health():
  t, clock, _wall = _tracker(staleness_s=6.0)
  # No heartbeat yet -> not healthy.
  assert t.vision_state().healthy is False
  t.note_vision_summary({"activity_score_ema": 0.07})
  t.note_vision_heartbeat({"last_frame_at": FRESH_FRAME})
  vs = t.vision_state()
  assert vs.activity_score_ema == 0.07
  assert vs.last_frame_at == FRESH_FRAME
  assert vs.healthy is True
  # Let the heartbeat go stale -> unhealthy.
  clock.t += 10.0
  assert t.vision_state().healthy is False


def test_vision_unhealthy_when_frames_stop_but_heartbeat_continues():
  """The five-day outage: v4l2src silently stopped delivering buffers while the
  video process kept heartbeating on schedule. Heartbeat freshness alone read
  green throughout; `last_frame_at` was the only field that knew."""
  t, clock, wall = _tracker(staleness_s=6.0, frame_staleness_s=30.0)
  t.note_vision_heartbeat({"last_frame_at": FRESH_FRAME})
  assert t.vision_state().healthy is True

  # Time advances on both clocks well past frame_staleness_s; the heartbeat
  # keeps arriving punctually, but carries the SAME frozen last_frame_at.
  for _ in range(30):
    clock.t += 2.0
    wall.t += 2.0
    t.note_vision_heartbeat({"last_frame_at": FRESH_FRAME})

  vs = t.vision_state()
  assert vs.last_frame_at == FRESH_FRAME
  # The heartbeat is fresh — the old check would still say healthy here.
  assert t._fresh(t._vision_hb_at) is True
  assert vs.healthy is False


def test_vision_healthy_while_frames_keep_advancing():
  """The control case for the above: same punctual heartbeats, but each one
  carries a newer frame stamp -> stays healthy."""
  t, clock, wall = _tracker(staleness_s=6.0, frame_staleness_s=30.0)
  for _ in range(10):
    clock.t += 2.0
    wall.t += 2.0
    stamp = datetime.fromtimestamp(wall.t, timezone.utc).isoformat()
    t.note_vision_heartbeat({"last_frame_at": stamp})
    assert t.vision_state().healthy is True


def test_vision_frame_staleness_boundary():
  t, _clock, wall = _tracker(staleness_s=600.0, frame_staleness_s=30.0)
  t.note_vision_heartbeat({"last_frame_at": FRESH_FRAME})
  # Just inside the bound -> healthy (staleness_s is wide so only frames decide).
  wall.t += 29.0
  assert t.vision_state().healthy is True
  # Past it -> unhealthy.
  wall.t += 2.0
  assert t.vision_state().healthy is False


def test_vision_unhealthy_when_last_frame_at_missing_or_unparseable():
  """A heartbeat that cannot say when it last saw a frame is not evidence that
  frames are flowing — fail safe rather than defaulting to healthy."""
  for payload in ({}, {"last_frame_at": None}, {"last_frame_at": "not-a-date"}):
    t, _clock, _wall = _tracker(staleness_s=6.0)
    t.note_vision_heartbeat(payload)
    assert t.vision_state().healthy is False, payload


def test_vision_tolerates_clock_skew_ahead_of_supervisor():
  """Pi and supervisor clocks drift independently; a frame stamped slightly in
  supervisor's future is skew, not a stall."""
  t, _clock, wall = _tracker(staleness_s=6.0, frame_staleness_s=30.0)
  t.note_vision_heartbeat({"last_frame_at": FRESH_FRAME})
  wall.t -= 5.0  # video's clock runs ahead of supervisor's
  assert t.vision_state().healthy is True


def test_audio_level_and_health():
  t, clock, _wall = _tracker(staleness_s=6.0)
  assert t.audio_state().healthy is False
  t.note_audio_level({"rms_db": -38.5})
  a = t.audio_state()
  assert a.level_db == -38.5
  assert a.healthy is True
  clock.t += 10.0
  assert t.audio_state().healthy is False


def test_violation_opens_entry_clear_flips_it():
  t, _clock, _wall = _tracker()
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
  t, _clock, _wall = _tracker()
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
  t, _clock, _wall = _tracker(log_cap=3)
  for i in range(5):
    t.note_event(AnomalyType.rest_violation, {"ts": f"t{i}"})
  log = t.full_log()
  assert len(log) == 3  # bounded
  # Newest first.
  assert [e.ts for e in log] == ["t4", "t3", "t2"]


def test_recent_capped_below_full_log():
  t, _clock, _wall = _tracker(log_cap=100, recent_cap=2)
  for i in range(5):
    t.note_event(AnomalyType.safe_box_violation, {"ts": f"t{i}"})
  assert len(t.recent()) == 2
  assert len(t.full_log()) == 5
