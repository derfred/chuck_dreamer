import pytest

from fessel_schemas import (
  Capabilities,
  LiveActivate,
  LiveState,
  LiveStateValue,
  ModeTriplet,
  PlugState,
  SafetyState,
  StateResponse,
  mode_from_canonical,
  mode_to_canonical,
)


def test_mode_canonical_roundtrip():
  m = ModeTriplet(resolution="1280x720", fps=30, bitrate_bps=2_500_000)
  s = mode_to_canonical(m)
  assert s == "1280x720@30@2500000"
  assert mode_from_canonical(s) == m


def test_mode_from_canonical_rejects_garbage():
  with pytest.raises(ValueError):
    mode_from_canonical("nonsense")
  with pytest.raises(ValueError):
    mode_from_canonical("1280x720@30")  # missing bitrate


def test_capabilities_payload():
  caps = Capabilities(modes=[ModeTriplet(resolution="640x480", fps=15, bitrate_bps=800_000)])
  dumped = caps.model_dump()
  assert dumped["modes"][0]["resolution"] == "640x480"


def test_live_activate_is_parameterless_but_wire_tolerant():
  # Parameterless activation (§2.2): no mode anywhere in the live path.
  a = LiveActivate()
  assert a.path is None
  assert not hasattr(a, "mode")
  # Wire tolerance: legacy callers still send {"path": ...}; it validates and
  # is ignored downstream.
  legacy = LiveActivate.model_validate({"path": "pi"})
  assert legacy.path == "pi"
  st = LiveState(state=LiveStateValue.running)
  assert st.model_dump()["state"] == "running"


def test_state_response_defaults():
  # The Slice-3 placeholder default: IDLE, no Jetson reached, no plugs verified.
  st = StateResponse()
  dumped = st.model_dump()
  assert dumped["safety_state"] == "IDLE"
  assert dumped["jetson"] is None
  assert dumped["plugs"] == {}
  assert dumped["camera"]["up"] is None


def test_state_response_with_plugs():
  st = StateResponse(
    safety_state=SafetyState.SHUTDOWN_ARM,
    jetson={"state": "stopped"},
    plugs={
      "arm": PlugState(on=False, verified=True, verified_at="2026-06-04T00:00:00Z"),
      "jetson": PlugState(on=True, verified=True, verified_at="2026-06-04T00:00:00Z"),
    },
  )
  dumped = st.model_dump()
  assert dumped["safety_state"] == "SHUTDOWN_ARM"
  assert dumped["plugs"]["arm"]["on"] is False
  assert dumped["plugs"]["arm"]["verified"] is True


def test_plug_state_unverified_default():
  # A plug whose verify failed: on may be known-or-unknown, verified is False.
  p = PlugState(on=False, verified=False, verified_at="2026-06-04T00:00:00Z")
  assert p.verified is False


# --- Slice 5 sensing shapes ---------------------------------------------------


def test_state_response_slice5_defaults():
  # Vision/audio default to not-healthy (no heartbeat seen yet); no anomalies.
  st = StateResponse()
  dumped = st.model_dump()
  assert dumped["vision"]["healthy"] is False
  assert dumped["vision"]["activity_score_ema"] == 0.0
  assert dumped["audio"]["healthy"] is False
  assert dumped["audio"]["level_db"] is None
  assert dumped["recent_anomalies"] == []


def test_anomaly_log_entry_roundtrip():
  from fessel_schemas import AnomalyLogEntry, AnomalyType, SafeBoxEvent

  ev = SafeBoxEvent(ts="2026-06-05T00:00:00Z", outside_fraction=0.18, threshold=0.05)
  entry = AnomalyLogEntry(
    ts=ev.ts,
    type=AnomalyType.safe_box_violation,
    anomaly_recording_id="rec-1",
    payload=ev.model_dump(mode="json"),
  )
  dumped = entry.model_dump(mode="json")
  assert dumped["type"] == "safe_box_violation"
  assert dumped["cleared"] is False
  assert dumped["payload"]["outside_fraction"] == 0.18


def test_anomaly_recording_metadata_is_listing_compatible():
  # S5.4: the anomaly metadata must carry the fields supervisor's /recordings
  # listing reads, plus a `type` discriminator.
  from fessel_schemas import AnomalyRecordingMetadata, AnomalyType

  meta = AnomalyRecordingMetadata(
    id="anom-1",
    anomaly_event_type=AnomalyType.audio_spike,
    trigger_ts="2026-06-05T00:00:00Z",
    started_at="2026-06-05T00:00:00Z",
    ended_at="2026-06-05T00:01:30Z",
    duration_seconds=90,
    segments=45,
  )
  dumped = meta.model_dump(mode="json")
  assert dumped["type"] == "anomaly"
  assert dumped["operator"] is None
  assert dumped["flagged_for_upload"] is False
  assert dumped["upload_state"] == "none"
