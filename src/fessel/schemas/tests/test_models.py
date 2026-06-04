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


def test_live_activate_and_state():
  a = LiveActivate(path="pi", mode=ModeTriplet(resolution="640x480", fps=30, bitrate_bps=1_000_000))
  assert a.path == "pi"
  st = LiveState(state=LiveStateValue.running, path="pi", mode=a.mode)
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
