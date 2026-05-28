import pytest

from fessel_schemas import (
  Capabilities,
  LiveActivate,
  LiveState,
  LiveStateValue,
  ModeTriplet,
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
