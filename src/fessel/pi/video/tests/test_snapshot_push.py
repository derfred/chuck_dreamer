"""Freeze-frame push: the PUT itself, and the capture-age accounting behind
the webui's "Xs ago" label.

The age is the whole reason these tests exist: dating a snapshot from when the
PUT landed makes a stalled capture side look perpetually fresh (the pusher
happily re-sends one cached frame every interval_s forever). The Pi therefore
declares how old the frame already was, measured on its own monotonic clock —
see snapshot_push.py.

The wire-level half is driven against an httpx MockTransport, mirroring
test_ingest_client.py. httpx reaches pi/video as a SYSTEM package on the Pi
(python3-httpx, in the dpkg Depends) rather than a pip dependency, so those
tests skip on a venv built without system site-packages; the VideoApp wiring
tests below need no HTTP at all and always run."""

import importlib.util

import pytest

from video.main import VideoApp
from video.snapshot_push import SnapshotPusher

requires_httpx = pytest.mark.skipif(
  importlib.util.find_spec("httpx") is None,
  reason="httpx is a system package on the Pi (python3-httpx), not a pip dep of pi/video",
)


def _pusher(handler) -> SnapshotPusher:
  import httpx

  p = SnapshotPusher(ingest_url_base="https://ingest.example:8443")
  p._client = httpx.Client(transport=httpx.MockTransport(handler))
  return p


def _capture(handler_status: int = 201):
  import httpx

  seen = {}

  def handler(req: httpx.Request) -> httpx.Response:
    seen["url"] = str(req.url)
    seen["body"] = req.content
    seen["ct"] = req.headers.get("content-type")
    seen["age"] = req.headers.get("x-snapshot-age-ms")
    return httpx.Response(handler_status)

  return seen, handler


@requires_httpx
def test_put_url_body_and_age_header():
  seen, handler = _capture()
  _pusher(handler).push(b"JPEGDATA", age_s=2.5)
  assert seen["url"] == "https://ingest.example:8443/snapshot"
  assert seen["body"] == b"JPEGDATA"
  assert seen["ct"] == "image/jpeg"
  assert seen["age"] == "2500"


@requires_httpx
def test_age_defaults_to_zero_and_clamps_negative():
  seen, handler = _capture()
  p = _pusher(handler)
  p.push(b"J")
  assert seen["age"] == "0"
  # A clock oddity must not turn into a "captured in the future" frame.
  p.push(b"J", age_s=-3.0)
  assert seen["age"] == "0"


@requires_httpx
def test_push_swallows_transport_and_status_errors():
  import httpx

  def boom(req: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("no route", request=req)

  _pusher(boom).push(b"J", age_s=1.0)  # no raise
  seen, handler = _capture(handler_status=502)
  _pusher(handler).push(b"J", age_s=1.0)  # no raise


# --- VideoApp wiring: the age comes from the CAPTURE instant -------------------


class _FakeSnapshotCapture:
  """Stands in for GstSnapshotPipeline: hands back a frame stamped with the
  monotonic instant the appsink delivered it."""

  def __init__(self, captured_at: float) -> None:
    self._captured_at = captured_at

  def latest(self):
    return b"JPEG", self._captured_at


class _RecordingPusher:
  def __init__(self) -> None:
    self.pushes: list[tuple[bytes, float]] = []

  def push(self, jpeg: bytes, age_s: float = 0.0) -> None:
    self.pushes.append((jpeg, age_s))


def _app_with_snapshot(captured_at: float, pusher: _RecordingPusher) -> VideoApp:
  app = VideoApp({"storage": {"ssd_path": "/tmp/ssd"}, "webui_base": "http://webui.x:8001"})
  app._gst_snapshot = _FakeSnapshotCapture(captured_at)
  app._snapshot_pusher = pusher
  return app


def test_push_declares_the_frames_age_not_zero():
  pusher = _RecordingPusher()
  # Frame captured at t=100, loop tick at t=112 -> the frame is already 12s old.
  app = _app_with_snapshot(100.0, pusher)
  app._maybe_push_snapshot(112.0)
  assert pusher.pushes == [(b"JPEG", 12.0)]


def test_stalled_capture_reports_a_growing_age_across_pushes():
  """The regression this guards: a wedged pipeline keeps handing back ONE
  frame, and each push must declare it older than the last (rather than every
  push looking brand new to the webui)."""
  pusher = _RecordingPusher()
  app = _app_with_snapshot(100.0, pusher)
  app._maybe_push_snapshot(110.0)
  app._maybe_push_snapshot(140.0)
  app._maybe_push_snapshot(170.0)
  assert [age for _, age in pusher.pushes] == [10.0, 40.0, 70.0]


def test_no_push_before_the_first_frame():
  pusher = _RecordingPusher()
  app = _app_with_snapshot(0.0, pusher)
  app._gst_snapshot = type("Empty", (), {"latest": staticmethod(lambda: None)})()
  app._maybe_push_snapshot(10.0)
  assert pusher.pushes == []
