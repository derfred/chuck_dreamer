"""Live WHIP branch lifecycle regressions from the first production deploy.

1. OPTIMISTIC connected: whipclientsink's outer bin never aggregates to
   PLAYING (webrtcsink adds internals dynamically), so waiting for the bus
   STATE_CHANGED signal let connect_timeout_s kill healthy sessions. A
   successful attach must fire on_connected directly.
2. The state machine must leave `stopping` even when GStreamer teardown is
   slow — covered indirectly: on_exited is driven by the (now non-blocking)
   teardown thread.
"""

from __future__ import annotations

from video.gst_pipeline_handle import GstLivePipeline


class FakeSM:
  def __init__(self) -> None:
    self.events: list[str] = []

  def on_connected(self) -> None:
    self.events.append("connected")

  def on_connect_failed(self) -> None:
    self.events.append("connect_failed")

  def on_disconnected(self) -> None:
    self.events.append("disconnected")

  def on_exited(self) -> None:
    self.events.append("exited")


class FakeCapture:
  """add_video_branch stand-in; returns a truthy branch handle on success."""

  def __init__(self, attach_ok: bool = True) -> None:
    self.attach_ok = attach_ok
    self.removed = []

  def add_video_branch(self, chain, *, tee_name, on_playing, on_error, on_removed):
    self._on_error = on_error
    if not self.attach_ok:
      on_error(False)
      return None
    return object()

  def remove_video_branch(self, branch) -> None:
    self.removed.append(branch)


def test_successful_attach_fires_connected_immediately():
  sm = FakeSM()
  h = GstLivePipeline(sm=sm, capture=FakeCapture(), whip_endpoint="http://x/whip/ingest")
  h.start()
  assert sm.events == ["connected"]


def test_failed_attach_fires_connect_failed_not_connected():
  sm = FakeSM()
  h = GstLivePipeline(sm=sm, capture=FakeCapture(attach_ok=False), whip_endpoint="http://x/whip/ingest")
  h.start()
  assert "connected" not in sm.events
  assert sm.events == ["connect_failed"]


def test_connected_fires_once_even_if_bus_playing_also_arrives():
  sm = FakeSM()
  h = GstLivePipeline(sm=sm, capture=FakeCapture(), whip_endpoint="http://x/whip/ingest")
  h.start()
  h._on_playing()  # the bus signal arriving later must dedupe
  assert sm.events == ["connected"]
