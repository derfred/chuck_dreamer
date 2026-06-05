"""Bandwidth coordinator (architecture §2.12; resolves Requirements OQ#6).

The cellular uplink is shared; the load-bearing rule (Requirements §4.2) is that
recording uploads **suspend fully whenever an operator is actively viewing** and
resume when the viewer leaves. This is a *policy loop that gates the
lowest-priority producer* (the uploader), not an inline traffic shaper.

It lives in `supervisor` because supervisor already owns both presence signals:

  - **Live viewer** — supervisor is in the live dispatch path and caches
    `arm/video/state/live`; `running` ⇒ a live viewer is present.
  - **Ring viewer** — every `GET /ring/*` the proxy serves bumps a last-read
    timestamp here; a read within a short window ⇒ a ring viewer is active.
    (This is the signal WebRTC-for-the-ring would have given for free; HLS
    requires recording it explicitly — see §2.11.)

A viewer is present if **either** holds. The coordinator publishes a single
retained gate (`arm/video/cmd/upload/gate`); the uploader checks it before each
pass. "Start simple": one gate, live + ring treated uniformly as "viewer";
tier-3 metrics/logs left on static caps. Dynamic bitrate reallocation and
ring-serve rate-limiting are deferred (§2.12).

This module is MQTT-free and clock-injectable so the decision logic is
unit-testable; the run loop is a daemon thread, matching ControlPlane's
background-refresh pattern (the supervisor control plane is threaded, not
asyncio — §5.6's asyncio note is a known doc/impl gap, left open for now).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from fessel_schemas import UploadGate
from fessel_shared import topics

log = logging.getLogger(__name__)

# A ring read counts as an "active viewer" for this long after the last served
# segment. Must comfortably exceed the gap between a player's segment fetches
# (segments are ~2 s; a player pulls one every couple of seconds) so a steadily
# scrubbing/playing viewer never flickers to "absent" between fetches.
DEFAULT_RING_ACTIVE_WINDOW_S = 10.0

# How often the loop re-evaluates presence (and so how fast uploads resume after
# a viewer leaves). Short: the ring-active window, not this tick, sets the
# debounce.
DEFAULT_TICK_S = 2.0

# A publish callback: (topic, payload-dict, qos, retain) -> None. Injected so
# this module carries no MQTT-client dependency (mirrors ControlPlane).
PublishFn = Callable[[str, dict, int, bool], None]


class BandwidthCoordinator:
  """Decides `uploads_allowed` from viewer presence and publishes the gate.

  Thread-safe: the HTTP/MQTT callbacks (`set_live_running`, `note_ring_read`)
  and the background loop all touch the small shared state under a lock.
  """

  def __init__(
    self,
    *,
    publish: PublishFn,
    ring_active_window_s: float = DEFAULT_RING_ACTIVE_WINDOW_S,
    tick_s: float = DEFAULT_TICK_S,
    monotonic: Callable[[], float] = time.monotonic,
  ) -> None:
    self._publish = publish
    self._ring_window = ring_active_window_s
    self._tick_s = tick_s
    self._now = monotonic

    self._lock = threading.Lock()
    self._live_running = False
    self._last_ring_read: float | None = None
    # Start with no decision published yet, so the first evaluation always emits
    # (the uploader must learn the initial gate even if it is the safe default).
    self._published_allowed: bool | None = None

    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._loop, name="bw-coordinator", daemon=True)

  # --- lifecycle -------------------------------------------------------------

  def start(self) -> None:
    self._thread.start()

  def close(self) -> None:
    self._stop.set()

  # --- presence inputs (thread-safe) ----------------------------------------

  def set_live_running(self, running: bool) -> None:
    """Called by the relay when the cached live state changes."""
    with self._lock:
      self._live_running = running
    # Re-decide promptly on a live transition rather than waiting for the tick:
    # the operator opening/closing the live feed should gate uploads at once.
    self._evaluate()

  def note_ring_read(self) -> None:
    """Called by RingProxy on every served ring file (playlist or segment)."""
    with self._lock:
      self._last_ring_read = self._now()
    # No immediate _evaluate() here: ring reads are frequent, and the first read
    # of a session is picked up within one tick. Avoids publish churn per
    # segment fetch.

  # --- decision --------------------------------------------------------------

  def _viewer_present(self) -> bool:
    if self._live_running:
      return True
    if self._last_ring_read is None:
      return False
    return (self._now() - self._last_ring_read) < self._ring_window

  def _evaluate(self) -> None:
    """Recompute the gate and publish only on change (debounced)."""
    with self._lock:
      viewer = self._viewer_present()
      allowed = not viewer
      if allowed == self._published_allowed:
        return  # no change; don't churn the retained topic
      self._published_allowed = allowed
      reason = self._reason(viewer)
    gate = UploadGate(uploads_allowed=allowed, reason=reason)
    self._publish(topics.CMD_UPLOAD_GATE, gate.model_dump(), topics.QOS_CMD, topics.RETAIN_UPLOAD_GATE)
    log.info(
      '{"event":"upload_gate","uploads_allowed":%s,"reason":"%s"}',
      "true" if allowed else "false",
      reason,
    )

  def _reason(self, viewer: bool) -> str:
    if not viewer:
      return "no viewer"
    return "live viewer" if self._live_running else "ring viewer"

  # --- background loop -------------------------------------------------------

  def _loop(self) -> None:
    # Publish the initial gate immediately, then re-evaluate every tick so a
    # ring viewer going idle (its window expiring) re-opens uploads.
    self._evaluate()
    while not self._stop.is_set():
      self._stop.wait(self._tick_s)
      if self._stop.is_set():
        return
      self._evaluate()
