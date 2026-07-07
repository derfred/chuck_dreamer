"""Bandwidth coordinator (architecture §2.12; resolves Requirements OQ#6).

The cellular uplink is shared; the load-bearing rule (Requirements §4.2) is that
recording uploads **suspend fully whenever an operator is actively viewing the
live stream** and resume when they stop. This is a *policy loop that gates the
lowest-priority producer* (the uploader), not an inline traffic shaper.

It lives in `supervisor` because supervisor owns the viewer signal: it is in the
live dispatch path and caches `arm/video/state/live`; `running` ⇒ a live viewer
is present. (There is no ring viewer to account for — the ring buffer is never
streamed back for viewing; it lives on the Pi behind the cellular link and only
backs recording look-back.)

The coordinator publishes a single retained gate (`arm/video/cmd/upload/gate`);
the uploader checks it before each pass. Dynamic bitrate reallocation is deferred
(§2.12).

This module is MQTT-free and clock-injectable so the decision logic is
unit-testable; the run loop is a daemon thread, matching ControlPlane's
background-refresh pattern.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from fessel_schemas import UploadGate
from fessel_shared import topics

log = logging.getLogger(__name__)

# How often the loop re-evaluates presence. A live transition re-evaluates
# immediately (set_live_running), so the tick is really just the initial-publish
# heartbeat / a safety re-eval.
DEFAULT_TICK_S = 2.0

# A publish callback: (topic, payload-dict, qos, retain) -> None. Injected so
# this module carries no MQTT-client dependency (mirrors ControlPlane).
PublishFn = Callable[[str, dict, int, bool], None]


class BandwidthCoordinator:
  """Decides `uploads_allowed` from live-viewer presence and publishes the gate.

  Thread-safe: the `set_live_running` callback and the background loop both touch
  the small shared state under a lock.
  """

  def __init__(
    self,
    *,
    publish: PublishFn,
    tick_s: float = DEFAULT_TICK_S,
    monotonic: Callable[[], float] = time.monotonic,
  ) -> None:
    self._publish = publish
    self._tick_s = tick_s
    self._now = monotonic

    self._lock = threading.Lock()
    self._live_running = False
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

  # --- presence input (thread-safe) -----------------------------------------

  def set_live_running(self, running: bool) -> None:
    """Called by the relay when the cached live state changes."""
    with self._lock:
      self._live_running = running
    # Re-decide promptly on a live transition rather than waiting for the tick:
    # the operator opening/closing the live feed should gate uploads at once.
    self._evaluate()

  # --- decision --------------------------------------------------------------

  def _evaluate(self) -> None:
    """Recompute the gate and publish only on change (debounced)."""
    with self._lock:
      viewer = self._live_running
      allowed = not viewer
      if allowed == self._published_allowed:
        return  # no change; don't churn the retained topic
      self._published_allowed = allowed
    reason = "live viewer" if viewer else "no viewer"
    gate = UploadGate(uploads_allowed=allowed, reason=reason)
    self._publish(topics.CMD_UPLOAD_GATE, gate.model_dump(), topics.QOS_CMD, topics.RETAIN_UPLOAD_GATE)
    log.info(
      '{"event":"upload_gate","uploads_allowed":%s,"reason":"%s"}',
      "true" if allowed else "false",
      reason,
    )

  # --- background loop -------------------------------------------------------

  def _loop(self) -> None:
    # Publish the initial gate immediately, then re-evaluate every tick as a
    # safety net (live transitions already re-evaluate inline).
    self._evaluate()
    while not self._stop.is_set():
      self._stop.wait(self._tick_s)
      if self._stop.is_set():
        return
      self._evaluate()
