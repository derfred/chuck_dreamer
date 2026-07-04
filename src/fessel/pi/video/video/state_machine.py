"""Live-branch state machine (V1.4, reshaped for WHIP §2.2/§2.3).

The live ENCODER is always warm (§2.2); this state machine governs the WHIP
sender branch, not the encoder. "Start" means attach the sender branch and
establish the WHIP session; "stop" means detach it. Activate/deactivate are
transition REQUESTS, not direct spawn/kill commands, and are PARAMETERLESS —
the live profile is a deploy-time config setting, not a per-session mode.
All transitions are serialised on a single worker thread so two sender
branches can never run at once and no branch leaks.

Transition table (architecture §2.3):

| From     | Event               | To       | Action                                    |
|----------|---------------------|----------|-------------------------------------------|
| off      | activate            | starting | attach WHIP branch; force IDR; arm timeout |
| starting | session established | running  | publish state                              |
| starting | connect fail/tmo    | off      | detach branch; log; await next activate    |
| starting | deactivate          | stopping | cancel pending start; detach branch        |
| running  | deactivate          | stopping | detach branch; teardown session            |
| running  | session dropped     | starting | re-attach with backoff                     |
| running  | activate            | running  | idempotent ack; no-op                      |
| stopping | exit                | off      | publish state                              |
| stopping | activate            | off->starting | queue; transition after exit          |
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from fessel_schemas import LiveStateValue

log = logging.getLogger(__name__)


class _EventKind(Enum):
  ACTIVATE = "activate"
  DEACTIVATE = "deactivate"
  CONNECTED = "connected"  # WHIP session established
  CONNECT_FAILED = "connect_failed"  # connect error or timeout
  DISCONNECTED = "disconnected"  # WHIP session dropped while running
  EXITED = "exited"  # sender branch finished tearing down


@dataclass
class _Event:
  kind: _EventKind


# A PipelineHandle abstracts the GStreamer WHIP sender branch so the state
# machine is testable without GStreamer. Implementations attach/detach and
# report connected/failed/disconnected/exited back via the supplied callbacks.
class PipelineHandle:
  def start(self) -> None:  # pragma: no cover - interface
    raise NotImplementedError

  def stop(self) -> None:  # pragma: no cover - interface
    """Request graceful detach; must eventually emit EXITED."""
    raise NotImplementedError

  def force_idr(self) -> None:  # pragma: no cover - interface
    """Ask the warm live encoder to emit an IDR keyframe at the next frame.

    Called once when the sender branch is attached (§2.2), so the stream
    starts with a decodable IDR rather than waiting for the next GOP
    boundary. A no-op by default: it is an optional best-effort
    optimisation, and an encoder that ignores the force-key-unit event
    simply falls back to the short-GOP + UI-spinner behaviour.
    """


PipelineFactory = Callable[[], PipelineHandle]
StatePublisher = Callable[[LiveStateValue], None]


class LiveStateMachine:
  def __init__(
    self,
    *,
    pipeline_factory: PipelineFactory,
    publish_state: StatePublisher,
    connect_timeout_s: float,
    reconnect_backoff_s: float = 1.0,
    reconnect_backoff_max_s: float = 15.0,
  ) -> None:
    self._make_pipeline = pipeline_factory
    self._publish_state = publish_state
    self._connect_timeout_s = connect_timeout_s
    self._backoff_base = reconnect_backoff_s
    self._backoff_max = reconnect_backoff_max_s

    self._state = LiveStateValue.off
    self._events: queue.Queue[_Event] = queue.Queue()
    self._thread = threading.Thread(target=self._run, name="live-sm", daemon=True)
    self._stop = threading.Event()

    self._pipeline: PipelineHandle | None = None
    self._queued_activate = False  # activate received during stopping
    self._connect_deadline: float | None = None
    self._backoff: float = reconnect_backoff_s
    self._reconnect_at: float | None = None  # when to retry after disconnect

  # --- public API (thread-safe; just enqueues) ---

  def start(self) -> None:
    self._thread.start()

  def shutdown(self) -> None:
    self._stop.set()
    self._events.put(_Event(_EventKind.DEACTIVATE))
    self._thread.join(timeout=5)

  @property
  def state(self) -> LiveStateValue:
    return self._state

  def request_activate(self) -> None:
    self._events.put(_Event(_EventKind.ACTIVATE))

  def request_deactivate(self) -> None:
    self._events.put(_Event(_EventKind.DEACTIVATE))

  # --- callbacks the pipeline implementation calls (thread-safe) ---

  def on_connected(self) -> None:
    self._events.put(_Event(_EventKind.CONNECTED))

  def on_connect_failed(self) -> None:
    self._events.put(_Event(_EventKind.CONNECT_FAILED))

  def on_disconnected(self) -> None:
    self._events.put(_Event(_EventKind.DISCONNECTED))

  def on_exited(self) -> None:
    self._events.put(_Event(_EventKind.EXITED))

  # --- worker loop ---

  def _run(self) -> None:
    self._set_state(LiveStateValue.off)
    while not self._stop.is_set() or self._state is not LiveStateValue.off:
      timeout = self._next_timeout()
      try:
        ev = self._events.get(timeout=timeout)
      except queue.Empty:
        self._handle_timers()
        continue
      self._dispatch(ev)

  def _next_timeout(self) -> float:
    """Soonest pending timer (connect deadline or reconnect), else a tick."""
    now = time.monotonic()
    deadlines = [d for d in (self._connect_deadline, self._reconnect_at) if d is not None]
    if not deadlines:
      return 1.0
    return max(0.0, min(deadlines) - now)

  def _handle_timers(self) -> None:
    now = time.monotonic()
    if self._connect_deadline is not None and now >= self._connect_deadline:
      log.warning("WHIP session connect timed out in %s", self._state)
      self._dispatch(_Event(_EventKind.CONNECT_FAILED))
    if self._reconnect_at is not None and now >= self._reconnect_at:
      self._reconnect_at = None
      self._spawn()

  def _dispatch(self, ev: _Event) -> None:
    handler = getattr(self, f"_on_{self._state.value}")
    handler(ev)

  # --- per-state handlers ---

  def _on_off(self, ev: _Event) -> None:
    if ev.kind is _EventKind.ACTIVATE:
      self._spawn()
    # deactivate/connected/etc. in off: ignore.

  def _on_starting(self, ev: _Event) -> None:
    if ev.kind is _EventKind.CONNECTED:
      self._connect_deadline = None
      self._backoff = self._backoff_base
      self._set_state(LiveStateValue.running)
    elif ev.kind is _EventKind.CONNECT_FAILED:
      self._connect_deadline = None
      self._teardown()
      self._set_state(LiveStateValue.off)
    elif ev.kind is _EventKind.DEACTIVATE:
      self._connect_deadline = None
      self._begin_teardown()
    elif ev.kind is _EventKind.ACTIVATE:
      pass  # already starting for a request; idempotent

  def _on_running(self, ev: _Event) -> None:
    if ev.kind is _EventKind.DEACTIVATE:
      self._begin_teardown()
    elif ev.kind is _EventKind.DISCONNECTED:
      # Detach the current sender branch and re-attach with backoff -> back to
      # starting (§2.3: uplink-blip recovery is the state machine's job).
      self._teardown()
      self._reconnect_at = time.monotonic() + self._backoff
      self._backoff = min(self._backoff * 2, self._backoff_max)
      self._set_state(LiveStateValue.starting)
      self._connect_deadline = None  # set when re-attach happens
    elif ev.kind is _EventKind.ACTIVATE:
      pass  # idempotent ack; no-op

  def _on_stopping(self, ev: _Event) -> None:
    if ev.kind is _EventKind.EXITED:
      self._pipeline = None
      self._set_state(LiveStateValue.off)
      if self._queued_activate:
        self._queued_activate = False
        self._spawn()
    elif ev.kind is _EventKind.ACTIVATE:
      # Queue; transition after current teardown completes.
      self._queued_activate = True

  # --- helpers ---

  def _spawn(self) -> None:
    self._pipeline = self._make_pipeline()
    self._connect_deadline = time.monotonic() + self._connect_timeout_s
    self._set_state(LiveStateValue.starting)
    self._pipeline.start()
    # Force an IDR on attach (§2.2/§2.3 off->starting action): the warm
    # encoder is mid-GOP, so the new WHIP session should start with a
    # decodable keyframe rather than waiting up to a full GOP. Once per
    # attach (incl. reconnect re-attach) and best-effort — a failure here
    # must not break the transition (short GOP covers it).
    try:
      self._pipeline.force_idr()
    except Exception:  # noqa: BLE001
      log.exception("force_idr failed (continuing; short GOP covers it)")

  def _begin_teardown(self) -> None:
    self._connect_deadline = None
    self._set_state(LiveStateValue.stopping)
    if self._pipeline is not None:
      self._pipeline.stop()  # graceful; emits EXITED
    else:
      self._dispatch(_Event(_EventKind.EXITED))

  def _teardown(self) -> None:
    """Hard teardown without waiting for graceful EXITED (failure paths)."""
    if self._pipeline is not None:
      try:
        self._pipeline.stop()
      except Exception:  # noqa: BLE001
        log.exception("error tearing down pipeline")
      self._pipeline = None

  def _set_state(self, state: LiveStateValue) -> None:
    self._state = state
    self._publish_state(state)
