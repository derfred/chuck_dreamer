"""Live-branch state machine (V1.4).

States: {off, starting, running, stopping}. Activate/deactivate are
transition REQUESTS, not direct spawn/kill commands. All transitions are
serialised on a single worker thread so two pipelines can never run at
once and no encoder process leaks (the specific failure the spike's shell
scripts exhibited).

Transition table:

| From     | Event             | To       | Action                                  |
|----------|-------------------|----------|-----------------------------------------|
| off      | activate(mode)    | starting | spawn pipeline; arm connect-timeout     |
| starting | srtsink connected | running  | publish state                           |
| starting | connect fail/tmo  | off      | tear down; log; await next activate     |
| starting | deactivate        | stopping | cancel pending start; tear down         |
| running  | deactivate        | stopping | send EOS; await graceful exit           |
| running  | srt disconnect    | starting | retry with backoff                      |
| running  | activate          | running  | idempotent ack; no-op                   |
| stopping | exit              | off      | publish state                           |
| stopping | activate          | off->starting | queue; transition after exit       |
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from fessel_schemas import LiveStateValue, ModeTriplet

log = logging.getLogger(__name__)


class _EventKind(Enum):
  ACTIVATE = "activate"
  DEACTIVATE = "deactivate"
  CONNECTED = "connected"  # srtsink connected
  CONNECT_FAILED = "connect_failed"  # connect error or timeout
  DISCONNECTED = "disconnected"  # srt disconnect while running
  EXITED = "exited"  # pipeline finished tearing down


@dataclass
class _Event:
  kind: _EventKind
  mode: ModeTriplet | None = None
  path: str | None = None


# A PipelineHandle abstracts the GStreamer pipeline so the state machine is
# testable without GStreamer. Implementations spawn/teardown and report
# connected/failed/disconnected/exited back via the supplied event callback.
class PipelineHandle:
  def start(self) -> None:  # pragma: no cover - interface
    raise NotImplementedError

  def stop(self) -> None:  # pragma: no cover - interface
    """Request graceful teardown (EOS); must eventually emit EXITED."""
    raise NotImplementedError

  def force_idr(self) -> None:  # pragma: no cover - interface
    """Ask the encoder to emit an IDR keyframe at the next frame (V2.2).

    Called once when the publisher connects, so a WHEP reader that joined
    during runOnDemand gets a keyframe immediately rather than waiting up
    to a full GOP. A no-op by default: it is an optional best-effort
    optimisation, and an encoder that ignores the force-key-unit event
    simply falls back to the short-GOP + UI-spinner behaviour.
    """


PipelineFactory = Callable[[ModeTriplet, str], PipelineHandle]
StatePublisher = Callable[[LiveStateValue, str | None, ModeTriplet | None], None]


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
    self._cur_mode: ModeTriplet | None = None
    self._cur_path: str | None = None
    self._queued_activate: _Event | None = None  # activate received during stopping
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

  def request_activate(self, mode: ModeTriplet, path: str) -> None:
    self._events.put(_Event(_EventKind.ACTIVATE, mode=mode, path=path))

  def request_deactivate(self, path: str | None = None) -> None:
    self._events.put(_Event(_EventKind.DEACTIVATE, path=path))

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
      log.warning("srtsink connect timed out in %s", self._state)
      self._dispatch(_Event(_EventKind.CONNECT_FAILED))
    if self._reconnect_at is not None and now >= self._reconnect_at:
      self._reconnect_at = None
      self._spawn(self._cur_mode, self._cur_path)  # type: ignore[arg-type]

  def _dispatch(self, ev: _Event) -> None:
    handler = getattr(self, f"_on_{self._state.value}")
    handler(ev)

  # --- per-state handlers ---

  def _on_off(self, ev: _Event) -> None:
    if ev.kind is _EventKind.ACTIVATE:
      self._spawn(ev.mode, ev.path)
    # deactivate/connected/etc. in off: ignore.

  def _on_starting(self, ev: _Event) -> None:
    if ev.kind is _EventKind.CONNECTED:
      self._connect_deadline = None
      self._backoff = self._backoff_base
      # Force an IDR now (V2.2): the publisher has just connected, so a WHEP
      # reader waiting on runOnDemand gets a keyframe immediately instead of
      # waiting up to a full GOP. Once per connect — not on every reconnect
      # retry's CONNECTED — and best-effort (no-op if the encoder ignores it).
      if self._pipeline is not None:
        try:
          self._pipeline.force_idr()
        except Exception:  # noqa: BLE001
          log.exception("force_idr failed (continuing; short GOP covers it)")
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
      # Tear down current pipeline and retry with backoff -> back to starting.
      self._teardown()
      self._reconnect_at = time.monotonic() + self._backoff
      self._backoff = min(self._backoff * 2, self._backoff_max)
      self._set_state(LiveStateValue.starting)
      self._connect_deadline = None  # set when respawn happens
    elif ev.kind is _EventKind.ACTIVATE:
      pass  # idempotent ack; no-op

  def _on_stopping(self, ev: _Event) -> None:
    if ev.kind is _EventKind.EXITED:
      self._pipeline = None
      self._set_state(LiveStateValue.off)
      if self._queued_activate is not None:
        q = self._queued_activate
        self._queued_activate = None
        self._spawn(q.mode, q.path)
    elif ev.kind is _EventKind.ACTIVATE:
      # Queue; transition after current teardown completes.
      self._queued_activate = ev

  # --- helpers ---

  def _spawn(self, mode: ModeTriplet | None, path: str | None) -> None:
    assert mode is not None and path is not None
    self._cur_mode, self._cur_path = mode, path
    self._pipeline = self._make_pipeline(mode, path)
    self._connect_deadline = time.monotonic() + self._connect_timeout_s
    self._set_state(LiveStateValue.starting)
    self._pipeline.start()

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
    path = self._cur_path if state is not LiveStateValue.off else None
    mode = self._cur_mode if state in (LiveStateValue.starting, LiveStateValue.running) else None
    self._publish_state(state, path, mode)
