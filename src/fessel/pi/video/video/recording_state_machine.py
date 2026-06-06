"""Explicit-recording state machine (V4.3).

States: {idle, starting, recording, stopping}. start/stop are transition
REQUESTS, not direct spawn/kill commands. Like the live state machine
(state_machine.py), all transitions are serialised on a single worker thread so
two recording pipelines can never run at once and no encoder process leaks.

This is a SEPARATE state machine from the live branch: the two share the encoder
hardware but are independent (the architecture, V4.3). Activating live while
recording is fine; stopping live while recording continues; etc.

Transition table:

| From      | Event                  | To        | Action                                       |
|-----------|------------------------|-----------|----------------------------------------------|
| idle      | start(id, mode, op)    | starting  | create dir, spawn recording branch, arm tmo  |
| starting  | first segment written  | recording | publish state                                |
| starting  | error / timeout        | idle      | tear down; leave partial output on disk      |
| starting  | stop                   | stopping  | cancel pending start; tear down              |
| recording | stop                   | stopping  | EOS the recording branch; await finalisation |
| recording | start(new id)          | recording | REJECT (only one active recording)           |
| stopping  | branch exited cleanly  | idle      | finalise playlist; write metadata; publish   |
| stopping  | start                  | stopping  | REJECT (one recording at a time)             |

A start that is rejected (already recording) is reported back to the caller via
`request_start`'s return value, so supervisor can surface a 409 (V4.3, S4.1).

> Note: the V4.3 plan's transition table had `stopping | start -> starting
> (queue)`. We deliberately simplified that to a synchronous REJECT: a start in
> ANY non-idle state (starting / recording / stopping) returns False from
> `request_start` so supervisor 409s immediately, consistent with the
> one-recording-at-a-time contract. Queuing a start across a teardown added a
> code path with no operator benefit (an explicit manual gesture should just be
> retried after the stop completes), so it was removed.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from fessel_schemas import ModeTriplet, RecordingMetadata, RecordingStateValue

log = logging.getLogger(__name__)


def _now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


class _EventKind(Enum):
  START = "start"
  STOP = "stop"
  SEGMENT = "segment"  # first segment written -> recording is live on disk
  FAILED = "failed"  # spawn/encode error or start timeout
  EXITED = "exited"  # branch finished tearing down (EOS handled)


@dataclass
class _Event:
  kind: _EventKind
  recording_id: str | None = None
  mode: ModeTriplet | None = None
  operator: str | None = None


# A RecordingPipelineHandle abstracts the GStreamer recording branch so the
# state machine is testable without GStreamer. Implementations spawn/teardown
# and report segment/failed/exited back via the supplied callbacks.
class RecordingPipelineHandle:
  def start(self) -> None:  # pragma: no cover - interface
    raise NotImplementedError

  def stop(self) -> None:  # pragma: no cover - interface
    """Request graceful teardown (EOS the recording branch only); must
    eventually emit EXITED."""
    raise NotImplementedError


# factory(recording_id, mode) -> handle. The handle reports back to the SM.
RecordingPipelineFactory = Callable[[str, ModeTriplet], RecordingPipelineHandle]
# publish_state(state, recording_id, started_at) -> None
RecordingStatePublisher = Callable[[RecordingStateValue, str | None, str | None], None]
# finalise(metadata) -> None  (write metadata.json; called on clean finalisation)
RecordingFinaliser = Callable[[RecordingMetadata], None]


class RecordingStateMachine:
  def __init__(
    self,
    *,
    pipeline_factory: RecordingPipelineFactory,
    publish_state: RecordingStatePublisher,
    finalise: RecordingFinaliser,
    start_timeout_s: float,
    count_segments: Callable[[str], int] | None = None,
  ) -> None:
    self._make_pipeline = pipeline_factory
    self._publish_state = publish_state
    self._finalise = finalise
    self._start_timeout_s = start_timeout_s
    # Used to stamp metadata.segments on finalisation; defaults to 0 (the count
    # is best-effort metadata, not load-bearing).
    self._count_segments = count_segments or (lambda _id: 0)

    self._state = RecordingStateValue.idle
    self._events: queue.Queue[_Event] = queue.Queue()
    self._thread = threading.Thread(target=self._run, name="rec-sm", daemon=True)
    self._stop = threading.Event()

    self._pipeline: RecordingPipelineHandle | None = None
    self._cur_id: str | None = None
    self._cur_mode: ModeTriplet | None = None
    self._cur_operator: str | None = None
    self._started_at: str | None = None
    self._start_deadline: float | None = None

  # --- public API (thread-safe) ----------------------------------------------

  def start(self) -> None:
    self._thread.start()

  def shutdown(self) -> None:
    self._stop.set()
    self._events.put(_Event(_EventKind.STOP))
    self._thread.join(timeout=5)

  @property
  def state(self) -> RecordingStateValue:
    return self._state

  def set_start_timeout(self, start_timeout_s: float) -> None:
    """Update the start-timeout (SIGHUP reload, §2.13). Takes effect on the next
    recording; an in-flight `starting` deadline is not retroactively changed."""
    self._start_timeout_s = float(start_timeout_s)

  @property
  def active_recording_id(self) -> str | None:
    return self._cur_id if self._state is not RecordingStateValue.idle else None

  def request_start(self, recording_id: str, mode: ModeTriplet, operator: str | None) -> bool:
    """Request a recording start. Returns False *synchronously* if a recording
    is already active (idle is required), so the HTTP caller can 409 without
    waiting for the async transition. Otherwise enqueues and returns True."""
    if self._state is not RecordingStateValue.idle:
      log.info("recording start rejected: state=%s active=%s", self._state, self._cur_id)
      return False
    self._events.put(
      _Event(_EventKind.START, recording_id=recording_id, mode=mode, operator=operator)
    )
    return True

  def request_stop(self) -> None:
    self._events.put(_Event(_EventKind.STOP))

  # --- callbacks the pipeline implementation calls (thread-safe) -------------

  def on_segment(self) -> None:
    self._events.put(_Event(_EventKind.SEGMENT))

  def on_failed(self) -> None:
    self._events.put(_Event(_EventKind.FAILED))

  def on_exited(self) -> None:
    self._events.put(_Event(_EventKind.EXITED))

  # --- worker loop -----------------------------------------------------------

  def _run(self) -> None:
    self._set_state(RecordingStateValue.idle)
    while not self._stop.is_set() or self._state is not RecordingStateValue.idle:
      timeout = self._next_timeout()
      try:
        ev = self._events.get(timeout=timeout)
      except queue.Empty:
        self._handle_timers()
        continue
      self._dispatch(ev)

  def _next_timeout(self) -> float:
    if self._start_deadline is None:
      return 1.0
    return max(0.0, self._start_deadline - time.monotonic())

  def _handle_timers(self) -> None:
    if self._start_deadline is not None and time.monotonic() >= self._start_deadline:
      log.warning("recording start timed out in %s (id=%s)", self._state, self._cur_id)
      self._dispatch(_Event(_EventKind.FAILED))

  def _dispatch(self, ev: _Event) -> None:
    handler = getattr(self, f"_on_{self._state.value}")
    handler(ev)

  # --- per-state handlers ----------------------------------------------------

  def _on_idle(self, ev: _Event) -> None:
    if ev.kind is _EventKind.START:
      self._spawn(ev.recording_id, ev.mode, ev.operator)
    # stop/segment/etc. in idle: ignore.

  def _on_starting(self, ev: _Event) -> None:
    if ev.kind is _EventKind.SEGMENT:
      self._start_deadline = None
      self._set_state(RecordingStateValue.recording)
    elif ev.kind is _EventKind.FAILED:
      self._start_deadline = None
      self._teardown()
      # Leave any partial output on disk; do NOT finalise (no metadata).
      self._reset_current()
      self._set_state(RecordingStateValue.idle)
    elif ev.kind is _EventKind.STOP:
      self._begin_teardown()
    # START in starting: ignore (request_start already rejected it).

  def _on_recording(self, ev: _Event) -> None:
    if ev.kind is _EventKind.STOP:
      self._begin_teardown()
    # START in recording is rejected synchronously in request_start, so it never
    # arrives here; a SEGMENT after the first is a no-op.

  def _on_stopping(self, ev: _Event) -> None:
    if ev.kind is _EventKind.EXITED:
      self._pipeline = None
      self._finalise_current()
      self._reset_current()
      self._set_state(RecordingStateValue.idle)
    # A START never reaches here: request_start rejects any non-idle state
    # synchronously (-> supervisor 409), so there is no start to queue.

  # --- helpers ---------------------------------------------------------------

  def _spawn(self, recording_id: str | None, mode: ModeTriplet | None, operator: str | None) -> None:
    assert recording_id is not None and mode is not None
    self._cur_id = recording_id
    self._cur_mode = mode
    self._cur_operator = operator
    self._started_at = _now_iso()
    self._pipeline = self._make_pipeline(recording_id, mode)
    self._start_deadline = time.monotonic() + self._start_timeout_s
    self._set_state(RecordingStateValue.starting)
    self._pipeline.start()

  def _begin_teardown(self) -> None:
    self._start_deadline = None
    self._set_state(RecordingStateValue.stopping)
    if self._pipeline is not None:
      self._pipeline.stop()  # graceful EOS; emits EXITED
    else:
      self._dispatch(_Event(_EventKind.EXITED))

  def _teardown(self) -> None:
    if self._pipeline is not None:
      try:
        self._pipeline.stop()
      except Exception:  # noqa: BLE001
        log.exception("error tearing down recording pipeline")
      self._pipeline = None

  def _finalise_current(self) -> None:
    """Write metadata.json for the recording that just finalised (V4.4)."""
    if self._cur_id is None or self._started_at is None:
      return
    ended_at = _now_iso()
    duration = _duration_seconds(self._started_at, ended_at)
    meta = RecordingMetadata(
      id=self._cur_id,
      started_at=self._started_at,
      ended_at=ended_at,
      operator=self._cur_operator,
      mode=self._cur_mode,
      duration_seconds=duration,
      segments=self._count_segments(self._cur_id),
    )
    try:
      self._finalise(meta)
    except Exception:  # noqa: BLE001 — a metadata write failure must not crash the SM
      log.exception("failed to write recording metadata for %s", self._cur_id)

  def _reset_current(self) -> None:
    self._cur_id = None
    self._cur_mode = None
    self._cur_operator = None
    self._started_at = None

  def _set_state(self, state: RecordingStateValue) -> None:
    self._state = state
    rid = self._cur_id if state is not RecordingStateValue.idle else None
    started = self._started_at if state is not RecordingStateValue.idle else None
    self._publish_state(state, rid, started)


def _duration_seconds(started_iso: str, ended_iso: str) -> int:
  try:
    a = datetime.fromisoformat(started_iso)
    b = datetime.fromisoformat(ended_iso)
    return max(0, int((b - a).total_seconds()))
  except ValueError:
    return 0
