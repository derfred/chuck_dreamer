"""Anomaly-triggered recording extraction (V5.7, the deferred Slice-4 task).

When any anomaly event fires (safe-box, rest, audio spike), video brackets the
moment with a finite HLS recording: `T_before` seconds lifted from the ring
buffer plus `T_after` seconds of fresh forward capture. The bundle is written
under recordings/anomaly/<id>/ and announced on MQTT so supervisor and the
dashboard can link the anomaly to its evidence.

This module is the orchestration core. The GStreamer-touching parts — the
forward-capture branch — are abstracted behind a `ForwardCaptureFactory` (same
pattern as the recording state machine's pipeline factory), so the
copy-from-ring + coalesce + metadata logic is unit-testable against a tmp dir
with a fake forward-capture handle. The ring-segment selection is mtime-based
(the plan's "quick first implementation"; a rotation-event index is the noted
upgrade).

Concurrency (plan §V5.7 "coalesce"): at most one anomaly recording captures at a
time. A trigger arriving while a capture is in flight extends `T_after` from the
new trigger and appends to `trigger_events`, rather than spawning a second
recording — avoiding an explosion of recordings during a sustained anomaly.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fessel_schemas import AnomalyRecordingMetadata, AnomalyType, ModeTriplet

log = logging.getLogger(__name__)


def _now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


# A ForwardCaptureHandle abstracts the GStreamer forward-capture branch: it
# writes new HLS segments into the anomaly dir and calls `on_closed` when it has
# run for its duration and finalised. Implementations are GStreamer; tests use a
# fake. `extend(extra_seconds)` lengthens an in-flight capture (coalesce).
class ForwardCaptureHandle:
  def start(self) -> None:  # pragma: no cover - interface
    raise NotImplementedError

  def extend(self, extra_seconds: float) -> None:  # pragma: no cover - interface
    """Lengthen an in-flight forward capture (coalesce)."""
    raise NotImplementedError

  def stop(self) -> None:  # pragma: no cover - interface
    raise NotImplementedError


# factory(anomaly_dir, after_seconds, on_closed) -> handle.
ForwardCaptureFactory = Callable[[Path, float, Callable[[], None]], ForwardCaptureHandle]


@dataclass
class AnomalyWindow:
  before_seconds: float = 30.0
  after_seconds: float = 60.0


@dataclass
class _Capture:
  """An in-flight anomaly recording."""

  id: str
  anomaly_event_type: AnomalyType
  trigger_ts: str
  started_at: str
  earliest_segment_ts: str | None
  trigger_events: list[dict] = field(default_factory=list)
  handle: ForwardCaptureHandle | None = None


def select_ring_segments(ring_dir: Path, start_epoch: float, end_epoch: float) -> list[Path]:
  """The ring .ts segments whose mtime falls in [start, end], in write order.

  mtime-based (the plan's quick first implementation). A segment is written when
  hlssink2 finishes it, so its mtime is ~the wall-clock of its last frame; we
  include any segment whose mtime is >= start (so the segment covering the
  window start is captured) and <= end + a small slack.
  """
  if not ring_dir.is_dir():
    return []
  segs = []
  for p in ring_dir.iterdir():
    if p.suffix != ".ts" or not p.name.startswith("seg-"):
      continue
    try:
      mtime = p.stat().st_mtime
    except OSError:
      continue
    if start_epoch <= mtime <= end_epoch:
      segs.append((mtime, p))
  segs.sort()
  return [p for _, p in segs]


def latest_ring_segment(ring_dir: Path) -> str | None:
  """Name of the most-recently-written ring segment (the `ring_segment_hint`
  stamped on anomaly events at trigger time, V5.3)."""
  if not ring_dir.is_dir():
    return None
  newest: tuple[float, str] | None = None
  for p in ring_dir.iterdir():
    if p.suffix != ".ts" or not p.name.startswith("seg-"):
      continue
    try:
      mtime = p.stat().st_mtime
    except OSError:
      continue
    if newest is None or mtime > newest[0]:
      newest = (mtime, p.name)
  return newest[1] if newest else None


class AnomalyRecorder:
  """Brackets anomalies with finite HLS recordings (V5.7).

  `trigger(event_type, ts, payload)` is called in-process by video's anomaly
  handler the moment an anomaly fires (no MQTT round-trip). All state is guarded
  by a lock; the forward-capture close callback runs on a GStreamer thread.
  """

  def __init__(
    self,
    *,
    ring_dir: Path,
    anomaly_dir: Path,
    forward_capture_factory: ForwardCaptureFactory,
    publish_event: Callable[[str, AnomalyType, str], None],
    window: AnomalyWindow,
    mode: ModeTriplet | None = None,
    count_segments: Callable[[str], int] | None = None,
    auto_upload: bool = False,
    flag_for_upload: Callable[[str], None] | None = None,
    now: Callable[[], float] = time.time,
    new_id: Callable[[], str] | None = None,
  ) -> None:
    self._ring_dir = Path(ring_dir)
    self._anomaly_dir = Path(anomaly_dir)
    self._make_forward = forward_capture_factory
    self._publish_event = publish_event
    self._window = window
    self._mode = mode
    self._count_segments = count_segments or (lambda _id: 0)
    self._auto_upload = auto_upload
    self._flag_for_upload = flag_for_upload
    self._now = now
    if new_id is None:
      import uuid

      new_id = lambda: str(uuid.uuid4())  # noqa: E731
    self._new_id = new_id
    self._lock = threading.Lock()
    self._current: _Capture | None = None

  def update_window(self, window: AnomalyWindow) -> None:
    """SIGHUP reload (§2.13): a new window applies to the next anomaly; an
    in-flight capture keeps its window."""
    with self._lock:
      self._window = window

  def update_auto_upload(self, auto_upload: bool) -> None:
    with self._lock:
      self._auto_upload = auto_upload

  def trigger(self, event_type: AnomalyType, trigger_ts: str, payload: dict) -> str | None:
    """Bracket the anomaly. Returns the anomaly-recording id (new or, on
    coalesce, the in-flight one), or None if extraction could not start."""
    with self._lock:
      if self._current is not None:
        # Coalesce: extend the in-flight capture's forward window from now.
        self._current.trigger_events.append({"type": event_type.value, **payload})
        if self._current.handle is not None:
          self._current.handle.extend(self._window.after_seconds)
        log.info(
          "anomaly recording %s coalesced new %s trigger",
          self._current.id,
          event_type.value,
        )
        return self._current.id
      return self._start_capture(event_type, trigger_ts, payload)

  def _start_capture(self, event_type: AnomalyType, trigger_ts: str, payload: dict) -> str | None:
    anomaly_id = self._new_id()
    rec_dir = self._anomaly_dir / anomaly_id
    try:
      rec_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
      log.exception("failed to create anomaly recording dir %s", rec_dir)
      return None

    # Copy (never move — the ring is still rolling) the ring segments covering
    # [now - before, now] into the anomaly dir, renumbered in order so the final
    # playlist references before+after contiguously.
    now_epoch = self._now()
    start_epoch = now_epoch - self._window.before_seconds
    src_segments = select_ring_segments(self._ring_dir, start_epoch, now_epoch)
    earliest_ts: str | None = None
    for i, src in enumerate(src_segments):
      dst = rec_dir / f"seg-{i:05d}.ts"
      try:
        shutil.copy2(src, dst)
      except OSError:
        log.warning("failed to copy ring segment %s into anomaly recording", src)
        continue
      if earliest_ts is None:
        earliest_ts = datetime.fromtimestamp(src.stat().st_mtime, timezone.utc).isoformat()

    cap = _Capture(
      id=anomaly_id,
      anomaly_event_type=event_type,
      trigger_ts=trigger_ts,
      started_at=_now_iso(),
      earliest_segment_ts=earliest_ts,
      trigger_events=[{"type": event_type.value, **payload}],
    )
    self._current = cap
    log.info(
      "anomaly recording %s started (%s, %d ring segments)",
      anomaly_id,
      event_type.value,
      len(src_segments),
    )
    # Spawn forward capture; it appends new segments and fires _on_forward_closed.
    cap.handle = self._make_forward(
      rec_dir, self._window.after_seconds, lambda: self._on_forward_closed(anomaly_id)
    )
    try:
      cap.handle.start()
    except Exception:  # noqa: BLE001 — a forward-capture failure should still
      # finalise what we copied from the ring, not crash.
      log.exception("anomaly forward capture failed to start; finalising before-only")
      self._finalise(anomaly_id)
    return anomaly_id

  def _on_forward_closed(self, anomaly_id: str) -> None:
    with self._lock:
      self._finalise(anomaly_id)

  def _finalise(self, anomaly_id: str) -> None:
    """Write index.m3u8 + metadata.json and publish the event. Caller holds the
    lock (or is the in-lock start-failure path)."""
    cap = self._current
    if cap is None or cap.id != anomaly_id:
      return
    rec_dir = self._anomaly_dir / anomaly_id
    segments = self._count_segments(anomaly_id)
    self._write_playlist(rec_dir)
    ended_at = _now_iso()
    meta = AnomalyRecordingMetadata(
      id=anomaly_id,
      anomaly_event_type=cap.anomaly_event_type,
      trigger_ts=cap.trigger_ts,
      window={
        "before_seconds": self._window.before_seconds,
        "after_seconds": self._window.after_seconds,
      },
      trigger_events=cap.trigger_events,
      started_at=cap.earliest_segment_ts or cap.started_at,
      ended_at=ended_at,
      operator=None,
      mode=self._mode,
      duration_seconds=_duration_seconds(cap.earliest_segment_ts or cap.started_at, ended_at),
      segments=segments,
    )
    self._write_metadata(rec_dir, meta)
    self._current = None
    self._publish_event(anomaly_id, cap.anomaly_event_type, cap.trigger_ts)
    if self._auto_upload and self._flag_for_upload is not None:
      try:
        self._flag_for_upload(anomaly_id)
      except Exception:  # noqa: BLE001 — auto-flag is best-effort
        log.exception("auto-flag of anomaly recording %s failed", anomaly_id)
    log.info("anomaly recording %s finalised (%d segments)", anomaly_id, segments)

  def _write_playlist(self, rec_dir: Path) -> None:
    """Write a finite VOD playlist referencing every .ts in the dir, in order.
    The copied ring segments + forward-capture segments together form the
    bracketed window; hlssink2 wrote its own playlist for the forward part, but
    we rewrite a combined one so playback spans before+after as one timeline."""
    segs = sorted(p.name for p in rec_dir.iterdir() if p.suffix == ".ts")
    lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:4", "#EXT-X-PLAYLIST-TYPE:VOD"]
    for name in segs:
      lines.append("#EXTINF:2.0,")
      lines.append(name)
    lines.append("#EXT-X-ENDLIST")
    (rec_dir / "index.m3u8").write_text("\n".join(lines) + "\n", encoding="utf-8")

  def _write_metadata(self, rec_dir: Path, meta: AnomalyRecordingMetadata) -> None:
    import json
    import os

    path = rec_dir / "metadata.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta.model_dump(mode="json"), indent=2), encoding="utf-8")
    os.replace(tmp, path)

  def shutdown(self) -> None:
    with self._lock:
      if self._current is not None and self._current.handle is not None:
        try:
          self._current.handle.stop()
        except Exception:  # noqa: BLE001
          log.exception("error stopping anomaly forward capture on shutdown")


def _duration_seconds(started_iso: str, ended_iso: str) -> int:
  try:
    a = datetime.fromisoformat(started_iso)
    b = datetime.fromisoformat(ended_iso)
    return max(0, int((b - a).total_seconds()))
  except ValueError:
    return 0
