"""Uploader core: the directory-watch upload pipeline (V4.7 / V5.5.1), MQTT-free
and sleep-free so it is unit-testable against a tmp SSD + a FakeIngestClient.

Driven by the upload_queue marker contract (V4.6): `flag-upload` (in video)
writes `/mnt/ssd/upload_queue/<id>.upload`; the uploader watches the directory
and processes any marker it finds. The marker's presence — not an in-memory
queue — is the durable signal, so the pipeline survives both video and uploader
restarts.

Per-recording outcome:
  - success      -> delete the marker; metadata.upload_state = "uploaded"
                    (+ uploaded_at); publish progress.
  - retryable    -> keep the marker, bump attempt count + backoff; metadata
                    stays "uploading"; publish progress with last_error.
  - permanent    -> rename marker to <id>.failed; metadata.upload_state =
                    "failed"; publish progress.

Backlog gauges (V4.8): count of pending markers + age of the oldest, published
on a periodic schedule. The Slice-7 ">3h" P2 alert lands against these.

Liveness heartbeat: published on the same schedule as the backlog gauges,
unconditionally (regardless of queue depth or the upload gate) — its absence,
not the backlog count, is what supervisor uses to tell "uploader crash-looping"
apart from "uploader idle with nothing queued" (the two look identical in the
backlog gauges alone).

Uploads are single-threaded (V4.7): cellular uplink is the bottleneck; parallel
uploads only fight each other for it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fessel_schemas import UploadProgress, UploadStateValue
from fessel_shared import Storage
from fessel_shared import topics

from .ingest import IngestClient, PermanentError, RetryableError

log = logging.getLogger(__name__)

# Default backoff schedule (seconds) for retryable failures (V4.7). The last
# value is the steady-state cap; the uploader never gives up on a retryable.
DEFAULT_BACKOFF_S = [30, 60, 300, 900, 1800]

# The recording's playlist + the metadata file. The completion ordering
# (V5.5.1) uploads segments first, then the playlist, then metadata.json LAST:
# metadata.json's presence is the backend's "fully uploaded" marker, so a
# partially-uploaded recording is never mistaken for complete.
PLAYLIST_FILENAME = "index.m3u8"
METADATA_FILENAME = "metadata.json"


def _now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


def _upload_rank(name: str) -> tuple[int, str]:
  """Sort key for the completion ordering (V5.5.1): segments (0) before the
  playlist (1) before metadata.json (2). Within a rank, lexicographic by name
  (so seg-00000 precedes seg-00001)."""
  if name == METADATA_FILENAME:
    return (2, name)
  if name == PLAYLIST_FILENAME:
    return (1, name)
  return (0, name)


@dataclass
class UploadOutcome:
  recording_id: str
  state: UploadStateValue
  attempts: int
  error: str | None = None


class Uploader:
  def __init__(
    self,
    *,
    storage: Storage,
    client: IngestClient,
    publish: Callable[[str, dict, int, bool], None],
    backoff_s: list[int] | None = None,
  ) -> None:
    self._storage = storage
    self._client = client
    self._publish = publish
    self._backoff = backoff_s or DEFAULT_BACKOFF_S
    # Per-recording attempt counters (in-memory; the marker is the durable
    # state, the count is observability that resets on uploader restart).
    self._attempts: dict[str, int] = {}

  def set_backoff(self, backoff_s: list[int]) -> None:
    """Replace the retry backoff schedule (SIGHUP reload, §2.13). Affects the
    next retryable failure; an in-flight backoff isn't retroactively changed."""
    if backoff_s:
      self._backoff = backoff_s

  # --- one pass over the queue ----------------------------------------------

  def process_pending(self) -> list[UploadOutcome]:
    """Process every pending marker once. Returns an outcome per marker.

    Retryable failures leave the marker in place (re-tried on the next pass);
    the caller (run loop) sleeps the backoff between passes."""
    outcomes: list[UploadOutcome] = []
    for marker in self._storage.list_upload_markers():
      recording_id = marker.stem  # "<id>.upload" -> "<id>"
      outcomes.append(self._process_one(recording_id, marker))
    return outcomes

  def _process_one(self, recording_id: str, marker: Path) -> UploadOutcome:
    attempt = self._attempts.get(recording_id, 0) + 1
    self._attempts[recording_id] = attempt

    rec_dir = self._storage.recording_dir(recording_id)
    if not rec_dir.is_dir():
      # The recording directory is gone — nothing to upload. Non-retryable.
      return self._fail(recording_id, marker, attempt, "recording directory missing")

    self._set_metadata_state(recording_id, UploadStateValue.uploading)
    self._publish_progress(recording_id, UploadStateValue.uploading, attempt)

    try:
      self._upload_dir(recording_id, rec_dir)
    except PermanentError as e:
      return self._fail(recording_id, marker, attempt, str(e))
    except RetryableError as e:
      # Keep the marker; metadata stays "uploading"; report the transient error.
      self._set_metadata_state(recording_id, UploadStateValue.uploading)
      self._publish_progress(recording_id, UploadStateValue.uploading, attempt, str(e))
      log.warning(
        '{"event":"upload_retry","recording_id":"%s","attempt":%d,"error":"%s"}',
        recording_id,
        attempt,
        e,
      )
      return UploadOutcome(recording_id, UploadStateValue.uploading, attempt, str(e))

    # Success: delete the marker, finalise metadata, publish.
    marker.unlink(missing_ok=True)
    self._attempts.pop(recording_id, None)
    self._set_metadata_state(recording_id, UploadStateValue.uploaded, uploaded=True)
    self._publish_progress(recording_id, UploadStateValue.uploaded, attempt)
    log.info(
      '{"event":"upload_complete","recording_id":"%s","attempts":%d}', recording_id, attempt
    )
    return UploadOutcome(recording_id, UploadStateValue.uploaded, attempt)

  def _upload_dir(self, recording_id: str, rec_dir: Path) -> None:
    """PUT every file of the recording to the backend's ingest endpoint, in the
    completion order (V5.5.1): segments first, then the playlist, then
    metadata.json LAST. metadata.json's arrival is the backend's "fully
    uploaded" marker, so this ordering is what makes `list()` show the recording
    as complete only once everything is up.

    A single retryable failure aborts the whole recording (the next pass
    re-uploads from the start — per-file PUTs are idempotent on the backend, so
    re-PUTting an already-stored file is harmless)."""
    files = [f for f in rec_dir.iterdir() if f.is_file()]
    for f in sorted(files, key=lambda p: _upload_rank(p.name)):
      self._client.put_file(recording_id, f.name, f)

  def _fail(self, recording_id: str, marker: Path, attempt: int, reason: str) -> UploadOutcome:
    # Rename the marker to <id>.failed and record the failure (V4.6/V4.7).
    failed = self._storage.upload_failed_path(recording_id)
    try:
      marker.replace(failed)
    except OSError:
      log.exception("could not rename marker for %s", recording_id)
    self._attempts.pop(recording_id, None)
    self._set_metadata_state(recording_id, UploadStateValue.failed)
    self._publish_progress(recording_id, UploadStateValue.failed, attempt, reason)
    log.error(
      '{"event":"upload_failed","recording_id":"%s","attempt":%d,"reason":"%s"}',
      recording_id,
      attempt,
      reason,
    )
    return UploadOutcome(recording_id, UploadStateValue.failed, attempt, reason)

  def backoff_for(self, attempt: int) -> int:
    """Seconds to wait before the next pass, given the highest attempt count
    seen this pass. Capped at the last schedule value (V4.7)."""
    idx = min(max(0, attempt - 1), len(self._backoff) - 1)
    return self._backoff[idx]

  # --- metadata + MQTT -------------------------------------------------------

  def _set_metadata_state(
    self, recording_id: str, state: UploadStateValue, *, uploaded: bool = False
  ) -> None:
    meta = self._storage.read_metadata(recording_id)
    if meta is None:
      return  # partial recording; nothing to update
    meta.upload_state = state
    if uploaded:
      meta.uploaded_at = _now_iso()
    self._storage.write_metadata(meta)

  def _publish_progress(
    self,
    recording_id: str,
    state: UploadStateValue,
    attempts: int,
    last_error: str | None = None,
  ) -> None:
    progress = UploadProgress(
      recording_id=recording_id,
      state=state,
      attempts=attempts,
      last_attempt_at=_now_iso(),
      last_error=last_error,
    )
    self._publish(
      topics.upload_progress(recording_id),
      progress.model_dump(),
      topics.QOS_UPLOAD,
      topics.RETAIN_UPLOAD,
    )

  # --- backlog gauges (V4.8) -------------------------------------------------

  def publish_backlog(self) -> tuple[int, int | None]:
    """Publish the upload-queue health gauges: count of pending markers and the
    age (seconds) of the oldest. Returns the (count, oldest_seconds) it
    published so callers/tests can assert without re-reading MQTT."""
    markers = self._storage.list_upload_markers()
    count = len(markers)
    oldest_seconds: int | None = None
    if markers:
      now = datetime.now(timezone.utc).timestamp()
      oldest_mtime = min(m.stat().st_mtime for m in markers)
      oldest_seconds = max(0, int(now - oldest_mtime))
    self._publish(topics.UPLOAD_BACKLOG_COUNT, {"value": count}, topics.QOS_UPLOAD, topics.RETAIN_UPLOAD)
    self._publish(
      topics.UPLOAD_BACKLOG_OLDEST,
      {"value": oldest_seconds},
      topics.QOS_UPLOAD,
      topics.RETAIN_UPLOAD,
    )
    return count, oldest_seconds

  def publish_heartbeat(self) -> None:
    """Publish the uploader liveness heartbeat. Called on the same cadence as
    `publish_backlog`, unconditionally — see the module docstring."""
    self._publish(
      topics.UPLOAD_HEARTBEAT,
      {"ts": _now_iso()},
      topics.QOS_UPLOAD_HEARTBEAT,
      topics.RETAIN_UPLOAD_HEARTBEAT,
    )
