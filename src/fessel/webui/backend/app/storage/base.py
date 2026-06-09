"""Storage backend interface (B5.5.1).

A small, backend-agnostic abstraction. Two implementations:
  - MinIO  (minio_backend.py): S3-compatible object store. Playback is a
    presigned-GET redirect — the browser fetches segments straight from MinIO,
    the backend never touches the bytes.
  - disk   (disk_backend.py): a mounted directory (PVC). Playback is served by
    the backend itself as byte-range GETs (read()).

The on-store layout is identical for both backends (architecture §4.3):
  recordings/<type>/<id>/{index.m3u8, seg-NNNNN.ts, metadata.json}
so a switch from one to the other is a copy, not a reformat.

Interface contract (enforced by the implementations + their tests):
  - store(...) STREAMS its byte iterator to the store incrementally — a 200 MB
    segment must not buffer fully in memory.
  - store(...) is IDEMPOTENT: a repeated PUT for the same (id, name) overwrites
    cleanly, which is what makes the Pi-side per-file retry safe.
  - store(...) is per-file ATOMIC: a call either completes the whole file or
    leaves the previous version (or absence) intact (disk: temp+rename; MinIO:
    the SDK's PutObject is atomic).
  - there are NO cross-recording transactions: a recording is "complete" only
    when all of its files are present; partial completeness is observable and
    consumers (list/playback) tolerate it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

# Recording type subdirectories (mirror the Pi-side storage layout: explicit
# operator recordings vs anomaly-triggered ones). `explicit` is the default
# when a file arrives before its metadata.json names the type.
EXPLICIT = "explicit"
ANOMALY = "anomaly"
DEFAULT_TYPE = EXPLICIT

PLAYLIST_FILENAME = "index.m3u8"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class RecordingMetadataView:
  """What `list()` reports per recording: the id, the files present, and the
  parsed metadata.json (or None if absent / unparseable — a partial upload).

  The backend's /api/recordings merges this with supervisor's Pi-local listing
  by recording_id (B5.5.5), so only the discriminating fields are surfaced
  here; the full metadata dict is carried through for the type/started_at sort.
  """

  recording_id: str
  rec_type: str = DEFAULT_TYPE
  files: tuple[str, ...] = ()
  metadata: dict | None = None  # parsed metadata.json; None until it lands

  @property
  def started_at(self) -> str:
    # Sort key for `list()` (newest first). Missing metadata sorts last.
    if self.metadata:
      return self.metadata.get("started_at") or ""
    return ""


# --- playback targets --------------------------------------------------------
# playback_url() returns one of these; the playback handler (B5.5.4) renders a
# 302 redirect for PresignedURL (MinIO) or a 200/206 byte stream for
# ServeLocally (disk). A uniform front, backend-specific behaviour.


@dataclass(frozen=True)
class PresignedURL:
  """MinIO: a short-TTL presigned GET URL. The handler 302-redirects to it."""

  url: str


@dataclass(frozen=True)
class ServeLocally:
  """Disk: the file is read from the backend and streamed (with Range)."""

  recording_id: str
  file_name: str


PlaybackTarget = PresignedURL | ServeLocally


@dataclass(frozen=True)
class ByteRange:
  """A parsed single-range HTTP Range request, resolved against a known size.

  HLS players issue `Range: bytes=START-END` (END optional). `start`/`end` are
  inclusive byte offsets; `length` is the number of bytes to serve; `total` is
  the file size. Used by the disk backend's read() to seek + bound the stream.
  """

  start: int
  end: int
  total: int

  @property
  def length(self) -> int:
    return self.end - self.start + 1


@dataclass(frozen=True)
class ReadResult:
  """A disk-backend read(): the byte iterator plus what the handler needs to
  build the response — content length, content type, and (for a Range request)
  the resolved range so it can emit 206 + Content-Range."""

  chunks: Iterator[bytes]
  content_type: str
  content_length: int
  byte_range: ByteRange | None = None  # set only for a Range request (206)


class StorageBackend:
  """The abstraction the rest of the backend depends on (B5.5.1)."""

  def store(self, recording_id: str, file_name: str, chunks: Iterable[bytes]) -> None:  # pragma: no cover - interface
    """Persist `chunks` under (recording_id, file_name), streaming. Idempotent,
    per-file atomic. The type subdir is read from a previously-stored
    metadata.json when available, else DEFAULT_TYPE."""
    raise NotImplementedError

  def exists(self, recording_id: str) -> bool:  # pragma: no cover - interface
    """True if any file for this recording is present."""
    raise NotImplementedError

  def list(self) -> list[RecordingMetadataView]:  # pragma: no cover - interface
    """Enumerate all recordings, newest first, with their parsed metadata."""
    raise NotImplementedError

  def playback_url(self, recording_id: str, file_name: str) -> PlaybackTarget | None:  # pragma: no cover - interface
    """A PlaybackTarget for one file, or None if it does not exist."""
    raise NotImplementedError

  def read(self, recording_id: str, file_name: str, http_range: str | None = None) -> ReadResult | None:  # pragma: no cover - interface
    """Disk-only: open the file and stream it (honouring an HTTP Range header).
    Returns None if the file is absent. The MinIO backend raises
    NotImplementedError — its playback is via redirect, never through the
    backend."""
    raise NotImplementedError


# --- shared helpers used by more than one backend ----------------------------


def content_type_for(name: str) -> str:
  """HLS-aware Content-Type. hls.js + Safari need the right type on .m3u8 and
  .ts or playback silently fails (F5.5.1 / B5.5.4)."""
  if name.endswith(".m3u8"):
    return "application/vnd.apple.mpegurl"
  if name.endswith(".ts"):
    return "video/mp2t"
  if name.endswith(".json"):
    return "application/json"
  return "application/octet-stream"


def parse_byte_range(header: str | None, total: int) -> ByteRange | None:
  """Parse a single `bytes=start-end` Range header against a known size.

  Returns a resolved ByteRange, or None when there is no Range header. Raises
  ValueError on a malformed or unsatisfiable range (the handler maps that to
  416). Only the single-range form HLS players use is supported; a multi-range
  header is rejected.
  """
  if not header:
    return None
  spec = header.strip()
  if not spec.lower().startswith("bytes="):
    raise ValueError(f"unsupported range unit: {header!r}")
  spec = spec[len("bytes=") :]
  if "," in spec:
    raise ValueError("multi-range requests are not supported")
  start_s, _, end_s = spec.partition("-")
  if start_s == "":
    # Suffix range: bytes=-N  -> the last N bytes.
    if end_s == "":
      raise ValueError("invalid range")
    n = int(end_s)
    if n <= 0:
      raise ValueError("invalid suffix range")
    start = max(0, total - n)
    end = total - 1
  else:
    start = int(start_s)
    end = int(end_s) if end_s else total - 1
  if start < 0 or end < start or start >= total:
    raise ValueError(f"unsatisfiable range: {header!r} against size {total}")
  end = min(end, total - 1)
  return ByteRange(start=start, end=end, total=total)
