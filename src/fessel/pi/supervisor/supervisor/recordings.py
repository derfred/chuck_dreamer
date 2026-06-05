"""Safe static-file serving for the ring buffer and explicit recordings (S4.3).

supervisor proxies the ring playlist + segments (and, for symmetry, local
explicit-recording playlists + segments) from the USB SSD to the cluster, which
has no direct filesystem access to the Pi. Two safety properties matter because
this endpoint is reachable from the cluster:

  - **No path traversal.** A requested name must resolve to a file *inside* the
    served directory; `..`, absolute paths, and symlinks pointing out are
    rejected (a directory traversal here is arbitrary Pi-file read).
  - **Range requests must work** — HLS players issue them for scrubbing.
    Starlette's FileResponse honours the Range header, so resolving to a
    FileResponse is enough; we just verify it.

Cache headers (S4.3): short on the rotating playlist, long-immutable on the
write-once segments.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

PLAYLIST_CACHE = "no-cache, max-age=1"
SEGMENT_CACHE = "public, max-age=31536000, immutable"


def _safe_child(base: Path, name: str) -> Path:
  """Resolve `name` strictly under `base`, or raise 404. Rejects traversal,
  absolute paths, and symlinks that escape the base."""
  base = base.resolve()
  # A name with a path separator or parent ref is suspicious; resolve and check
  # containment rather than trusting the string.
  candidate = (base / name).resolve()
  try:
    candidate.relative_to(base)
  except ValueError as e:
    raise HTTPException(status_code=404, detail="not found") from e
  if not candidate.is_file():
    raise HTTPException(status_code=404, detail="not found")
  return candidate


def _serve(path: Path) -> FileResponse:
  is_playlist = path.suffix == ".m3u8"
  media_type = "application/vnd.apple.mpegurl" if is_playlist else "video/mp2t"
  cache = PLAYLIST_CACHE if is_playlist else SEGMENT_CACHE
  # FileResponse handles Range requests (206 + Content-Range) automatically.
  return FileResponse(path, media_type=media_type, headers={"Cache-Control": cache})


class RingProxy:
  """Serves the ring playlist + segments and explicit-recording files from the
  SSD, with traversal protection (S4.3, B4.3/B4.4 open seam).

  `on_ring_read` (set by create_app to the bandwidth coordinator's
  `note_ring_read`) fires whenever a ring file is served, so the coordinator
  knows a ring viewer is active and suspends uploads (§2.12). It is the explicit
  presence signal HLS needs (WebRTC would have provided it implicitly). Only
  ring reads fire it — explicit-recording review is out of the §2.12 "start
  simple" scope.
  """

  def __init__(
    self,
    ring_dir: Path,
    explicit_dir: Path,
    on_ring_read: Callable[[], None] | None = None,
  ) -> None:
    self._ring_dir = ring_dir
    self._explicit_dir = explicit_dir
    self._on_ring_read = on_ring_read

  def _note_read(self) -> None:
    if self._on_ring_read is not None:
      self._on_ring_read()

  def ring_playlist(self) -> FileResponse:
    path = self._ring_dir / "index.m3u8"
    if not path.is_file():
      raise HTTPException(status_code=404, detail="ring playlist not available")
    self._note_read()
    return _serve(path)

  def ring_segment(self, name: str) -> FileResponse:
    resp = _serve(_safe_child(self._ring_dir, name))
    self._note_read()
    return resp

  def recording_playlist(self, recording_id: str) -> FileResponse:
    rec_dir = _recording_dir(self._explicit_dir, recording_id)
    path = rec_dir / "index.m3u8"
    if not path.is_file():
      raise HTTPException(status_code=404, detail="recording playlist not found")
    return _serve(path)

  def recording_segment(self, recording_id: str, name: str) -> FileResponse:
    rec_dir = _recording_dir(self._explicit_dir, recording_id)
    return _serve(_safe_child(rec_dir, name))


def _recording_dir(explicit_dir: Path, recording_id: str) -> Path:
  """Resolve a recording directory strictly under explicit/, rejecting a
  recording_id that contains traversal segments."""
  base = explicit_dir.resolve()
  candidate = (base / recording_id).resolve()
  try:
    candidate.relative_to(base)
  except ValueError as e:
    raise HTTPException(status_code=404, detail="not found") from e
  if not candidate.is_dir():
    raise HTTPException(status_code=404, detail="recording not found")
  return candidate
