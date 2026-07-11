"""Snapshot pusher: PUTs the latest freeze-frame JPEG to webui's tailnet-only
ingest listener (Monitor UX — a recent still shown before Stream On, so the
operator doesn't pay for a WHIP activation just to see the scene).

  PUT <ingest_url_base>/snapshot
      body: JPEG bytes
      Content-Type: image/jpeg

Best-effort: a failed PUT (cellular blip, webui down) is logged and dropped —
the next tick tries again with a fresher frame, so there is nothing to retry or
queue (mirrors the uploader's ingest client shape, minus the retry taxonomy,
which that module needs for recordings but a snapshot does not: a stale-by-one-
tick miss is invisible to the operator).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class SnapshotPusher:
  def __init__(self, *, ingest_url_base: str, timeout_s: float = 10.0, verify_tls: bool = True) -> None:
    import httpx

    self._url = ingest_url_base.rstrip("/") + "/snapshot"
    self._client = httpx.Client(
      timeout=timeout_s, verify=verify_tls, headers={"User-Agent": "fessel-video-snapshot"}
    )

  def push(self, jpeg: bytes) -> None:
    import httpx

    try:
      resp = self._client.put(self._url, content=jpeg, headers={"Content-Type": "image/jpeg"})
    except httpx.HTTPError as e:
      log.warning("snapshot push failed: %s: %s", type(e).__name__, e)
      return
    if resp.status_code not in (200, 201, 204):
      log.warning("snapshot push -> %d", resp.status_code)
