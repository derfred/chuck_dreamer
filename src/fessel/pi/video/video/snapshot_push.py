"""Snapshot pusher: PUTs the latest freeze-frame JPEG to webui's tailnet-only
ingest listener (Monitor UX — a recent still shown before Stream On, so the
operator doesn't pay for a WHIP activation just to see the scene).

  PUT <ingest_url_base>/snapshot
      body: JPEG bytes
      Content-Type: image/jpeg
      X-Snapshot-Age-Ms: how long ago the frame was CAPTURED (see below)

The age header is what makes the webui's "Xs ago" label honest: without it the
webui can only date the frame from when the PUT landed, which hides a stalled
pipeline (a wedged camera pushes the same cached frame forever, each push
looking brand new). It is sent as an *age* rather than a capture timestamp on
purpose — an age is measured entirely with the Pi's monotonic clock and
resolved against the webui's wall clock on arrival, so a Pi whose wall clock is
skewed (no NTP yet after a boot, cellular-only) can't misreport freshness.

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

  def push(self, jpeg: bytes, age_s: float = 0.0) -> None:
    """PUT the frame, declaring how old it already is (seconds since capture).
    A negative age (clock oddity) is clamped to 0 rather than sent."""
    import httpx

    headers = {
      "Content-Type": "image/jpeg",
      "X-Snapshot-Age-Ms": str(int(max(0.0, age_s) * 1000)),
    }
    try:
      resp = self._client.put(self._url, content=jpeg, headers=headers)
    except httpx.HTTPError as e:
      log.warning("snapshot push failed: %s: %s", type(e).__name__, e)
      return
    if resp.status_code not in (200, 201, 204):
      log.warning("snapshot push -> %d", resp.status_code)
