// Footage page (`/footage`): everything on disk, replayed over HLS — no live
// stream involved. The rolling ring buffer (a scrubbable hero with an anomaly
// lane) sits above the explicit + anomaly recordings table (the embedded
// Recordings view).
//
// Merges the old /ring and /recordings routes: they are the two cheap,
// HLS-backed ways to look at past footage, distinct from the expensive live
// stream that now lives on Monitor.

import { useRef } from "react";
import { RING_PLAYLIST_URL } from "./api";
import { AnomalyLane } from "./AnomalyLane";
import { Recordings } from "./Recordings";
import { useAnomalies } from "./useAnomalies";
import { useHls } from "./useHls";
import type { AnomalyLogEntry } from "../../shared/schemas";

const RING_WINDOW_MS = 120_000;

export function Footage() {
  return (
    <div style={{ padding: 16, maxWidth: 1080, margin: "0 auto" }}>
      <h1 style={{ fontSize: 22, letterSpacing: "-0.02em", margin: "10px 0 18px" }}>Footage</h1>
      <RingHero />
      <section style={{ marginTop: 28 }}>
        <h2 style={{ fontSize: 16, margin: "0 0 12px" }}>Recordings</h2>
        <Recordings embedded />
      </section>
    </div>
  );
}

// The ring buffer as a scrubbable HLS window with an anomaly lane beneath the
// player. Live-following (live=true) so it opens at the edge; scrub back to
// review. No WebRTC stream is opened — this is the recorder's own segments.
function RingHero() {
  const anomalies = useAnomalies();
  const videoRef = useRef<HTMLVideoElement>(null);
  useHls(videoRef, RING_PLAYLIST_URL, { live: true });

  const seek = (entry: AnomalyLogEntry) => {
    const video = videoRef.current;
    if (!video) return;
    const ageMs = Date.now() - Date.parse(entry.ts);
    if (!Number.isFinite(ageMs)) return;
    const buffered = video.buffered;
    if (buffered.length === 0) return;
    const end = buffered.end(buffered.length - 1);
    video.currentTime = Math.max(0, end - ageMs / 1000);
  };

  return (
    <section>
      <h2 style={{ fontSize: 16, margin: "0 0 12px" }}>Ring buffer</h2>
      <div style={{ border: "1px solid #dde3e6", borderRadius: 14, overflow: "hidden", background: "#fff" }}>
        <video
          ref={videoRef}
          controls
          autoPlay
          playsInline
          muted
          style={{ width: "100%", display: "block", background: "#000", aspectRatio: "16 / 9" }}
        />
        <div style={{ padding: "10px 12px 12px" }}>
          <AnomalyLane anomalies={anomalies} windowMs={RING_WINDOW_MS} onSeek={seek} />
        </div>
      </div>
    </section>
  );
}
