// /ring view (F4.1): the always-on ring buffer as a scrubbable HLS window of
// recent footage. No activity-score sparkline yet (vision dependency, Slice 5).
//
// The ring is live and rolling, so the player defaults to the live edge and the
// playlist is refreshed periodically (hls.js polls the playlist itself; we just
// keep the element pointed at the live URL). Behind oauth2-proxy like the rest
// of the operator UI; a 401 on load escalates to re-auth.

import { useRef } from "react";
import { RING_PLAYLIST_URL } from "./api";
import { useHls } from "./useHls";

export function Ring() {
  const videoRef = useRef<HTMLVideoElement>(null);
  // Always-live source: hls.js follows the rolling playlist; live=true keeps
  // the player near the edge as new segments are written.
  useHls(videoRef, RING_PLAYLIST_URL, { live: true });

  return (
    <div style={{ padding: 16 }}>
      <h1>Fessel — Ring buffer</h1>
      <p style={{ color: "#666", fontSize: 13 }}>
        The last few minutes of footage, always recording. Scrub back to review;
        the window rolls forward as new footage is captured.
      </p>
      <video
        ref={videoRef}
        controls
        autoPlay
        playsInline
        muted
        style={{ width: "100%", maxWidth: 960, background: "#000" }}
      />
    </div>
  );
}
