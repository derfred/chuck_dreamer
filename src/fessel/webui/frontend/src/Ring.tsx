// /ring view (F4.1 + F5.4): the always-on ring buffer as a scrubbable HLS window
// of recent footage, with a live activity-score sparkline below the player.
//
// The ring is live and rolling, so the player defaults to the live edge and the
// playlist is refreshed periodically (hls.js polls the playlist itself; we just
// keep the element pointed at the live URL). Behind oauth2-proxy like the rest
// of the operator UI; a 401 on load escalates to re-auth.
//
// The activity sparkline (F5.4) is aligned to "now" — a short client-side
// history of activity scores polled from /api/state. Aligning it to the
// *scrubbed* playhead needs a server-side time-indexed score history; that is
// deferred (Slice 7). The live-to-now sparkline is sufficient for Slice 5.

import { useEffect, useRef, useState } from "react";
import { AuthError, fetchState, RING_PLAYLIST_URL, redirectToLogin } from "./api";
import { config } from "./config";
import { Sparkline } from "./Sparkline";
import { useHls } from "./useHls";

export function Ring() {
  const videoRef = useRef<HTMLVideoElement>(null);
  // Always-live source: hls.js follows the rolling playlist; live=true keeps
  // the player near the edge as new segments are written.
  useHls(videoRef, RING_PLAYLIST_URL, { live: true });

  const [history, setHistory] = useState<number[]>([]);
  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      fetchState()
        .then((s) => {
          if (cancelled) return;
          const score = s.vision?.activity_score_ema;
          if (typeof score === "number") {
            setHistory((h) => [...h, score].slice(-120));
          }
        })
        .catch((e) => {
          if (e instanceof AuthError) redirectToLogin();
          // Other errors: keep the last-known history (informational overlay).
        });
    };
    poll();
    const id = window.setInterval(poll, config.statePollMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

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
      <div style={{ maxWidth: 960, marginTop: 8 }}>
        <div style={{ color: "#666", fontSize: 13 }}>Activity (last ~2 min, live)</div>
        <Sparkline
          values={history}
          width={960}
          height={36}
          max={1}
          ariaLabel="ring activity score history"
        />
      </div>
    </div>
  );
}
