// /live page (F2.2). The full UX state machine (Idle → Requesting → Signaling
// → WaitingForVideo → Playing, with Stalled/Error recovery), media-liveness
// detection and auto-reconnect (in useLiveSession), and re-auth escalation on
// a 401. There is NO resolution/mode selector: the live profile is fixed
// (architecture §4.4); the WHEP offer goes same-origin to /whep.

import { useRef } from "react";
import { useLiveSession } from "./useLiveSession";

// Per-state display: spinner/banner copy + whether the video element shows.
// Stalled deliberately KEEPS the video element on screen (last good frame
// stays visible while reconnect runs — recovery, not a UI flash).
const SPINNER: Partial<Record<string, string>> = {
  Requesting: "Connecting…",
  Signaling: "Establishing connection…",
  WaitingForVideo: "Waiting for video…",
};

export function Live() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const session = useLiveSession(videoRef);

  const { state, detail, attempt } = session;
  const spinner = SPINNER[state];
  const showVideo = state === "Playing" || state === "WaitingForVideo" || state === "Stalled";

  return (
    <div style={{ fontFamily: "sans-serif", padding: 16 }}>
      <h1>Fessel — Live</h1>

      <div>
        {state === "Idle" || state === "Error" ? (
          <button onClick={session.start}>Start live view</button>
        ) : (
          <button onClick={session.stop}>Stop</button>
        )}
      </div>

      <div style={{ position: "relative", width: 640, marginTop: 12 }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ width: 640, background: "#000", display: showVideo ? "block" : "none" }}
        />

        {spinner && (
          <Overlay>
            <Spinner />
            <div>{spinner}</div>
          </Overlay>
        )}

        {state === "Stalled" && (
          <Banner color="#b8860b">
            Stream stalled — reconnecting… (attempt {attempt})
            {detail ? ` — ${detail}` : ""}
          </Banner>
        )}

        {state === "Error" && (
          <div style={{ marginTop: 8 }}>
            <p style={{ color: "crimson" }}>{detail || "Something went wrong."}</p>
            <button onClick={session.retry}>Retry</button>
          </div>
        )}
      </div>

      <p style={{ color: "#666", marginTop: 12 }}>State: {state}</p>
    </div>
  );
}

function Overlay({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 8,
        color: "#fff",
        background: "rgba(0,0,0,0.6)",
        minHeight: 360,
      }}
    >
      {children}
    </div>
  );
}

function Banner({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        bottom: 0,
        padding: "6px 10px",
        background: color,
        color: "#fff",
        fontSize: 13,
      }}
    >
      {children}
    </div>
  );
}

function Spinner() {
  return (
    <div
      style={{
        width: 28,
        height: 28,
        border: "3px solid rgba(255,255,255,0.3)",
        borderTopColor: "#fff",
        borderRadius: "50%",
        animation: "fessel-spin 0.8s linear infinite",
      }}
    />
  );
}
