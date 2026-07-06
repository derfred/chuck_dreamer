// Monitor video panel: a single stage with a Ring/Live source toggle (the
// bandwidth-aware heart of the redesign).
//
// Two sources, two very different costs:
//   - Ring (default): HLS replay of the rolling ring buffer the recorder is
//     already writing to disk. Cheap, buffered, always safe to leave running.
//   - Live: a dedicated WebRTC/WHEP stream with sub-second latency. Expensive —
//     it is the ONLY surface that opens a real-time stream, so it never
//     auto-starts and tears itself down on tab-hide (useLiveSession) and on
//     switch-back-to-Ring (here) and on navigate-away (unmount).
//
// Each source owns its own <video> element so switching is a clean stop/start,
// never a src/srcObject swap on a shared element. The inactive source's element
// is unmounted, which stops its playback (HLS destroy / live teardown).

import { useEffect, useRef, useState } from "react";
import { AnomalyLane } from "./AnomalyLane";
import { RING_PLAYLIST_URL } from "./api";
import { useHls } from "./useHls";
import { useLiveSession } from "./useLiveSession";
import type { AnomalyLogEntry } from "../../shared/schemas";

type Source = "ring" | "live";

// Ring window length (ms) for the anomaly lane's time mapping. Matches the
// backend ring buffer (~120s); a knob only if the buffer length changes.
const RING_WINDOW_MS = 120_000;

const SPINNER: Partial<Record<string, string>> = {
  Requesting: "Connecting…",
  Signaling: "Establishing connection…",
  WaitingForVideo: "Waiting for video…",
};

export function MonitorVideo({ anomalies }: { anomalies: AnomalyLogEntry[] }) {
  const [source, setSource] = useState<Source>("ring");
  return (
    <div style={{ border: "1px solid #dde3e6", borderRadius: 14, overflow: "hidden", background: "#fff" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "12px 14px",
          borderBottom: "1px solid #eaeef0",
        }}
      >
        <SourceToggle source={source} onChange={setSource} />
        <CostTag source={source} />
      </div>
      {source === "ring" ? (
        <RingStage anomalies={anomalies} onGoLive={() => setSource("live")} />
      ) : (
        <LiveStage onFellBack={() => setSource("ring")} />
      )}
    </div>
  );
}

// --- source toggle -----------------------------------------------------------

function SourceToggle({ source, onChange }: { source: Source; onChange: (s: Source) => void }) {
  return (
    <div
      role="group"
      aria-label="Video source"
      style={{
        display: "inline-flex",
        background: "#eef1f3",
        borderRadius: 10,
        padding: 3,
        border: "1px solid #dde3e6",
      }}
    >
      <SegButton on="ring" active={source === "ring"} onClick={() => onChange("ring")} />
      <SegButton on="live" active={source === "live"} onClick={() => onChange("live")} />
    </div>
  );
}

function SegButton({ on, active, onClick }: { on: Source; active: boolean; onClick: () => void }) {
  const label = on === "ring" ? "Ring" : "Live";
  const activeBg = on === "live" ? "#c2410c" : "#fff";
  const activeColor = on === "live" ? "#fff" : "#0d6b57";
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      style={{
        font: "inherit",
        fontSize: 13,
        fontWeight: 600,
        cursor: "pointer",
        border: "none",
        background: active ? activeBg : "transparent",
        color: active ? activeColor : "#5a6b68",
        padding: "6px 14px",
        borderRadius: 7,
        boxShadow: active && on === "ring" ? "0 1px 2px rgba(0,0,0,.06)" : "none",
      }}
    >
      {label}
    </button>
  );
}

function CostTag({ source }: { source: Source }) {
  const cheap = source === "ring";
  return (
    <span
      style={{
        marginLeft: "auto",
        fontSize: 12,
        fontWeight: 600,
        padding: "3px 9px",
        borderRadius: 999,
        color: cheap ? "#0d6b57" : "#c2410c",
        background: cheap ? "rgba(23,150,122,0.12)" : "#fdf1ec",
      }}
    >
      {cheap ? "◇ Buffered · low bandwidth" : "⚡ WebRTC · dedicated stream"}
    </span>
  );
}

// --- ring stage (cheap, default) --------------------------------------------

function RingStage({
  anomalies,
  onGoLive,
}: {
  anomalies: AnomalyLogEntry[];
  onGoLive: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  useHls(videoRef, RING_PLAYLIST_URL, { live: true });

  const seek = (entry: AnomalyLogEntry) => {
    // Best-effort playhead jump: map the anomaly's age onto the buffered range.
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
    <div>
      <video
        ref={videoRef}
        controls
        autoPlay
        playsInline
        muted
        style={{ width: "100%", display: "block", background: "#000", aspectRatio: "16 / 10" }}
      />
      <div style={{ display: "flex", justifyContent: "flex-end", padding: "8px 12px 0" }}>
        <button
          type="button"
          onClick={onGoLive}
          style={{
            font: "inherit",
            fontSize: 12.5,
            fontWeight: 650,
            cursor: "pointer",
            color: "#fff",
            background: "#c2410c",
            border: "none",
            padding: "6px 12px",
            borderRadius: 8,
          }}
        >
          Go Live ⚡
        </button>
      </div>
      <div style={{ padding: "8px 12px 12px" }}>
        <AnomalyLane anomalies={anomalies} windowMs={RING_WINDOW_MS} onSeek={seek} />
      </div>
    </div>
  );
}

// --- live stage (expensive, explicit) ---------------------------------------

function LiveStage({ onFellBack }: { onFellBack: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const session = useLiveSession(videoRef);
  const { state, detail, attempt } = session;

  // Start the WebRTC session as soon as the Live source is selected (mounted).
  // Stop it on unmount (switching back to Ring / leaving Monitor) — the
  // useLiveSession unmount effect tears the stream down.
  useEffect(() => {
    session.start();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const spinner = SPINNER[state];
  const showVideo = state === "Playing" || state === "WaitingForVideo" || state === "Stalled";

  return (
    <div>
      <div style={{ position: "relative", background: "#000", aspectRatio: "16 / 10" }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{
            width: "100%",
            height: "100%",
            objectFit: "contain",
            display: showVideo ? "block" : "none",
          }}
        />
        {spinner && (
          <Overlay>
            <Spinner />
            <div>{spinner}</div>
          </Overlay>
        )}
        {state === "Stalled" && (
          <Banner color="#b8860b">
            Stream stalled — reconnecting… (attempt {attempt}){detail ? ` — ${detail}` : ""}
          </Banner>
        )}
        {state === "Error" && (
          <Overlay>
            <p style={{ color: "#ffb4a4", margin: 0, textAlign: "center", padding: "0 16px" }}>
              {detail || "Something went wrong."}
            </p>
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <button type="button" onClick={session.retry} style={liveBtn}>
                Retry
              </button>
              <button type="button" onClick={onFellBack} style={liveBtnGhost}>
                Back to Ring
              </button>
            </div>
          </Overlay>
        )}
      </div>
    </div>
  );
}

const liveBtn: React.CSSProperties = {
  font: "inherit",
  fontSize: 13,
  fontWeight: 600,
  cursor: "pointer",
  color: "#fff",
  background: "#c2410c",
  border: "none",
  padding: "7px 14px",
  borderRadius: 8,
};

const liveBtnGhost: React.CSSProperties = {
  ...liveBtn,
  background: "transparent",
  border: "1px solid rgba(255,255,255,0.4)",
};

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
