// Monitor video panel: the live WebRTC view with a Stream On/Off toggle above it.
//
// The ring buffer lives on the Pi behind the cellular link, so it is NEVER
// streamed back for viewing — its only role is retroactive capture (the record
// dialog's look-back). The one viewable surface here is the dedicated
// WebRTC/WHEP live stream, and the toggle controls whether that stream is
// active at all. It defaults to OFF (0 bandwidth): the operator turns it on
// deliberately, and it tears itself down on tab-hide (useLiveSession) and on
// navigate-away (unmount).

import { useEffect, useRef, useState } from "react";
import { useLiveSession } from "./useLiveSession";

const SPINNER: Partial<Record<string, string>> = {
  Requesting: "Connecting…",
  Signaling: "Establishing connection…",
  WaitingForVideo: "Waiting for video…",
};

export function MonitorVideo() {
  // Default OFF: opening Monitor must not open a stream. The operator turns the
  // live view on when they want it (and pays the bandwidth then, not before).
  const [streamOn, setStreamOn] = useState(false);
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
        <StreamToggle on={streamOn} onChange={setStreamOn} />
        <CostTag on={streamOn} />
      </div>
      {streamOn ? <LiveStage /> : <StreamOffStage onTurnOn={() => setStreamOn(true)} />}
    </div>
  );
}

// --- stream on/off toggle ----------------------------------------------------

function StreamToggle({ on, onChange }: { on: boolean; onChange: (on: boolean) => void }) {
  return (
    <div
      role="group"
      aria-label="Live stream"
      style={{
        display: "inline-flex",
        background: "#eef1f3",
        borderRadius: 10,
        padding: 3,
        border: "1px solid #dde3e6",
      }}
    >
      <SegButton label="Stream On" mode="on" active={on} onClick={() => onChange(true)} />
      <SegButton label="Stream Off" mode="off" active={!on} onClick={() => onChange(false)} />
    </div>
  );
}

function SegButton({
  label,
  mode,
  active,
  onClick,
}: {
  label: string;
  mode: "on" | "off";
  active: boolean;
  onClick: () => void;
}) {
  // Active "On" is the warm live colour (streaming = costs bandwidth); active
  // "Off" is a neutral raised chip (the safe resting state).
  const activeBg = mode === "on" ? "#c2410c" : "#fff";
  const activeColor = mode === "on" ? "#fff" : "#16211f";
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
        boxShadow: active && mode === "off" ? "0 1px 2px rgba(0,0,0,.06)" : "none",
      }}
    >
      {label}
    </button>
  );
}

function CostTag({ on }: { on: boolean }) {
  return (
    <span
      style={{
        marginLeft: "auto",
        fontSize: 12,
        fontWeight: 600,
        padding: "3px 9px",
        borderRadius: 999,
        color: on ? "#c2410c" : "#8a9895",
        background: on ? "#fdf1ec" : "#eef1f3",
      }}
    >
      {on ? "⚡ WebRTC · live" : "◇ stream off · 0 bandwidth"}
    </span>
  );
}

// --- stream-off resting stage ------------------------------------------------

function StreamOffStage({ onTurnOn }: { onTurnOn: () => void }) {
  return (
    <div
      style={{
        position: "relative",
        background: "#05100e",
        aspectRatio: "16 / 10",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 12,
        color: "#cfe0db",
      }}
    >
      <div style={{ fontSize: 13.5 }}>Stream is off — no bandwidth in use.</div>
      <button
        type="button"
        onClick={onTurnOn}
        style={{
          font: "inherit",
          fontSize: 13,
          fontWeight: 650,
          cursor: "pointer",
          color: "#fff",
          background: "#c2410c",
          border: "none",
          padding: "8px 16px",
          borderRadius: 9,
        }}
      >
        Turn stream on
      </button>
    </div>
  );
}

// --- live stage (WebRTC) -----------------------------------------------------

function LiveStage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const session = useLiveSession(videoRef);
  const { state, detail, attempt } = session;

  // Start the WebRTC session as soon as the stream is turned on (mounted). Stop
  // it on unmount (toggled off / left Monitor) — useLiveSession's unmount effect
  // tears the stream down and releases the WHEP viewer.
  useEffect(() => {
    session.start();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const spinner = SPINNER[state];
  const showVideo = state === "Playing" || state === "WaitingForVideo" || state === "Stalled";

  return (
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
          <button type="button" onClick={session.retry} style={liveBtn}>
            Retry
          </button>
        </Overlay>
      )}
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
  marginTop: 12,
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
