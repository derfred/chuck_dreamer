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
import { AuthError, fetchSnapshotMeta, reauthenticate } from "./api";
import { config } from "./config";
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

// Cache-bust the <img> src on every successful meta poll, so the browser
// re-fetches the JPEG each time a fresher one has landed rather than serving
// its own cached copy of an older frame.
function snapshotUrl(receivedAt: string): string {
  return `/api/snapshot?t=${encodeURIComponent(receivedAt)}`;
}

function ageStr(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h`;
}

function StreamOffStage({ onTurnOn }: { onTurnOn: () => void }) {
  const [meta, setMeta] = useState<{ receivedAt: string } | null>(null);
  const [now, setNow] = useState<number>(() => Date.now());
  const [lightboxOpen, setLightboxOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      fetchSnapshotMeta()
        .then((m) => {
          if (cancelled) return;
          setMeta(m.available && m.received_at ? { receivedAt: m.received_at } : null);
        })
        .catch((e) => {
          if (cancelled) return;
          if (e instanceof AuthError) {
            reauthenticate();
            return;
          }
          // Network / 5xx: keep showing the last-known snapshot
        });
    };
    poll();
    const id = window.setInterval(poll, config.snapshotPollMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  // Ticks "now" every second so the age label advances between polls.
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const ageMs = meta ? now - Date.parse(meta.receivedAt) : null;
  // Colour-shift toward amber as the frame approaches the documented ~30s
  // staleness bound, so a stuck Pi push (not just a normal 10s-old frame) is
  // visually distinct.
  const stale = ageMs != null && ageMs > 25_000;

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
      {meta && (
        <button
          type="button"
          onClick={() => setLightboxOpen(true)}
          aria-label="View full-size snapshot"
          style={{
            position: "absolute",
            inset: 0,
            padding: 0,
            border: "none",
            cursor: "zoom-in",
            background: "transparent",
          }}
        >
          <img
            src={snapshotUrl(meta.receivedAt)}
            alt="Recent camera snapshot (not live)"
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </button>
      )}
      <div
        style={{
          position: "relative",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 10,
          padding: meta ? "10px 14px" : 0,
          borderRadius: 12,
          background: meta ? "rgba(5,16,14,0.72)" : "transparent",
        }}
      >
        {meta ? (
          <span
            style={{
              fontSize: 12,
              fontWeight: 650,
              padding: "3px 9px",
              borderRadius: 999,
              color: stale ? "#2a1600" : "#0a2e26",
              background: stale ? "#e0a955" : "#9fe0cf",
            }}
          >
            ● snapshot · {ageStr(ageMs!)} ago — not live
          </span>
        ) : (
          <div style={{ fontSize: 13.5 }}>Stream is off — no bandwidth in use.</div>
        )}
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
      {lightboxOpen && meta && (
        <SnapshotLightbox receivedAt={meta.receivedAt} ageMs={ageMs!} onClose={() => setLightboxOpen(false)} />
      )}
    </div>
  );
}

// Full-size view (click-to-enlarge without starting the live stream). A
// custom lightbox mirroring RecordModal's dialog pattern (no UI library in
// this codebase): Escape/outside-click to close, full-viewport overlay.
function SnapshotLightbox({
  receivedAt,
  ageMs,
  onClose,
}: {
  receivedAt: string;
  ageMs: number;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Full-size snapshot"
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(6,12,11,0.85)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 12,
        padding: 20,
        zIndex: 60,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span
          style={{
            fontSize: 12,
            fontWeight: 650,
            padding: "3px 9px",
            borderRadius: 999,
            color: "#0a2e26",
            background: "#9fe0cf",
          }}
        >
          ● snapshot · {ageStr(ageMs)} ago — not live
        </span>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          style={{
            font: "inherit",
            fontSize: 18,
            lineHeight: 1,
            color: "#fff",
            background: "none",
            border: 0,
            cursor: "pointer",
            padding: "2px 6px",
          }}
        >
          ✕
        </button>
      </div>
      <img
        src={snapshotUrl(receivedAt)}
        alt="Recent camera snapshot (not live), full size"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: "100%", maxHeight: "80vh", objectFit: "contain", borderRadius: 8 }}
      />
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
