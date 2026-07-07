// Start-recording dialog with look-back (the ring-buffer reframe).
//
// The ring buffer lives on the Pi behind the cellular link, so it is never
// streamed back for viewing. Its value is retroactive capture: when you start a
// recording you can reach the start point BACK into the buffer to include
// footage from before you hit record. This dialog picks that look-back on a
// timeline spanning [−max_lookback, now], with anomaly markers to anchor to, and
// an "upload when done" toggle. There is NO quality/resolution picker —
// recording resolution is a deploy setting (capabilities.recording_mode).

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AnomalyLogEntry } from "../../shared/schemas";

const ANOMALY_META: Record<string, { label: string; color: string }> = {
  safe_box_violation: { label: "Safe-box violation", color: "#b22222" },
  rest_violation: { label: "Rest-period motion", color: "#cc7000" },
  audio_spike: { label: "Audio spike", color: "#cc9a00" },
};
const DEFAULT_META = { label: "Anomaly", color: "#666" };

// Quick-pick look-back offsets (seconds). Capped to maxLookback at render.
const PRESETS = [0, 30, 60, 120];

export interface RecordModalProps {
  maxLookbackSeconds: number;
  anomalies: AnomalyLogEntry[];
  busy: boolean;
  onCancel: () => void;
  onConfirm: (opts: { lookbackSeconds: number; uploadWhenDone: boolean }) => void;
}

export function RecordModal({
  maxLookbackSeconds,
  anomalies,
  busy,
  onCancel,
  onConfirm,
}: RecordModalProps) {
  const [lookback, setLookback] = useState<number>(0);
  const [upload, setUpload] = useState<boolean>(false);
  const trackRef = useRef<HTMLDivElement>(null);
  const dragging = useRef<boolean>(false);

  const clamp = useCallback(
    (s: number) => Math.max(0, Math.min(maxLookbackSeconds, s)),
    [maxLookbackSeconds],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  // Map a pointer x within the track to a look-back (left edge = max, right = now).
  const seekFromClientX = useCallback(
    (clientX: number) => {
      const el = trackRef.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
      setLookback(clamp(maxLookbackSeconds * (1 - frac)));
    },
    [clamp, maxLookbackSeconds],
  );

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (dragging.current) seekFromClientX(e.clientX);
    };
    const onUp = () => (dragging.current = false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [seekFromClientX]);

  // Anomaly markers within the look-back window, positioned by age.
  const now = useMemo(() => Date.now(), []);
  const marks = useMemo(() => {
    const windowStart = now - maxLookbackSeconds * 1000;
    return anomalies
      .map((a) => ({ a, at: Date.parse(a.ts) }))
      .filter((m) => Number.isFinite(m.at) && m.at >= windowStart && m.at <= now)
      .map((m) => ({
        entry: m.a,
        leftPct: ((m.at - windowStart) / (maxLookbackSeconds * 1000)) * 100,
        ageSeconds: (now - m.at) / 1000,
        meta: ANOMALY_META[m.a.type] ?? DEFAULT_META,
      }));
  }, [anomalies, now, maxLookbackSeconds]);

  // handle position: 0s look-back = now = right edge (100%).
  const handlePct = (1 - lookback / maxLookbackSeconds) * 100;
  const nearEdge = lookback > maxLookbackSeconds - 15;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Start recording"
      style={overlay}
      onClick={onCancel}
    >
      <div style={modal} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: "flex", alignItems: "center", padding: "18px 20px 0" }}>
          <h2 style={{ fontSize: 18, margin: 0 }}>Start recording</h2>
          <button aria-label="Close" onClick={onCancel} style={closeBtn}>
            ✕
          </button>
        </div>

        <div style={{ padding: "14px 20px 4px" }}>
          <p style={{ color: "#5a6b68", fontSize: 13.5, margin: "2px 0 18px" }}>
            Reach back into the on-Pi buffer to capture footage from before now — anchor
            the start to a moment or an anomaly.
          </p>

          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
            <span style={eyebrow}>Look-back start</span>
            <span style={{ fontFamily: "monospace", fontSize: 13, color: "#c2410c", fontWeight: 650 }}>
              {lookback < 1 ? "from now (no look-back)" : `−${fmt(lookback)} before now`}
            </span>
          </div>

          {/* timeline */}
          <div style={{ position: "relative", height: 56, userSelect: "none" }}>
            <div
              ref={trackRef}
              onMouseDown={(e) => {
                dragging.current = true;
                seekFromClientX(e.clientX);
              }}
              style={track}
            >
              {/* recorded region: handle -> now */}
              <div
                style={{
                  position: "absolute",
                  left: `${handlePct}%`,
                  right: 0,
                  top: 0,
                  bottom: 0,
                  background: "rgba(194,65,12,0.28)",
                  borderRadius: 6,
                }}
              />
            </div>
            {/* anomaly ticks */}
            {marks.map((m, i) => (
              <button
                key={`${m.entry.ts}-${i}`}
                type="button"
                title={`${m.meta.label} · −${fmt(m.ageSeconds)}`}
                onClick={() => setLookback(clamp(m.ageSeconds))}
                aria-label={`Anchor look-back to ${m.meta.label}`}
                style={{
                  position: "absolute",
                  left: `${m.leftPct}%`,
                  top: 6,
                  width: 3,
                  height: 22,
                  padding: 0,
                  border: "none",
                  borderRadius: 2,
                  background: m.meta.color,
                  transform: "translateX(-50%)",
                  cursor: "pointer",
                }}
              />
            ))}
            {/* draggable start handle */}
            <div
              role="slider"
              tabIndex={0}
              aria-label="Recording start point"
              aria-valuemin={0}
              aria-valuemax={Math.round(maxLookbackSeconds)}
              aria-valuenow={Math.round(lookback)}
              onMouseDown={(e) => {
                e.stopPropagation();
                dragging.current = true;
              }}
              onKeyDown={(e) => {
                if (e.key === "ArrowLeft") setLookback((s) => clamp(s + 5));
                if (e.key === "ArrowRight") setLookback((s) => clamp(s - 5));
              }}
              style={{ ...handle, left: `${handlePct}%` }}
            />
            <div style={scale}>
              <span>−{fmt(maxLookbackSeconds)}</span>
              <span>now</span>
            </div>
          </div>

          {/* presets */}
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", margin: "16px 0 6px" }}>
            {PRESETS.filter((s) => s <= maxLookbackSeconds).map((s) => (
              <button
                key={s}
                type="button"
                aria-pressed={Math.abs(s - lookback) < 0.5}
                onClick={() => setLookback(s)}
                style={preset(Math.abs(s - lookback) < 0.5)}
              >
                {s === 0 ? "Now" : `−${fmt(s)}`}
              </button>
            ))}
          </div>

          {/* upload-when-done toggle */}
          <label style={uploadRow}>
            <span style={{ display: "flex", flexDirection: "column" }}>
              <b style={{ fontSize: 13.5 }}>Upload when done</b>
              <span style={{ fontSize: 12, color: "#5a6b68" }}>
                Flag for upload the moment the recording stops.
              </span>
            </span>
            <input
              type="checkbox"
              checked={upload}
              onChange={(e) => setUpload(e.target.checked)}
              aria-label="Upload when done"
              style={{ marginLeft: "auto", width: 18, height: 18 }}
            />
          </label>
        </div>

        <div style={foot}>
          {nearEdge && (
            <span style={{ fontSize: 12, color: "#b8860b" }}>
              ⚠ near the edge of the on-Pi buffer
            </span>
          )}
          <span style={{ flex: 1 }} />
          <button onClick={onCancel} style={cancelBtn}>
            Cancel
          </button>
          <button
            onClick={() => onConfirm({ lookbackSeconds: lookback, uploadWhenDone: upload })}
            disabled={busy}
            style={confirmBtn}
          >
            {busy ? "…" : "Start recording"}
          </button>
        </div>
      </div>
    </div>
  );
}

function fmt(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// --- styles ------------------------------------------------------------------
const overlay: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(6,12,11,0.5)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 20,
  zIndex: 60,
};
const modal: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #dde3e6",
  borderRadius: 16,
  width: 560,
  maxWidth: "100%",
  boxShadow: "0 24px 60px rgba(0,0,0,0.28)",
  overflow: "hidden",
};
const closeBtn: React.CSSProperties = {
  marginLeft: "auto",
  font: "inherit",
  fontSize: 20,
  lineHeight: 1,
  color: "#8a9895",
  background: "none",
  border: 0,
  cursor: "pointer",
  padding: "2px 6px",
};
const eyebrow: React.CSSProperties = {
  fontSize: 12,
  letterSpacing: "0.04em",
  textTransform: "uppercase",
  color: "#8a9895",
  fontWeight: 650,
};
const track: React.CSSProperties = {
  position: "absolute",
  left: 0,
  right: 0,
  top: 22,
  height: 12,
  borderRadius: 6,
  background: "#eef1f3",
  border: "1px solid #dde3e6",
  cursor: "pointer",
};
const handle: React.CSSProperties = {
  position: "absolute",
  top: 14,
  width: 14,
  height: 28,
  borderRadius: 5,
  background: "#fff",
  border: "2px solid #c2410c",
  boxShadow: "0 2px 6px rgba(0,0,0,0.18)",
  cursor: "ew-resize",
  transform: "translateX(-50%)",
  zIndex: 4,
};
const scale: React.CSSProperties = {
  position: "absolute",
  left: 0,
  right: 0,
  bottom: 0,
  display: "flex",
  justifyContent: "space-between",
  fontSize: 10.5,
  color: "#8a9895",
  fontFamily: "monospace",
};
const uploadRow: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 12,
  margin: "16px 0 4px",
  padding: "12px 14px",
  borderRadius: 12,
  border: "1px solid #eaeef0",
  cursor: "pointer",
};
const foot: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "16px 20px 20px",
  borderTop: "1px solid #eaeef0",
  marginTop: 14,
};
const cancelBtn: React.CSSProperties = {
  font: "inherit",
  fontSize: 13,
  fontWeight: 600,
  cursor: "pointer",
  padding: "9px 16px",
  borderRadius: 9,
  border: "1px solid #dde3e6",
  background: "#fff",
  color: "#16211f",
};
const confirmBtn: React.CSSProperties = {
  font: "inherit",
  fontSize: 13,
  fontWeight: 700,
  cursor: "pointer",
  padding: "9px 18px",
  borderRadius: 9,
  border: 0,
  background: "#c0392b",
  color: "#fff",
};
function preset(active: boolean): React.CSSProperties {
  return {
    font: "inherit",
    fontSize: 12.5,
    fontWeight: 600,
    cursor: "pointer",
    padding: "6px 12px",
    borderRadius: 999,
    border: `1px solid ${active ? "#16211f" : "#dde3e6"}`,
    background: active ? "#16211f" : "#fff",
    color: active ? "#fff" : "#5a6b68",
  };
}
