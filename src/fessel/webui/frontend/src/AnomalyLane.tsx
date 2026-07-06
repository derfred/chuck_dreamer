// Anomaly marker lane under a ring/scrubber timeline (F5.2 + F5.4). Renders the
// recent anomaly log as ticks positioned along a time window, colour-coded by
// type, hoverable for a label + timestamp, clickable to jump the playhead.
//
// The ring is a rolling live window of the last `windowMs` up to "now"; a
// marker's horizontal position is (ts - windowStart) / windowMs. Markers older
// than the window (or in the future) are dropped. "Now" ticks on a slow cadence
// so markers drift left as the window rolls — matching the live edge.
//
// Aligning to a *scrubbed* playhead needs a server-side time-indexed history
// (deferred, Slice 7); a live-to-now lane is sufficient here. The colour map is
// shared with the dashboard's anomaly panel (ANOMALY_LABELS).

import { useEffect, useMemo, useState } from "react";
import type { AnomalyLogEntry } from "../../shared/schemas";

// Type -> {label, colour}. Semantic colours (critical vs warning), not the
// accent. Kept in sync with Dashboard's ANOMALY_LABELS.
const ANOMALY_META: Record<string, { label: string; color: string }> = {
  safe_box_violation: { label: "Safe-box violation", color: "#b22222" },
  rest_violation: { label: "Rest-period motion", color: "#cc7000" },
  audio_spike: { label: "Audio spike", color: "#cc9a00" },
};

const DEFAULT_META = { label: "Anomaly", color: "#666" };

export interface AnomalyLaneProps {
  anomalies: AnomalyLogEntry[];
  // Length of the visible window in ms (e.g. the ring's 120s buffer).
  windowMs: number;
  // Called with the entry when a marker is clicked (jump the playhead).
  onSeek?: (entry: AnomalyLogEntry) => void;
}

export function AnomalyLane({ anomalies, windowMs, onSeek }: AnomalyLaneProps) {
  // Tick "now" so markers drift as the live window rolls forward.
  const [now, setNow] = useState<number>(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const windowStart = now - windowMs;
  const marks = useMemo(() => {
    return anomalies
      .map((a) => ({ entry: a, at: Date.parse(a.ts) }))
      .filter((m) => Number.isFinite(m.at) && m.at >= windowStart && m.at <= now)
      .map((m) => ({
        entry: m.entry,
        pct: ((m.at - windowStart) / windowMs) * 100,
        meta: ANOMALY_META[m.entry.type] ?? DEFAULT_META,
      }));
  }, [anomalies, windowStart, now, windowMs]);

  // The legend lists only the types actually present in the window.
  const present = useMemo(() => {
    const seen = new Map<string, { label: string; color: string }>();
    for (const m of marks) seen.set(m.entry.type, m.meta);
    return [...seen.values()];
  }, [marks]);

  return (
    <div
      style={{ display: "flex", alignItems: "center", gap: 12, padding: "0 2px 4px" }}
      aria-label="anomalies in view"
    >
      <div style={{ position: "relative", flex: 1, height: 14 }}>
        {marks.map((m, i) => (
          <button
            key={`${m.entry.ts}-${i}`}
            type="button"
            title={`${m.meta.label} · ${fmtTime(m.entry.ts)}`}
            onClick={() => onSeek?.(m.entry)}
            aria-label={`Jump to ${m.meta.label} at ${fmtTime(m.entry.ts)}`}
            style={{
              position: "absolute",
              left: `${m.pct}%`,
              top: 0,
              width: 3,
              height: 14,
              padding: 0,
              border: "none",
              borderRadius: 2,
              background: m.meta.color,
              cursor: onSeek ? "pointer" : "default",
              transform: "translateX(-50%)",
              opacity: m.entry.cleared ? 0.5 : 1,
            }}
          />
        ))}
      </div>
      {present.length > 0 && (
        <span
          style={{
            display: "inline-flex",
            gap: 12,
            fontSize: 11,
            color: "#8a9895",
            whiteSpace: "nowrap",
          }}
        >
          {present.map((p) => (
            <span key={p.label} style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
              <span
                style={{ width: 8, height: 8, borderRadius: 2, background: p.color }}
                aria-hidden
              />
              {p.label}
            </span>
          ))}
        </span>
      )}
    </div>
  );
}

function fmtTime(ts: string): string {
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? ts : d.toLocaleTimeString();
}
