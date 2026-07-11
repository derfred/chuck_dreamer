// Corner Pi-connection health light + drill-down (health-check spec §4).
//
// A small always-visible status dot in the app chrome (rendered by App.tsx on
// every route). It polls /api/health/pi, renders the rollup `light`, and opens
// a drill-down popover listing the per-fact breakdown root-cause-first. The
// light is presentation; all the substance is in the facts the backend computes
// (reachability, liveness, version sync) and their masking + rollup.
//
// Two staleness sources collapse to grey/unknown:
//   - the backend's rollup itself can be `unknown` (before first probe, or a
//     masked-everything picture);
//   - the FRONTEND overrides to grey if its own last SUCCESSFUL fetch is older
//     than config.healthStaleMs (§4.4) — a green light the browser can't
//     refresh is not a green light.

import { useEffect, useRef, useState } from "react";
import {
  AuthError,
  fetchPiHealth,
  reauthenticate,
  runConnectionCheck,
  type ConnectionCheckResult,
  type HealthFact,
  type HealthState,
  type PiHealth,
} from "./api";
import { config } from "./config";

const DOT_COLOR: Record<HealthState, string> = {
  green: "#2a7",
  yellow: "#cc9a00",
  red: "#b22222",
  unknown: "#999",
};

// Terse label beside the dot (spec §4.1). Kept short; the detail lives in the
// drill-down.
const LIGHT_LABEL: Record<HealthState, string> = {
  green: "Pi: OK",
  yellow: "Pi: degraded",
  red: "Pi: problem",
  unknown: "Pi: checking…",
};

function ageStr(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h`;
}

export function PiHealthIndicator() {
  const [health, setHealth] = useState<PiHealth | null>(null);
  // Timestamp (ms) of the last SUCCESSFUL fetch — drives the §4.4 staleness
  // override. null until the first success.
  const [lastSuccess, setLastSuccess] = useState<number | null>(null);
  const [open, setOpen] = useState(false);
  // A ticking "now" so the "stale — Xs ago" label and the staleness threshold
  // re-evaluate even when no new fetch has landed (a wedged backend).
  const [now, setNow] = useState<number>(() => Date.now());

  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      fetchPiHealth()
        .then((h) => {
          if (cancelled) return;
          setHealth(h);
          setLastSuccess(Date.now());
        })
        .catch((e) => {
          if (cancelled) return;
          if (e instanceof AuthError) {
            reauthenticate();
            return;
          }
          // Network / 5xx: do NOT blank the light — keep the last known value
          // and let the staleness clock (below) take over (§4.2).
        });
    };
    poll();
    const id = window.setInterval(poll, config.healthPollMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  // Tick "now" on a 1s cadence so the staleness override and age label update
  // between polls.
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  // Frontend staleness override (§4.4): if we've never succeeded, or the last
  // success is older than the threshold, force grey/unknown regardless of the
  // last-known light.
  const staleAge = lastSuccess == null ? Infinity : now - lastSuccess;
  const stale = staleAge > config.healthStaleMs;
  const light: HealthState = stale ? "unknown" : (health?.light ?? "unknown");
  const label = stale
    ? lastSuccess == null
      ? "Pi: checking…"
      : `Pi: stale — ${ageStr(staleAge)} ago`
    : LIGHT_LABEL[light];

  const ref = useRef<HTMLDivElement>(null);
  // Close the drill-down on an outside click or Escape.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-label={label}
        aria-expanded={open}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          background: "none",
          border: "none",
          cursor: "pointer",
          padding: "2px 4px",
          font: "inherit",
          color: "#444",
          fontSize: 13,
        }}
      >
        <span
          aria-hidden
          style={{
            width: 10,
            height: 10,
            borderRadius: 5,
            background: DOT_COLOR[light],
            display: "inline-block",
          }}
        />
        {label}
      </button>
      {open && <DrillDown health={health} stale={stale} staleAge={staleAge} />}
    </div>
  );
}

function DrillDown({
  health,
  stale,
  staleAge,
}: {
  health: PiHealth | null;
  stale: boolean;
  staleAge: number;
}) {
  return (
    <div
      role="dialog"
      aria-label="Pi connection health"
      style={{
        position: "absolute",
        top: "calc(100% + 6px)",
        right: 0,
        minWidth: 280,
        background: "#fff",
        border: "1px solid #ddd",
        borderRadius: 6,
        boxShadow: "0 4px 16px rgba(0,0,0,0.12)",
        padding: 12,
        zIndex: 1000,
      }}
    >
      {stale && (
        <div
          role="status"
          style={{ background: "#f5deb3", padding: "4px 8px", fontSize: 12, marginBottom: 8 }}
        >
          Health data is stale — last updated {ageStr(staleAge)} ago. Retrying…
        </div>
      )}
      {health == null ? (
        <p style={{ color: "#666", fontSize: 13, margin: 0 }}>Checking…</p>
      ) : (
        <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
          {health.facts.map((f) => (
            <FactRow key={f.id} fact={f} />
          ))}
        </ul>
      )}
      <ConnectionCheckPanel />
    </div>
  );
}

// On-demand latency + bandwidth probe to the Pi (cluster-side measurement,
// distinct from the passively-polled health facts above). Idle until the
// operator clicks it — the probe actively drives traffic and takes
// noticeably longer than a status read, so it must never auto-run or poll.
function ConnectionCheckPanel() {
  const [state, setState] = useState<
    { status: "idle" } | { status: "running" } | { status: "done"; result: ConnectionCheckResult } | { status: "error"; message: string }
  >({ status: "idle" });

  const run = () => {
    setState({ status: "running" });
    runConnectionCheck()
      .then((result) => setState({ status: "done", result }))
      .catch((e) => {
        if (e instanceof AuthError) {
          reauthenticate();
          return;
        }
        setState({ status: "error", message: e instanceof Error ? e.message : String(e) });
      });
  };

  return (
    <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid #eee" }}>
      <button
        type="button"
        onClick={run}
        disabled={state.status === "running"}
        style={{
          width: "100%",
          padding: "6px 10px",
          fontSize: 12,
          fontWeight: 600,
          border: "1px solid #ccc",
          borderRadius: 6,
          background: "#fafafa",
          cursor: state.status === "running" ? "default" : "pointer",
        }}
      >
        {state.status === "running" ? "Checking connection…" : "Check connection to Pi"}
      </button>
      {state.status === "done" && (
        <p style={{ fontSize: 12, color: "#444", margin: "6px 0 0" }}>
          Latency {state.result.latency_ms.toFixed(0)} ms · Throughput{" "}
          {state.result.throughput_mbps.toFixed(1)} Mbps
        </p>
      )}
      {state.status === "error" && (
        <p style={{ fontSize: 12, color: "#b22222", margin: "6px 0 0" }}>{state.message}</p>
      )}
    </div>
  );
}

function FactRow({ fact }: { fact: HealthFact }) {
  // Masked (unknown) facts are de-emphasised so the eye lands on the red root
  // cause first (§4.3).
  const masked = fact.state === "unknown";
  return (
    <li
      style={{
        display: "flex",
        alignItems: "baseline",
        gap: 8,
        padding: "5px 0",
        borderBottom: "1px solid #f2f2f2",
        opacity: masked ? 0.55 : 1,
      }}
    >
      <span
        aria-hidden
        style={{
          width: 8,
          height: 8,
          borderRadius: 4,
          marginTop: 4,
          background: DOT_COLOR[fact.state],
          flex: "0 0 auto",
        }}
      />
      <span style={{ fontSize: 13, fontWeight: 600, minWidth: 96 }}>{fact.label}</span>
      <span style={{ fontSize: 12, color: "#666" }}>{fact.detail}</span>
    </li>
  );
}
