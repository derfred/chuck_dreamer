// Operator dashboard (F3.1–F3.4). Three regions: a read-only status panel,
// the control buttons, and a link to /live. The dashboard polls /api/state on
// a short interval (state can change from outside the UI — direct supervisor
// calls, or the Slice 6 safety machine), surfaces a transient "state
// unavailable" banner on a 5xx WITHOUT clearing the last-known state, and
// escalates to re-auth on a 401.
//
// Destructive actions (Stop, Power off arm/Jetson) require a confirmation
// modal whose default focus is Cancel and which is Escape-dismissable — the
// failure mode guarded against is "clicked Stop by accident, hit Enter".

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AnomalyLogEntry,
  Capabilities,
  ModeTriplet,
  PlugState,
  StateResponse,
} from "../../shared/schemas";
import {
  AuthError,
  ControlError,
  fetchAnomalies,
  fetchCapabilities,
  fetchState,
  modeToCanonical,
  postControl,
  reauthenticate,
  startRecording,
  stopRecording,
  type ControlAction,
} from "./api";
import { config } from "./config";
import { Sparkline } from "./Sparkline";

type ButtonStatus =
  | { kind: "idle" }
  | { kind: "inflight" }
  | { kind: "error"; message: string }
  | { kind: "success" };

interface ActionDef {
  action: ControlAction;
  label: string;
  destructive: boolean;
  // Modal copy for destructive actions: names the specific action + target and
  // makes the operator responsible for the resulting safe state (F3.3).
  confirm?: string;
}

const ACTIONS: ActionDef[] = [
  { action: "pause", label: "Pause", destructive: false },
  { action: "resume", label: "Resume", destructive: false },
  {
    action: "stop",
    label: "Stop",
    destructive: true,
    confirm:
      "Stop the arm? The current episode ends and the arm returns home and idles. " +
      "You are responsible for confirming the arm is in a safe state afterwards.",
  },
  {
    action: "shutdown/arm",
    label: "Power off arm",
    destructive: true,
    confirm:
      "Cut power to the arm? The arm will lose power immediately and the operator " +
      "who initiated this is responsible for confirming the arm is in a safe state.",
  },
  {
    action: "shutdown/jetson",
    label: "Power off Jetson",
    destructive: true,
    confirm:
      "Cut power to the Jetson? Control and learning software stop immediately. " +
      "You are responsible for confirming this is safe.",
  },
  { action: "poweron/arm", label: "Power on arm", destructive: false },
  { action: "poweron/jetson", label: "Power on Jetson", destructive: false },
];

export function Dashboard() {
  const [state, setState] = useState<StateResponse | null>(null);
  const [stateError, setStateError] = useState<boolean>(false);
  const [statuses, setStatuses] = useState<Record<string, ButtonStatus>>({});
  const [pending, setPending] = useState<ActionDef | null>(null);
  // A short client-side history of the activity score (F5.1 sparkline). Capped
  // at ~60 samples (= ~60s at the 1Hz-ish state poll); a ring, oldest dropped.
  const [activityHistory, setActivityHistory] = useState<number[]>([]);

  const setStatus = useCallback((action: string, s: ButtonStatus) => {
    setStatuses((prev) => ({ ...prev, [action]: s }));
  }, []);

  // --- state polling (F3.4) --------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      fetchState()
        .then((s) => {
          if (cancelled) return;
          setState(s);
          setStateError(false);
          // Append the latest activity score to the sparkline history (F5.1/F5.4).
          const score = s.vision?.activity_score_ema;
          if (typeof score === "number") {
            setActivityHistory((h) => [...h, score].slice(-60));
          }
        })
        .catch((e) => {
          if (cancelled) return;
          // 401 -> session lapsed; hand off to the proxy login (do NOT loop).
          if (e instanceof AuthError) {
            reauthenticate();
            return;
          }
          // 5xx/network -> transient banner; keep the last-known state on screen.
          setStateError(true);
        });
    };
    poll();
    const id = window.setInterval(poll, config.statePollMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  // --- issuing an action (F3.2) ---------------------------------------------
  const run = useCallback(
    (def: ActionDef) => {
      setStatus(def.action, { kind: "inflight" });
      postControl(def.action)
        .then(() => {
          setStatus(def.action, { kind: "success" });
          // Brief success indicator, then back to idle.
          window.setTimeout(() => setStatus(def.action, { kind: "idle" }), 1500);
        })
        .catch((e) => {
          if (e instanceof AuthError) {
            reauthenticate();
            return;
          }
          const message = e instanceof ControlError ? e.message : String(e);
          setStatus(def.action, { kind: "error", message });
        });
    },
    [setStatus],
  );

  const onClick = useCallback(
    (def: ActionDef) => {
      // Clear a prior inline error on the next click (F3.2).
      setStatus(def.action, { kind: "idle" });
      if (def.destructive) setPending(def);
      else run(def);
    },
    [run, setStatus],
  );

  const confirm = useCallback(() => {
    if (pending) {
      const def = pending;
      setPending(null);
      run(def);
    }
  }, [pending, run]);

  return (
    <div style={{ padding: 16 }}>
      <h1>Fessel</h1>

      <StatusPanel state={state} error={stateError} />

      <SensingStrip
        vision={state?.vision}
        audio={state?.audio}
        history={activityHistory}
      />

      <RecentAnomaliesPanel />

      <h2 style={{ fontSize: 16, marginTop: 24 }}>Controls</h2>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
        {ACTIONS.map((def) => (
          <ControlButton
            key={def.action}
            def={def}
            status={statuses[def.action] ?? { kind: "idle" }}
            onClick={() => onClick(def)}
          />
        ))}
      </div>

      <h2 style={{ fontSize: 16, marginTop: 24 }}>Recording</h2>
      <RecordingControls recording={state?.recording} />
      <BacklogIndicator backlog={state?.upload_backlog} />

      <p style={{ marginTop: 24, display: "flex", gap: 16 }}>
        <a href="/live" onClick={(e) => (e.preventDefault(), navigate("/live"))}>
          Go to live view →
        </a>
        <a href="/ring" onClick={(e) => (e.preventDefault(), navigate("/ring"))}>
          View ring buffer →
        </a>
        <a href="/recordings" onClick={(e) => (e.preventDefault(), navigate("/recordings"))}>
          Browse recordings →
        </a>
      </p>

      {pending && (
        <ConfirmModal def={pending} onCancel={() => setPending(null)} onConfirm={confirm} />
      )}
    </div>
  );
}

// --- recording controls (F4.5) ----------------------------------------------
// Start/Stop derived from /api/state's `recording` field. Idle -> pick a mode +
// "Start recording"; recording -> "Stop recording" (red, no confirmation modal
// — stopping a recording is not destructive in the Slice-3 modal sense).
// starting/stopping -> a disabled spinner.

function RecordingControls({ recording }: { recording: StateResponse["recording"] | undefined }) {
  const [modes, setModes] = useState<ModeTriplet[]>([]);
  const [selected, setSelected] = useState<number>(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchCapabilities()
      .then((c: Capabilities) => setModes(c.modes))
      .catch((e) => {
        if (e instanceof AuthError) reauthenticate();
      });
  }, []);

  const state = recording?.state ?? "idle";

  const onStart = useCallback(() => {
    const mode = modes[selected];
    if (!mode) return;
    setBusy(true);
    setError(null);
    startRecording(mode)
      .then(() => setBusy(false))
      .catch((e) => {
        if (e instanceof AuthError) {
          reauthenticate();
          return;
        }
        setBusy(false);
        setError(e instanceof ControlError ? e.message : String(e));
      });
  }, [modes, selected]);

  const onStop = useCallback(() => {
    setBusy(true);
    setError(null);
    stopRecording()
      .then(() => setBusy(false))
      .catch((e) => {
        if (e instanceof AuthError) {
          reauthenticate();
          return;
        }
        setBusy(false);
        setError(e instanceof ControlError ? e.message : String(e));
      });
  }, []);

  const transitioning = state === "starting" || state === "stopping" || busy;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
      {state === "idle" ? (
        <>
          <label>
            Mode:{" "}
            <select
              value={selected}
              onChange={(e) => setSelected(Number(e.target.value))}
              disabled={transitioning || modes.length === 0}
            >
              {modes.map((m, i) => (
                <option key={i} value={i}>
                  {modeToCanonical(m)}
                </option>
              ))}
            </select>
          </label>
          <button onClick={onStart} disabled={transitioning || modes.length === 0}>
            {busy ? "…" : "Start recording"}
          </button>
        </>
      ) : state === "recording" ? (
        <button
          onClick={onStop}
          disabled={transitioning}
          style={{ background: "#b22222", color: "#fff", padding: "6px 12px" }}
        >
          {busy ? "…" : "Stop recording"}
        </button>
      ) : (
        <button disabled style={{ padding: "6px 12px" }}>
          {state === "starting" ? "Starting…" : "Stopping…"}
        </button>
      )}
      <span style={{ color: "#666", fontFamily: "monospace", fontSize: 13 }}>
        state: {state}
        {recording?.active_recording_id ? ` (${recording.active_recording_id})` : ""}
      </span>
      {error && <span style={{ color: "crimson", fontSize: 12 }}>{error}</span>}
    </div>
  );
}

// --- upload backlog indicator (F4.5) ----------------------------------------
// Plain text count + oldest-pending age. Becomes alert-styled once it crosses
// the Slice-7 thresholds (the architecture's "oldest pending > 3h" P2); for now
// it is informational.

const BACKLOG_ALERT_SECONDS = 3 * 3600;

function BacklogIndicator({ backlog }: { backlog: StateResponse["upload_backlog"] | undefined }) {
  const count = backlog?.count ?? 0;
  const oldest = backlog?.oldest_pending_seconds ?? null;
  const alert = oldest != null && oldest > BACKLOG_ALERT_SECONDS;
  return (
    <p
      style={{
        marginTop: 8,
        fontSize: 13,
        color: alert ? "#fff" : "#666",
        background: alert ? "#b22222" : "transparent",
        display: "inline-block",
        padding: alert ? "2px 8px" : 0,
      }}
      role={alert ? "alert" : undefined}
    >
      Upload backlog: {count} pending
      {oldest != null ? `, oldest ${fmtAge(oldest)}` : ""}
    </p>
  );
}

function fmtAge(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  return `${Math.floor(seconds / 3600)}h`;
}

// SPA navigation mirror of App.tsx's navigate (kept local to avoid a shared
// router dependency for a two-route app).
function navigate(to: string): void {
  window.history.pushState({}, "", to);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

// --- sensing strip (F5.1) ----------------------------------------------------
// Live activity score (sparkline of the last ~60s) + audio level meter +
// vision/audio health pills. A quick-glance health strip, NOT the live video
// feed (/live). Reads from the same /api/state poll as the rest of the page.

function SensingStrip({
  vision,
  audio,
  history,
}: {
  vision: StateResponse["vision"] | undefined;
  audio: StateResponse["audio"] | undefined;
  history: number[];
}) {
  const ema = vision?.activity_score_ema ?? 0;
  const levelDb = audio?.level_db ?? null;
  return (
    <section style={{ marginTop: 16 }}>
      <h2 style={{ fontSize: 16 }}>Sensing</h2>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 24, alignItems: "center" }}>
        <div>
          <div style={{ color: "#666", fontSize: 13 }}>
            Activity{" "}
            <span style={{ fontFamily: "monospace" }}>{ema.toFixed(3)}</span>
          </div>
          <Sparkline values={history} ariaLabel="activity score history" max={1} />
        </div>
        <div>
          <div style={{ color: "#666", fontSize: 13 }}>Audio level</div>
          <AudioMeter levelDb={levelDb} />
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <HealthPill label="Vision" healthy={vision?.healthy ?? false} />
          <HealthPill label="Audio" healthy={audio?.healthy ?? false} />
        </div>
      </div>
    </section>
  );
}

// Map a dBFS level (~[-60, 0]) to a 0..1 bar fill. Quiet is empty, loud full.
function AudioMeter({ levelDb }: { levelDb: number | null }) {
  if (levelDb == null) {
    return <span style={{ color: "#999", fontSize: 13 }}>—</span>;
  }
  const frac = Math.min(1, Math.max(0, (levelDb + 60) / 60));
  return (
    <div
      role="meter"
      aria-valuenow={Math.round(levelDb)}
      aria-valuemin={-60}
      aria-valuemax={0}
      style={{ width: 160, height: 14, background: "#eee", borderRadius: 3, overflow: "hidden" }}
    >
      <div
        style={{
          width: `${frac * 100}%`,
          height: "100%",
          background: frac > 0.85 ? "#b22222" : "#2a7",
        }}
      />
      <span style={{ fontFamily: "monospace", fontSize: 11, color: "#666" }}>
        {levelDb.toFixed(1)} dB
      </span>
    </div>
  );
}

function HealthPill({ label, healthy }: { label: string; healthy: boolean }) {
  return (
    <span
      style={{
        fontSize: 12,
        padding: "2px 8px",
        borderRadius: 10,
        color: "#fff",
        background: healthy ? "#2a7" : "#b22222",
      }}
      title={`${label} ${healthy ? "healthy" : "degraded (heartbeat stale)"}`}
    >
      {label} {healthy ? "●" : "○"}
    </span>
  );
}

// --- recent anomalies panel (F5.2) -------------------------------------------
// A short list of the last few anomalies from /api/anomalies. Each row links to
// the anomaly recording's playback (the Slice-4 /recordings path) when one
// exists. Auto-refreshes on the same cadence as the dashboard state poll.

const ANOMALY_LABELS: Record<string, { label: string; color: string }> = {
  safe_box_violation: { label: "Safe-box violation", color: "#b22222" },
  rest_violation: { label: "Rest-period motion", color: "#cc7000" },
  audio_spike: { label: "Audio spike", color: "#7a3" },
};

function RecentAnomaliesPanel() {
  const [anomalies, setAnomalies] = useState<AnomalyLogEntry[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      fetchAnomalies()
        .then((a) => {
          if (!cancelled) setAnomalies(a);
        })
        .catch((e) => {
          if (e instanceof AuthError) reauthenticate();
          // Other errors: keep the last-known list silently (informational panel).
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
    <section style={{ marginTop: 16 }}>
      <h2 style={{ fontSize: 16 }}>Recent anomalies</h2>
      {anomalies == null ? (
        <p style={{ color: "#666" }}>Loading…</p>
      ) : anomalies.length === 0 ? (
        <p style={{ color: "#666", fontSize: 13 }}>No anomalies recorded.</p>
      ) : (
        <ul style={{ listStyle: "none", padding: 0, margin: 0, maxWidth: 640 }}>
          {anomalies.slice(0, 8).map((a, i) => (
            <AnomalyRow key={`${a.ts}-${i}`} entry={a} />
          ))}
        </ul>
      )}
    </section>
  );
}

function AnomalyRow({ entry }: { entry: AnomalyLogEntry }) {
  const meta = ANOMALY_LABELS[entry.type] ?? { label: entry.type, color: "#666" };
  const rid = entry.anomaly_recording_id ?? null;
  return (
    <li
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "4px 0",
        borderBottom: "1px solid #f0f0f0",
        fontSize: 13,
      }}
    >
      <span style={{ width: 8, height: 8, borderRadius: 4, background: meta.color }} />
      <span style={{ color: "#666", fontFamily: "monospace" }}>{fmtTimeShort(entry.ts)}</span>
      <span>{meta.label}</span>
      {entry.cleared ? <span style={{ color: "#999" }}>(cleared)</span> : null}
      {rid ? (
        <a
          href="/recordings"
          onClick={(e) => (e.preventDefault(), navigate("/recordings"))}
          style={{ marginLeft: "auto" }}
        >
          View recording →
        </a>
      ) : null}
    </li>
  );
}

function fmtTimeShort(ts: string | null | undefined): string {
  if (!ts) return "—";
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts;
  return d.toLocaleTimeString();
}

// --- status panel (read-only facts, F3.1) -----------------------------------

function StatusPanel({ state, error }: { state: StateResponse | null; error: boolean }) {
  return (
    <section>
      <h2 style={{ fontSize: 16 }}>Status</h2>
      {error && (
        <div
          style={{ background: "#f5deb3", padding: "4px 8px", fontSize: 13, marginBottom: 8 }}
          role="status"
        >
          State unavailable — showing last known. Retrying…
        </div>
      )}
      {state === null ? (
        <p style={{ color: "#666" }}>Loading…</p>
      ) : (
        <table style={{ borderCollapse: "collapse" }}>
          <tbody>
            <Row label="Safety state" value={state.safety_state ?? "—"} />
            <Row label="Jetson" value={jetsonSummary(state.jetson)} />
            <Row label="Arm plug" value={plugSummary(state.plugs?.arm)} />
            <Row label="Jetson plug" value={plugSummary(state.plugs?.jetson)} />
            <Row label="Camera" value={cameraSummary(state.camera?.up)} />
          </tbody>
        </table>
      )}
    </section>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <tr>
      <td style={{ padding: "2px 12px 2px 0", color: "#666" }}>{label}</td>
      <td style={{ padding: "2px 0", fontFamily: "monospace" }}>{value}</td>
    </tr>
  );
}

function jetsonSummary(jetson: StateResponse["jetson"]): string {
  if (jetson == null) return "unknown";
  const s = (jetson as { state?: unknown }).state;
  return typeof s === "string" ? s : "reported";
}

function plugSummary(plug: PlugState | undefined): string {
  if (!plug) return "unknown";
  const power = plug.on === true ? "on" : plug.on === false ? "off" : "unknown";
  // An unverified read is the load-bearing fact (S3.1): show it, never hide it.
  return plug.verified ? power : `${power} (unverified)`;
}

function cameraSummary(up: boolean | null | undefined): string {
  return up === true ? "up" : up === false ? "down" : "unknown";
}

// --- control button (F3.2) ---------------------------------------------------

function ControlButton({
  def,
  status,
  onClick,
}: {
  def: ActionDef;
  status: ButtonStatus;
  onClick: () => void;
}) {
  const inflight = status.kind === "inflight";
  return (
    <div style={{ display: "flex", flexDirection: "column", minWidth: 140 }}>
      <button onClick={onClick} disabled={inflight} style={{ padding: "8px 12px" }}>
        {inflight ? "…" : def.label}
        {status.kind === "success" ? " ✓" : ""}
      </button>
      {status.kind === "error" && (
        <span style={{ color: "crimson", fontSize: 12, marginTop: 4 }}>{status.message}</span>
      )}
    </div>
  );
}

// --- confirmation modal (F3.3) ----------------------------------------------

function ConfirmModal({
  def,
  onCancel,
  onConfirm,
}: {
  def: ActionDef;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const cancelRef = useRef<HTMLButtonElement>(null);

  // Default focus is Cancel, not Confirm (guards "clicked by accident, hit
  // Enter"). Escape dismisses the modal.
  useEffect(() => {
    cancelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.5)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
      onClick={onCancel}
    >
      <div
        style={{ background: "#fff", padding: 20, maxWidth: 420, borderRadius: 4 }}
        onClick={(e) => e.stopPropagation()}
      >
        <p style={{ marginTop: 0 }}>{def.confirm}</p>
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button ref={cancelRef} onClick={onCancel}>
            Cancel
          </button>
          <button onClick={onConfirm} style={{ background: "#b22222", color: "#fff" }}>
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
}
