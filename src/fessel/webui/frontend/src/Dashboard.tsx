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
import type { Capabilities, ModeTriplet, PlugState, StateResponse } from "../../shared/schemas";
import {
  AuthError,
  ControlError,
  fetchCapabilities,
  fetchState,
  modeToCanonical,
  postControl,
  redirectToLogin,
  startRecording,
  stopRecording,
  type ControlAction,
} from "./api";
import { config } from "./config";

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
        })
        .catch((e) => {
          if (cancelled) return;
          // 401 -> session lapsed; hand off to the proxy login (do NOT loop).
          if (e instanceof AuthError) {
            redirectToLogin();
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
            redirectToLogin();
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
        if (e instanceof AuthError) redirectToLogin();
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
          redirectToLogin();
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
          redirectToLogin();
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
