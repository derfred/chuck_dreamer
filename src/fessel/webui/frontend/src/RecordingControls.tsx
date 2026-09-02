// Recording controls (F4.5 + ring-buffer look-back reframe), rendered in the
// Monitor video panel's header bar — recording is an action *on the scene*,
// so it lives with the video, not down in the side rail's control soup.
//
// Start/Stop derived from /api/state's `recording` field. Idle -> "Record…"
// opens the look-back dialog (RecordModal: pick how far to reach back into the
// on-Pi ring + upload-when-done; NO resolution picker — that's a deploy
// setting). recording -> "Stop" (red, with a pulsing dot). starting/stopping ->
// disabled.
//
// The header bar is narrow, so the states are expressed as one compact chip
// pair (button + status pill) that mirrors the Stream On/Off toggle opposite
// it; the upload backlog rides along only when there IS a backlog.

import { useCallback, useEffect, useState } from "react";
import type { Capabilities, StateResponse } from "../../shared/schemas";
import {
  AuthError,
  ControlError,
  fetchCapabilities,
  reauthenticate,
  startRecording,
  stopRecording,
} from "./api";
import { RecordModal } from "./RecordModal";
import { useAnomalies } from "./useAnomalies";

// Backlog crosses into alert styling at the architecture's "oldest pending
// > 3h" P2 threshold; below that it is quiet informational text.
const BACKLOG_ALERT_SECONDS = 3 * 3600;

export function RecordingControls({
  recording,
  backlog,
}: {
  recording: StateResponse["recording"] | undefined;
  backlog: StateResponse["upload_backlog"] | undefined;
}) {
  const [maxLookback, setMaxLookback] = useState<number>(0);
  const [modalOpen, setModalOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const anomalies = useAnomalies();

  useEffect(() => {
    fetchCapabilities()
      .then((c: Capabilities) => setMaxLookback(c.max_lookback_seconds))
      .catch((e) => {
        if (e instanceof AuthError) reauthenticate();
      });
  }, []);

  const state = recording?.state ?? "idle";

  const onConfirm = useCallback((opts: { lookbackSeconds: number; uploadWhenDone: boolean }) => {
    setBusy(true);
    setError(null);
    startRecording(opts)
      .then(() => {
        setBusy(false);
        setModalOpen(false);
      })
      .catch((e) => {
        if (e instanceof AuthError) {
          reauthenticate();
          return;
        }
        setBusy(false);
        setError(e instanceof ControlError ? e.message : String(e));
      });
  }, []);

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
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginLeft: "auto" }}>
      {error && (
        <span style={{ color: "#c0392b", fontSize: 12, maxWidth: 220 }} role="alert">
          {error}
        </span>
      )}
      <BacklogChip backlog={backlog} />
      {state === "recording" && <RecordingStatusChip recording={recording} />}
      {state === "idle" ? (
        <button
          type="button"
          onClick={() => setModalOpen(true)}
          disabled={maxLookback <= 0}
          style={{
            ...chipButton,
            border: "1px solid #dde3e6",
            background: "#fff",
            color: maxLookback <= 0 ? "#a8b5b2" : "#16211f",
            cursor: maxLookback <= 0 ? "default" : "pointer",
          }}
        >
          <Dot color="#c0392b" />
          Record…
        </button>
      ) : state === "recording" ? (
        <button
          type="button"
          onClick={onStop}
          disabled={transitioning}
          style={{ ...chipButton, border: "none", background: "#c0392b", color: "#fff" }}
        >
          {busy ? "…" : "Stop recording"}
        </button>
      ) : (
        <button type="button" disabled style={{ ...chipButton, ...pendingChip }}>
          {state === "starting" ? "Starting…" : "Stopping…"}
        </button>
      )}

      {modalOpen && (
        <RecordModal
          maxLookbackSeconds={maxLookback}
          anomalies={anomalies}
          busy={busy}
          onCancel={() => setModalOpen(false)}
          onConfirm={onConfirm}
        />
      )}
    </div>
  );
}

const chipButton: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 7,
  font: "inherit",
  fontSize: 13,
  fontWeight: 600,
  padding: "6px 13px",
  borderRadius: 999,
  cursor: "pointer",
};

const pendingChip: React.CSSProperties = {
  border: "1px solid #dde3e6",
  background: "#eef1f3",
  color: "#5a6b68",
  cursor: "default",
};

// The live "REC" pill: the same pill geometry as the stream cost tag it
// replaced, in the recording red, so an active recording is legible from
// across the room without reading the word.
function RecordingStatusChip({ recording }: { recording: StateResponse["recording"] | undefined }) {
  return (
    <span
      title={recording?.active_recording_id ? `Recording ${recording.active_recording_id}` : undefined}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        fontSize: 12,
        fontWeight: 700,
        letterSpacing: "0.04em",
        padding: "3px 9px",
        borderRadius: 999,
        color: "#c0392b",
        background: "#fdecea",
      }}
    >
      <Dot color="#c0392b" pulse />
      REC
    </span>
  );
}

function Dot({ color, pulse = false }: { color: string; pulse?: boolean }) {
  return (
    <span
      aria-hidden
      style={{
        width: 7,
        height: 7,
        borderRadius: "50%",
        background: color,
        animation: pulse ? "fessel-rec-pulse 1.4s ease-in-out infinite" : undefined,
      }}
    />
  );
}

// Backlog rides in the header only when there IS one — a "0 pending" chip is
// noise in a bar that is otherwise about the live scene.
function BacklogChip({ backlog }: { backlog: StateResponse["upload_backlog"] | undefined }) {
  const count = backlog?.count ?? 0;
  const oldest = backlog?.oldest_pending_seconds ?? null;
  if (count <= 0) return null;
  const alert = oldest != null && oldest > BACKLOG_ALERT_SECONDS;
  return (
    <span
      role={alert ? "alert" : undefined}
      title={`Upload backlog: ${count} pending${oldest != null ? `, oldest ${fmtAge(oldest)}` : ""}`}
      style={{
        fontSize: 12,
        fontWeight: 600,
        padding: "3px 9px",
        borderRadius: 999,
        color: alert ? "#fff" : "#5a6b68",
        background: alert ? "#c0392b" : "#eef1f3",
      }}
    >
      ↑ {count} pending
      {oldest != null ? `, oldest ${fmtAge(oldest)}` : ""}
    </span>
  );
}

function fmtAge(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  return `${Math.floor(seconds / 3600)}h`;
}
