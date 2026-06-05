// /recordings view (F4.2–F4.4): the list of explicit recordings, playback in a
// modal, and the flag-for-upload control.
//
// - Table sorted by started_at desc (the backend already sorts; we render as-is).
// - Play opens an hls.js modal pointed at /api/recordings/<id>/playlist; the
//   backend decides MinIO-redirect vs supervisor-proxy.
// - Flag is enabled only for not-yet-flagged recordings whose upload_state is
//   "none" or "failed" (a failed upload can be re-flagged). Optimistic UI:
//   mark flagged immediately, revert on error.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AuthError,
  ControlError,
  fetchRecordings,
  flagForUpload,
  recordingPlaylistUrl,
  redirectToLogin,
  type RecordingItem,
} from "./api";
import { config } from "./config";
import { useHls } from "./useHls";

export function Recordings() {
  const [items, setItems] = useState<RecordingItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState<RecordingItem | null>(null);
  const [flagging, setFlagging] = useState<Record<string, boolean>>({});
  const [flagError, setFlagError] = useState<Record<string, string>>({});

  const load = useCallback(() => {
    fetchRecordings()
      .then((r) => {
        setItems(r);
        setError(null);
      })
      .catch((e) => {
        if (e instanceof AuthError) {
          redirectToLogin();
          return;
        }
        setError(String(e));
      });
  }, []);

  useEffect(() => {
    load();
    // Poll on a slow interval so upload_state advances visibly (F4.6 / B4.6).
    const id = window.setInterval(load, config.statePollMs * 3);
    return () => window.clearInterval(id);
  }, [load]);

  const onFlag = useCallback(
    (rec: RecordingItem) => {
      // Optimistic: mark flagged immediately; revert on error.
      setFlagging((p) => ({ ...p, [rec.recording_id]: true }));
      setFlagError((p) => ({ ...p, [rec.recording_id]: "" }));
      setItems((prev) =>
        prev
          ? prev.map((r) =>
              r.recording_id === rec.recording_id
                ? { ...r, flagged_for_upload: true, upload_state: "queued" }
                : r,
            )
          : prev,
      );
      flagForUpload(rec.recording_id)
        .then(() => {
          setFlagging((p) => ({ ...p, [rec.recording_id]: false }));
          load(); // pick up the authoritative state on the next read
        })
        .catch((e) => {
          if (e instanceof AuthError) {
            redirectToLogin();
            return;
          }
          // Revert the optimistic change.
          setItems((prev) =>
            prev
              ? prev.map((r) =>
                  r.recording_id === rec.recording_id
                    ? { ...r, flagged_for_upload: rec.flagged_for_upload, upload_state: rec.upload_state }
                    : r,
                )
              : prev,
          );
          const msg = e instanceof ControlError ? e.message : String(e);
          setFlagging((p) => ({ ...p, [rec.recording_id]: false }));
          setFlagError((p) => ({ ...p, [rec.recording_id]: msg }));
        });
    },
    [load],
  );

  return (
    <div style={{ padding: 16 }}>
      <h1>Fessel — Recordings</h1>
      {error && (
        <div style={{ background: "#f5deb3", padding: "4px 8px", fontSize: 13 }} role="status">
          Could not load recordings — retrying…
        </div>
      )}
      {items === null ? (
        <p style={{ color: "#666" }}>Loading…</p>
      ) : items.length === 0 ? (
        <p style={{ color: "#666" }}>
          No recordings yet. Start one from the dashboard with “Start recording”.
        </p>
      ) : (
        <table style={{ borderCollapse: "collapse", width: "100%", maxWidth: 960 }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid #ddd" }}>
              <Th>Started</Th>
              <Th>Ended</Th>
              <Th>Duration</Th>
              <Th>Operator</Th>
              <Th>Upload</Th>
              <Th>Actions</Th>
            </tr>
          </thead>
          <tbody>
            {items.map((rec) => (
              <Row
                key={rec.recording_id}
                rec={rec}
                onPlay={() => setPlaying(rec)}
                onFlag={() => onFlag(rec)}
                flagging={!!flagging[rec.recording_id]}
                flagError={flagError[rec.recording_id]}
              />
            ))}
          </tbody>
        </table>
      )}

      {playing && <PlayerModal rec={playing} onClose={() => setPlaying(null)} />}
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return <th style={{ padding: "4px 12px 4px 0", color: "#666", fontWeight: 600 }}>{children}</th>;
}

function Td({ children }: { children: React.ReactNode }) {
  return <td style={{ padding: "4px 12px 4px 0", fontFamily: "monospace" }}>{children}</td>;
}

function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function fmtDuration(seconds: number | null): string {
  if (seconds == null) return "—";
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

// A failed upload can be re-flagged to retry (F4.4); "none" can be flagged for
// the first time. Anything else (queued/uploading/uploaded) is not re-flaggable.
function canFlag(rec: RecordingItem): boolean {
  if (rec.flagged_for_upload && rec.upload_state !== "failed") return false;
  return rec.upload_state === "none" || rec.upload_state === "failed";
}

function Row({
  rec,
  onPlay,
  onFlag,
  flagging,
  flagError,
}: {
  rec: RecordingItem;
  onPlay: () => void;
  onFlag: () => void;
  flagging: boolean;
  flagError?: string;
}) {
  const flaggable = canFlag(rec) && !flagging;
  return (
    <tr style={{ borderBottom: "1px solid #f0f0f0" }}>
      <Td>{fmtTime(rec.started_at)}</Td>
      <Td>{fmtTime(rec.ended_at)}</Td>
      <Td>{fmtDuration(rec.duration_seconds)}</Td>
      <Td>{rec.operator ?? "—"}</Td>
      <Td>{rec.upload_state}</Td>
      <td style={{ padding: "4px 0" }}>
        <button onClick={onPlay} style={{ marginRight: 8 }}>
          Play
        </button>
        <button onClick={onFlag} disabled={!flaggable} title={flaggable ? "" : "Already flagged / uploaded"}>
          {flagging ? "…" : rec.upload_state === "failed" ? "Re-flag" : "Flag for upload"}
        </button>
        {flagError ? (
          <span style={{ color: "crimson", fontSize: 12, marginLeft: 8 }}>{flagError}</span>
        ) : null}
      </td>
    </tr>
  );
}

// --- playback modal (F4.3) ---------------------------------------------------

function PlayerModal({ rec, onClose }: { rec: RecordingItem; onClose: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  // A finite recording: live=false so the operator can scrub from the start.
  useHls(videoRef, recordingPlaylistUrl(rec.recording_id), { live: false });

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
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.6)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
      onClick={onClose}
    >
      <div style={{ background: "#fff", padding: 12, borderRadius: 4 }} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
          <strong style={{ fontFamily: "monospace" }}>{rec.recording_id}</strong>
          <button onClick={onClose}>Close</button>
        </div>
        <video
          ref={videoRef}
          controls
          autoPlay
          playsInline
          style={{ width: "80vw", maxWidth: 960, background: "#000" }}
        />
      </div>
    </div>
  );
}
