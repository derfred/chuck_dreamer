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
  deleteRecording,
  fetchRecordings,
  flagForUpload,
  recordingPlaylistUrl,
  reauthenticate,
  type RecordingItem,
} from "./api";
import { config } from "./config";
import { useHls } from "./useHls";

// `embedded` drops the standalone "Fessel — Recordings" heading and outer
// padding so the table can sit inside the Footage page under its own heading.
// Standalone (default) preserves the original chrome for the direct-render tests.
export function Recordings({ embedded = false }: { embedded?: boolean } = {}) {
  const [items, setItems] = useState<RecordingItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState<RecordingItem | null>(null);
  const [flagging, setFlagging] = useState<Record<string, boolean>>({});
  const [flagError, setFlagError] = useState<Record<string, string>>({});
  const [deleting, setDeleting] = useState<Record<string, boolean>>({});
  const [confirmDelete, setConfirmDelete] = useState<RecordingItem | null>(null);

  const load = useCallback(() => {
    fetchRecordings()
      .then((r) => {
        setItems(r);
        setError(null);
      })
      .catch((e) => {
        if (e instanceof AuthError) {
          reauthenticate();
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
            reauthenticate();
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

  // Destructive: NOT optimistic. Only after the backend confirms do we reload
  // the authoritative list — showing a row as gone before the delete lands
  // would be a lie about data that no longer exists anywhere in the cluster.
  const onDelete = useCallback(
    (rec: RecordingItem) => {
      const id = rec.recording_id;
      setDeleting((p) => ({ ...p, [id]: true }));
      setFlagError((p) => ({ ...p, [id]: "" }));
      deleteRecording(id)
        .then(() => {
          setConfirmDelete(null);
          setDeleting((p) => ({ ...p, [id]: false }));
          load();
        })
        .catch((e) => {
          if (e instanceof AuthError) {
            reauthenticate();
            return;
          }
          const msg = e instanceof ControlError ? e.message : String(e);
          setDeleting((p) => ({ ...p, [id]: false }));
          setConfirmDelete(null);
          setFlagError((p) => ({ ...p, [id]: msg }));
        });
    },
    [load],
  );

  return (
    <div style={{ padding: embedded ? 0 : 16 }}>
      {!embedded && <h1>Fessel — Recordings</h1>}
      {error && (
        <div style={{ background: "#f5deb3", padding: "4px 8px", fontSize: 13 }} role="status">
          Could not load recordings — retrying…
        </div>
      )}
      {items === null ? (
        <p style={{ color: "#666" }}>Loading…</p>
      ) : items.length === 0 ? (
        <p style={{ color: "#666" }}>
          No recordings yet. Start one from Monitor with “Start recording”.
        </p>
      ) : (
        <table style={{ borderCollapse: "collapse", width: "100%", maxWidth: 960 }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid #ddd" }}>
              <Th>Type</Th>
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
                onDelete={() => setConfirmDelete(rec)}
                flagging={!!flagging[rec.recording_id]}
                deleting={!!deleting[rec.recording_id]}
                flagError={flagError[rec.recording_id]}
              />
            ))}
          </tbody>
        </table>
      )}

      {playing && <PlayerModal rec={playing} onClose={() => setPlaying(null)} />}
      {confirmDelete && (
        <ConfirmDeleteModal
          rec={confirmDelete}
          busy={!!deleting[confirmDelete.recording_id]}
          onCancel={() => setConfirmDelete(null)}
          onConfirm={() => onDelete(confirmDelete)}
        />
      )}
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

// Discriminate explicit (operator) vs anomaly (auto-triggered) recordings (F5.3).
function TypeBadge({ type }: { type: "explicit" | "anomaly" }) {
  const anomaly = type === "anomaly";
  return (
    <span
      style={{
        fontSize: 11,
        padding: "1px 6px",
        borderRadius: 3,
        color: anomaly ? "#fff" : "#333",
        background: anomaly ? "#b22222" : "#e8e8e8",
      }}
    >
      {anomaly ? "anomaly" : "explicit"}
    </span>
  );
}

// A failed upload can be re-flagged to retry (F4.4); "none" can be flagged for
// the first time. Anything else (queued/uploading/uploaded) is not re-flaggable.
function canFlag(rec: RecordingItem): boolean {
  if (rec.flagged_for_upload && rec.upload_state !== "failed") return false;
  return rec.upload_state === "none" || rec.upload_state === "failed";
}

// Why Flag is disabled. `upload_state` already distinguishes the cases, so the
// tooltip names the actual one rather than the old catch-all "already flagged /
// uploaded" (which read as wrong for a queued upload that had not happened yet).
function flagDisabledReason(rec: RecordingItem): string {
  switch (rec.upload_state) {
    case "uploaded":
      return "Already uploaded";
    case "queued":
      return "Already flagged — waiting for the uploader";
    case "uploading":
      return "Upload in progress";
    default:
      return "Already flagged for upload";
  }
}

// Playability is an AVAILABILITY question, not an upload-state one. The backend
// serves playback from the store when the recording is there (available_remote)
// and otherwise proxies it from the Pi (available_local) — so a recording that
// is merely `queued`/`uploading` is still perfectly playable while its Pi-side
// copy exists. What is NOT playable is a recording that has left the Pi but has
// not fully landed in the store yet: mergedRecordings reports exactly that as
// available_local=false + available_remote=true with upload_state "uploading"
// (metadata.json is the uploader's last file, so its absence means partial).
function canPlay(rec: RecordingItem): boolean {
  if (rec.available_local) return true;
  return rec.available_remote && rec.upload_state === "uploaded";
}

// Why Play is disabled — shown as the button's tooltip so a greyed control is
// never unexplained.
function playDisabledReason(rec: RecordingItem): string {
  if (rec.available_remote && rec.upload_state === "uploading") {
    return "Upload still in progress — not all segments have arrived yet";
  }
  return "Recording files are not available";
}

// Delete removes the CLUSTER-STORE copy, so it is offered only when there IS
// one and the upload has fully landed. Deleting mid-upload would race the
// uploader (which would simply re-PUT the files), and there is nothing to
// delete for a recording that only exists on the Pi — that copy is reclaimed
// automatically once its upload succeeds.
function canDelete(rec: RecordingItem): boolean {
  return rec.available_remote && rec.upload_state === "uploaded";
}

function Row({
  rec,
  onPlay,
  onFlag,
  onDelete,
  flagging,
  deleting,
  flagError,
}: {
  rec: RecordingItem;
  onPlay: () => void;
  onFlag: () => void;
  onDelete: () => void;
  flagging: boolean;
  deleting: boolean;
  flagError?: string;
}) {
  const flaggable = canFlag(rec) && !flagging;
  const playable = canPlay(rec);
  return (
    <tr style={{ borderBottom: "1px solid #f0f0f0" }}>
      <Td>
        <TypeBadge type={rec.type} />
      </Td>
      <Td>{fmtTime(rec.started_at)}</Td>
      <Td>{fmtTime(rec.ended_at)}</Td>
      <Td>{fmtDuration(rec.duration_seconds)}</Td>
      <Td>{rec.operator ?? "—"}</Td>
      <Td>{rec.upload_state}</Td>
      <td style={{ padding: "4px 0" }}>
        <button
          onClick={onPlay}
          disabled={!playable}
          title={playable ? "" : playDisabledReason(rec)}
          style={{ marginRight: 8 }}
        >
          Play
        </button>
        <button onClick={onFlag} disabled={!flaggable} title={flaggable ? "" : flagDisabledReason(rec)}>
          {flagging ? "…" : rec.upload_state === "failed" ? "Re-flag" : "Flag for upload"}
        </button>
        {canDelete(rec) && (
          <button
            onClick={onDelete}
            disabled={deleting}
            title="Delete the cluster copy of this recording"
            style={{ marginLeft: 8, color: "crimson" }}
          >
            {deleting ? "…" : "Delete"}
          </button>
        )}
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
  // A finite recording — the operator scrubs from the start.
  useHls(videoRef, recordingPlaylistUrl(rec.recording_id));

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

// --- delete confirmation -----------------------------------------------------

// Deleting the cluster copy is irreversible, so it takes an explicit
// confirmation naming the recording. The dialog states precisely what is and is
// not removed — an operator should never have to guess whether "delete" also
// wiped the Pi.
function ConfirmDeleteModal({
  rec,
  busy,
  onCancel,
  onConfirm,
}: {
  rec: RecordingItem;
  busy: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  useEffect(() => {
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
      aria-label="Confirm delete"
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.6)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
      onClick={onCancel}
    >
      <div
        style={{ background: "#fff", padding: 16, borderRadius: 4, maxWidth: 460 }}
        onClick={(e) => e.stopPropagation()}
      >
        <h3 style={{ marginTop: 0 }}>Delete recording?</h3>
        <p style={{ fontFamily: "monospace", fontSize: 13 }}>{rec.recording_id}</p>
        <p style={{ fontSize: 13, color: "#444" }}>
          This permanently deletes the stored copy of this recording. It cannot be undone, and the
          footage will no longer be playable.
        </p>
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
          <button onClick={onCancel} disabled={busy}>
            Cancel
          </button>
          <button onClick={onConfirm} disabled={busy} style={{ color: "crimson" }}>
            {busy ? "Deleting…" : "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
}
