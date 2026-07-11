// Backend API client + the re-auth escalation primitive (F2.1, F2.5).
//
// The app is AUTH-MECHANISM AGNOSTIC. An auth proxy (oauth2-proxy today, but
// the app neither knows nor cares which) guards the webui and settles the flat
// in/out access decision upstream; the backend only trusts the identity it
// injects. A 401 reaching the app means a request slipped past the proxy
// without authenticated identity — almost always an expired session caught on
// a background XHR rather than a navigation. The app does NOT know any login
// endpoint: it simply re-navigates (reload) so the proxy can re-authenticate
// the navigation and either restore the session or serve its own denial page.
// The app must NOT loop on 401; a one-shot guard (reauthenticate) backs off to
// a terminal error if a reload still comes back unauthenticated.

import type { ZodType } from "zod";
import type {
  AnomalyLogEntry,
  Capabilities,
  ModeTriplet,
  StateResponse,
} from "../../shared/schemas";
import { CapabilitiesSchema, StateResponseSchema } from "./generated/validators";

// Thrown on a 401 from any backend call. Callers treat this as terminal for
// the current session and hand off to reauthenticate() rather than retrying.
export class AuthError extends Error {
  constructor() {
    super("authentication required");
    this.name = "AuthError";
  }
}

export interface Identity {
  user: string;
  email: string | null;
  groups: string[];
}

// sessionStorage key for the one-shot re-auth guard (see reauthenticate).
const REAUTH_GUARD_KEY = "fessel-reauth-attempted";

// Re-trigger authentication after a 401, without naming any auth mechanism
// (F2.5). The app re-navigates to the current URL; the auth proxy in front of
// the webui intercepts that navigation and runs whatever login/denial flow it
// is configured for (the app is agnostic to which). A loop guard makes this
// safe: if we already reloaded once for auth and the very next load still
// 401s — a misconfigured or absent proxy that lets unauthenticated navigations
// through — we stop reloading and let the caller surface a terminal error
// instead of bouncing forever. The guard is cleared by clearReauthGuard() once
// an authenticated request succeeds.
export function reauthenticate(): boolean {
  if (sessionStorage.getItem(REAUTH_GUARD_KEY)) {
    // Already bounced once and still unauthenticated — don't loop.
    return false;
  }
  sessionStorage.setItem(REAUTH_GUARD_KEY, "1");
  // Re-request the current URL so the auth proxy can re-authenticate it.
  window.location.reload();
  return true;
}

// Clear the re-auth loop guard. Called after any authenticated request
// succeeds, so a later genuine session expiry can reload again.
export function clearReauthGuard(): void {
  sessionStorage.removeItem(REAUTH_GUARD_KEY);
}

// Thrown when a 2xx response body fails its runtime schema validation. The
// backend/Pi sent something off-contract — distinct from a network/HTTP error.
export class SchemaError extends Error {
  constructor(url: string, detail: string) {
    super(`${url} returned an off-contract body: ${detail}`);
    this.name = "SchemaError";
  }
}

// Fetch JSON and, when a Zod schema is supplied, validate the body at the
// trust boundary (parse-don't-cast). The schemas are GENERATED from the Python
// models (../../shared/validators), so they cannot drift from the wire shape.
// A validation failure is logged via console.error — a hook for future
// monitoring — and surfaced as SchemaError so callers handle it like any other
// failed read. Endpoints without a generated model (e.g. /api/me -> Identity,
// an auth-layer dataclass) pass no validator and keep the plain cast.
async function getJson<T>(url: string, validate?: ZodType<T>): Promise<T> {
  const res = await fetch(url);
  if (res.status === 401) throw new AuthError();
  // A non-401 response proves auth is working — reset the re-auth loop guard so
  // a later genuine session expiry can reload again.
  clearReauthGuard();
  if (!res.ok) throw new Error(`${url} failed: ${res.status}`);
  const body = await res.json();
  if (!validate) return body as T;
  const result = validate.safeParse(body);
  if (!result.success) {
    // Structured-ish line so a log scraper can pick it up later. The Zod issue
    // list says exactly which field was off-contract.
    console.error(`[schema] ${url} validation failed`, result.error.issues);
    throw new SchemaError(url, result.error.message);
  }
  return result.data;
}

export function fetchMe(): Promise<Identity> {
  // No generated validator: Identity is a backend auth dataclass, not a
  // fessel_schemas wire model. Keep the typed cast.
  return getJson<Identity>("/api/me");
}

export function fetchCapabilities(): Promise<Capabilities> {
  return getJson<Capabilities>("/api/capabilities", CapabilitiesSchema);
}

// Canonical mode string "<W>x<H>@<fps>@<bitrate_bps>" (X0.2). Must match the
// Python serialiser (fessel_schemas.mode_to_canonical) exactly. resolution is
// already "<W>x<H>", so the form is resolution@fps@bitrate.
export function modeToCanonical(m: ModeTriplet): string {
  return `${m.resolution}@${m.fps}@${m.bitrate_bps}`;
}

// --- Slice 3 control plane (F3.1–F3.4) --------------------------------------

// The seven operator control actions. The string is the /api/control/<action>
// path segment AND the audit action name; it matches CONTROL_ACTIONS on the
// backend exactly.
export type ControlAction =
  | "pause"
  | "stop"
  | "resume"
  | "shutdown/arm"
  | "shutdown/jetson"
  | "poweron/arm"
  | "poweron/jetson";

// Thrown when a control POST fails with a non-2xx that is NOT a 401. Carries
// the human-facing diagnostic the backend/supervisor returned (e.g. "plug
// reported on after retries") so the button can render it inline (F3.2).
export class ControlError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ControlError";
    this.status = status;
  }
}

// Pull a human-readable message out of supervisor's structured 5xx body. The
// body may be {detail: {error, message, ...}} (supervisor via the backend),
// {error, message} (backend's own 502), or a bare {detail: "..."} string.
function diagnosticOf(body: unknown, status: number): string {
  const d = (body as { detail?: unknown })?.detail ?? body;
  if (typeof d === "string") return d;
  if (d && typeof d === "object") {
    const o = d as { message?: string; error?: string };
    if (o.message) return o.message;
    if (o.error) return o.error;
  }
  return `request failed (${status})`;
}

// Poll the aggregated control-plane state. A 401 throws AuthError (the caller
// escalates to re-auth); any other failure (HTTP, network, or an off-contract
// body via SchemaError) throws so the dashboard shows its transient "state
// unavailable" banner while retrying. The body is validated against the
// generated StateResponse schema (the owned fields strictly; the opaque
// `jetson` passthrough stays permissive, matching the Python model).
export function fetchState(): Promise<StateResponse> {
  return getJson<StateResponse>("/api/state", StateResponseSchema);
}

// --- Pi-connection health (health-check spec §2.2 / §4) ----------------------

// One health fact. `unknown` is first-class (distinct from red): before the
// first poll, or when a Pi-observed fact is masked because pi_control is red.
export type HealthState = "green" | "yellow" | "red" | "unknown";

export interface HealthFact {
  id: string;
  label: string;
  state: HealthState;
  detail: string;
  observed_at: string;
}

// The composite /api/health/pi response: the rollup `light` + the per-fact
// breakdown in root-cause-first display order. A backend-internal projection,
// not a fessel_schemas wire model, so it is typed here (like RecordingItem).
export interface PiHealth {
  light: HealthState;
  generated_at: string;
  facts: HealthFact[];
}

// Poll the composite Pi-connection health. A 401 throws AuthError (the caller
// escalates to re-auth); any other failure throws so the corner light can begin
// its staleness clock (spec §4.2/§4.4) rather than blanking.
export function fetchPiHealth(): Promise<PiHealth> {
  return getJson<PiHealth>("/api/health/pi");
}

// Result of an operator-triggered connection check: a timed round trip to
// supervisor (latency) plus a timed download of a known-size payload
// (throughput), both measured cluster-side against the Pi over the tailnet.
export interface ConnectionCheckResult {
  latency_ms: number;
  throughput_mbps: number;
  payload_bytes: number;
}

// POSTs /api/connection-check. This actively drives traffic to the Pi and
// takes noticeably longer than a status read, so it is operator-triggered
// (a button), never polled. A 401 throws AuthError; any other failure
// (including a 502 "Pi unreachable") throws a plain Error the caller shows
// inline next to the button.
export async function runConnectionCheck(): Promise<ConnectionCheckResult> {
  const res = await fetch("/api/connection-check", { method: "POST" });
  if (res.status === 401) throw new AuthError();
  clearReauthGuard();
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.message ?? `request failed (${res.status})`);
  }
  return res.json();
}

// --- Slice 4 recordings + ring (F4.1–F4.5) ----------------------------------

// One row of /api/recordings (B4.2). Shape matches the backend's merged item
// (Pi-local + MinIO), not a single fessel_schemas model, so it is typed here.
export interface RecordingItem {
  recording_id: string;
  // Slice 5 (S5.4): explicit (operator) vs anomaly (auto-triggered) recording.
  type: "explicit" | "anomaly";
  started_at: string | null;
  ended_at: string | null;
  duration_seconds: number | null;
  operator: string | null;
  flagged_for_upload: boolean;
  upload_state: "none" | "queued" | "uploading" | "uploaded" | "failed";
  available_local: boolean;
  available_remote: boolean;
}

export function fetchRecordings(): Promise<RecordingItem[]> {
  return getJson<RecordingItem[]>("/api/recordings");
}

// The in-memory anomaly log (B5.2 / F5.2). Newest first; each entry carries the
// full original event payload + the linked anomaly-recording id (if any).
export function fetchAnomalies(): Promise<AnomalyLogEntry[]> {
  return getJson<AnomalyLogEntry[]>("/api/anomalies");
}

// Options for starting an explicit recording. Recording resolution is a deploy
// setting (recordings reuse the ring encode), so there is no mode here.
// `lookbackSeconds` reaches the recording's start back into the ring buffer
// (0 = start now); `uploadWhenDone` flags it for upload on finalise.
export interface StartRecordingOpts {
  lookbackSeconds?: number;
  uploadWhenDone?: boolean;
}

// Start an explicit recording. Returns the recording id the supervisor minted.
// Throws AuthError on 401, ControlError otherwise (F4.5).
export async function startRecording(opts: StartRecordingOpts = {}): Promise<string> {
  const res = await fetch("/api/recording/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      lookback_seconds: opts.lookbackSeconds ?? 0,
      upload_when_done: opts.uploadWhenDone ?? false,
    }),
  });
  if (res.status === 401) throw new AuthError();
  if (!res.ok) throw new ControlError(await diagnostic(res), res.status);
  const body = (await res.json()) as { recording_id?: string };
  return body.recording_id ?? "";
}

export async function stopRecording(): Promise<void> {
  const res = await fetch("/api/recording/stop", { method: "POST" });
  if (res.status === 401) throw new AuthError();
  if (!res.ok) throw new ControlError(await diagnostic(res), res.status);
}

// Flag a finalised recording for upload (F4.4). Non-destructive; the dashboard
// applies it optimistically and reverts on error.
export async function flagForUpload(recordingId: string): Promise<void> {
  const res = await fetch("/api/recording/flag-upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ recording_id: recordingId }),
  });
  if (res.status === 401) throw new AuthError();
  if (!res.ok) throw new ControlError(await diagnostic(res), res.status);
}

// URL the hls.js player points at for a recording's playlist (B4.3). The
// backend decides MinIO-redirect vs supervisor-proxy; the frontend just loads
// this URL.
export function recordingPlaylistUrl(recordingId: string): string {
  return `/api/recordings/${encodeURIComponent(recordingId)}/playlist`;
}

async function diagnostic(res: Response): Promise<string> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // non-JSON error body; fall through to status-only message.
  }
  return diagnosticOf(body, res.status);
}

// Issue a control action. Resolves on success; throws AuthError on 401 (the
// session lapsed) or ControlError carrying the diagnostic on any other
// failure. The frontend does NOT retry — one click, one POST (matches the
// backend's one-intent-one-forward contract).
export async function postControl(action: ControlAction): Promise<void> {
  const res = await fetch(`/api/control/${action}`, { method: "POST" });
  if (res.status === 401) throw new AuthError();
  clearReauthGuard();
  if (res.ok) return;
  throw new ControlError(await diagnostic(res), res.status);
}
