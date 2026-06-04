// Backend API client + the re-auth escalation primitive (F2.1, F2.5).
//
// All /api/... endpoints sit behind oauth2-proxy. A 401 means the proxy is
// no longer forwarding authenticated requests (session expired/revoked). The
// frontend must NOT loop on 401 — it escalates to the proxy's login flow.

import type { ZodType } from "zod";
import { config } from "./config";
import type { Capabilities, ModeTriplet, StateResponse } from "../../shared/schemas";
import { CapabilitiesSchema, StateResponseSchema } from "./generated/validators";

// Thrown on a 401 from any backend call. Callers treat this as terminal for
// the current session and hand off to redirectToLogin() rather than retrying.
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

// Redirect the browser to oauth2-proxy's login start, asking it to return to
// the current URL after a successful GitHub OIDC login (F2.5).
export function redirectToLogin(): void {
  const rd = encodeURIComponent(window.location.href);
  window.location.href = `${config.oauthStartPath}?rd=${rd}`;
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

// Mint a fresh signed WHEP URL. A 401 throws AuthError (session expired);
// callers escalate to re-auth rather than retrying (F2.4 rule).
export async function fetchWhepUrl(path: string, mode: ModeTriplet): Promise<string> {
  const params = new URLSearchParams({ path, mode: modeToCanonical(mode) });
  const res = await fetch(`/api/auth/whep-url?${params.toString()}`);
  if (res.status === 401) throw new AuthError();
  if (!res.ok) throw new Error(`whep-url failed: ${res.status}`);
  const body = (await res.json()) as { url: string };
  return body.url;
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

// Issue a control action. Resolves on success; throws AuthError on 401 (the
// session lapsed) or ControlError carrying the diagnostic on any other
// failure. The frontend does NOT retry — one click, one POST (matches the
// backend's one-intent-one-forward contract).
export async function postControl(action: ControlAction): Promise<void> {
  const res = await fetch(`/api/control/${action}`, { method: "POST" });
  if (res.status === 401) throw new AuthError();
  if (res.ok) return;
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // non-JSON error body; fall back to the status-only message.
  }
  throw new ControlError(diagnosticOf(body, res.status), res.status);
}
