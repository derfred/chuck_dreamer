// Backend API client + the re-auth escalation primitive (F2.1, F2.5).
//
// All /api/... endpoints sit behind oauth2-proxy. A 401 means the proxy is
// no longer forwarding authenticated requests (session expired/revoked). The
// frontend must NOT loop on 401 — it escalates to the proxy's login flow.

import { config } from "./config";
import type { Capabilities, ModeTriplet } from "../../shared/schemas";

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

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (res.status === 401) throw new AuthError();
  if (!res.ok) throw new Error(`${url} failed: ${res.status}`);
  return (await res.json()) as T;
}

export function fetchMe(): Promise<Identity> {
  return getJson<Identity>("/api/me");
}

export function fetchCapabilities(): Promise<Capabilities> {
  return getJson<Capabilities>("/api/capabilities");
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
