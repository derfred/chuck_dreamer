// Tunable knobs for the /live UX state machine (F2.2–F2.4). These are
// starting points drawn from the architecture's discussion (§4.4) and are
// expected to be revisited once the system runs on real cellular — they are
// knobs, not hardcoded constants. Overridable at build time via Vite env
// (VITE_*), falling back to the documented defaults.

function num(name: string, fallback: number): number {
  const raw = import.meta.env[name as keyof ImportMetaEnv] as string | undefined;
  if (raw === undefined || raw === "") return fallback;
  const v = Number(raw);
  return Number.isFinite(v) ? v : fallback;
}

function str(name: string, fallback: string): string {
  const raw = import.meta.env[name as keyof ImportMetaEnv] as string | undefined;
  return raw && raw !== "" ? raw : fallback;
}

export const config = {
  // mediamtx source path for the single live feed.
  livePath: str("VITE_LIVE_PATH", "pi"),
  // getStats() / currentTime polling interval while Playing/WaitingForVideo.
  statsPollMs: num("VITE_STATS_POLL_MS", 1500),
  // Both bytesReceived and currentTime must fail to advance across this
  // window (with ICE still connected) to count as a silent media stall.
  stallWindowMs: num("VITE_STALL_WINDOW_MS", 3000),
  // How long WaitingForVideo waits for the first decoded frame before
  // giving up to Error (keyframe wait should be ~1–2s with a short GOP).
  firstFrameTimeoutMs: num("VITE_FIRST_FRAME_TIMEOUT_MS", 10000),
  // Auto-reconnect backoff schedule (ms). Capped at the last value.
  reconnectBackoffMs: [1000, 2000, 4000, 8000, 8000, 8000],
  // Give up reconnecting (-> Error) after this many failed attempts.
  reconnectMaxAttempts: num("VITE_RECONNECT_MAX_ATTEMPTS", 6),
  // oauth2-proxy login start path; the proxy bounces the browser to GitHub.
  // Exact path depends on the proxy config, hence a knob.
  oauthStartPath: str("VITE_OAUTH_START_PATH", "/oauth2/start"),
  // Dashboard /api/state poll interval (F3.4). Short, because state can change
  // from outside the UI (direct supervisor calls, or the Slice 6 safety state
  // machine acting on its own) and the dashboard must reflect that.
  statePollMs: num("VITE_STATE_POLL_MS", 2000),
};
