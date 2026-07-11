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

export const config = {
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
  // Dashboard /api/state poll interval (F3.4). Short, because state can change
  // from outside the UI (direct supervisor calls, or the Slice 6 safety state
  // machine acting on its own) and the dashboard must reflect that.
  statePollMs: num("VITE_STATE_POLL_MS", 2000),
  // Corner-light /api/health/pi poll interval (health-check spec §4.2). Aligned
  // with the backend refresh (default 5s) — polling faster only reads the same
  // cached snapshot.
  healthPollMs: num("VITE_HEALTH_POLL_MS", 5000),
  // If the last SUCCESSFUL health fetch is older than this, the light degrades
  // to grey/"stale" regardless of the last-known colour (spec §4.4): a green
  // light the browser can't refresh must not keep showing green.
  healthStaleMs: num("VITE_HEALTH_STALE_MS", 20000),
  // Monitor freeze-frame meta poll interval. The Pi pushes a fresh snapshot
  // roughly every 10s (video.snapshot.interval_s), so polling much faster than
  // that only re-reads the same cached age.
  snapshotPollMs: num("VITE_SNAPSHOT_POLL_MS", 5000),
};
