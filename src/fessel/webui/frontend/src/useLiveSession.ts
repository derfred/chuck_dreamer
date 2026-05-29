// The /live UX state machine (F2.2), media-liveness detection (F2.3),
// auto-reconnect with backoff (F2.4), and re-auth escalation (F2.5).
//
// Why a state machine and not "connected/disconnected": ICE state is NOT a
// reliable proxy for media liveness. A WebRTC peer connection can report
// `connected` indefinitely with no frames arriving, and <video> will sit at
// readyState=4, paused=false over a frozen frame. This machine treats
// stalled media as a DISTINCT condition (Stalled), caught by polling stats,
// not by an ICE event (WebRTC fires none for "media stopped while ICE up").

import { useCallback, useEffect, useRef, useState } from "react";
import type { ModeTriplet } from "../../shared/schemas";
import { config } from "./config";
import { AuthError, fetchWhepUrl, redirectToLogin } from "./api";
import { startWhep } from "./whep";

export type LiveState =
  | "Idle"
  | "Requesting"
  | "Signaling"
  | "WaitingForVideo"
  | "Playing"
  | "Stalled"
  | "Error";

export interface LiveSession {
  state: LiveState;
  // Short human-readable diagnostic shown in Error/Stalled.
  detail: string;
  // Reconnect attempt counter (shown in the Stalled banner).
  attempt: number;
  start: (mode: ModeTriplet) => void;
  stop: () => void;
  retry: () => void;
}

interface Sample {
  bytes: number;
  currentTime: number;
  at: number;
}

// Drive the live session for a given <video> element. The caller owns the
// element ref and the mode selection.
export function useLiveSession(videoRef: React.RefObject<HTMLVideoElement>): LiveSession {
  const [state, setState] = useState<LiveState>("Idle");
  const [detail, setDetail] = useState<string>("");
  const [attempt, setAttempt] = useState<number>(0);

  // Mutable refs so async loops/timers read live values without stale closures.
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const modeRef = useRef<ModeTriplet | null>(null);
  const stateRef = useRef<LiveState>("Idle");
  const attemptRef = useRef<number>(0);
  const pollRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);
  const samplesRef = useRef<Sample[]>([]);
  // Monotonically increasing token; bumped on every teardown so stale async
  // callbacks (signaling, stats, scheduled reconnects) become no-ops.
  const genRef = useRef<number>(0);

  const setLiveState = useCallback((s: LiveState, d = "") => {
    stateRef.current = s;
    setState(s);
    setDetail(d);
  }, []);

  const clearTimers = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Close the peer connection cleanly. The dead PC is never reused (F2.4).
  const closePc = useCallback(() => {
    if (pcRef.current) {
      try {
        pcRef.current.close();
      } catch {
        /* already closed */
      }
      pcRef.current = null;
    }
  }, []);

  // Full teardown: invalidate in-flight async work, clear timers, close PC.
  const teardown = useCallback(() => {
    genRef.current += 1;
    clearTimers();
    closePc();
    samplesRef.current = [];
  }, [clearTimers, closePc]);

  // Escalate to the proxy login. Tear down cleanly first so an already-issued
  // token's in-flight media isn't left dangling against a closed door (F2.5;
  // satisfies architecture §4.5 rule 6 — we don't hammer, we tear down).
  const escalateReauth = useCallback(() => {
    teardown();
    setLiveState("Error", "Authentication required — redirecting to sign in…");
    redirectToLogin();
  }, [teardown, setLiveState]);

  // Sample inboundRtp.bytesReceived + <video>.currentTime. Stall = BOTH
  // failed to advance across a window exceeding the stall threshold while ICE
  // is still connected (F2.3). Requiring both protects against either lying:
  // bytes-without-time = player not decoding; time-without-bytes = stale stats.
  const poll = useCallback(
    async (gen: number) => {
      const pc = pcRef.current;
      const video = videoRef.current;
      if (!pc || !video || gen !== genRef.current) return;

      // ICE off `connected` is a separate, faster path to Stalled.
      const ice = pc.iceConnectionState;
      if (ice === "failed" || ice === "disconnected" || ice === "closed") {
        beginReconnect(gen, `connection ${ice}`);
        return;
      }

      let bytes = 0;
      try {
        const stats = await pc.getStats();
        stats.forEach((r) => {
          if (r.type === "inbound-rtp" && (r as RTCInboundRtpStreamStats).kind === "video") {
            bytes += (r as RTCInboundRtpStreamStats).bytesReceived ?? 0;
          }
        });
      } catch {
        return; // transient getStats failure; try again next tick
      }
      if (gen !== genRef.current) return;

      const now = performance.now();
      const sample: Sample = { bytes, currentTime: video.currentTime, at: now };
      const samples = samplesRef.current;
      samples.push(sample);
      // Keep only samples within the stall window plus one for comparison.
      while (samples.length > 1 && now - samples[0].at > config.stallWindowMs) {
        if (now - samples[1].at > config.stallWindowMs) samples.shift();
        else break;
      }

      const playing = video.currentTime > 0 && bytes > 0;

      if (stateRef.current === "WaitingForVideo") {
        if (playing) {
          setLiveState("Playing");
        }
        return;
      }

      // Playing: look for a silent stall across the full window.
      const oldest = samples[0];
      if (samples.length >= 2 && now - oldest.at >= config.stallWindowMs) {
        const bytesStalled = sample.bytes <= oldest.bytes;
        const timeStalled = sample.currentTime <= oldest.currentTime;
        if (bytesStalled && timeStalled) {
          beginReconnect(gen, "media stalled");
        }
      }
    },
    // beginReconnect/setLiveState defined below; refs make the closure safe.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [videoRef, setLiveState],
  );

  // Run one connect attempt: mint a fresh WHEP URL, build a fresh PC, signal,
  // then enter WaitingForVideo and start polling for first frame / stall.
  const connectOnce = useCallback(
    async (gen: number) => {
      const mode = modeRef.current;
      const video = videoRef.current;
      if (!mode || !video || gen !== genRef.current) return;
      try {
        let url: string;
        try {
          url = await fetchWhepUrl(config.livePath, mode);
        } catch (e) {
          if (e instanceof AuthError) {
            escalateReauth();
            return;
          }
          throw e;
        }
        if (gen !== genRef.current) return;
        setLiveState("Signaling");
        const pc = await startWhep(url, video);
        if (gen !== genRef.current) {
          pc.close();
          return;
        }
        pcRef.current = pc;
        setLiveState("WaitingForVideo");
        samplesRef.current = [];

        // First-frame timeout: WaitingForVideo masks the keyframe wait, but
        // give up to Error if no frame ever arrives.
        clearTimers();
        timerRef.current = window.setTimeout(() => {
          if (gen === genRef.current && stateRef.current === "WaitingForVideo") {
            teardown();
            setLiveState("Error", "Camera unavailable — no video received");
          }
        }, config.firstFrameTimeoutMs);

        pollRef.current = window.setInterval(() => void poll(gen), config.statsPollMs);
      } catch (e) {
        if (gen !== genRef.current) return;
        // Signaling/network failure during a fresh start vs a reconnect.
        beginReconnect(gen, e instanceof Error ? e.message : "connection failed");
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [videoRef, setLiveState, escalateReauth, clearTimers, teardown, poll],
  );

  // Transition Playing/WaitingForVideo -> Stalled and schedule a reconnect
  // with exponential backoff. Keep the <video> element on screen (the caller
  // never clears srcObject in Stalled) so the last good frame stays visible.
  const beginReconnect = useCallback(
    (gen: number, reason: string) => {
      if (gen !== genRef.current) return;
      clearTimers();
      closePc();
      // New generation for the upcoming attempt's async work.
      genRef.current += 1;
      const nextGen = genRef.current;

      const nextAttempt = attemptRef.current + 1;
      attemptRef.current = nextAttempt;
      setAttempt(nextAttempt);

      if (nextAttempt > config.reconnectMaxAttempts) {
        setLiveState("Error", `Reconnect failed after ${config.reconnectMaxAttempts} attempts`);
        return;
      }

      setLiveState("Stalled", reason);
      const idx = Math.min(nextAttempt - 1, config.reconnectBackoffMs.length - 1);
      const delay = config.reconnectBackoffMs[idx];
      timerRef.current = window.setTimeout(() => void connectOnce(nextGen), delay);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [clearTimers, closePc, setLiveState, connectOnce],
  );

  const start = useCallback(
    (mode: ModeTriplet) => {
      teardown();
      modeRef.current = mode;
      attemptRef.current = 0;
      setAttempt(0);
      genRef.current += 1;
      const gen = genRef.current;
      setLiveState("Requesting");
      void connectOnce(gen);
    },
    [teardown, setLiveState, connectOnce],
  );

  const stop = useCallback(() => {
    teardown();
    attemptRef.current = 0;
    setAttempt(0);
    if (videoRef.current) videoRef.current.srcObject = null;
    setLiveState("Idle");
  }, [teardown, setLiveState, videoRef]);

  const retry = useCallback(() => {
    const mode = modeRef.current;
    if (mode) start(mode);
    else setLiveState("Idle");
  }, [start, setLiveState]);

  // Tear down on unmount so timers/PC don't leak.
  useEffect(() => () => teardown(), [teardown]);

  return { state, detail, attempt, start, stop, retry };
}
