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
import { config } from "./config";
import { AuthError, reauthenticate } from "./api";
import { deleteWhepSession, startWhep } from "./whep";

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
  start: () => void;
  stop: () => void;
  retry: () => void;
}

interface Sample {
  bytes: number;
  currentTime: number;
  at: number;
}

// Drive the live session for a given <video> element. The caller owns the
// element ref. There is no mode: the live profile is fixed (§4.4 — no
// resolution selector).
export function useLiveSession(videoRef: React.RefObject<HTMLVideoElement | null>): LiveSession {
  const [state, setState] = useState<LiveState>("Idle");
  const [detail, setDetail] = useState<string>("");
  const [attempt, setAttempt] = useState<number>(0);

  // Mutable refs so async loops/timers read live values without stale closures.
  const pcRef = useRef<RTCPeerConnection | null>(null);
  // The /whep/{id} resource of the current session (from the Location header);
  // DELETEd on any teardown so the server releases the viewer promptly.
  const locationRef = useRef<string | null>(null);
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

  // Close the peer connection cleanly and release the server-side WHEP
  // resource (fire-and-forget DELETE). The dead PC is never reused (F2.4).
  const closePc = useCallback(() => {
    if (pcRef.current) {
      try {
        pcRef.current.close();
      } catch {
        /* already closed */
      }
      pcRef.current = null;
    }
    if (locationRef.current) {
      deleteWhepSession(locationRef.current);
      locationRef.current = null;
    }
  }, []);

  // Full teardown: invalidate in-flight async work, clear timers, close PC.
  const teardown = useCallback(() => {
    genRef.current += 1;
    clearTimers();
    closePc();
    samplesRef.current = [];
  }, [clearTimers, closePc]);

  // Escalate re-authentication. Tear down cleanly first so in-flight media
  // isn't left dangling against a closed door (F2.5; architecture §4.5 — we
  // don't hammer, we tear down). The app re-navigates so the auth proxy can
  // re-authenticate; it names no login mechanism (auth is infra's concern).
  const escalateReauth = useCallback(() => {
    teardown();
    setLiveState("Error", "Authentication required — re-authenticating…");
    reauthenticate();
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

  // Run one connect attempt: build a fresh PC, POST the offer same-origin to
  // /whep, then enter WaitingForVideo and start polling for first frame /
  // stall. The /whep answer can BLOCK up to the server's activation timeout
  // (~15s) on the first viewer — Signaling ("Establishing connection…")
  // deliberately covers that wait.
  const connectOnce = useCallback(
    async (gen: number) => {
      const video = videoRef.current;
      if (!video || gen !== genRef.current) return;
      try {
        let session;
        try {
          // Requesting ("Connecting…") covers building the offer; the callback
          // flips to Signaling when the (possibly ~15s-blocking) POST starts.
          session = await startWhep(video, () => {
            if (gen === genRef.current) setLiveState("Signaling");
          });
        } catch (e) {
          if (e instanceof AuthError) {
            escalateReauth();
            return;
          }
          throw e;
        }
        if (gen !== genRef.current) {
          session.pc.close();
          if (session.location) deleteWhepSession(session.location);
          return;
        }
        pcRef.current = session.pc;
        locationRef.current = session.location;
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
        // Giving up: carry the last failure's operator-facing reason (e.g. the
        // /whep 503/504 differentiated reasons) into the Error display.
        const suffix = reason ? ` — ${reason}` : "";
        setLiveState(
          "Error",
          `Reconnect failed after ${config.reconnectMaxAttempts} attempts${suffix}`,
        );
        return;
      }

      setLiveState("Stalled", reason);
      const idx = Math.min(nextAttempt - 1, config.reconnectBackoffMs.length - 1);
      const delay = config.reconnectBackoffMs[idx];
      timerRef.current = window.setTimeout(() => void connectOnce(nextGen), delay);
    },
    [clearTimers, closePc, setLiveState, connectOnce],
  );

  const start = useCallback(() => {
    teardown();
    attemptRef.current = 0;
    setAttempt(0);
    genRef.current += 1;
    const gen = genRef.current;
    setLiveState("Requesting");
    void connectOnce(gen);
  }, [teardown, setLiveState, connectOnce]);

  const stop = useCallback(() => {
    teardown();
    attemptRef.current = 0;
    setAttempt(0);
    if (videoRef.current) videoRef.current.srcObject = null;
    setLiveState("Idle");
  }, [teardown, setLiveState, videoRef]);

  const retry = useCallback(() => {
    start();
  }, [start]);

  // Bandwidth safeguard: stop the live stream when the tab is hidden. Live is
  // the only surface that holds a dedicated WebRTC stream open, so a
  // backgrounded tab must not keep paying for frames nobody is watching. Only
  // fires when a session is actually active (not Idle/Error) — hiding the tab on
  // the Ring/idle view is a no-op. Navigating away unmounts the component, which
  // the unmount effect below already tears down.
  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden && stateRef.current !== "Idle" && stateRef.current !== "Error") {
        stop();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, [stop]);

  // Tear down on unmount so timers/PC don't leak.
  useEffect(() => () => teardown(), [teardown]);

  return { state, detail, attempt, start, stop, retry };
}
