// Tests for the /live UX state machine — the heart of Slice 2 (F2.2–F2.5).
//
// These cover the transitions the architecture (§4.4) calls load-bearing and
// that nothing else exercises:
//   - the happy path Idle → Requesting → Signaling → WaitingForVideo → Playing;
//   - WaitingForVideo masking the keyframe wait (no premature Playing);
//   - silent media stall (bytes frozen, ICE still "connected") → Stalled, then
//     auto-reconnect on backoff with a FRESH peer connection (dead PC reused
//     never), returning to Playing when media flows again;
//   - an ICE drop as the separate faster path to Stalled;
//   - a 401 on the /whep POST escalating to re-auth (redirect), NOT a retry;
//   - the reconnect-attempt budget terminating in Error, carrying the
//     backend's differentiated reason (503/504 body) in the message;
//   - the first-frame timeout giving up to Error when no frame ever arrives;
//   - stop() returning cleanly to Idle, tearing the PC down, and DELETEing
//     the /whep/{id} resource the Location header named.
//
// The hook drives an HTMLVideoElement via a ref. We render it with renderHook
// and hand it a ref to a <video> we create and control directly, so the tests
// never depend on testing-library DOM queries (which fight fake timers). Time
// is fully faked: advanceTimersByTimeAsync both fires timers AND drains the
// async connect/poll microtasks.

import { renderHook, act } from "@testing-library/react";
import { createRef } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createdPcs, installFakeRtc, type FakeRTCPeerConnection } from "./test/fakeRtc";
import { useLiveSession } from "./useLiveSession";

// Defaults from config.ts that the assertions below depend on.
const POLL_MS = 1500;
// The stall window (config.stallWindowMs, 3000ms) equals two POLL_MS steps, so
// the tests advance two polls to elapse it rather than referencing it directly.
const FIRST_FRAME_TIMEOUT_MS = 10_000;
const BACKOFF = [1000, 2000, 4000, 8000, 8000, 8000];
const MAX_ATTEMPTS = 6;

// Mount the hook with a ref to a real <video> we control.
function mountSession() {
  const ref = createRef<HTMLVideoElement>();
  const video = document.createElement("video");
  // @ts-expect-error test-only: assign the readonly ref to our element.
  ref.current = video;
  const { result } = renderHook(() => useLiveSession(ref));
  return { result, video };
}

// Build a fetch response the whep client understands: status + optional SDP
// answer + optional JSON error body + the Location header a 201 carries.
function whepResponse(status: number, opts: { body?: unknown; location?: string } = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (h: string) => (h === "Location" ? (opts.location ?? null) : null) },
    json: async () => opts.body ?? {},
    text: async () => "v=0\r\n",
  };
}

// /whep answers 201 + SDP + Location (the DELETE teardown resource).
function mockWhepOk(location = "/whep/42") {
  const fn = vi.fn().mockResolvedValue(whepResponse(201, { location }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

// /whep returns 401 (session expired at the auth proxy) → AuthError → re-auth.
function mockWhep401() {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(whepResponse(401)));
}

// Advance fake timers AND let the async connect/poll bodies drain.
async function tick(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

function pc(): FakeRTCPeerConnection {
  return createdPcs[createdPcs.length - 1];
}

// Drive a fresh session to Playing: start, let signaling complete, deliver
// advancing media so the next poll promotes WaitingForVideo → Playing.
async function reachPlaying(h: ReturnType<typeof mountSession>) {
  act(() => h.result.current.start());
  await tick(0); // drain the /whep POST → WaitingForVideo
  expect(h.result.current.state).toBe("WaitingForVideo");
  pc().bytesReceived = 1000;
  h.video.currentTime = 0.1;
  await tick(POLL_MS);
  expect(h.result.current.state).toBe("Playing");
}

beforeEach(() => {
  // Fake `performance` too: the stall detector measures the window with
  // performance.now(), so it must advance in lockstep with the timers. Without
  // this it returns wall-clock (~constant in a sync test) and the window never
  // elapses.
  vi.useFakeTimers({
    toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "performance", "Date"],
  });
  installFakeRtc();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("happy path", () => {
  it("advances Idle → Requesting → WaitingForVideo → Playing", async () => {
    mockWhepOk();
    const h = mountSession();
    expect(h.result.current.state).toBe("Idle");

    // Requesting is entered synchronously on start (before the awaits resolve).
    act(() => h.result.current.start());
    expect(h.result.current.state).toBe("Requesting");

    await reachPlaying(h);
    // The inbound stream got wired onto the <video> element.
    expect(h.video.srcObject).not.toBeNull();
  });

  it("WaitingForVideo masks the keyframe wait: no Playing until media advances", async () => {
    mockWhepOk();
    const h = mountSession();
    act(() => h.result.current.start());
    await tick(0);
    expect(h.result.current.state).toBe("WaitingForVideo");

    // Bytes flowing but the decoder hasn't advanced currentTime yet → still
    // waiting (requiring BOTH guards against a black "Playing" frame).
    pc().bytesReceived = 1000;
    h.video.currentTime = 0;
    await tick(POLL_MS);
    expect(h.result.current.state).toBe("WaitingForVideo");

    // currentTime advances → Playing.
    h.video.currentTime = 0.2;
    await tick(POLL_MS);
    expect(h.result.current.state).toBe("Playing");
  });
});

describe("media liveness (F2.3) + auto-reconnect (F2.4)", () => {
  it("silent stall (bytes frozen, ICE still connected) → Stalled → reconnect → Playing", async () => {
    mockWhepOk();
    const h = mountSession();
    await reachPlaying(h);
    const stalledPc = pc();

    // Freeze media: bytes and currentTime stop advancing, ICE stays connected.
    // Poll in single steps so we land on Stalled BEFORE the first backoff (1s)
    // elapses — a single >window jump would trip the stall and start the
    // reconnect in one tick, hiding the Stalled state. The window is measured
    // from the oldest retained sample (the one that promoted to Playing), so it
    // takes two polls (2×1500ms = 3000ms = the stall window) to elapse.
    await tick(POLL_MS); // frozen sample, window not yet elapsed
    await tick(POLL_MS); // 3000ms since the oldest sample → stall trips
    expect(h.result.current.state).toBe("Stalled");
    expect(h.result.current.attempt).toBe(1);
    // The dead PC was closed (F2.4: never reused).
    expect(stalledPc.closed).toBe(true);

    // First backoff (1s) elapses → a FRESH PC is built and signals.
    await tick(BACKOFF[0]);
    expect(h.result.current.state).toBe("WaitingForVideo");
    expect(pc()).not.toBe(stalledPc);

    // Media flows again on the new connection → back to Playing.
    pc().bytesReceived = 5000;
    h.video.currentTime = 1.0;
    await tick(POLL_MS);
    expect(h.result.current.state).toBe("Playing");
  });

  it("ICE drop is the separate, faster path to Stalled", async () => {
    mockWhepOk();
    const h = mountSession();
    await reachPlaying(h);

    // ICE goes non-connected; the very next poll routes straight to Stalled
    // without waiting out the full stall window.
    pc().iceConnectionState = "disconnected";
    await tick(POLL_MS);
    expect(h.result.current.state).toBe("Stalled");
    expect(h.result.current.detail).toContain("disconnected");
  });

  it("gives up to Error after the budget, surfacing the backend's 503 reason", async () => {
    mockWhepOk();
    const h = mountSession();
    await reachPlaying(h);

    // Make every reconnect fail with the health-gate's differentiated 503 so
    // attempts climb to the budget and the reason reaches the operator.
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          whepResponse(503, { body: { error: "live_unavailable", reason: "Pi unreachable" } }),
        ),
    );
    // Trip the stall (two polls span the 3000ms window; see the test above).
    await tick(POLL_MS);
    await tick(POLL_MS);
    expect(h.result.current.state).toBe("Stalled");

    // Run out the backoff schedule; each attempt fails signaling and reschedules.
    // The +POLL_MS slack lets the failing connect attempt settle each round.
    for (const d of BACKOFF) {
      await tick(d + POLL_MS);
    }
    expect(h.result.current.state).toBe("Error");
    expect(h.result.current.detail).toContain(String(MAX_ATTEMPTS));
    // The differentiated reason from the /whep error body reaches the display.
    expect(h.result.current.detail).toContain("Pi unreachable");
  });
});

describe("re-auth escalation (F2.5)", () => {
  it("a 401 on the /whep POST re-authenticates (reload) and does NOT retry", async () => {
    mockWhep401();
    const h = mountSession();
    act(() => h.result.current.start());
    await tick(0);

    // Escalation is immediate: Error state + a re-navigation (reload) so the
    // auth proxy can re-auth. Auth-mechanism agnostic: no named login endpoint.
    expect(h.result.current.state).toBe("Error");
    expect(globalThis.__reloadCount).toBeGreaterThanOrEqual(1);
    expect(globalThis.__lastNavigation).toBeUndefined();
    expect(h.result.current.detail).toContain("Authentication");

    // No reconnect was scheduled (no looping on a closed door): advancing time
    // does not produce another fetch attempt.
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const calls = fetchMock.mock.calls.length;
    await tick(BACKOFF[0] + BACKOFF[1]);
    expect(fetchMock.mock.calls.length).toBe(calls);
  });
});

describe("first-frame timeout + stop", () => {
  it("times out to Error if no frame ever arrives in WaitingForVideo", async () => {
    mockWhepOk();
    const h = mountSession();
    act(() => h.result.current.start());
    await tick(0);
    expect(h.result.current.state).toBe("WaitingForVideo");

    // Never deliver media; the first-frame timeout fires.
    await tick(FIRST_FRAME_TIMEOUT_MS);
    expect(h.result.current.state).toBe("Error");
    expect(h.result.current.detail).toContain("no video");
  });

  it("stop() returns to Idle, clears the video, closes the PC, and DELETEs the WHEP resource", async () => {
    const fetchMock = mockWhepOk("/whep/42");
    const h = mountSession();
    await reachPlaying(h);
    const p = pc();

    act(() => h.result.current.stop());
    expect(h.result.current.state).toBe("Idle");
    expect(h.video.srcObject).toBeNull();
    expect(p.closed).toBe(true);

    // Clean viewer teardown: the Location the 201 named is DELETEd.
    const deletes = fetchMock.mock.calls.filter(
      (c) => (c[1] as RequestInit | undefined)?.method === "DELETE",
    );
    expect(deletes).toHaveLength(1);
    expect(deletes[0][0]).toBe("/whep/42");

    // No stray timers fire afterwards (teardown invalidated the poll loop).
    await tick(POLL_MS * 3);
    expect(h.result.current.state).toBe("Idle");
  });
});
