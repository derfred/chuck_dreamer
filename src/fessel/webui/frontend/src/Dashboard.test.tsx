// Dashboard tests (F3.1–F3.4). The load-bearing behaviours:
//   - the status panel renders the facts from /api/state, including an
//     UNVERIFIED plug read (the S3.1 signal must never be hidden);
//   - a non-destructive action POSTs directly (no modal);
//   - a destructive action opens a confirmation modal whose default focus is
//     Cancel; Confirm issues the POST, Cancel does not;
//   - a control 5xx renders the diagnostic inline next to the button;
//   - a 401 (on poll or action) escalates to re-auth (redirect), not a retry;
//   - a 5xx on the state poll shows a transient banner but keeps last state.

import { render, screen, fireEvent, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Dashboard } from "./Dashboard";

const STATE = {
  safety_state: "SHUTDOWN_ARM",
  jetson: { state: "stopped" },
  plugs: {
    arm: { on: false, verified: true, verified_at: "t" },
    jetson: { on: true, verified: false, verified_at: "t" },
  },
  camera: { up: false },
};

// A fetch mock that answers GET /api/state with STATE and lets each test
// decide what POST /api/control/* returns (default 200).
function installFetch(controlResponder?: (url: string) => { status: number; body: unknown }) {
  const fn = vi.fn((url: string, init?: RequestInit) => {
    if (!init || init.method !== "POST") {
      // The dashboard's recording mode selector fetches capabilities on mount;
      // answer it with an empty mode list so it doesn't consume the STATE body.
      if (url.startsWith("/api/capabilities")) {
        return Promise.resolve({ ok: true, status: 200, json: async () => ({ modes: [] }) });
      }
      // Slice 5: the recent-anomalies panel polls /api/anomalies; answer with an
      // empty log so it doesn't consume the STATE body.
      if (url.startsWith("/api/anomalies")) {
        return Promise.resolve({ ok: true, status: 200, json: async () => [] });
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => STATE,
      });
    }
    const r = controlResponder ? controlResponder(url) : { status: 200, body: {} };
    return Promise.resolve({
      ok: r.status >= 200 && r.status < 300,
      status: r.status,
      json: async () => r.body,
    });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.runOnlyPendingTimers();
  vi.useRealTimers();
});

// Render + flush the initial state poll so the panel is populated.
async function renderReady(fn = installFetch()) {
  render(<Dashboard />);
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
  return fn;
}

describe("status panel (F3.1)", () => {
  it("renders the facts, showing an unverified plug read distinctly", async () => {
    await renderReady();
    expect(screen.getByText("SHUTDOWN_ARM")).toBeTruthy();
    expect(screen.getByText("stopped")).toBeTruthy();
    expect(screen.getByText("off")).toBeTruthy(); // arm plug, verified off
    // jetson plug is on but unverified -> the unverified marker is shown.
    expect(screen.getByText("on (unverified)")).toBeTruthy();
    expect(screen.getByText("down")).toBeTruthy(); // camera
  });
});

// Click a control button and flush the resulting async state update.
async function clickAndFlush(label: string) {
  await act(async () => {
    fireEvent.click(screen.getByText(label));
    await vi.advanceTimersByTimeAsync(0);
  });
}

describe("non-destructive action (F3.2)", () => {
  it("POSTs directly with no modal", async () => {
    const fn = await renderReady();
    await clickAndFlush("Pause");
    const posted = fn.mock.calls.find((c) => c[1]?.method === "POST");
    expect(posted?.[0]).toBe("/api/control/pause");
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("destructive action confirmation (F3.3)", () => {
  it("opens a modal with Cancel focused; Confirm issues the POST", async () => {
    const fn = await renderReady();
    await act(async () => {
      fireEvent.click(screen.getByText("Power off arm"));
    });
    const dialog = screen.getByRole("dialog");
    expect(dialog).toBeTruthy();
    // No POST yet — the modal gates it.
    expect(fn.mock.calls.some((c) => c[1]?.method === "POST")).toBe(false);
    // Default focus is Cancel.
    expect((document.activeElement as HTMLElement)?.textContent).toBe("Cancel");

    await clickAndFlush("Confirm");
    const posted = fn.mock.calls.find((c) => c[1]?.method === "POST");
    expect(posted?.[0]).toBe("/api/control/shutdown/arm");
  });

  it("Cancel closes the modal without issuing the POST", async () => {
    const fn = await renderReady();
    await act(async () => {
      fireEvent.click(screen.getByText("Stop"));
    });
    await act(async () => {
      fireEvent.click(screen.getByText("Cancel"));
    });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(fn.mock.calls.some((c) => c[1]?.method === "POST")).toBe(false);
  });

  it("Escape dismisses the modal", async () => {
    await renderReady();
    await act(async () => {
      fireEvent.click(screen.getByText("Stop"));
    });
    expect(screen.getByRole("dialog")).toBeTruthy();
    await act(async () => {
      fireEvent.keyDown(window, { key: "Escape" });
    });
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("action failure (F3.2)", () => {
  it("renders the supervisor diagnostic inline on a 5xx", async () => {
    await renderReady(
      installFetch(() => ({
        status: 503,
        body: {
          detail: { error: "plug_verify_failed", message: "arm plug reported on after retries" },
        },
      })),
    );
    await clickAndFlush("Power on arm"); // non-destructive, no modal
    expect(screen.getByText("arm plug reported on after retries")).toBeTruthy();
  });
});

// The dashboard also fetches /api/capabilities on mount (Slice-4 recording
// mode selector). These tests target the /api/state poll specifically, so they
// answer the capabilities call with an empty mode list and key the state
// behaviour off the /api/state URL.
const EMPTY_CAPS = { modes: [] };

describe("re-auth escalation (F3.4)", () => {
  it("re-authenticates (reload) on a 401 from the state poll", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        // capabilities may 401 too; either way the state poll's 401 drives the
        // re-auth. Answer both with 401 to exercise the escalation.
        void url;
        return Promise.resolve({ ok: false, status: 401, json: async () => ({}) });
      }),
    );
    render(<Dashboard />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    // Auth-mechanism agnostic: the app re-navigates (reload) so the proxy can
    // re-auth; it must NOT navigate to any named login endpoint.
    expect(globalThis.__reloadCount).toBeGreaterThanOrEqual(1);
    expect(globalThis.__lastNavigation).toBeUndefined();
  });
});

describe("state poll resilience (F3.4)", () => {
  it("shows the banner on a 5xx but keeps the last-known state", async () => {
    let stateCalls = 0;
    const fn = vi.fn((url: string) => {
      if (url.startsWith("/api/capabilities")) {
        return Promise.resolve({ ok: true, status: 200, json: async () => EMPTY_CAPS });
      }
      if (url.startsWith("/api/anomalies")) {
        return Promise.resolve({ ok: true, status: 200, json: async () => [] });
      }
      // /api/state: first poll succeeds; subsequent polls 503.
      stateCalls += 1;
      if (stateCalls === 1)
        return Promise.resolve({ ok: true, status: 200, json: async () => STATE });
      return Promise.resolve({ ok: false, status: 503, json: async () => ({}) });
    });
    vi.stubGlobal("fetch", fn);
    render(<Dashboard />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByText("SHUTDOWN_ARM")).toBeTruthy();
    // Trigger the next poll (config.statePollMs = 2000).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(screen.getByText(/State unavailable/)).toBeTruthy();
    // Last-known state is still on screen.
    expect(screen.getByText("SHUTDOWN_ARM")).toBeTruthy();
  });
});

// --- Slice 4: recording controls + backlog (F4.5) ---------------------------

const CAPS = { modes: [{ resolution: "1280x720", fps: 30, bitrate_bps: 2500000 }] };

function installFetchWithRecording(
  recording: unknown,
  backlog: unknown,
  recResponder?: (url: string) => { status: number; body: unknown },
) {
  const stateBody = { ...STATE, recording, upload_backlog: backlog };
  const fn = vi.fn((url: string, init?: RequestInit) => {
    if (url.startsWith("/api/capabilities")) {
      return Promise.resolve({ ok: true, status: 200, json: async () => CAPS });
    }
    if (url.startsWith("/api/anomalies")) {
      return Promise.resolve({ ok: true, status: 200, json: async () => [] });
    }
    if (init?.method === "POST") {
      const r = recResponder ? recResponder(url) : { status: 200, body: { recording_id: "x" } };
      return Promise.resolve({
        ok: r.status >= 200 && r.status < 300,
        status: r.status,
        json: async () => r.body,
      });
    }
    return Promise.resolve({ ok: true, status: 200, json: async () => stateBody });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

describe("recording controls (F4.5)", () => {
  it("shows Start recording + mode selector when idle and POSTs on click", async () => {
    const fn = installFetchWithRecording(
      { state: "idle", active_recording_id: null, started_at: null },
      { count: 0, oldest_pending_seconds: null },
    );
    render(<Dashboard />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    const start = screen.getByText("Start recording");
    await act(async () => {
      fireEvent.click(start);
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(fn.mock.calls.some((c) => c[0] === "/api/recording/start")).toBe(true);
  });

  it("shows Stop recording when a recording is active", async () => {
    installFetchWithRecording(
      { state: "recording", active_recording_id: "rec-1", started_at: "t" },
      { count: 1, oldest_pending_seconds: 120 },
    );
    render(<Dashboard />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByText("Stop recording")).toBeTruthy();
    // Backlog indicator surfaces count + oldest age.
    expect(screen.getByText(/Upload backlog: 1 pending/)).toBeTruthy();
  });

  it("disables the control while starting", async () => {
    installFetchWithRecording(
      { state: "starting", active_recording_id: "rec-1", started_at: "t" },
      { count: 0, oldest_pending_seconds: null },
    );
    render(<Dashboard />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    const btn = screen.getByText("Starting…") as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });
});
