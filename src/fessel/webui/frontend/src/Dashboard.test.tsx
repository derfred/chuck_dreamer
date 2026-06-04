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

describe("re-auth escalation (F3.4)", () => {
  it("redirects to login on a 401 from the state poll", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 401, json: async () => ({}) }),
    );
    render(<Dashboard />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(globalThis.__lastNavigation).toContain("/oauth2/start");
  });
});

describe("state poll resilience (F3.4)", () => {
  it("shows the banner on a 5xx but keeps the last-known state", async () => {
    let calls = 0;
    const fn = vi.fn(() => {
      calls += 1;
      // First poll succeeds; subsequent polls 503.
      if (calls === 1) return Promise.resolve({ ok: true, status: 200, json: async () => STATE });
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
