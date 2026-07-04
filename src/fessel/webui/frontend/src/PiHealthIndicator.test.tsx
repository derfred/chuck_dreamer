// Corner Pi-health indicator tests (health-check spec §4). Load-bearing
// behaviours:
//   - the rollup `light` drives the dot + terse label;
//   - clicking opens a drill-down listing the facts in backend order;
//   - masked (unknown) facts render de-emphasised but still show their detail;
//   - a 401 on the poll escalates to re-auth (redirect), not a retry;
//   - if the last successful fetch ages past the staleness threshold, the light
//     degrades to grey/"stale" regardless of the last-known colour (§4.4).

import { render, screen, fireEvent, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PiHealthIndicator } from "./PiHealthIndicator";
import { config } from "./config";

const HEALTH_RED = {
  light: "red",
  generated_at: "2026-07-01T10:00:00Z",
  facts: [
    {
      id: "pi_control",
      label: "Pi control",
      state: "red",
      detail: "unreachable — no contact this session",
      observed_at: "2026-07-01T10:00:00Z",
    },
    {
      id: "relay",
      label: "live relay",
      state: "green",
      detail: "healthy",
      observed_at: "2026-07-01T10:00:00Z",
    },
    {
      id: "video",
      label: "video process",
      state: "unknown",
      detail: "unknown — Pi unreachable",
      observed_at: "2026-07-01T10:00:00Z",
    },
    {
      id: "camera",
      label: "camera",
      state: "unknown",
      detail: "unknown — Pi unreachable",
      observed_at: "2026-07-01T10:00:00Z",
    },
    {
      id: "version_sync",
      label: "Component versions",
      state: "unknown",
      detail: "unknown — can't read Pi version",
      observed_at: "2026-07-01T10:00:00Z",
    },
  ],
};

const HEALTH_GREEN = {
  light: "green",
  generated_at: "2026-07-01T10:00:00Z",
  facts: [
    {
      id: "pi_control",
      label: "Pi control",
      state: "green",
      detail: "reachable — 2s ago",
      observed_at: "2026-07-01T10:00:00Z",
    },
  ],
};

function installFetch(responder: () => { status: number; body: unknown }) {
  const fn = vi.fn(() => {
    const r = responder();
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
  vi.unstubAllGlobals();
});

async function renderReady(body: unknown = HEALTH_GREEN, status = 200) {
  const fn = installFetch(() => ({ status, body }));
  render(<PiHealthIndicator />);
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
  return fn;
}

describe("corner light", () => {
  it("renders the rollup label from /api/health/pi", async () => {
    await renderReady(HEALTH_GREEN);
    expect(screen.getByText("Pi: OK")).toBeTruthy();
  });

  it("opens a drill-down listing the facts, masked ones showing their detail", async () => {
    await renderReady(HEALTH_RED);
    expect(screen.getByText("Pi: problem")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Pi: problem/ }));
    // Root-cause fact + masked facts' details are all visible. "unknown — Pi
    // unreachable" appears for both video and camera, so match all.
    expect(screen.getByText("unreachable — no contact this session")).toBeTruthy();
    expect(screen.getAllByText("unknown — Pi unreachable")).toHaveLength(2);
    expect(screen.getByText("unknown — can't read Pi version")).toBeTruthy();
  });
});

describe("re-auth on 401", () => {
  it("reloads (hands off to the proxy) rather than retrying", async () => {
    // window.location.reload is stubbed in test/setup.ts as a reload counter.
    globalThis.__reloadCount = 0;
    sessionStorage.clear();
    await renderReady({}, 401);
    // A 401 hands off to the proxy login; the light stays "checking…" (grey).
    expect(screen.getByText("Pi: checking…")).toBeTruthy();
    expect(globalThis.__reloadCount).toBeGreaterThanOrEqual(1);
  });
});

describe("frontend staleness override (§4.4)", () => {
  it("degrades a green light to grey/stale once the last success ages out", async () => {
    // First poll succeeds green, then all further fetches reject (backend gone).
    let fail = false;
    installFetch(() => {
      if (fail) throw new Error("network");
      return { status: 200, body: HEALTH_GREEN };
    });
    render(<PiHealthIndicator />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByText("Pi: OK")).toBeTruthy();
    fail = true;
    // Advance past the staleness threshold; the ticking clock re-evaluates.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(config.healthStaleMs + 1500);
    });
    expect(screen.getByText(/Pi: stale/)).toBeTruthy();
    expect(screen.queryByText("Pi: OK")).toBeNull();
  });
});
