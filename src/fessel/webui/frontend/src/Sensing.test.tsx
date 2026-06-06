// Slice-5 sensing surfaces (F5.1/F5.2): the dashboard sensing strip renders the
// live activity score, audio level, and health pills from /api/state, and the
// recent-anomalies panel renders the /api/anomalies log with a recording link.

import { render, screen, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Dashboard } from "./Dashboard";

const STATE = {
  safety_state: "PAUSED",
  jetson: null,
  plugs: {},
  camera: { up: true },
  recording: { state: "idle" },
  upload_backlog: { count: 0 },
  vision: { activity_score_ema: 0.42, last_frame_at: "2026-06-05T00:00:00Z", healthy: true },
  audio: { level_db: -22.0, healthy: false },
  recent_anomalies: [],
};

const ANOMALIES = [
  {
    ts: "2026-06-05T00:00:01Z",
    type: "safe_box_violation",
    anomaly_recording_id: "rec-1",
    cleared: false,
    payload: { outside_fraction: 0.2 },
  },
  {
    ts: "2026-06-05T00:00:05Z",
    type: "audio_spike",
    anomaly_recording_id: null,
    cleared: true,
    payload: { peak_db: -3 },
  },
];

function installFetch(anomalies: unknown = ANOMALIES) {
  const fn = vi.fn((url: string) => {
    if (url.startsWith("/api/capabilities")) {
      return Promise.resolve({ ok: true, status: 200, json: async () => ({ modes: [] }) });
    }
    if (url.startsWith("/api/anomalies")) {
      return Promise.resolve({ ok: true, status: 200, json: async () => anomalies });
    }
    return Promise.resolve({ ok: true, status: 200, json: async () => STATE });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  vi.runOnlyPendingTimers();
  vi.useRealTimers();
});

async function renderReady() {
  installFetch();
  render(<Dashboard />);
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
}

describe("sensing strip (F5.1)", () => {
  it("shows the activity score, audio level, and health pills", async () => {
    await renderReady();
    // Activity score value (3 dp).
    expect(screen.getByText("0.420")).toBeTruthy();
    // Audio level meter shows the dB value.
    expect(screen.getByText(/-22.0 dB/)).toBeTruthy();
    // Health pills: vision healthy (●), audio degraded (○).
    expect(screen.getByText("Vision ●")).toBeTruthy();
    expect(screen.getByText("Audio ○")).toBeTruthy();
  });
});

describe("recent anomalies panel (F5.2)", () => {
  it("lists anomalies with labels, cleared state, and a recording link", async () => {
    await renderReady();
    expect(screen.getByText("Safe-box violation")).toBeTruthy();
    expect(screen.getByText("Audio spike")).toBeTruthy();
    // The audio spike is cleared.
    expect(screen.getByText("(cleared)")).toBeTruthy();
    // The safe-box violation links to its recording; the audio spike (no id) does not.
    const links = screen.getAllByText("View recording →");
    expect(links.length).toBe(1);
  });

  it("shows an empty state when there are no anomalies", async () => {
    installFetch([]);
    render(<Dashboard />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByText("No anomalies recorded.")).toBeTruthy();
  });
});
