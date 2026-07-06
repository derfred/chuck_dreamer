// AnomalyLane tests (F5.2/F5.4): markers within the ring window render as
// positioned ticks with a type legend; out-of-window entries are dropped; a
// click seeks. Time is faked so "now" — and thus each marker's position — is
// deterministic.

import { render, screen, fireEvent, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AnomalyLane } from "./AnomalyLane";
import type { AnomalyLogEntry } from "../../shared/schemas";

const NOW = Date.parse("2026-07-06T12:00:00.000Z");
const WINDOW_MS = 120_000;

function entry(agoMs: number, type: string, over: Partial<AnomalyLogEntry> = {}): AnomalyLogEntry {
  return {
    ts: new Date(NOW - agoMs).toISOString(),
    type,
    anomaly_recording_id: null,
    cleared: false,
    payload: {},
    ...over,
  } as AnomalyLogEntry;
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date", "setInterval", "clearInterval"] });
  vi.setSystemTime(NOW);
});
afterEach(() => {
  vi.useRealTimers();
});

describe("anomaly markers", () => {
  it("renders a positioned tick per in-window anomaly, with the type in the legend", () => {
    const anomalies = [
      entry(30_000, "safe_box_violation"), // 90s into a 120s window -> 75%
      entry(60_000, "audio_spike"), // 60s -> 50%
    ];
    render(<AnomalyLane anomalies={anomalies} windowMs={WINDOW_MS} />);
    const marks = screen.getAllByRole("button");
    expect(marks).toHaveLength(2);
    // Positions map (ts - windowStart)/window; assert the computed left values.
    const lefts = marks.map((m) => (m as HTMLElement).style.left).sort();
    expect(lefts).toEqual(["50%", "75%"]);
    // Legend lists both present types.
    expect(screen.getByText("Safe-box violation")).toBeTruthy();
    expect(screen.getByText("Audio spike")).toBeTruthy();
  });

  it("drops anomalies older than the window", () => {
    const anomalies = [
      entry(200_000, "safe_box_violation"), // older than 120s -> dropped
      entry(10_000, "audio_spike"), // in window
    ];
    render(<AnomalyLane anomalies={anomalies} windowMs={WINDOW_MS} />);
    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(screen.queryByText("Safe-box violation")).toBeNull();
    expect(screen.getByText("Audio spike")).toBeTruthy();
  });

  it("calls onSeek with the entry when a marker is clicked", () => {
    const onSeek = vi.fn();
    const a = entry(20_000, "safe_box_violation");
    render(<AnomalyLane anomalies={[a]} windowMs={WINDOW_MS} onSeek={onSeek} />);
    act(() => {
      fireEvent.click(screen.getByRole("button"));
    });
    expect(onSeek).toHaveBeenCalledWith(a);
  });

  it("renders no legend when there are no in-window anomalies", () => {
    render(<AnomalyLane anomalies={[]} windowMs={WINDOW_MS} />);
    expect(screen.queryAllByRole("button")).toHaveLength(0);
  });
});
