// RecordModal tests (ring-buffer look-back reframe): the dialog has no quality
// picker, presets + anomaly ticks set the look-back, the upload toggle is
// carried, and confirm reports {lookbackSeconds, uploadWhenDone}.

import { render, screen, fireEvent, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RecordModal } from "./RecordModal";
import type { AnomalyLogEntry } from "../../shared/schemas";

const NOW = Date.parse("2026-07-07T12:00:00.000Z");

function anomaly(agoMs: number, type: string): AnomalyLogEntry {
  return {
    ts: new Date(NOW - agoMs).toISOString(),
    type,
    anomaly_recording_id: null,
    cleared: false,
    payload: {},
  } as AnomalyLogEntry;
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(NOW);
});
afterEach(() => vi.useRealTimers());

function renderModal(props: Partial<React.ComponentProps<typeof RecordModal>> = {}) {
  const onConfirm = vi.fn();
  const onCancel = vi.fn();
  render(
    <RecordModal
      maxLookbackSeconds={120}
      anomalies={props.anomalies ?? []}
      busy={false}
      onCancel={onCancel}
      onConfirm={onConfirm}
    />,
  );
  return { onConfirm, onCancel };
}

describe("record dialog", () => {
  it("has no quality/resolution picker (recording resolution is a deploy setting)", () => {
    renderModal();
    expect(screen.queryByText(/quality/i)).toBeNull();
    expect(screen.queryByRole("combobox")).toBeNull();
  });

  it("defaults to no look-back and reports 0 on confirm", () => {
    const { onConfirm } = renderModal();
    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    expect(onConfirm).toHaveBeenCalledWith({ lookbackSeconds: 0, uploadWhenDone: false });
  });

  it("a preset sets the look-back", () => {
    const { onConfirm } = renderModal();
    fireEvent.click(screen.getByText("−1:00")); // 60s preset
    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    expect(onConfirm).toHaveBeenCalledWith({ lookbackSeconds: 60, uploadWhenDone: false });
  });

  it("the upload toggle is carried through", () => {
    const { onConfirm } = renderModal();
    fireEvent.click(screen.getByLabelText("Upload when done"));
    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    expect(onConfirm).toHaveBeenCalledWith({ lookbackSeconds: 0, uploadWhenDone: true });
  });

  it("clicking an anomaly tick anchors the look-back to that anomaly's age", () => {
    // A safe-box violation 40s ago -> clicking its tick sets lookback ~40s.
    const { onConfirm } = renderModal({ anomalies: [anomaly(40_000, "safe_box_violation")] });
    fireEvent.click(screen.getByRole("button", { name: /Anchor look-back to Safe-box violation/ }));
    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    const arg = onConfirm.mock.calls[0][0];
    expect(arg.lookbackSeconds).toBeCloseTo(40, 0);
  });

  it("Escape cancels", () => {
    const { onCancel } = renderModal();
    act(() => {
      fireEvent.keyDown(window, { key: "Escape" });
    });
    expect(onCancel).toHaveBeenCalled();
  });
});
