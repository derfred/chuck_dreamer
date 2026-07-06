// MonitorVideo tests: the Ring/Live source toggle and its bandwidth framing.
//   - defaults to Ring (cheap) with the low-bandwidth cost tag + a Go Live CTA;
//   - selecting Live flips to the dedicated-stream cost tag and mounts the live
//     stage (which opens the WebRTC session);
//   - switching back to Ring tears the live session down (the PC is closed).
//
// jsdom has no HLS/MSE (useHls no-ops) and no WebRTC, so we install the fake
// RTC peer + a WHEP-shaped fetch; the assertions stay on the toggle/cost DOM.

import { render, screen, fireEvent, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MonitorVideo } from "./MonitorVideo";
import { createdPcs, installFakeRtc } from "./test/fakeRtc";

function whepResponse(status: number, location = "/whep/1") {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (h: string) => (h === "Location" ? location : null) },
    json: async () => ({}),
    text: async () => "v=0\r\n",
  };
}

beforeEach(() => {
  vi.useFakeTimers();
  installFakeRtc();
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(whepResponse(201)));
});
afterEach(() => {
  vi.runOnlyPendingTimers();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function tick(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

describe("source toggle", () => {
  it("defaults to Ring with the low-bandwidth cost tag and a Go Live CTA", () => {
    render(<MonitorVideo anomalies={[]} />);
    const ring = screen.getByRole("button", { name: "Ring" });
    const live = screen.getByRole("button", { name: "Live" });
    expect(ring.getAttribute("aria-pressed")).toBe("true");
    expect(live.getAttribute("aria-pressed")).toBe("false");
    expect(screen.getByText(/low bandwidth/)).toBeTruthy();
    expect(screen.getByText("Go Live ⚡")).toBeTruthy();
  });

  it("selecting Live shows the dedicated-stream cost tag and opens a WebRTC session", async () => {
    render(<MonitorVideo anomalies={[]} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Live" }));
    });
    await tick(0); // drain the WHEP POST
    expect(screen.getByRole("button", { name: "Live" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByText(/dedicated stream/)).toBeTruthy();
    // The live stage mounted and started a peer connection.
    expect(createdPcs.length).toBeGreaterThanOrEqual(1);
  });

  it("switching Live -> Ring tears the live session down (PC closed)", async () => {
    render(<MonitorVideo anomalies={[]} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Live" }));
    });
    await tick(0);
    const pc = createdPcs[createdPcs.length - 1];
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Ring" }));
    });
    await tick(0);
    // Back on Ring: cheap tag restored, and the WebRTC PC was closed on unmount.
    expect(screen.getByText(/low bandwidth/)).toBeTruthy();
    expect(pc.closed).toBe(true);
  });
});
