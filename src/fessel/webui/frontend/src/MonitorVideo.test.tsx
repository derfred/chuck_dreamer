// MonitorVideo tests: the Stream On/Off toggle and its bandwidth framing.
//   - defaults to OFF (0 bandwidth) — opening Monitor must not open a stream;
//   - turning the stream on flips to the live cost tag and opens a WebRTC
//     session (the PC is created);
//   - turning it back off tears the live session down (the PC is closed).
//
// jsdom has no WebRTC, so we install the fake RTC peer + a WHEP-shaped fetch;
// the assertions stay on the toggle/cost DOM + PC lifecycle.

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

describe("stream on/off toggle", () => {
  it("defaults to Stream Off (0 bandwidth) and opens NO WebRTC session", () => {
    render(<MonitorVideo />);
    expect(screen.getByRole("button", { name: "Stream Off" }).getAttribute("aria-pressed")).toBe(
      "true",
    );
    expect(screen.getByRole("button", { name: "Stream On" }).getAttribute("aria-pressed")).toBe(
      "false",
    );
    expect(screen.getByText(/0 bandwidth/)).toBeTruthy();
    // The resting stage offers to turn the stream on; no stream is running.
    expect(screen.getByText("Turn stream on")).toBeTruthy();
    expect(createdPcs.length).toBe(0);
  });

  it("turning the stream on shows the live cost tag and opens a WebRTC session", async () => {
    render(<MonitorVideo />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Stream On" }));
    });
    await tick(0); // drain the WHEP POST
    expect(screen.getByRole("button", { name: "Stream On" }).getAttribute("aria-pressed")).toBe(
      "true",
    );
    expect(screen.getByText(/WebRTC · live/)).toBeTruthy();
    expect(createdPcs.length).toBeGreaterThanOrEqual(1);
  });

  it("the resting 'Turn stream on' button also starts the stream", async () => {
    render(<MonitorVideo />);
    await act(async () => {
      fireEvent.click(screen.getByText("Turn stream on"));
    });
    await tick(0);
    expect(createdPcs.length).toBeGreaterThanOrEqual(1);
  });

  it("turning the stream off tears the live session down (PC closed)", async () => {
    render(<MonitorVideo />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Stream On" }));
    });
    await tick(0);
    const pc = createdPcs[createdPcs.length - 1];
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Stream Off" }));
    });
    await tick(0);
    // Back off: 0-bandwidth tag restored, and the WebRTC PC was closed on unmount.
    expect(screen.getByText(/0 bandwidth/)).toBeTruthy();
    expect(pc.closed).toBe(true);
  });
});
