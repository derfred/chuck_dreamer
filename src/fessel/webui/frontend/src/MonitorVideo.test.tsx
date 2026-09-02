// MonitorVideo tests: the Stream On/Off toggle and its bandwidth framing.
//   - defaults to OFF (0 bandwidth) — opening Monitor must not open a stream;
//   - turning the stream on opens a WebRTC session (the PC is created);
//   - turning it back off tears the live session down (the PC is closed).
//
// The bar carries no stream cost badge any more (the toggle's own pressed
// state says whether the stream is on); its right-hand side is the recording
// control, so the on/off assertions key off aria-pressed + the PC lifecycle.
//
// jsdom has no WebRTC, so we install the fake RTC peer + a WHEP-shaped fetch;
// the assertions stay on the toggle/cost DOM + PC lifecycle.

import { render, screen, fireEvent, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { config } from "./config";
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
  it("defaults to Stream Off (0 bandwidth) and opens NO WebRTC session", async () => {
    render(<MonitorVideo />);
    // Drain the resting stage's snapshot-meta poll (fires on mount) so its
    // state update lands inside act() rather than after the test returns.
    await tick(0);
    expect(screen.getByRole("button", { name: "Stream Off" }).getAttribute("aria-pressed")).toBe(
      "true",
    );
    expect(screen.getByRole("button", { name: "Stream On" }).getAttribute("aria-pressed")).toBe(
      "false",
    );
    // The recording control shares the header bar and is independent of the
    // stream: it is available with the stream off.
    expect(screen.getByText("Record…")).toBeTruthy();
    // The resting stage offers to turn the stream on; no stream is running.
    expect(screen.getByText("Turn stream on")).toBeTruthy();
    expect(createdPcs.length).toBe(0);
  });

  it("turning the stream on opens a WebRTC session", async () => {
    render(<MonitorVideo />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Stream On" }));
    });
    await tick(0); // drain the WHEP POST
    expect(screen.getByRole("button", { name: "Stream On" }).getAttribute("aria-pressed")).toBe(
      "true",
    );
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
    // Back off: the toggle reads Off again and the WebRTC PC was closed on unmount.
    expect(screen.getByRole("button", { name: "Stream Off" }).getAttribute("aria-pressed")).toBe(
      "true",
    );
    expect(pc.closed).toBe(true);
  });
});

describe("resting-stage freeze-frame", () => {
  it("shows no snapshot chip when the backend has none yet", async () => {
    render(<MonitorVideo />);
    await tick(0);
    expect(screen.getByText("Stream is off — no bandwidth in use.")).toBeTruthy();
    expect(screen.queryByText(/ago — not live/)).toBeNull();
  });

  it("shows the snapshot age chip and a click-to-enlarge lightbox once a snapshot exists", async () => {
    const capturedAt = new Date().toISOString();
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockImplementation((url: string) => {
      if (url.includes("/api/snapshot/meta")) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({ available: true, captured_at: capturedAt }),
        });
      }
      return Promise.resolve(whepResponse(201));
    });

    render(<MonitorVideo />);
    await tick(0);
    expect(screen.getByText(/ago — not live/)).toBeTruthy();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "View full-size snapshot" }));
    });
    expect(screen.getByRole("dialog", { name: "Full-size snapshot" })).toBeTruthy();

    await act(async () => {
      fireEvent.keyDown(window, { key: "Escape" });
    });
    expect(screen.queryByRole("dialog", { name: "Full-size snapshot" })).toBeNull();
    await tick(config.snapshotPollMs); // drain the next scheduled poll before unmount
  });

  // The label dates the FRAME, not the push: a Pi that keeps re-pushing one
  // stale cached frame reports a capture time that stays put, and the chip
  // must show that growing age (and go amber) rather than resetting to 0s.
  it("ages the chip from the capture time, not from when the push landed", async () => {
    const capturedAt = new Date(Date.now() - 90_000).toISOString();
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockImplementation((url: string) => {
      if (url.includes("/api/snapshot/meta")) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({ available: true, captured_at: capturedAt }),
        });
      }
      return Promise.resolve(whepResponse(201));
    });

    render(<MonitorVideo />);
    await tick(0);
    expect(screen.getByText(/1m ago — not live/)).toBeTruthy();
    await tick(config.snapshotPollMs); // drain the next scheduled poll before unmount
  });
});
