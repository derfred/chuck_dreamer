// /recordings view tests (F4.2–F4.4): the table renders the list, Play opens a
// player modal, and Flag-for-upload is gated by upload state + applies
// optimistically (reverting on error).

import { render, screen, fireEvent, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Recordings } from "./Recordings";

const LIST = [
  {
    recording_id: "r-new",
    type: "explicit",
    started_at: "2026-06-04T02:00:00+00:00",
    ended_at: "2026-06-04T02:01:00+00:00",
    duration_seconds: 60,
    operator: "octocat",
    flagged_for_upload: false,
    upload_state: "none",
    available_local: true,
    available_remote: false,
  },
  {
    recording_id: "r-uploaded",
    type: "anomaly",
    started_at: "2026-06-04T01:00:00+00:00",
    ended_at: "2026-06-04T01:05:00+00:00",
    duration_seconds: 300,
    operator: "octocat",
    flagged_for_upload: true,
    upload_state: "uploaded",
    available_local: true,
    available_remote: true,
  },
];

function installFetch(opts: { list?: unknown; flagResponder?: () => { status: number } } = {}) {
  const list = opts.list ?? LIST;
  const fn = vi.fn((url: string, init?: RequestInit) => {
    if (init?.method === "POST" && url === "/api/recording/flag-upload") {
      const r = opts.flagResponder ? opts.flagResponder() : { status: 200 };
      return Promise.resolve({
        ok: r.status >= 200 && r.status < 300,
        status: r.status,
        json: async () => ({}),
      });
    }
    // GET /api/recordings
    return Promise.resolve({ ok: true, status: 200, json: async () => list });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  vi.runOnlyPendingTimers();
  vi.useRealTimers();
});

async function renderReady(fn = installFetch()) {
  render(<Recordings />);
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
  return fn;
}

describe("recordings list (F4.2)", () => {
  it("renders a row per recording with operator + upload state", async () => {
    await renderReady();
    // Two rows, both by octocat.
    expect(screen.getAllByText("octocat", { selector: "td" }).length).toBe(2);
    expect(screen.getByText("none")).toBeTruthy(); // r-new
    expect(screen.getByText("uploaded")).toBeTruthy(); // r-uploaded
  });

  it("shows an empty state when there are no recordings", async () => {
    await renderReady(installFetch({ list: [] }));
    expect(screen.getByText(/No recordings yet/)).toBeTruthy();
  });

  it("shows a type badge discriminating explicit vs anomaly (F5.3)", async () => {
    await renderReady();
    expect(screen.getByText("explicit")).toBeTruthy(); // r-new
    expect(screen.getByText("anomaly")).toBeTruthy(); // r-uploaded
  });
});

// The enabled flag button is the one whose row is in upload_state "none"/"failed";
// the uploaded row's button is disabled.
function enabledFlagButton(): HTMLButtonElement {
  const buttons = screen
    .getAllByRole("button")
    .filter((b) => b.textContent === "Flag for upload") as HTMLButtonElement[];
  const enabled = buttons.find((b) => !b.disabled);
  if (!enabled) throw new Error("no enabled Flag button");
  return enabled;
}

describe("flag-for-upload gating (F4.4)", () => {
  it("enables Flag for an unflagged 'none' recording and disables it once uploaded", async () => {
    await renderReady();
    const buttons = screen
      .getAllByRole("button")
      .filter((b) => b.textContent === "Flag for upload") as HTMLButtonElement[];
    // One enabled (r-new, none) and one disabled (r-uploaded, uploaded).
    expect(buttons.some((b) => !b.disabled)).toBe(true);
    expect(buttons.some((b) => b.disabled)).toBe(true);
  });

  it("optimistically flags then reverts on error", async () => {
    const fn = await renderReady(installFetch({ flagResponder: () => ({ status: 503 }) }));
    await act(async () => {
      fireEvent.click(enabledFlagButton());
      await vi.advanceTimersByTimeAsync(0);
    });
    // POST was issued.
    expect(fn.mock.calls.some((c) => c[0] === "/api/recording/flag-upload")).toBe(true);
    // On error the optimistic "queued" reverts back to "none" (still listed).
    expect(screen.getByText("none")).toBeTruthy();
  });
});

describe("playback (F4.3)", () => {
  it("opens a player modal on Play and closes it", async () => {
    await renderReady();
    const playButtons = screen.getAllByText("Play");
    await act(async () => {
      fireEvent.click(playButtons[0]);
    });
    expect(screen.getByRole("dialog")).toBeTruthy();
    await act(async () => {
      fireEvent.keyDown(window, { key: "Escape" });
    });
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
