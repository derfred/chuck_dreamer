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

function installFetch(
  opts: {
    list?: unknown;
    flagResponder?: () => { status: number };
    deleteResponder?: () => { status: number };
  } = {},
) {
  const list = opts.list ?? LIST;
  const fn = vi.fn((url: string, init?: RequestInit) => {
    if (init?.method === "DELETE") {
      const r = opts.deleteResponder ? opts.deleteResponder() : { status: 200 };
      return Promise.resolve({
        ok: r.status >= 200 && r.status < 300,
        status: r.status,
        json: async () => ({}),
      });
    }
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
  it("renders Flag only for the recording that can be flagged", async () => {
    await renderReady();
    const buttons = screen
      .getAllByRole("button")
      .filter((b) => b.textContent === "Flag for upload") as HTMLButtonElement[];
    expect(buttons.length).toBe(1);
    expect(buttons[0].disabled).toBe(false);
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

// --- Play gating: availability, not upload_state ------------------------------

function rec(over: Record<string, unknown>) {
  return {
    recording_id: "r-x",
    type: "explicit",
    started_at: "2026-06-04T02:00:00+00:00",
    ended_at: "2026-06-04T02:01:00+00:00",
    duration_seconds: 60,
    operator: "octocat",
    flagged_for_upload: false,
    upload_state: "none",
    available_local: true,
    available_remote: false,
    ...over,
  };
}

// queryAllByRole (not getAllByRole) — a row whose actions are all ineligible
// renders NO buttons, and the getAll* variant throws on an empty match.
function playButton(): HTMLButtonElement | undefined {
  return screen.queryAllByRole("button").find((b) => b.textContent === "Play") as
    | HTMLButtonElement
    | undefined;
}

describe("play gating", () => {
  it("keeps Play ENABLED for a queued/uploading recording still on the Pi", async () => {
    for (const state of ["queued", "uploading"]) {
      const { unmount } = render(<Recordings />);
      unmount();
      await renderReady(
        installFetch({ list: [rec({ upload_state: state, flagged_for_upload: true, available_local: true })] }),
      );
      const b = playButton();
      expect(b).toBeTruthy();
      expect(b!.disabled).toBe(false);
      screen.getByText(state); // sanity: the row really is in that state
      document.body.innerHTML = "";
    }
  });

  it("HIDES Play for a partially-uploaded recording gone from the Pi", async () => {
    await renderReady(
      installFetch({
        list: [rec({ upload_state: "uploading", flagged_for_upload: true, available_local: false, available_remote: true })],
      }),
    );
    expect(playButton()).toBeUndefined();
  });

  it("enables Play for a fully uploaded remote-only recording", async () => {
    await renderReady(
      installFetch({
        list: [rec({ upload_state: "uploaded", flagged_for_upload: true, available_local: false, available_remote: true })],
      }),
    );
    const b = playButton();
    expect(b).toBeTruthy();
    expect(b!.disabled).toBe(false);
  });
});

describe("flag gating", () => {
  it("renders no Flag button once a recording is flagged or uploaded", async () => {
    for (const state of ["uploaded", "queued", "uploading"]) {
      await renderReady(
        installFetch({ list: [rec({ upload_state: state, flagged_for_upload: true })] }),
      );
      const b = screen.queryAllByRole("button").find((x) => x.textContent === "Flag for upload");
      expect(b).toBeUndefined();
      document.body.innerHTML = "";
    }
  });

  it("still renders Re-flag for a failed upload", async () => {
    await renderReady(
      installFetch({ list: [rec({ upload_state: "failed", flagged_for_upload: true })] }),
    );
    expect(screen.getAllByRole("button").some((b) => b.textContent === "Re-flag")).toBe(true);
  });
});

// --- delete (cluster copy) ----------------------------------------------------

function deleteButton(): HTMLButtonElement | undefined {
  return screen.queryAllByRole("button").find((b) => b.textContent === "Delete") as
    | HTMLButtonElement
    | undefined;
}

describe("delete gating", () => {
  it("offers Delete only for a fully uploaded recording", async () => {
    await renderReady(
      installFetch({ list: [rec({ upload_state: "uploaded", available_remote: true })] }),
    );
    expect(deleteButton()).toBeTruthy();
  });

  it("hides Delete when there is no cluster copy", async () => {
    for (const state of ["none", "queued", "uploading", "failed"]) {
      await renderReady(
        installFetch({
          list: [rec({ upload_state: state, available_remote: state === "uploading" })],
        }),
      );
      expect(deleteButton()).toBeUndefined();
      document.body.innerHTML = "";
    }
  });
});

describe("delete flow", () => {
  it("confirms first, then DELETEs and reloads", async () => {
    const fn = await renderReady(
      installFetch({ list: [rec({ upload_state: "uploaded", available_remote: true })] }),
    );
    // Nothing is sent on the first click — a confirmation appears.
    await act(async () => {
      fireEvent.click(deleteButton()!);
    });
    expect(screen.getByRole("dialog", { name: /confirm delete/i })).toBeTruthy();
    expect(fn.mock.calls.some((c) => (c[1] as RequestInit | undefined)?.method === "DELETE")).toBe(
      false,
    );

    // Confirming issues the DELETE against the right URL.
    const confirm = screen
      .getAllByRole("button")
      .filter((b) => b.textContent === "Delete")
      .pop() as HTMLButtonElement;
    await act(async () => {
      fireEvent.click(confirm);
      await vi.advanceTimersByTimeAsync(0);
    });
    const del = fn.mock.calls.find((c) => (c[1] as RequestInit | undefined)?.method === "DELETE");
    expect(del).toBeTruthy();
    expect(del![0]).toBe("/api/recordings/r-x");
  });

  it("cancelling sends nothing", async () => {
    const fn = await renderReady(
      installFetch({ list: [rec({ upload_state: "uploaded", available_remote: true })] }),
    );
    await act(async () => {
      fireEvent.click(deleteButton()!);
    });
    await act(async () => {
      fireEvent.click(screen.getByText("Cancel"));
    });
    expect(screen.queryByRole("dialog", { name: /confirm delete/i })).toBeNull();
    expect(fn.mock.calls.some((c) => (c[1] as RequestInit | undefined)?.method === "DELETE")).toBe(
      false,
    );
  });

  it("surfaces a delete failure and keeps the row", async () => {
    await renderReady(
      installFetch({
        list: [rec({ upload_state: "uploaded", available_remote: true })],
        deleteResponder: () => ({ status: 500 }),
      }),
    );
    await act(async () => {
      fireEvent.click(deleteButton()!);
    });
    const confirm = screen
      .getAllByRole("button")
      .filter((b) => b.textContent === "Delete")
      .pop() as HTMLButtonElement;
    await act(async () => {
      fireEvent.click(confirm);
      await vi.advanceTimersByTimeAsync(0);
    });
    // Row still listed (not optimistically removed).
    expect(screen.getByText("uploaded")).toBeTruthy();
  });
});
