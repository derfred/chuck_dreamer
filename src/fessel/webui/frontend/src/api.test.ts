// Backend API client tests (F2.1, F2.5). The load-bearing behaviours:
//   - a 401 from any /api call throws AuthError (so callers escalate to
//     re-auth instead of looping — F2.4/F2.5), while other failures throw a
//     plain Error;
//   - redirectToLogin() points at the proxy's start path with rd=<current-url>;
//   - modeToCanonical() matches the Python serialiser EXACTLY, since the signer
//     (backend) and the URL query must agree on the string.

import { describe, expect, it, vi } from "vitest";
import {
  AuthError,
  ControlError,
  fetchCapabilities,
  fetchMe,
  fetchState,
  fetchWhepUrl,
  modeToCanonical,
  postControl,
  redirectToLogin,
  SchemaError,
} from "./api";
import type { ModeTriplet } from "../../shared/schemas";

function mockFetch(status: number, body: unknown) {
  const res = {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
  };
  const fn = vi.fn().mockResolvedValue(res);
  vi.stubGlobal("fetch", fn);
  return fn;
}

const MODE: ModeTriplet = { resolution: "1280x720", fps: 30, bitrate_bps: 2_500_000 };

describe("modeToCanonical", () => {
  it("produces <W>x<H>@<fps>@<bitrate_bps> (matches mode_to_canonical)", () => {
    expect(modeToCanonical(MODE)).toBe("1280x720@30@2500000");
    expect(modeToCanonical({ resolution: "640x480", fps: 15, bitrate_bps: 1_000_000 })).toBe(
      "640x480@15@1000000",
    );
  });
});

describe("redirectToLogin", () => {
  it("navigates to the proxy start path with rd=<current href>", () => {
    redirectToLogin();
    const dest = new URL(globalThis.__lastNavigation as string, "https://webui.example.com");
    expect(dest.pathname).toBe("/oauth2/start");
    expect(dest.searchParams.get("rd")).toBe("https://webui.example.com/live");
  });
});

describe("fetchWhepUrl", () => {
  it("requests with path+canonical mode and returns the signed url", async () => {
    const fn = mockFetch(200, { url: "https://media/pi/whep?mode=1280x720@30@2500000&jwt=x" });
    const url = await fetchWhepUrl("pi", MODE);
    expect(url).toContain("/pi/whep");
    const called = fn.mock.calls[0][0] as string;
    expect(called).toContain("path=pi");
    // mode is URL-encoded; the @ separators survive as %40.
    expect(decodeURIComponent(called)).toContain("mode=1280x720@30@2500000");
  });

  it("throws AuthError on 401 (caller escalates to re-auth, does not retry)", async () => {
    mockFetch(401, {});
    await expect(fetchWhepUrl("pi", MODE)).rejects.toBeInstanceOf(AuthError);
  });

  it("throws a plain Error on other failures (a retryable network fault)", async () => {
    mockFetch(503, {});
    const err = await fetchWhepUrl("pi", MODE).catch((e) => e);
    expect(err).toBeInstanceOf(Error);
    expect(err).not.toBeInstanceOf(AuthError);
  });
});

describe("fetchMe / fetchCapabilities", () => {
  it("fetchMe returns the identity object", async () => {
    mockFetch(200, { user: "octocat", email: "o@example.com", groups: [] });
    await expect(fetchMe()).resolves.toEqual({
      user: "octocat",
      email: "o@example.com",
      groups: [],
    });
  });

  it("fetchMe throws AuthError on 401", async () => {
    mockFetch(401, {});
    await expect(fetchMe()).rejects.toBeInstanceOf(AuthError);
  });

  it("fetchCapabilities throws AuthError on 401", async () => {
    mockFetch(401, {});
    await expect(fetchCapabilities()).rejects.toBeInstanceOf(AuthError);
  });
});

describe("fetchState", () => {
  it("returns the validated StateResponse on 200", async () => {
    const body = { safety_state: "PAUSED", plugs: {}, camera: { up: true } };
    mockFetch(200, body);
    const got = await fetchState();
    // Owned fields survive; Zod fills the optional opaque `jetson` default.
    expect(got.safety_state).toBe("PAUSED");
    expect(got.camera).toEqual({ up: true });
    expect(got.jetson).toBeNull();
  });

  it("keeps the opaque jetson body untouched (passthrough, not validated)", async () => {
    const jetson = { state: "active", anything: { nested: 1 } };
    mockFetch(200, { safety_state: "RUNNING", jetson, plugs: {}, camera: { up: true } });
    const got = await fetchState();
    expect(got.jetson).toEqual(jetson); // permissive on the unowned shape
  });

  it("throws AuthError on 401 (caller escalates to re-auth)", async () => {
    mockFetch(401, {});
    await expect(fetchState()).rejects.toBeInstanceOf(AuthError);
  });

  it("throws a plain Error on 5xx (dashboard shows transient banner)", async () => {
    mockFetch(503, {});
    const err = await fetchState().catch((e) => e);
    expect(err).toBeInstanceOf(Error);
    expect(err).not.toBeInstanceOf(AuthError);
  });

  it("throws SchemaError + console.error on an off-contract body", async () => {
    // safety_state is not one of the enum values -> validation fails.
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    mockFetch(200, { safety_state: "NOT_A_STATE", plugs: {}, camera: {} });
    const err = await fetchState().catch((e) => e);
    expect(err).toBeInstanceOf(SchemaError);
    expect(spy).toHaveBeenCalled(); // hook for future monitoring
    spy.mockRestore();
  });

  it("rejects a plug entry that is the wrong shape", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    // `verified` must be boolean; a string is off-contract.
    mockFetch(200, {
      safety_state: "IDLE",
      plugs: { arm: { on: false, verified: "yes" } },
      camera: { up: null },
    });
    await expect(fetchState()).rejects.toBeInstanceOf(SchemaError);
    spy.mockRestore();
  });
});

describe("postControl", () => {
  it("POSTs /api/control/<action> and resolves on success", async () => {
    const fn = mockFetch(200, { ok: true });
    await expect(postControl("pause")).resolves.toBeUndefined();
    expect(fn.mock.calls[0][0]).toBe("/api/control/pause");
    expect(fn.mock.calls[0][1]).toMatchObject({ method: "POST" });
  });

  it("encodes the slash in shutdown/arm into the path", async () => {
    const fn = mockFetch(200, {});
    await postControl("shutdown/arm");
    expect(fn.mock.calls[0][0]).toBe("/api/control/shutdown/arm");
  });

  it("throws AuthError on 401 (does NOT retry)", async () => {
    mockFetch(401, {});
    await expect(postControl("stop")).rejects.toBeInstanceOf(AuthError);
  });

  it("throws ControlError carrying supervisor's diagnostic on a 5xx", async () => {
    mockFetch(503, {
      detail: { error: "plug_verify_failed", message: "arm plug reported on after retries" },
    });
    const err = await postControl("shutdown/arm").catch((e) => e);
    expect(err).toBeInstanceOf(ControlError);
    expect(err.message).toBe("arm plug reported on after retries");
    expect(err.status).toBe(503);
  });

  it("falls back to the error code when the body has no message", async () => {
    mockFetch(502, { error: "supervisor_unreachable" });
    const err = await postControl("resume").catch((e) => e);
    expect(err).toBeInstanceOf(ControlError);
    expect(err.message).toBe("supervisor_unreachable");
  });
});
