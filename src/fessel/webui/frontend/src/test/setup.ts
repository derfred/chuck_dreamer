// Global test setup. Vitest + jsdom give us window/document; this file fills
// the gaps jsdom leaves for our code: WebRTC (RTCPeerConnection) and a
// navigable window.location. Per-test fetch stubbing and timer control live in
// the individual test files.

import { afterEach, vi } from "vitest";
import { installFakeRtc } from "./fakeRtc";

// HTMLMediaElement.currentTime is read by the stall detector. jsdom backs the
// <video> element but leaves currentTime at 0 with no decode pipeline; tests
// set it directly, so make it a plain writable property.
Object.defineProperty(HTMLMediaElement.prototype, "currentTime", {
  configurable: true,
  get(this: { _ct?: number }) {
    return this._ct ?? 0;
  },
  set(this: { _ct?: number }, v: number) {
    this._ct = v;
  },
});

// jsdom throws "Not implemented: navigation" on a real location assignment or
// reload(). The app is auth-mechanism agnostic: on a 401 it re-navigates via
// window.location.reload() so the auth proxy can re-authenticate (it names no
// login endpoint). Capture reload() (and href, still used elsewhere) so the
// re-auth test can assert the app tried to re-navigate.
declare global {
  // `var` is required here: global augmentation only works with `var`, not
  // let/const. (tseslint's no-var doesn't flag declare-global, so no disable.)
  var __lastNavigation: string | undefined;
  var __reloadCount: number;
}
const INITIAL_HREF = "https://webui.example.com/live";
let _href = INITIAL_HREF;
Object.defineProperty(window, "location", {
  configurable: true,
  value: {
    pathname: "/live",
    get href() {
      return _href;
    },
    set href(v: string) {
      _href = v;
      globalThis.__lastNavigation = v;
    },
    reload() {
      globalThis.__reloadCount = (globalThis.__reloadCount ?? 0) + 1;
    },
  },
});

installFakeRtc();

afterEach(() => {
  globalThis.__lastNavigation = undefined;
  globalThis.__reloadCount = 0;
  _href = INITIAL_HREF;
  // The re-auth loop guard lives in sessionStorage; clear it between tests so
  // one test's reauthenticate() doesn't trip the next test's guard.
  sessionStorage.clear();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  // Re-install the RTC fake: a test that stubbed globals via vi may have
  // cleared it.
  installFakeRtc();
});
