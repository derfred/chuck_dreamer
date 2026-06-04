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

// jsdom throws "Not implemented: navigation" on a real location assignment.
// redirectToLogin() sets window.location.href; capture it instead so the
// re-auth test can assert where it tried to go.
declare global {
  // `var` is required here: global augmentation only works with `var`, not
  // let/const. (tseslint's no-var doesn't flag declare-global, so no disable.)
  var __lastNavigation: string | undefined;
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
  },
});

installFakeRtc();

afterEach(() => {
  globalThis.__lastNavigation = undefined;
  _href = INITIAL_HREF;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  // Re-install the RTC fake: a test that stubbed globals via vi may have
  // cleared it.
  installFakeRtc();
});
