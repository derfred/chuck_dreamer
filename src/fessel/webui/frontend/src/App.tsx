// App shell + lightweight routing. Two routes: `/` (Monitor — live-view +
// controls) and `/footage` (ring buffer + recordings). No router dependency —
// the app is a two-route shell; a path-based switch is enough.
//
// The shell trusts that oauth2-proxy already authenticated the request that
// served this page (the app is served from behind the proxy). It probes
// /api/me on load purely to detect a lapsed session early (a 401 hands off to
// the proxy's login flow rather than rendering a broken shell); the operator
// identity itself is not displayed in the chrome.

import { useEffect, useState } from "react";
import { AuthError, fetchMe, reauthenticate } from "./api";
import { Footage } from "./Footage";
import { Monitor } from "./Monitor";
import { navigate } from "./nav";
import { PiHealthIndicator } from "./PiHealthIndicator";

function usePath(): string {
  const [path, setPath] = useState<string>(window.location.pathname);
  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  return path;
}

const NAV: { path: string; label: string }[] = [
  { path: "/", label: "Monitor" },
  { path: "/footage", label: "Footage" },
];

export function App() {
  const path = usePath();

  useEffect(() => {
    fetchMe().catch((e) => {
      // A 401 on the shell's own identity probe means the proxy session is
      // gone; bounce to login rather than render an unauthenticated shell.
      if (e instanceof AuthError) reauthenticate();
    });
  }, []);

  return (
    <div style={{ fontFamily: "sans-serif" }}>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 18,
          padding: "10px 16px",
          borderBottom: "1px solid #ddd",
        }}
      >
        <span style={{ display: "flex", alignItems: "center", gap: 9, fontWeight: 700 }}>
          <span
            style={{
              width: 9,
              height: 9,
              borderRadius: "50%",
              background: "#17967a",
              boxShadow: "0 0 0 3px rgba(23,150,122,0.22)",
            }}
            aria-hidden
          />
          Fessel
        </span>
        <nav style={{ display: "flex", gap: 4 }}>
          {NAV.map((n) => {
            const active = n.path === "/" ? path === "/" : path.startsWith(n.path);
            return (
              <a
                key={n.path}
                href={n.path}
                aria-current={active ? "page" : undefined}
                onClick={(e) => (e.preventDefault(), navigate(n.path))}
                style={{
                  fontSize: 14,
                  fontWeight: 550,
                  textDecoration: "none",
                  color: active ? "#16211f" : "#5a6b68",
                  background: active ? "#eef1f3" : "transparent",
                  padding: "7px 13px",
                  borderRadius: 8,
                }}
              >
                {n.label}
              </a>
            );
          })}
        </nav>
        <span style={{ flex: 1 }} />
        <PiHealthIndicator />
      </header>

      {route(path)}
    </div>
  );
}

function route(path: string) {
  if (path.startsWith("/footage")) return <Footage />;
  return <Monitor />;
}
