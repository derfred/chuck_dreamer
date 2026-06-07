// App shell + lightweight routing (F2.1). Two routes: `/` (dashboard
// placeholder, filled in later slices) and `/live`. No router dependency —
// the app is tiny; path-based switch is enough.
//
// The shell trusts that oauth2-proxy already authenticated the request that
// served this page (the app is served from behind the proxy). It displays the
// operator identity from /api/me; a 401 anywhere means the session lapsed, so
// it hands off to the proxy's login flow rather than rendering a broken shell.

import { useEffect, useState } from "react";
import { AuthError, fetchMe, reauthenticate, type Identity } from "./api";
import { Dashboard } from "./Dashboard";
import { Live } from "./Live";
import { Recordings } from "./Recordings";
import { Ring } from "./Ring";

function usePath(): string {
  const [path, setPath] = useState<string>(window.location.pathname);
  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  return path;
}

function navigate(to: string): void {
  window.history.pushState({}, "", to);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export function App() {
  const path = usePath();
  const [identity, setIdentity] = useState<Identity | null>(null);

  useEffect(() => {
    fetchMe()
      .then(setIdentity)
      .catch((e) => {
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
          justifyContent: "space-between",
          alignItems: "center",
          padding: "8px 16px",
          borderBottom: "1px solid #ddd",
        }}
      >
        <nav style={{ display: "flex", gap: 12 }}>
          <a href="/" onClick={(e) => (e.preventDefault(), navigate("/"))}>
            Dashboard
          </a>
          <a href="/live" onClick={(e) => (e.preventDefault(), navigate("/live"))}>
            Live
          </a>
          <a href="/ring" onClick={(e) => (e.preventDefault(), navigate("/ring"))}>
            Ring
          </a>
          <a href="/recordings" onClick={(e) => (e.preventDefault(), navigate("/recordings"))}>
            Recordings
          </a>
        </nav>
        <span style={{ color: "#666", fontSize: 13 }}>{identity ? identity.user : "…"}</span>
      </header>

      {route(path)}
    </div>
  );
}

function route(path: string) {
  if (path === "/live") return <Live />;
  if (path === "/ring") return <Ring />;
  if (path === "/recordings") return <Recordings />;
  return <Dashboard />;
}
