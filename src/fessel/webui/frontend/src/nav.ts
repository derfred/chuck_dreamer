// SPA navigation primitive shared by the app shell and any component that
// links between routes. A path-based pushState + a synthetic popstate so the
// App shell's usePath() re-renders. Kept as a tiny standalone module (no router
// dependency) — the app is a two-route shell (Monitor `/`, Footage `/footage`).

export function navigate(to: string): void {
  window.history.pushState({}, "", to);
  window.dispatchEvent(new PopStateEvent("popstate"));
}
