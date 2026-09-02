// Monitor page (`/`): the operator's live-view + control surface. Two columns —
// the live video panel (Stream On/Off toggle + recording controls,
// MonitorVideo) and a side rail carrying the sensing strip + control buttons
// (the embedded Dashboard).
//
// The page owns the single /api/state poll and hands the slices to both halves
// (useMonitorState), so the two columns can't drift apart or double the poll
// rate on the cellular link.
//
// The stream defaults to OFF (0 bandwidth); the operator turns the live WebRTC
// view on deliberately, and it auto-tears-down on tab-hide / navigate-away
// (MonitorVideo / useLiveSession). The ring buffer is never viewed here — it is
// on the Pi behind cellular and only serves the record dialog's look-back.

import { Dashboard } from "./Dashboard";
import { MonitorVideo } from "./MonitorVideo";
import { useMonitorState } from "./useMonitorState";

export function Monitor() {
  const monitorState = useMonitorState();
  return (
    <div style={{ padding: 16, maxWidth: 1080, margin: "0 auto" }}>
      <h1 style={{ fontSize: 22, letterSpacing: "-0.02em", margin: "10px 0 18px" }}>Monitor</h1>
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 320px", gap: 20, alignItems: "start" }}>
        <MonitorVideo
          recording={monitorState.state?.recording}
          backlog={monitorState.state?.upload_backlog}
        />
        <div>
          <Dashboard embedded injectedState={monitorState} />
        </div>
      </div>
    </div>
  );
}
