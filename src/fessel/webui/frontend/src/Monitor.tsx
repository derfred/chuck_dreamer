// Monitor page (`/`): the operator's live-view + control surface. Two columns —
// the video panel (Ring/Live toggle, MonitorVideo) and a side rail carrying the
// sensing strip + control buttons (the embedded Dashboard).
//
// The video defaults to the cheap Ring source; Live is an explicit, clearly
// costed opt-in that auto-tears-down (MonitorVideo / useLiveSession). Anomaly
// markers under the video come from a shared /api/anomalies poll (useAnomalies).

import { Dashboard } from "./Dashboard";
import { MonitorVideo } from "./MonitorVideo";
import { useAnomalies } from "./useAnomalies";

export function Monitor() {
  const anomalies = useAnomalies();
  return (
    <div style={{ padding: 16, maxWidth: 1080, margin: "0 auto" }}>
      <h1 style={{ fontSize: 22, letterSpacing: "-0.02em", margin: "10px 0 18px" }}>Monitor</h1>
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 320px", gap: 20, alignItems: "start" }}>
        <MonitorVideo anomalies={anomalies} />
        <div>
          <Dashboard embedded />
        </div>
      </div>
    </div>
  );
}
