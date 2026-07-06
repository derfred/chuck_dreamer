// Poll the anomaly log (/api/anomalies) on the dashboard cadence. Shared by the
// Monitor video's anomaly lane and the Footage ring hero. A 401 escalates to
// re-auth; other errors keep the last-known list (informational overlay).

import { useEffect, useState } from "react";
import { AuthError, fetchAnomalies, reauthenticate } from "./api";
import { config } from "./config";
import type { AnomalyLogEntry } from "../../shared/schemas";

export function useAnomalies(): AnomalyLogEntry[] {
  const [anomalies, setAnomalies] = useState<AnomalyLogEntry[]>([]);
  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      fetchAnomalies()
        .then((a) => {
          if (!cancelled) setAnomalies(a);
        })
        .catch((e) => {
          if (e instanceof AuthError) reauthenticate();
        });
    };
    poll();
    const id = window.setInterval(poll, config.statePollMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);
  return anomalies;
}
