// The shared /api/state poll for the Monitor page.
//
// Both halves of Monitor need the same state document — the video panel's
// recording chip and the side rail's sensing strip + controls — so the page
// polls ONCE and hands the slices down, rather than each component opening its
// own 2s poll of the same endpoint.
//
// Error handling matches what the Dashboard did on its own: a 401 hands off to
// the proxy login (never loops), and a 5xx/network blip raises a transient
// `error` flag WITHOUT clearing the last-known state.

import { useEffect, useState } from "react";
import type { StateResponse } from "../../shared/schemas";
import { AuthError, fetchState, reauthenticate } from "./api";
import { config } from "./config";

export interface MonitorState {
  state: StateResponse | null;
  error: boolean;
  // A short client-side history of the activity score (F5.1 sparkline), capped
  // at ~60 samples (= ~2min at the 2s poll); a ring, oldest dropped.
  activityHistory: number[];
}

export function useMonitorState(): MonitorState {
  const [state, setState] = useState<StateResponse | null>(null);
  const [error, setError] = useState<boolean>(false);
  const [activityHistory, setActivityHistory] = useState<number[]>([]);

  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      fetchState()
        .then((s) => {
          if (cancelled) return;
          setState(s);
          setError(false);
          const score = s.vision?.activity_score_ema;
          if (typeof score === "number") {
            setActivityHistory((h) => [...h, score].slice(-60));
          }
        })
        .catch((e) => {
          if (cancelled) return;
          if (e instanceof AuthError) {
            reauthenticate();
            return;
          }
          setError(true);
        });
    };
    poll();
    const id = window.setInterval(poll, config.statePollMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  return { state, error, activityHistory };
}
