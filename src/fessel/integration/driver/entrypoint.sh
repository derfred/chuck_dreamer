#!/usr/bin/env bash
# Start Xvfb so Chromium runs headed (H.264 decoder available), then run the
# driver. Exit code propagates so the Job fails on assertion failure.
set -euo pipefail

export DISPLAY=:99
Xvfb :99 -screen 0 1280x720x24 -nolisten tcp &
XVFB_PID=$!

# Give Xvfb a moment.
for _ in $(seq 1 20); do
  if xdpyinfo -display :99 >/dev/null 2>&1; then break; fi
  sleep 0.5
done

set +e
python /driver/run_tests.py
RC=$?
set -e

kill "$XVFB_PID" 2>/dev/null || true
exit "$RC"
