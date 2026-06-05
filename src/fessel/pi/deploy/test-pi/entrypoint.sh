#!/usr/bin/env bash
# Test-Pi container entrypoint. systemd-in-container is avoided on the
# rke2/containerd cluster; instead we run the SAME ExecStart commands the
# dpkg's systemd units define (units stay authoritative — we parse them).
#
# Order: mosquitto -> video + supervisor. The camera is video's built-in
# videotestsrc (no /dev/video0, no kernel module), so no feed process.
set -euo pipefail

log() { echo "[entrypoint] $*"; }

# Pull ExecStart out of an installed unit so the unit remains the source of
# truth for how each process is launched.
execstart() {
  awk -F= '/^ExecStart=/{sub(/^ExecStart=/,""); print; exit}' "/lib/systemd/system/$1"
}

term() {
  log "shutting down"
  kill $(jobs -p) 2>/dev/null || true
  wait || true
}
trap term TERM INT

# 1. mosquitto (broker)
log "starting mosquitto"
$(execstart fessel-mosquitto.service) &
sleep 1

# 2. supervisor (control plane + relay), video (pipeline + state machine), and
# uploader. ExecStart is /usr/bin/fessel-{supervisor,video,uploader} (the dpkg's
# launchers) — run them as-is. All three read the single fessel.yaml via
# FESSEL_CONFIG (each loads its own subtree + the shared mqtt/storage sections).
export FESSEL_CONFIG=/etc/fessel/fessel.yaml

log "starting supervisor"
$(execstart fessel-supervisor.service) &

log "starting video"
$(execstart fessel-video.service) &

log "starting uploader"
$(execstart fessel-uploader.service) &

log "all processes started"
wait -n
log "a process exited; tearing down"
term
