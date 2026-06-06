#!/usr/bin/env bash
# Deploy the full Fessel system into a fresh namespace for an integration
# test run, rendered from the jsonnet deploy library (deploy/jsonnet) — the
# single source of truth shared with production (bugzoo-infrastructure).
#
# Two tailnet-substituted endpoints become in-cluster Service DNS (the
# library's webrtc.mode=podip + includeTestPi handle this):
#   - mediamtx -> supervisor (runOnDemand): supervisor.<ns>.svc:8443
#   - video -> mediamtx (SRT publish): mediamtx-srt.<ns>.svc:8890
#
# Required env:
#   NS                 target namespace (created if missing)
#   IMAGE_TAG          run-id tag for the built images
#   REGISTRY           e.g. ghcr.io/derfred
#   FESSEL_WHEP_SECRET shared HMAC/JWT secret (one value -> backend + mediamtx)
set -euo pipefail
: "${NS:?}" "${IMAGE_TAG:?}" "${REGISTRY:?}" "${FESSEL_WHEP_SECRET:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSONNET="$HERE/../deploy/jsonnet"

echo "== rendering integration env from jsonnet =="
render() {
  tk eval "$JSONNET/envs/integration.jsonnet" \
    -V ns="$NS" -V image_tag="$IMAGE_TAG" -V registry="$REGISTRY" \
    -V whep_secret="$FESSEL_WHEP_SECRET"
}

echo "== applying (creates namespace + infra) =="
render | kubectl apply -f -

echo "== waiting for rollouts =="
rollout() {
  local d="$1" t="$2"
  if ! kubectl -n "$NS" rollout status "deploy/$d" --timeout="$t"; then
    echo "!! rollout failed for $d — diagnostics:"
    kubectl -n "$NS" get pods -o wide || true
    kubectl -n "$NS" describe deploy "$d" | tail -30 || true
    for p in $(kubectl -n "$NS" get pods -l app="$d" -o name); do
      echo "--- describe $p ---"; kubectl -n "$NS" describe "$p" | tail -40 || true
      echo "--- logs $p (current) ---"; kubectl -n "$NS" logs "$p" --tail=50 || true
      echo "--- logs $p (previous) ---"; kubectl -n "$NS" logs "$p" --previous --tail=50 || true
    done
    return 1
  fi
}
rollout mediamtx 120s
rollout webui 120s
rollout pi 180s
# jetson-mock only exists in the integration env (includeControlMocks); gate on
# it when present so a mock crash is an explicit deploy failure with diagnostics,
# not a confusing later failure of the Slice-3 control_* assertions.
if kubectl -n "$NS" get deploy jetson-mock >/dev/null 2>&1; then
  rollout jetson-mock 120s
fi
# minio only exists in the integration env (includeMinio); gate on it so the
# recording E2E (X4.3) has its object store up before the test Job runs.
if kubectl -n "$NS" get deploy minio >/dev/null 2>&1; then
  rollout minio 120s
fi

# Post-rollout stability gate: a pod can pass rollout then crash-loop.
echo "== stability check (no restarts after settle) =="
sleep 15
unstable=0
stability_targets="mediamtx webui pi"
if kubectl -n "$NS" get deploy jetson-mock >/dev/null 2>&1; then
  stability_targets="$stability_targets jetson-mock"
fi
for d in $stability_targets; do
  for p in $(kubectl -n "$NS" get pods -l app="$d" -o name); do
    rc=$(kubectl -n "$NS" get "$p" -o jsonpath='{.status.containerStatuses[0].restartCount}')
    echo "  $p restarts=$rc"
    if [ "${rc:-0}" -gt 0 ]; then
      unstable=1
      echo "  !! $p is unstable — logs:"
      kubectl -n "$NS" logs "$p" --previous --tail=40 || true
    fi
  done
done
[ "$unstable" -eq 0 ] || { echo "deploy unstable"; exit 1; }

echo "== deploy complete in $NS =="
