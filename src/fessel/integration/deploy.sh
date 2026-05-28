#!/usr/bin/env bash
# Deploy the full Fessel system into a fresh namespace for an integration
# test run, render-substituting environment values. Stands in for the Tanka
# test environment (T1.4): same shape, but self-contained in the repo under
# test so CI needs no cross-repo Tanka render.
#
# Two tailnet-substituted endpoints become in-cluster Service DNS:
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

export NS IMAGE_TAG REGISTRY FESSEL_WHEP_SECRET

echo "== creating namespace $NS =="
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -

echo "== rendering + applying infra manifests (00-30) =="
for tmpl in "$HERE"/manifests/0*.yaml.tmpl "$HERE"/manifests/[123]*.yaml.tmpl; do
  "$HERE/render.sh" < "$tmpl"
  echo "---"
done | kubectl apply -n "$NS" -f -

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

echo "== deploy complete in $NS =="
