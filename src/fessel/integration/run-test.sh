#!/usr/bin/env bash
# Run the test driver Job in the (already deployed) namespace, wait for it,
# collect JUnit XML from the pod, and propagate pass/fail.
#
# Required env: NS, IMAGE_TAG, REGISTRY
set -euo pipefail
: "${NS:?}" "${IMAGE_TAG:?}" "${REGISTRY:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NS IMAGE_TAG REGISTRY
RESULTS_DIR="${RESULTS_DIR:-$HERE/../../../integration-results}"
mkdir -p "$RESULTS_DIR"

echo "== applying test Job =="
kubectl delete job fessel-test -n "$NS" --ignore-not-found
"$HERE/render.sh" < "$HERE/manifests/40-testjob.yaml.tmpl" | kubectl apply -n "$NS" -f -

echo "== waiting for test Job to complete (timeout 8m) =="
# Wait for either completion or failure.
kubectl wait --for=condition=complete job/fessel-test -n "$NS" --timeout=480s &
WAIT_OK=$!
kubectl wait --for=condition=failed job/fessel-test -n "$NS" --timeout=480s &
WAIT_FAIL=$!
wait -n "$WAIT_OK" "$WAIT_FAIL" || true

POD=$(kubectl get pods -n "$NS" -l app=fessel-test -o jsonpath='{.items[0].metadata.name}')
echo "== driver pod: $POD =="
echo "== driver logs =="
kubectl logs -n "$NS" "$POD" || true

# Extract JUnit: the driver prints it to stdout between the testsuite tags.
kubectl logs -n "$NS" "$POD" 2>/dev/null \
  | awk '/<\?xml/{f=1} f{print} /<\/testsuite>/{f=0}' > "$RESULTS_DIR/junit.xml" || true
echo "== junit written to $RESULTS_DIR/junit.xml =="

# Job success condition.
SUCCEEDED=$(kubectl get job fessel-test -n "$NS" -o jsonpath='{.status.succeeded}')
if [ "$SUCCEEDED" = "1" ]; then
  echo "INTEGRATION: PASS"
  exit 0
fi

# On failure, dump full diagnostics so the CI log is self-contained (no live
# cluster access needed for a post-mortem). This is the first place to look
# when the integration check is red.
echo "############################################################"
echo "## INTEGRATION FAILED — diagnostics for namespace $NS"
echo "############################################################"
echo "== pods =="
kubectl get pods -n "$NS" -o wide || true
echo "== recent events =="
kubectl get events -n "$NS" --sort-by=.lastTimestamp 2>/dev/null | tail -30 || true
for app in mediamtx webui pi; do
  echo "== describe $app =="
  kubectl describe deploy "$app" -n "$NS" 2>/dev/null | tail -25 || true
  for p in $(kubectl get pods -n "$NS" -l app="$app" -o name 2>/dev/null); do
    echo "== logs $p (current) =="
    kubectl logs -n "$NS" "$p" --tail=80 --all-containers 2>&1 || true
    echo "== logs $p (previous, if restarted) =="
    kubectl logs -n "$NS" "$p" --previous --tail=80 --all-containers 2>&1 || true
  done
done
echo "INTEGRATION: FAIL"
exit 1
