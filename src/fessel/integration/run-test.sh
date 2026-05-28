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
echo "INTEGRATION: FAIL"
exit 1
