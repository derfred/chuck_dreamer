#!/usr/bin/env bash
# Tear down a live-preview namespace. Required env: NS.
set -euo pipefail
: "${NS:?}"
echo "== deleting namespace $NS =="
kubectl delete namespace "$NS" --ignore-not-found --wait=false
echo "done"
