#!/usr/bin/env bash
# Tear down a live-preview namespace. Required env: NS.
set -euo pipefail
: "${NS:?}"

# Stash the Let's Encrypt TLS secret before the namespace dies: deploy.sh
# restores it, so cert-manager reuses the still-valid certificate instead of
# ordering a new one per preview cycle (5 identical orders/week hits LE's
# duplicate-certificate rate limit — learned the hard way).
STASH_NS="fessel-preview-certs"
if kubectl -n "$NS" get secret fessel-webui-tls >/dev/null 2>&1; then
  kubectl create namespace "$STASH_NS" --dry-run=client -o yaml | kubectl apply -f -
  kubectl -n "$NS" get secret fessel-webui-tls -o json     | jq 'del(.metadata.namespace,.metadata.uid,.metadata.resourceVersion,.metadata.creationTimestamp,.metadata.ownerReferences,.metadata.annotations["kubectl.kubernetes.io/last-applied-configuration"])'     | kubectl -n "$STASH_NS" apply -f -
  echo "== stashed fessel-webui-tls to $STASH_NS =="
fi

echo "== deleting namespace $NS =="
kubectl delete namespace "$NS" --ignore-not-found --wait=false
echo "done"
