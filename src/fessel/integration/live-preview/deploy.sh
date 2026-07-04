#!/usr/bin/env bash
# Deploy the Fessel stack for an on-demand LIVE PREVIEW: production WebRTC
# exposure (node public IPs + NodePort) and HTTPS Ingresses so a REAL
# browser can watch the stream. Runs the control-plane assertions, then
# leaves the stack UP and prints the URL. Does NOT tear down (use
# teardown.sh or the workflow's TTL).
#
# Required env: NS, IMAGE_TAG, REGISTRY, BASE_DOMAIN
# Optional: WRTC_UDP_NODEPORT (default 31554) — the viewer media UDP NodePort
# (No streaming secret and no TCP NodePort: the in-process Pion
# relay is in-process; UDP-only ICE, no JWT.)
set -euo pipefail
: "${NS:?}" "${IMAGE_TAG:?}" "${REGISTRY:?}" "${BASE_DOMAIN:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSONNET="$HERE/../../deploy/jsonnet"

WRTC_UDP_NODEPORT="${WRTC_UDP_NODEPORT:-31554}"
# Subdomain override: lets a preview move to a fresh hostname when the
# default one has burned Let's Encrypt's duplicate-certificate quota (or is
# HSTS-pinned in a browser against a fallback cert).
WEBUI_SUBDOMAIN="${WEBUI_SUBDOMAIN:-fessel-live}"
WEBUI_HOST="${WEBUI_SUBDOMAIN}.${BASE_DOMAIN}"

# Node public IPs (comma-separated) for the relay's viewer ICE candidates
# (FESSEL_VIEWER_PUBLIC_IPS). Explicit env wins; otherwise discover from the
# node objects' ExternalIP. On clusters whose nodes register only an
# InternalIP (Talos on bugzoo: the public addresses live outside k8s), the
# discovery finds nothing and the env MUST be provided — fail loud rather
# than deploy a relay that advertises no reachable candidate.
if [ -z "${NODE_PUBLIC_IPS:-}" ]; then
  NODE_PUBLIC_IPS="$(
    kubectl get nodes -o jsonpath='{range .items[*]}{.spec.unschedulable}{" "}{.status.addresses[?(@.type=="ExternalIP")].address}{"\n"}{end}' \
    | awk '$1!="true" && $2!="" {printf "%s%s", sep, $2; sep=","}'
  )"
fi
if [ -z "$NODE_PUBLIC_IPS" ]; then
  echo "ERROR: no node public IPs — nodes register no ExternalIP here; set NODE_PUBLIC_IPS" >&2
  exit 1
fi
echo "== node public IPs: ${NODE_PUBLIC_IPS} =="

# Nodes that actually own those public IPs (comma-separated hostnames,
# optional). With externalTrafficPolicy: Local, a webui pod scheduled on a
# node WITHOUT a public IP (worker-zrh1) would break viewer media silently —
# pin the pod when the cluster is heterogeneous.
NODE_HOSTNAMES="${NODE_HOSTNAMES:-}"

echo "== render + apply live-preview env (jsonnet, nodeport mode) =="
tk eval "$JSONNET/envs/live-preview.jsonnet" \
  -V ns="$NS" -V image_tag="$IMAGE_TAG" -V registry="$REGISTRY" \
  -V base_domain="$BASE_DOMAIN" \
  -V node_public_ips="$NODE_PUBLIC_IPS" \
  -V webui_host="$WEBUI_HOST" \
  -V node_hostnames="$NODE_HOSTNAMES" \
  -V udp_nodeport="$WRTC_UDP_NODEPORT" \
  | kubectl apply -f -

# Restore a previously-stashed TLS secret (see teardown.sh) so cert-manager
# adopts the existing valid certificate instead of ordering a new one.
STASH_NS="fessel-preview-certs"
if kubectl -n "$STASH_NS" get secret fessel-webui-tls >/dev/null 2>&1; then
  kubectl -n "$STASH_NS" get secret fessel-webui-tls -o json \
    | jq 'del(.metadata.namespace,.metadata.uid,.metadata.resourceVersion,.metadata.creationTimestamp,.metadata.ownerReferences)' \
    | kubectl -n "$NS" apply -f -
  echo "== restored stashed TLS secret =="
fi

echo "== wait for rollouts =="
kubectl -n "$NS" rollout status deploy/webui --timeout=120s
kubectl -n "$NS" rollout status deploy/pi --timeout=180s

echo "== control-plane assertions (data plane is verified manually) =="
kubectl delete job fessel-cp -n "$NS" --ignore-not-found
cat <<EOF | kubectl apply -n "$NS" -f -
apiVersion: batch/v1
kind: Job
metadata: { name: fessel-cp }
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: driver
          image: ${REGISTRY}/fessel-test-driver:${IMAGE_TAG}
          imagePullPolicy: Always  # stable tag, rebuilt each preview
          env:
            - { name: WEBUI, value: "http://webui:8000" }
            - { name: MEDIA, value: "http://webui:8000" }
            - { name: SUPERVISOR, value: "http://supervisor:8443" }
            - { name: CONTROL_PLANE_ONLY, value: "1" }
            - { name: AUTH_BYPASS, value: "1" }  # webui runs FESSEL_DEV_IDENTITY here
            - { name: RECONNECT_CYCLES, value: "10" }
            - { name: JUNIT_OUT, value: "/tmp/junit.xml" }
EOF
kubectl wait --for=condition=complete job/fessel-cp -n "$NS" --timeout=240s &
W1=$!; kubectl wait --for=condition=failed job/fessel-cp -n "$NS" --timeout=240s & W2=$!
wait -n "$W1" "$W2" || true
kubectl logs -n "$NS" job/fessel-cp 2>&1 | grep -E '\[PASS\]|\[FAIL\]|INTEGRATION:' || true
CP_OK=$(kubectl get job fessel-cp -n "$NS" -o jsonpath='{.status.succeeded}')
[ "$CP_OK" = "1" ] && echo "control plane: PASS" || echo "control plane: FAIL (continuing; preview still up)"

cat <<EOF

============================================================
  FESSEL LIVE PREVIEW IS UP
  Open:  https://${WEBUI_HOST}/
  (pick a mode, click "Start live view", confirm moving video)

  (WHEP signaling is same-origin on the webui host; media on UDP
   NodePort ${WRTC_UDP_NODEPORT})
  Namespace: ${NS}
  Tear down: integration/live-preview/teardown.sh  (NS=${NS})
============================================================
EOF
