#!/usr/bin/env bash
# Render mediamtx.yml from the template by substituting deploy-time values.
# mediamtx config is environment-specific (node IPs, NodePorts, the
# supervisor egress URL); it is templated and rendered, never hand-edited.
#
# Usage:
#   WEBRTC_ADDITIONAL_HOSTS='"1.2.3.4","5.6.7.8","9.10.11.12"' \
#   WEBRTC_UDP_PORT=30890 WEBRTC_TCP_PORT=30891 \
#   STUN_URL='stun:stun.l.google.com:19302' \
#   BACKEND_JWKS_URL='http://fessel-backend.fessel.svc.cluster.local:8000/jwks' \
#   SUPERVISOR_BASE='http://supervisor.fessel.svc.cluster.local:8443' \
#   LIVE_PATH=pi \
#   tools/render-mediamtx-config.sh < webui/deploy/mediamtx.yml.template > rendered.yml
set -euo pipefail

: "${WEBRTC_ADDITIONAL_HOSTS:?}"
: "${WEBRTC_UDP_PORT:?}"
: "${WEBRTC_TCP_PORT:?}"
: "${STUN_URL:?}"
: "${BACKEND_JWKS_URL:?}"
: "${SUPERVISOR_BASE:?}"
: "${LIVE_PATH:?}"

# Only substitute the deploy-time vars; leave mediamtx's own $MTX_* runtime
# vars (expanded by mediamtx itself in runOnDemand) untouched.
envsubst '
  ${WEBRTC_ADDITIONAL_HOSTS}
  ${WEBRTC_UDP_PORT}
  ${WEBRTC_TCP_PORT}
  ${STUN_URL}
  ${BACKEND_JWKS_URL}
  ${SUPERVISOR_BASE}
  ${LIVE_PATH}
'
