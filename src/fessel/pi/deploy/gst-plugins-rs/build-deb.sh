#!/usr/bin/env bash
# Build gstreamer1.0-plugins-rs-fessel_<ver>_arm64.deb locally (buildx + QEMU;
# CI runs the same Dockerfile — see .github/workflows/fessel-gst-plugins-rs.yml).
#
# The cargo build under QEMU takes a long time (~30-60 min on a Mac); the
# buildx layer cache makes rebuilds of only the packaging stage fast.
#
# Usage:
#   ./build-deb.sh              # -> dist/gstreamer1.0-plugins-rs-fessel_*.deb
#   DEB_REVISION=2 ./build-deb.sh   # repackage bump of the same upstream
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEB_REVISION="${DEB_REVISION:-1}"
BRANCH="${GST_PLUGINS_RS_BRANCH:-0.13}"

docker buildx build \
  --platform linux/arm64 \
  --build-arg DEB_REVISION="$DEB_REVISION" \
  --build-arg GST_PLUGINS_RS_BRANCH="$BRANCH" \
  --output "type=local,dest=$HERE/dist" \
  "$HERE"

ls -la "$HERE/dist"
