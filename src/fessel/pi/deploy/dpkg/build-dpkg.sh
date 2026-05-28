#!/usr/bin/env bash
# Build the fessel-monitor .deb. Runs inside a Debian container (the host
# may be macOS without dpkg-deb). Produces dist/fessel-monitor_<ver>_all.deb.
#
# The package installs `supervisor` and `video` as system-site-packages
# venvs under /opt/fessel, registers the systemd units, declares the
# GStreamer/PyGObject apt dependencies, and seeds /etc/fessel config as
# dpkg conffiles (so operator edits survive upgrades — T1.8).
#
# Usage (in a debian:bookworm container with the repo at /src):
#   pi/deploy/dpkg/build-dpkg.sh <version>
set -euo pipefail

VERSION="${1:-0.1.0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FESSEL_ROOT="$(cd "$HERE/../../.." && pwd)"   # src/fessel
STAGE="$(mktemp -d)"
OUT_DIR="$FESSEL_ROOT/dist"
mkdir -p "$OUT_DIR"

echo "== staging package tree at $STAGE =="

# --- payload: per-process venvs under /opt/fessel ---
# Build clean venvs against system python3 with --system-site-packages so
# `video` can bind to the apt-installed gi/GStreamer. supervisor doesn't
# need gi but a single venv strategy keeps the package simple.
# Build each venv at its FINAL install path (/opt/fessel/...) so the venv
# shebangs and pyvenv paths are correct, then copy the tree into the
# staging dir for packaging. (Building under $STAGE would bake the staging
# path into the venv and break it after install.)
for proc in supervisor video; do
  FINAL="/opt/fessel/$proc"
  rm -rf "$FINAL"
  install -d "$FINAL"
  python3 -m venv --system-site-packages "$FINAL/.venv"
  "$FINAL/.venv/bin/pip" install --no-cache-dir -q --upgrade pip
  "$FINAL/.venv/bin/pip" install --no-cache-dir -q \
    "$FESSEL_ROOT/schemas" "$FESSEL_ROOT/pi/shared" "$FESSEL_ROOT/pi/$proc"
  # Ensure a real pydantic lives in the venv (a system namespace pkg can
  # shadow the dependency-installed one).
  "$FINAL/.venv/bin/pip" install --no-cache-dir -q \
    --ignore-installed "pydantic>=2.6" pydantic-core
  install -d "$STAGE/opt/fessel"
  cp -a "$FINAL" "$STAGE/opt/fessel/$proc"
done

# --- systemd units ---
install -d "$STAGE/lib/systemd/system"
install -m 644 "$FESSEL_ROOT/pi/deploy/systemd/supervisor.service" "$STAGE/lib/systemd/system/fessel-supervisor.service"
install -m 644 "$FESSEL_ROOT/pi/deploy/systemd/video.service" "$STAGE/lib/systemd/system/fessel-video.service"
install -m 644 "$FESSEL_ROOT/pi/deploy/systemd/mosquitto.service.example" "$STAGE/lib/systemd/system/fessel-mosquitto.service"

# --- config (conffiles: preserved on upgrade) ---
install -d "$STAGE/etc/fessel"
install -m 644 "$FESSEL_ROOT/pi/deploy/mosquitto.conf" "$STAGE/etc/fessel/mosquitto.conf"
install -m 644 "$FESSEL_ROOT/pi/deploy/supervisor.yaml.example" "$STAGE/etc/fessel/supervisor.yaml"
install -m 644 "$FESSEL_ROOT/pi/deploy/video.yaml.example" "$STAGE/etc/fessel/video.yaml"

# --- DEBIAN control metadata ---
install -d "$STAGE/DEBIAN"
sed "s/@VERSION@/$VERSION/" "$HERE/control.in" > "$STAGE/DEBIAN/control"
install -m 644 "$HERE/conffiles" "$STAGE/DEBIAN/conffiles"
install -m 755 "$HERE/postinst" "$STAGE/DEBIAN/postinst"
install -m 755 "$HERE/prerm" "$STAGE/DEBIAN/prerm"

DEB="$OUT_DIR/fessel-monitor_${VERSION}_all.deb"
echo "== building $DEB =="
dpkg-deb --build --root-owner-group "$STAGE" "$DEB"
echo "built: $DEB"
rm -rf "$STAGE"
