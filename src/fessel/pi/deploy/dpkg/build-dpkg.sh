#!/usr/bin/env bash
# Build the fessel-monitor .deb. Runs inside a Debian (trixie) container (the
# host may be macOS without dpkg-deb). Produces dist/fessel-monitor_<ver>_all.deb.
#
# The package ships the Fessel Python modules as plain files under
# /opt/fessel/lib (NO venv, NO pip), relies on Debian python3-* packages for
# all libraries (declared in control.in Depends), installs two launcher
# wrappers in /usr/bin, registers the systemd units, and seeds /etc/fessel
# config as dpkg conffiles (so operator edits survive upgrades).
#
# The package version is the SINGLE SOURCE OF TRUTH in
# schemas/pyproject.toml — the same value the webui image tag and the
# release workflow derive from. Pass an explicit version only to override
# for a one-off local build; normally omit it and let the pyproject win.
#
# Usage (in a debian:trixie container with the repo at /src):
#   pi/deploy/dpkg/build-dpkg.sh            # version from schemas/pyproject.toml
#   pi/deploy/dpkg/build-dpkg.sh <version>  # explicit override
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FESSEL_ROOT="$(cd "$HERE/../../.." && pwd)"   # src/fessel

# Read the canonical version from schemas/pyproject.toml (first `version = "…"`
# under [project]). No toml parser needed in the build container — a tight grep.
pyproject_version() {
  sed -n 's/^version = "\([^"]*\)".*/\1/p' "$FESSEL_ROOT/schemas/pyproject.toml" | head -n1
}
VERSION="${1:-$(pyproject_version)}"
if [ -z "$VERSION" ]; then
  echo "could not determine version from schemas/pyproject.toml" >&2
  exit 1
fi
STAGE="$(mktemp -d)"
OUT_DIR="$FESSEL_ROOT/dist"
mkdir -p "$OUT_DIR"

echo "== staging package tree at $STAGE =="

# --- payload: pure-Python modules under /opt/fessel/lib (no venv, no pip) ---
# The four modules are pure Python and import each other by top-level name
# (fessel_schemas, fessel_shared) + relative imports, so a flat directory on
# sys.path resolves correctly. /opt/fessel/lib is private (not dist-packages)
# so generic names `supervisor`/`video` can't collide with other system
# packages; the /usr/bin wrappers put it on sys.path for our processes only.
install -d "$STAGE/opt/fessel/lib"
cp -a "$FESSEL_ROOT/schemas/fessel_schemas"      "$STAGE/opt/fessel/lib/"
cp -a "$FESSEL_ROOT/pi/shared/fessel_shared"     "$STAGE/opt/fessel/lib/"
cp -a "$FESSEL_ROOT/pi/supervisor/supervisor"    "$STAGE/opt/fessel/lib/"
cp -a "$FESSEL_ROOT/pi/video/video"              "$STAGE/opt/fessel/lib/"
# Don't ship build-host bytecode.
find "$STAGE/opt/fessel/lib" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$STAGE/opt/fessel/lib" -type f -name '*.pyc' -delete

# --- release version stamp ---
# fessel_version() reads this at runtime; supervisor/video surface it on
# /healthz. It is the SAME string as the package version and the webui image
# tag (one release == one version), so a deployed Pi and cluster can be
# compared for drift. Not a conffile: it's owned by the package, replaced on
# every upgrade.
printf '%s\n' "$VERSION" > "$STAGE/opt/fessel/VERSION"

# --- launcher wrappers in /usr/bin ---
install -d "$STAGE/usr/bin"
install -m 755 "$HERE/bin/fessel-supervisor" "$STAGE/usr/bin/fessel-supervisor"
install -m 755 "$HERE/bin/fessel-video"      "$STAGE/usr/bin/fessel-video"

# --- systemd units (source filenames match the installed fessel-* names;
# mosquitto ships from the .example which the package treats as the real unit) ---
install -d "$STAGE/lib/systemd/system"
install -m 644 "$FESSEL_ROOT/pi/deploy/systemd/fessel-supervisor.service" "$STAGE/lib/systemd/system/fessel-supervisor.service"
install -m 644 "$FESSEL_ROOT/pi/deploy/systemd/fessel-video.service" "$STAGE/lib/systemd/system/fessel-video.service"
install -m 644 "$FESSEL_ROOT/pi/deploy/systemd/fessel-mosquitto.service.example" "$STAGE/lib/systemd/system/fessel-mosquitto.service"

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
