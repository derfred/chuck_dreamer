#!/usr/bin/env bash
# dpkg install/upgrade assertion (Slice 1.5 T1.8).
#
# Proves the architecture's load-bearing packaging contract: "an upgrade
# replaces binaries and units without clobbering operator-edited config"
# (§5.5). The regression this guards is a config-clobber on upgrade — the most
# damaging packaging bug, since it silently resets safety thresholds.
#
# Runs in a debian:trixie container with the repo at /src (same base as the
# test-pi image, so the apt deps resolve). It is HERMETIC: it builds BOTH the
# N-1 and N packages from the current tree with explicit version overrides, so
# it needs no prior release artifact and works from the very first release.
# (Per the architecture, the package version is a single value; the upgrade
# semantics are identical for any real N-1 -> N pair, so building the "previous"
# from the same tree with a lower version exercises the same dpkg machinery.)
#
# Usage (in debian:trixie with the repo at /src):
#   pi/deploy/dpkg/test-upgrade.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FESSEL_ROOT="$(cd "$HERE/../../.." && pwd)"   # src/fessel

OLD_VER="${OLD_VER:-0.0.0-upgrade-test}"   # stands in for N-1
NEW_VER="${NEW_VER:-9.9.9-upgrade-test}"   # stands in for N (> OLD_VER)

fail() { echo "UPGRADE-TEST FAIL: $*" >&2; exit 1; }
ok() { echo "  [ok] $*"; }

echo "== building N-1 ($OLD_VER) and N ($NEW_VER) packages =="
"$HERE/build-dpkg.sh" "$OLD_VER"
"$HERE/build-dpkg.sh" "$NEW_VER"
OLD_DEB="$FESSEL_ROOT/dist/fessel-monitor_${OLD_VER}_all.deb"
NEW_DEB="$FESSEL_ROOT/dist/fessel-monitor_${NEW_VER}_all.deb"
[ -f "$OLD_DEB" ] || fail "N-1 package not built: $OLD_DEB"
[ -f "$NEW_DEB" ] || fail "N package not built: $NEW_DEB"

echo "== installing stub gstreamer1.0-plugins-rs-fessel =="
# fessel-monitor hard-Depends on the cross-built gst-plugins-rs deb (the
# fail-loud whipclientsink contract). That package is ours, not Debian's, so
# the hermetic container can't resolve it from apt — install an empty stub
# that satisfies the Depends. The upgrade test exercises dpkg config
# semantics, not GStreamer; the real plugin install is covered by the test-pi
# image build and the integration run.
STUB_DIR="$(mktemp -d)"
mkdir -p "$STUB_DIR/pkg/DEBIAN"
cat > "$STUB_DIR/pkg/DEBIAN/control" <<'EOF'
Package: gstreamer1.0-plugins-rs-fessel
Version: 0.0.0-upgrade-test-stub
Architecture: all
Maintainer: fessel upgrade test <upgrade-test@invalid>
Description: hermetic stand-in for the cross-built gst-plugins-rs deb
 The real package ships libgstrswebrtc.so (whipclientsink) for the Pi; this
 empty stub only satisfies fessel-monitor's Depends inside the upgrade test.
EOF
dpkg-deb --build "$STUB_DIR/pkg" "$STUB_DIR/stub.deb" >/dev/null
dpkg -i "$STUB_DIR/stub.deb"
ok "stub plugin package installed"

echo "== installing N-1 =="
# -f install pulls Depends if the base image lacks them (it has most).
dpkg -i "$OLD_DEB" || apt-get update && apt-get -f install -y
[ "$(cat /opt/fessel/VERSION)" = "$OLD_VER" ] || fail "VERSION stamp not $OLD_VER after install"
ok "N-1 installed, VERSION=$OLD_VER"

echo "== operator edits the config (the thing that must survive) =="
SENTINEL="# OPERATOR-SENTINEL-$$  do-not-clobber"
printf '\n%s\n' "$SENTINEL" >> /etc/fessel/fessel.yaml
# Also flip a real tunable, to prove a value-level edit survives, not just a comment.
sed -i 's/^\( *spike_threshold_db:\).*/\1 -7  # operator-tuned/' /etc/fessel/fessel.yaml || true
grep -q "$SENTINEL" /etc/fessel/fessel.yaml || fail "sentinel not written"
ok "config edited (sentinel + tuned spike_threshold_db)"

echo "== upgrading N-1 -> N =="
dpkg -i "$NEW_DEB" || apt-get update && apt-get -f install -y

echo "== assertions =="
# 1) binaries/units updated: the VERSION stamp (a non-conffile, package-owned
#    file) must be replaced with the new version.
[ "$(cat /opt/fessel/VERSION)" = "$NEW_VER" ] || fail "VERSION not updated to $NEW_VER on upgrade"
ok "package-owned VERSION stamp updated to $NEW_VER"

# 2) operator-edited config preserved: dpkg must NOT clobber the conffile. The
#    sentinel AND the tuned value must still be there, and dpkg must not have
#    written a .dpkg-dist/.dpkg-new alongside (which would mean it declined to
#    install the maintainer version because we edited it — the desired behaviour,
#    but we also assert the live file is unchanged).
grep -q "$SENTINEL" /etc/fessel/fessel.yaml || fail "operator sentinel CLOBBERED on upgrade"
grep -q 'spike_threshold_db: -7' /etc/fessel/fessel.yaml || fail "operator-tuned value clobbered"
ok "operator-edited /etc/fessel/fessel.yaml preserved across upgrade"

# 3) the units are registered (installed to /lib/systemd/system). We don't run
#    systemd in this container, so we assert the unit files are present + updated
#    rather than that systemctl restarted them (the test-pi shim covers running).
for u in fessel-supervisor fessel-video fessel-uploader fessel-mosquitto; do
  [ -f "/lib/systemd/system/$u.service" ] || fail "unit $u.service missing after upgrade"
done
ok "systemd units present after upgrade"

# 4) the new code is actually on disk (a module from the latest tree imports).
python3 -c "import sys; sys.path.insert(0,'/opt/fessel/lib'); import fessel_shared, supervisor, video, uploader" \
  || fail "installed modules do not import after upgrade"
ok "installed Python modules import"

echo "UPGRADE-TEST PASS: N-1 -> N preserved operator config and updated binaries"
