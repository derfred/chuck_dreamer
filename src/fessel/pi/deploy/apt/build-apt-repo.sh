#!/usr/bin/env bash
# Build/refresh the signed Fessel apt repo (flat: suite=stable, component=main,
# arch all + arm64) using apt-ftparchive + gpg. ACCUMULATES: new .deb(s) are
# added to the existing pool; prior versions are retained and re-indexed, so
# old releases stay installable.
#
# Usage:
#   build-apt-repo.sh <repo-root> <deb> [<deb> ...]
#
# <repo-root> is the (already-checked-out) gh-pages tree. Requires the signing
# GPG key already imported into the default keyring; the key id/email is
# auto-discovered (or set FESSEL_GPG_KEY_ID).
#
# Produces under <repo-root>:
#   pool/main/<x>/<package>/*.deb          (pool dir derived per package)
#   dists/stable/main/binary-all/Packages[.gz]
#   dists/stable/main/binary-arm64/Packages[.gz]
#   dists/stable/Release, Release.gpg, InRelease
#   fessel-archive-keyring.gpg   (exported public key, for the Pi)
#
# Both binary-<arch> indexes list the whole pool: apt on the Pi (arm64)
# resolves Architecture:all packages (fessel-monitor) and arm64 ones
# (gstreamer1.0-plugins-rs-fessel) from whichever index it fetches, and a
# pre-existing `arch=all` source line keeps working during migration.
set -euo pipefail

REPO="${1:?usage: build-apt-repo.sh <repo-root> <deb>...}"; shift
[ "$#" -ge 1 ] || { echo "no .deb files given" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DIST="$REPO/dists/stable"
mkdir -p "$DIST/main/binary-all" "$DIST/main/binary-arm64"

echo "== adding $# package(s) to the pool =="
for deb in "$@"; do
  # Debian pool layout: pool/main/<first letter>/<package name>/.
  pkg="$(dpkg-deb -f "$deb" Package)"
  pooldir="$REPO/pool/main/${pkg:0:1}/$pkg"
  mkdir -p "$pooldir"
  cp -f "$deb" "$pooldir/"
  echo "   + $pkg <- $(basename "$deb")"
done

echo "== generating Packages indexes over the whole pool =="
PKGIDX="$(mktemp)"
( cd "$REPO" && apt-ftparchive packages pool > "$PKGIDX" )
for arch in all arm64; do
  BINDIR="$DIST/main/binary-$arch"
  cp "$PKGIDX" "$BINDIR/Packages"
  gzip -9c "$BINDIR/Packages" > "$BINDIR/Packages.gz"
done
rm -f "$PKGIDX"

echo "== generating Release =="
( cd "$REPO" && apt-ftparchive -c "$HERE/apt-ftparchive.conf" release dists/stable > "$DIST/Release" )

# Discover the signing key if not provided.
KEY_ID="${FESSEL_GPG_KEY_ID:-$(gpg --list-secret-keys --with-colons | awk -F: '/^sec:/{print $5; exit}')}"
[ -n "$KEY_ID" ] || { echo "no GPG secret key available to sign with" >&2; exit 1; }
echo "== signing Release with key $KEY_ID =="
gpg --batch --yes --local-user "$KEY_ID" --armor --detach-sign \
  --output "$DIST/Release.gpg" "$DIST/Release"
gpg --batch --yes --local-user "$KEY_ID" --clearsign \
  --output "$DIST/InRelease" "$DIST/Release"

echo "== exporting public keyring for the Pi =="
gpg --export "$KEY_ID" > "$REPO/fessel-archive-keyring.gpg"

# Pages serves .nojekyll-friendly trees as-is; keep the pool/dists visible.
touch "$REPO/.nojekyll"

echo "== done. repo at $REPO =="
ls -R "$DIST" "$REPO/pool" | sed 's/^/   /'
