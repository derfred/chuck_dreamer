#!/usr/bin/env bash
# Build/refresh the signed Fessel apt repo (flat: suite=stable, component=main,
# arch=all) using apt-ftparchive + gpg. ACCUMULATES: new .deb(s) are added to
# the existing pool; prior versions are retained and re-indexed, so old
# releases stay installable.
#
# Usage:
#   build-apt-repo.sh <repo-root> <deb> [<deb> ...]
#
# <repo-root> is the (already-checked-out) gh-pages tree. Requires the signing
# GPG key already imported into the default keyring; the key id/email is
# auto-discovered (or set FESSEL_GPG_KEY_ID).
#
# Produces under <repo-root>:
#   pool/main/f/fessel-monitor/*.deb
#   dists/stable/main/binary-all/Packages[.gz]
#   dists/stable/Release, Release.gpg, InRelease
#   fessel-archive-keyring.gpg   (exported public key, for the Pi)
set -euo pipefail

REPO="${1:?usage: build-apt-repo.sh <repo-root> <deb>...}"; shift
[ "$#" -ge 1 ] || { echo "no .deb files given" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

POOL="$REPO/pool/main/f/fessel-monitor"
DIST="$REPO/dists/stable"
BINDIR="$DIST/main/binary-all"
mkdir -p "$POOL" "$BINDIR"

echo "== adding $# package(s) to the pool =="
for deb in "$@"; do
  cp -f "$deb" "$POOL/"
  echo "   + $(basename "$deb")"
done

echo "== generating Packages index over the whole pool =="
( cd "$REPO" && apt-ftparchive packages pool > "$BINDIR/Packages" )
gzip -9c "$BINDIR/Packages" > "$BINDIR/Packages.gz"

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
ls -R "$DIST" "$POOL" | sed 's/^/   /'
