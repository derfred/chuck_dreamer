#!/usr/bin/env bash
# Build/refresh the signed Fessel apt repo (suite=stable, components main +
# prerelease, arch all + arm64) using apt-ftparchive + gpg. ACCUMULATES: new
# .deb(s) are added to the existing pool; prior versions are retained and
# re-indexed, so old releases stay installable.
#
# Usage:
#   build-apt-repo.sh [-c <component>] <repo-root> <deb> [<deb> ...]
#
# -c <component>   which repo component to publish into (default: main).
#                  Use `prerelease` for edge/rc builds so a stable Pi's
#                  `apt upgrade` never sees them unless it opts in (see the
#                  fessel-channel helper + apt/README.md "Edge channel").
#
# <repo-root> is the (already-checked-out) gh-pages tree. Requires the signing
# GPG key already imported into the default keyring; the key id/email is
# auto-discovered (or set FESSEL_GPG_KEY_ID).
#
# Produces under <repo-root>:
#   pool/<component>/<x>/<package>/*.deb    (pool dir derived per package)
#   dists/stable/<component>/binary-all/Packages[.gz]
#   dists/stable/<component>/binary-arm64/Packages[.gz]
#   dists/stable/Release, Release.gpg, InRelease   (covers ALL components)
#   fessel-archive-keyring.gpg   (exported public key, for the Pi)
#
# Each component's Packages index lists only that component's pool subtree, but
# the single signed Release covers every component present in the tree, so a
# Pi that lists `main` and one that also lists `prerelease` both verify against
# the same InRelease. Both binary-<arch> indexes of a component list its whole
# subtree: apt on the Pi (arm64) resolves Architecture:all packages
# (fessel-monitor) and arm64 ones (gstreamer1.0-plugins-rs-fessel) from
# whichever index it fetches.
set -euo pipefail

COMPONENT="main"
while getopts "c:" opt; do
  case "$opt" in
    c) COMPONENT="$OPTARG" ;;
    *) echo "usage: build-apt-repo.sh [-c <component>] <repo-root> <deb>..." >&2; exit 2 ;;
  esac
done
shift "$((OPTIND - 1))"

REPO="${1:?usage: build-apt-repo.sh [-c <component>] <repo-root> <deb>...}"; shift
[ "$#" -ge 1 ] || { echo "no .deb files given" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DIST="$REPO/dists/stable"

echo "== adding $# package(s) to component '$COMPONENT' =="
for deb in "$@"; do
  # Debian pool layout: pool/<component>/<first letter>/<package name>/.
  pkg="$(dpkg-deb -f "$deb" Package)"
  pooldir="$REPO/pool/$COMPONENT/${pkg:0:1}/$pkg"
  mkdir -p "$pooldir"
  cp -f "$deb" "$pooldir/"
  echo "   + $pkg <- $(basename "$deb")"
done

# Regenerate the Packages index for EVERY component already present in the
# tree (not just the one we touched) so the shared Release checksums stay
# consistent — a signed Release that references a stale Packages fails apt's
# verification. Components are discovered from pool/ subdirs, always including
# the one we just wrote.
components=()
for d in "$REPO"/pool/*/; do
  [ -d "$d" ] || continue
  components+=("$(basename "$d")")
done
# Ensure the just-written component is present even on a first-ever build.
case " ${components[*]} " in *" $COMPONENT "*) ;; *) components+=("$COMPONENT") ;; esac

echo "== generating Packages indexes for components: ${components[*]} =="
for comp in "${components[@]}"; do
  PKGIDX="$(mktemp)"
  # Scan only this component's pool subtree; rewrite the recorded Filename so
  # it stays repo-root-relative (apt-ftparchive records the path it scanned).
  ( cd "$REPO" && apt-ftparchive packages "pool/$comp" > "$PKGIDX" )
  for arch in all arm64; do
    BINDIR="$DIST/$comp/binary-$arch"
    mkdir -p "$BINDIR"
    cp "$PKGIDX" "$BINDIR/Packages"
    gzip -9c "$BINDIR/Packages" > "$BINDIR/Packages.gz"
  done
  rm -f "$PKGIDX"
done

echo "== generating Release (components: ${components[*]}) =="
# apt-ftparchive reads the component list from the config; pass the discovered
# set so a repo that only has `main` doesn't advertise an empty `prerelease`.
comp_str="${components[*]}"
( cd "$REPO" && APT_FTPARCHIVE_COMPONENTS="$comp_str" \
    apt-ftparchive -c "$HERE/apt-ftparchive.conf" \
    -o "APT::FTPArchive::Release::Components=$comp_str" \
    release dists/stable > "$DIST/Release" )

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
