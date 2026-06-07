#!/usr/bin/env bash
# Cut a Fessel release. This is the SINGLE entry point for bumping the version:
# it rewrites the one source of truth (schemas/pyproject.toml), regenerates the
# committed deploy/jsonnet/VERSION (which the jsonnet default webui image tag
# derives from), commits both, and pushes the v<version> tag. The tag push
# triggers .github/workflows/fessel-release.yml, whose `setup` job re-derives
# the version from pyproject.toml and asserts the tag matches — so this script
# and the workflow cross-check each other.
#
# VERSION is committed (not generated at deploy time) because `tk eval` renders
# the jsonnet tree on every PR (integration / live-preview), and a missing
# importstr target fails the render. This script is the only thing that should
# ever touch it — never hand-edit deploy/jsonnet/VERSION.
#
# Dry run by default: it prints the planned edits without writing or pushing.
# Pass --push to actually edit files, commit, tag, and push.
#
# Usage:
#   release.sh <version>            # dry run: show what would change
#   release.sh <version> --push     # bump, commit, tag, push (triggers release)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FESSEL_ROOT="$HERE"                               # src/fessel
PYPROJECT="$FESSEL_ROOT/schemas/pyproject.toml"
VERSION_FILE="$FESSEL_ROOT/deploy/jsonnet/VERSION"
RELEASE_BRANCH="main"

die() { echo "release: $*" >&2; exit 1; }

# --- args ---------------------------------------------------------------------
VERSION=""
PUSH=0
for arg in "$@"; do
  case "$arg" in
    --push) PUSH=1 ;;
    -*) die "unknown flag: $arg" ;;
    *) [ -z "$VERSION" ] || die "unexpected extra argument: $arg"; VERSION="$arg" ;;
  esac
done
[ -n "$VERSION" ] || die "usage: release.sh <version> [--push]"
# Semver X.Y.Z (the tag/image-tag/dpkg version shape used across the release).
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "version must be X.Y.Z, got '$VERSION'"

# --- read current version (same sed idiom as build-dpkg.sh / the workflow) ----
pyproject_version() {
  sed -n 's/^version = "\([^"]*\)".*/\1/p' "$PYPROJECT" | head -n1
}
CURRENT="$(pyproject_version)"
[ -n "$CURRENT" ] || die "could not read current version from $PYPROJECT"

echo "== fessel release =="
echo "  current: $CURRENT"
echo "  new:     $VERSION"

[ "$VERSION" != "$CURRENT" ] || die "new version equals current ($CURRENT) — nothing to release"
# Refuse a non-increasing bump (sort -V: largest last must be the new version).
GREATEST="$(printf '%s\n%s\n' "$CURRENT" "$VERSION" | sort -V | tail -n1)"
[ "$GREATEST" = "$VERSION" ] || die "new version $VERSION is not greater than current $CURRENT"

# --- preflight: clean tree on the release branch ------------------------------
cd "$FESSEL_ROOT"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "$RELEASE_BRANCH" ] || die "on branch '$BRANCH', expected '$RELEASE_BRANCH'"
[ -z "$(git status --porcelain)" ] || die "working tree is dirty — commit or stash first"
TAG="v$VERSION"
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  die "tag $TAG already exists"
fi

echo
echo "  will edit:   schemas/pyproject.toml  (version -> $VERSION)"
echo "  will write:  deploy/jsonnet/VERSION  ($VERSION)"
echo "  will commit: fessel: release $VERSION"
echo "  will tag:    $TAG"
echo "  will push:   $RELEASE_BRANCH + $TAG -> origin"

if [ "$PUSH" -ne 1 ]; then
  echo
  echo "== dry run (no changes made). re-run with --push to release. =="
  exit 0
fi

# --- apply --------------------------------------------------------------------
# Anchored to the same `^version = "..."` line build-dpkg.sh / the workflow read.
# Use a temp file so a sed failure can't truncate pyproject.toml.
tmp="$(mktemp)"
sed 's/^version = "[^"]*"/version = "'"$VERSION"'"/' "$PYPROJECT" > "$tmp"
mv "$tmp" "$PYPROJECT"
[ "$(pyproject_version)" = "$VERSION" ] || die "pyproject bump did not take — aborting"

printf '%s\n' "$VERSION" > "$VERSION_FILE"

git add schemas/pyproject.toml deploy/jsonnet/VERSION
git commit -m "fessel: release $VERSION"
git tag "$TAG"

echo
echo "== pushing $RELEASE_BRANCH and $TAG to origin =="
git push origin "$RELEASE_BRANCH"
git push origin "$TAG"

echo
echo "== released $VERSION — fessel-release.yml will publish the webui image + Pi dpkg =="
