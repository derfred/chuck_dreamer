#!/usr/bin/env bash
# Portable ${VAR} substitution for a fixed allowlist of vars, so the harness
# doesn't depend on `envsubst` (gettext) being present on the runner. Reads
# stdin, writes stdout. Only substitutes the known vars; leaves mediamtx's
# own $MTX_* runtime vars untouched.
set -euo pipefail

content="$(cat)"
for var in NS IMAGE_TAG REGISTRY FESSEL_WHEP_SECRET; do
  value="${!var-}"
  content="${content//\$\{$var\}/$value}"
done
# printf '%s\n' guarantees a trailing newline so concatenated docs with a
# following '---' separator stay on their own lines.
printf '%s\n' "$content"
