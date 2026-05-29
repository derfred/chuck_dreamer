#!/usr/bin/env bash
# ONE-TIME: generate the dedicated GPG key that signs the Fessel apt repo.
# Run this once locally, then:
#   - put the PRIVATE key (printed) into the GitHub Actions secret
#     FESSEL_APT_GPG_PRIVATE_KEY
#   - the PUBLIC key is published to the repo by CI (build-apt-repo.sh exports
#     fessel-archive-keyring.gpg); the Pi fetches it from the Pages URL.
#
# The private key NEVER gets committed. This script uses an ephemeral GNUPGHOME
# so it doesn't touch your personal keyring.
set -euo pipefail

NAME="${1:-Fessel apt signing key}"
EMAIL="${2:-fessel-apt@derfred.com}"
GNUPGHOME="$(mktemp -d)"; export GNUPGHOME
chmod 700 "$GNUPGHOME"
trap 'rm -rf "$GNUPGHOME"' EXIT

echo "== generating key for $NAME <$EMAIL> =="
gpg --batch --gen-key <<EOF
%no-protection
Key-Type: eddsa
Key-Curve: ed25519
Key-Usage: sign
Name-Real: $NAME
Name-Email: $EMAIL
Expire-Date: 0
%commit
EOF

KEY_ID="$(gpg --list-secret-keys --with-colons | awk -F: '/^sec:/{print $5; exit}')"
echo
echo "== KEY ID: $KEY_ID =="
echo
echo "----- PRIVATE KEY (paste into GH secret FESSEL_APT_GPG_PRIVATE_KEY) -----"
gpg --armor --export-secret-keys "$KEY_ID"
echo "----- END PRIVATE KEY -----"
echo
echo "Public keyring written to ./fessel-archive-keyring.gpg (committed for reference)."
gpg --export "$KEY_ID" > fessel-archive-keyring.gpg
