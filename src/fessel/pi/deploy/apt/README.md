# Fessel apt repo

The Pi installs and upgrades `fessel-monitor` from a **signed apt repository**
published to GitHub Pages by CI. The package ships the Fessel Python modules as
plain files under `/opt/fessel/lib` and relies on Debian **trixie** `python3-*`
packages (no venv, no pip).

Repo URL: `https://derfred.github.io/chuck_dreamer`
Layout: flat — suite `stable`, component `main`, architecture `all`.

## Pi onboarding (one-time)

```sh
# 1. Trust the signing key (published alongside the repo)
sudo curl -fsSL https://derfred.github.io/chuck_dreamer/fessel-archive-keyring.gpg \
  -o /usr/share/keyrings/fessel-archive-keyring.gpg

# 2. Add the signed source
echo "deb [arch=all signed-by=/usr/share/keyrings/fessel-archive-keyring.gpg] https://derfred.github.io/chuck_dreamer stable main" \
  | sudo tee /etc/apt/sources.list.d/fessel.list

# 3. Install
sudo apt update
sudo apt install fessel-monitor
```

Thereafter, upgrades are just:
```sh
sudo apt update && sudo apt upgrade
```
A specific version still resolves (old versions stay in the pool):
```sh
sudo apt install fessel-monitor=0.1.0
```

## How releases work

- Push a git tag `vX.Y.Z` → the `fessel release` workflow builds
  `fessel-monitor_X.Y.Z_all.deb` (in a `debian:trixie` container via
  `pi/deploy/dpkg/build-dpkg.sh`), adds it to the repo's `pool/`, regenerates +
  GPG-signs `dists/stable/{Release,Release.gpg,InRelease}`, and pushes the
  updated tree to the `gh-pages` branch.
- Manual fallback: run the workflow with an explicit `version` input.

## Signing key (one-time setup, done by a maintainer)

The repo is signed by a dedicated GPG key (separate from any personal key).

1. Generate it once: `pi/deploy/apt/gen-signing-key.sh`.
2. Paste the printed **private** key into the GitHub Actions secret
   `FESSEL_APT_GPG_PRIVATE_KEY` (Settings → Secrets → Actions).
3. The **public** key is exported to `fessel-archive-keyring.gpg` and published
   by CI; the Pi fetches it (step 1 above). The private key is never committed.

## One-time repo setup

- Create the `gh-pages` branch (empty is fine) and enable **GitHub Pages** to
  serve from it (Settings → Pages → branch `gh-pages`, root).
- CI accumulates into that branch; do not hand-edit it.

## Slice 4: capture layer (ring buffer, recordings, uploader)

The package now also ships the **uploader** process (`fessel-uploader.service`,
launcher `/usr/bin/fessel-uploader`) alongside supervisor and video. All three
read a **single** config conffile `/etc/fessel/fessel.yaml` (Requirements §6):
shared `mqtt`/`storage` sections at the top level plus per-process subtrees
(`video:` / `supervisor:` / `uploader:`).

### USB SSD layout (one-time, per Pi)

video (ring buffer + explicit recordings) and the uploader read/write a single
USB SSD mount. The mount path is the **one** shared `storage.ssd_path` in
`fessel.yaml` (default `/mnt/ssd`), read by all three processes — no longer
duplicated per file. The processes create the sub-layout at startup, but the
mount itself is the operator's responsibility:

```sh
# Mount the USB SSD at the configured path and make it writable by the Fessel
# processes (run as root by systemd here). Persist via /etc/fstab.
sudo mkdir -p /mnt/ssd
sudo mount /dev/sdX1 /mnt/ssd          # the SSD's partition
# Resulting layout (created by the processes):
#   /mnt/ssd/ring/                 rolling HLS ring buffer (always-on)
#   /mnt/ssd/recordings/explicit/  per-recording HLS dirs + metadata.json
#   /mnt/ssd/upload_queue/         flag-for-upload markers (.upload / .failed)
```

Sizing (ring duration vs. SSD size vs. bitrate) is a deployment choice — see the
`video.ring` block in `fessel.yaml`. The architecture prefers a short ring
(recent context, not archival); long retention is the upload-to-cluster path's
job.

### MinIO upload (uploader section)

The uploader ships **flagged** recordings to the cluster MinIO over Tailscale.
Set `uploader.minio.endpoint` / `bucket` / `access_key` / `secret_key` in
`/etc/fessel/fessel.yaml`. The production driver (`minio`) needs the
`python3-minio` package — it is in the deb's **Recommends** (installed by default
with `apt install`, but the import is lazy so the service still starts without
it; only the `minio` driver requires it). Credentials live in the conffile for
now; a later packaging slice wires them from a managed secret.
