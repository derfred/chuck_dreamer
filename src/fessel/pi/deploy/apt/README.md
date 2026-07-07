# Fessel apt repo

The Pi installs and upgrades `fessel-monitor` from a **signed apt repository**
published to GitHub Pages by CI. The package ships the Fessel Python modules as
plain files under `/opt/fessel/lib` and relies on Debian **trixie** `python3-*`
packages (no venv, no pip).

Repo URL: `https://derfred.github.io/chuck_dreamer`
Layout: suite `stable`, components `main` (real releases) + `prerelease`
(edge/rc builds — see [Edge channel](#edge-channel-prereleases)), architectures
`all` + `arm64`.

## Pi onboarding (one-time)

```sh
# 1. Trust the signing key (published alongside the repo)
sudo curl -fsSL https://derfred.github.io/chuck_dreamer/fessel-archive-keyring.gpg \
  -o /usr/share/keyrings/fessel-archive-keyring.gpg

# 2. Add the signed source (arch=all,arm64: fessel-monitor is arch-independent,
#    gstreamer1.0-plugins-rs-fessel ships compiled arm64 plugins)
echo "deb [arch=all,arm64 signed-by=/usr/share/keyrings/fessel-archive-keyring.gpg] https://derfred.github.io/chuck_dreamer stable main" \
  | sudo tee /etc/apt/sources.list.d/fessel.list

# 3. Install
sudo apt update
sudo apt install fessel-monitor gstreamer1.0-plugins-rs-fessel
```

(Pis onboarded before the plugin package existed have `arch=all` in
`fessel.list` — edit it to `arch=all,arm64` once.)

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
- Manual fallback: run the `fessel release` workflow with **no**
  `prerelease_suffix` — it builds from the current `schemas/pyproject.toml`
  version into `main`, same as a tag push.
- `gstreamer1.0-plugins-rs-fessel` (the WHIP uplink plugins, arm64) is
  published into the SAME repo by its own workflow — it versions on
  gst-plugins-rs upstream + the Pi's GStreamer minor, not on Fessel releases.
  See `pi/deploy/gst-plugins-rs/README.md`. The two publish jobs share a
  `concurrency: fessel-apt-repo` group so gh-pages pushes never race.

## Edge channel (prereleases)

The **edge** channel lets you push a build to the Pi **and** the cluster without
cutting a real release — no `pyproject` bump, no `vX.Y.Z` tag. It's the repo's
second component, `prerelease`, kept entirely separate from `main` so a stable
Pi never sees an edge build unless it opts in.

### Cut a prerelease (maintainer)

Actions → **fessel release** → *Run workflow* → set **`prerelease_suffix`**
(e.g. `rc1`). The effective version is `<pyproject>-<suffix>`:

- Pi dpkg: `fessel-monitor_0.5.0~rc1_all.deb`, published into the `prerelease`
  component. The `~` makes `0.5.0~rc1` sort **below** the stable `0.5.0`, so the
  eventual real release cleanly supersedes it.
- Cluster image: `ghcr.io/<owner>/fessel-webui:0.5.0-rc1` (Docker tags forbid
  `~`, so the image uses `-`). No `:latest`/stable pointer is moved. Deploy it
  by pinning that tag in the cluster manifest.

Leave `prerelease_suffix` empty for a normal release into `main`.

### Opt a Pi into edge

Use the `fessel-channel` helper (shipped in `/usr/bin` by the package):

```sh
sudo fessel-channel edge      # add the `prerelease` component + pin, apt update
sudo apt upgrade              # now tracks the newest prerelease
sudo fessel-channel status    # show active channel + installed version
```

`edge` installs `/etc/apt/preferences.d/fessel-edge` pinning the `prerelease`
component to priority 1001. That pin — not the version number — is what makes
`apt upgrade` prefer an rc (whose version sorts *below* stable); without it,
merely listing the component would never pull one. So edge is a channel that
**auto-tracks the newest prerelease**, not a per-version manual pin.

### Return a Pi to stable

```sh
sudo fessel-channel stable    # remove the pin + component, apt update
# if the box is currently on a prerelease, step it back down explicitly:
sudo apt install --allow-downgrades fessel-monitor
```

(You can still pin one exact prerelease by hand without switching channels:
`sudo apt install fessel-monitor=0.5.0~rc1` — note the `~`.)

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

### Recording upload (uploader section)

The uploader PUTs **flagged** recordings to webui-backend's tailnet-only
recording-ingest endpoint over Tailscale (Slice 5.5). Set
`uploader.ingest_url_base` in `/etc/fessel/fessel.yaml` to the backend's
recording-ingest Tailscale ingress (the `webui-recording-ingest` Service's
magicDNS name + port, e.g. `https://fessel-ingest.<tailnet>.ts.net:8443`). The
uploader is an **HTTPS client only** (`python3-httpx`, in **Depends**) — it
holds **no cluster-store credentials**; auth is by Tailscale identity at the
network layer.

**Upgrade note (from a pre-5.5 install):** delete the obsolete
`uploader.minio` block (endpoint + `access_key`/`secret_key`) from
`/etc/fessel/fessel.yaml` — those S3 credentials no longer belong on the Pi.
`python3-minio` is no longer a dependency; nothing on the Pi imports the MinIO
SDK. Which store the recordings land in (MinIO or disk) is now a cluster-side
choice on webui-backend (`recordings_storage.backend`), invisible to the Pi.
