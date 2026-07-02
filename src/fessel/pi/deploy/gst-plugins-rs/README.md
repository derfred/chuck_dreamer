# gstreamer1.0-plugins-rs-fessel — WHIP plugins for the Pi, as a .deb

Debian does not package gst-plugins-rs' `net/webrtc` plugin, so the Pi's WHIP
uplink elements are cross-built here and shipped as a proper .deb through the
existing signed Fessel apt repo (`../apt/README.md`). This replaces the manual
scp/install flow in `whip-relay/deploy/pi-plugin/`.

The package ships **both** plugins the uplink needs:

| File | Element | Why |
|---|---|---|
| `libgstrswebrtc.so` | `whipclientsink` | the WHIP publisher |
| `libgstrsrtp.so` | `rtpgccbwe` | GCC bandwidth estimation — without it congestion control is disabled and the bitrate bounces instead of adapting (confirmed in the 2026-07-02 uplink bake-off) |

## Install on the Pi

The Pi already trusts the Fessel apt repo, so:

```sh
sudo apt update
sudo apt install gstreamer1.0-plugins-rs-fessel
gst-inspect-1.0 whipclientsink   # sanity check
```

If the source line in `/etc/apt/sources.list.d/fessel.list` still pins
`arch=all` (pre-plugin onboarding), change it to `arch=all,arm64` first — see
`../apt/README.md`.

## Versioning and rebuild triggers

Version = `<upstream>+fessel<rev>+gst<minor>`, e.g. `0.13.6+fessel1+gst1.26`.

The plugin ABI tracks the **GStreamer minor** it was built against; the
package Depends on `libgstreamer1.0-0 (<< next-minor)` so a Pi OS upgrade to a
newer GStreamer refuses at apt level instead of failing mysteriously at
runtime. When that happens: pick the gst-plugins-rs branch targeting the new
GStreamer (0.13 ↔ 1.26), rebuild, and the `+gst` suffix moves.

Library Depends are derived with `dpkg-shlibdeps` inside the build container
(don't hand-write them — trixie's `t64` package renames alone would get it
wrong). `gstreamer1.0-plugins-bad` (webrtcbin) and `gstreamer1.0-nice` (ICE)
are added manually: elements load dynamically, invisible to shlibdeps, and a
missing libnice fails as an opaque "Failed to request pad from webrtcbin".

## Building

CI: the **fessel gst-plugins-rs dpkg** workflow
(`.github/workflows/fessel-gst-plugins-rs.yml`) — manual dispatch (inputs:
deb revision, gst-plugins-rs branch), also runs on changes to this directory.
It cross-builds under QEMU (~45–90 min, cached) and publishes the .deb into
the signed apt repo on gh-pages. Deliberately separate from the `fessel
release` workflow: the plugin versions on upstream + the Pi's GStreamer, not
on Fessel releases, and the build is far too slow to hang on every tag.

Locally (Docker + buildx):

```sh
./build-deb.sh                    # -> dist/gstreamer1.0-plugins-rs-fessel_*.deb
DEB_REVISION=2 ./build-deb.sh     # repackage of the same upstream
```

## If Debian ever packages it

Prefer the official package and retire this one (`apt remove
gstreamer1.0-plugins-rs-fessel` first — both would ship the same plugin
filenames, so dpkg will refuse to install them side by side, which is the
correct failure).
