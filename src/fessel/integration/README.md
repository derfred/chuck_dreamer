# Fessel integration test (Slice 1.5)

Deploys the full Fessel system into a fresh Kubernetes namespace and drives
the streaming start/stop chain end-to-end, then tears the namespace down.
Runs on every PR via `.github/workflows/fessel-integration.yml`:
build+push images (GitHub-hosted) → deploy + test on the self-hosted
`bugzoo` runner → publish JUnit as a separate "fessel integration" check.

## What it proves

| Assertion | What it validates |
|---|---|
| `happy_path_activation` | Browser WHEP → webui `/auth` (JWT) → mediamtx `runOnDemand` → supervisor (in-cluster DNS) → MQTT → `video` state machine `off→starting→running` → SRT publish to mediamtx (`is publishing … H264`). |
| `teardown_returns_off` | Disconnect → `runOnDemandCloseAfter` → deactivate → state machine returns to `off`. |
| `token_rejection_no_activation` | A tampered token is rejected at mediamtx/webui **before** any Pi activation (state stays `off`). The core security property. |
| `rapid_reconnect_no_leak` | N rapid connect/disconnect cycles leave the state machine in `off` with no leaked encoder — the regression the live state machine exists to prevent. |

These exercise the real installation assets: the Pi container installs the
built **dpkg**; mediamtx + webui are deployed from the built images.

## Tailnet substitutions (per the Slice 1.5 plan)

The two cluster↔Pi legs use in-cluster Service DNS instead of Tailscale:
- mediamtx → supervisor (`runOnDemand`): `http://supervisor:8443` (egress stand-in)
- video → mediamtx (SRT publish): `mediamtx-srt:8890` (ingress stand-in)

Both are config-only on the Fessel side, so the same dpkg/images run unmodified.

## Known gap: browser WebRTC media reception

`data_plane_media_flows` is a **non-blocking xfail** (`<skipped>` in JUnit).
ICE between the in-cluster **headless Chrome** test client and mediamtx does
not complete (`peer connection state: connecting → deadline exceeded`),
despite both pods having routable IPs and JWT auth + SDP signaling
succeeding. This is a containerized-Chrome / k8s UDP-ICE limitation of the
**test client**, not a Fessel defect — the Pi *is* publishing H264 to
mediamtx (verified in mediamtx logs), so the media plane works up to the
browser's ICE agent.

This sits alongside the other Slice 1.5 §6 coverage gaps (hardware encoder,
Tailscale Services, production NodePort/public-IP WebRTC, real cellular).
Cover real WebRTC media reception in the physical-Pi / real-browser tier.

Set `DATA_PLANE_BLOCKING=1` on the test Job to make this assertion blocking
again once such a tier exists.

## Live preview (manual video verification)

Because the automated browser can't complete WebRTC ICE in-cluster, there's
an on-demand **live preview** that puts a *real* browser (yours) at the end:
`.github/workflows/fessel-live-preview.yml` (`live-preview/`).

It deploys the same system but with **production WebRTC exposure** — WebRTC
media over a NodePort on the node public IPs (`webrtcAdditionalHosts`), WHEP
signaling + the `/live` page behind HTTPS Ingresses — so a public browser can
actually connect. It runs the control-plane assertions (`CONTROL_PLANE_ONLY`),
then leaves the stack up and prints the URL.

Usage (Actions → "fessel live preview" → Run workflow):
- `action=up` (default): build images (tag `live-preview`), deploy, print
  `https://fessel-live.derfred.com/`, keep alive for `ttl_minutes` (default
  30), then auto-teardown. Open the URL, pick a mode, click "Start live
  view", and confirm the moving test pattern.
- `action=down`: tear down immediately.

DNS for `*.derfred.com` resolves to the cluster ingress; WebRTC media flows
directly to the node public IPs on NodePorts 31554 (UDP) / 31555 (TCP).
This exercises the production NodePort/public-IP WebRTC path that the per-PR
test overrides — closing that coverage gap manually, on demand.

## Other deviations from the plan

- **Camera**: uses `video`'s built-in `videotestsrc` (moving pattern), not
  `v4l2loopback`. The loopback module isn't loaded on the bugzoo nodes and
  loading it means modifying shared production nodes. The MJPG `v4l2src`
  capture preamble is therefore not exercised here (already a §6 gap).
- **Deploy**: self-contained templated manifests (`manifests/*.yaml.tmpl`
  rendered by `render.sh`) instead of a cross-repo Tanka environment, so CI
  in this repo needs no Tanka render. Same shape; same single-Secret wiring.
