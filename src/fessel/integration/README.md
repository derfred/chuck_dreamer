# Fessel integration test

Deploys the full Fessel system into a fresh Kubernetes namespace and drives
the live-streaming chain end-to-end, then tears the namespace down.
Runs on every PR via `.github/workflows/fessel-integration.yml`:
build+push images (GitHub-hosted) → deploy + test on the self-hosted
`bugzoo` runner → publish JUnit as a separate "fessel integration" check.

The chain under test is

```
driver POST /whep (identity header)
  → webui health gate (cached Pi health; ambiguous → proceed)
  → first viewer: POST supervisor:/control/live/activate  (blocks ≤15s)
  → supervisor → MQTT → video attaches rtph264pay ! whipclientsink
  → WHIP POST to webui:8001/whip/ingest  (ingest listener)
  → RTP into the relay's permanent shared track
  → 201 + SDP answer to the driver → ICE/DTLS → decoded frames in aiortc
```

## What it proves

| Assertion | What it validates |
|---|---|
| `happy_path_activation` | `POST /whep` with the oauth2-proxy identity header → 201 + SDP answer + `Location: /whep/{id}` → supervisor live state `off→starting→running` → **data plane, ingest leg**: `/metrics` shows `fessel_relay_ingest_live 1` and `fessel_relay_ingest_packets_total` *increasing* across two samples → **data plane, viewer leg**: the aiortc peer reaches `connected` and decodes a video frame (podip mode makes in-cluster ICE work). |
| `teardown_returns_off` | `DELETE /whep/{id}` → viewer count 0 → relay idle timeout (`FESSEL_LIVE_IDLE_TIMEOUT_S`, 10s) → supervisor deactivate → live state returns to `off`, `fessel_relay_ingest_live 0`, viewer gauge 0. |
| `whep_requires_auth` | `POST /whep` **without** the identity header → 401 `{"detail": "authentication required"}`, and **no side effects**: live state stays `off`, `fessel_relay_activations_total` and `fessel_relay_viewer_sessions_total` unchanged |
| `rapid_reconnect_no_leak` | N rapid connect/DELETE cycles leave viewers=0, then `off` with `ingest_live 0` — no leaked encoder, ingest session, or viewer PeerConnection. The regression the live state machine + permanent-shared-track design exist to prevent. |
| `control_*` (Slice 3) | Backend auth gate + `/api/control` forwarders → supervisor → mocked leaf actuators (jetson-mock pod, in-process fake WiZ), incl. send-and-verify failure surfacing. Unchanged. |
| `recording_roundtrip_via_ingest` (T5.5.2) | Start → stop → flag-for-upload → the test-Pi's `uploader` (real `http` driver) PUTs each file to webui's ingest listener → the **disk** backend persists to the PVC → upload reaches `uploaded`, listing shows `available_remote` → playback: playlist 200 + a `Range` segment fetch returns **206** with the right byte count. |
| `ingest_listener_accepts_put` (T5.5.3) | A PUT to `/recording-ingest/...` on the ingest listener (`:8001`) is accepted (201) — positive control for the bypass test. |
| `ingest_not_reachable_via_public_listener` (T5.5.3) | A PUT to `/recording-ingest/...` on the **public** listener (`:8000`) is **not** accepted (404/405 — the route exists only on the ingest listener). |
| `recordings_require_auth` (T5.5.3) | `/api/recordings`, `/playlist`, `/segment/<name>` all 401 `{"detail": "authentication required"}` without the oauth2-proxy identity header. |

These exercise the real installation assets: the Pi container installs the
built **dpkg** (incl. the cross-built `whipclientsink` GStreamer plugin);
webui is deployed from the built Go image (`webui/deploy/go.Dockerfile`).

## Two listeners

The Go webui runs two HTTP listeners (B5.5.7):

- **public `:8000`** — `/api/*`, `POST /whep` (+ `DELETE /whep/{id}`),
  `/metrics`, `/healthz`, static frontend. Behind oauth2-proxy in production.
- **ingest `:8001`** — ALL Pi→webui traffic: `PUT /recording-ingest/{id}/{name}`
  and `POST /whip/ingest` (WHIP signaling), plus its own `/healthz`
  (`listener: "ingest"`). Tailnet-only in production.

## Substitutions vs. production

- **Tailnet → in-cluster Service DNS.** webui→supervisor
  (`http://supervisor:8443`, live activate/deactivate + control relays) and
  Pi→webui ingest (`http://webui:8001`, WHIP publish + recording uploads) use
  Service DNS instead of Tailscale. Config-only; the same dpkg/images run
  unmodified. The production Tailscale ingest ingress is therefore not
  exercised here (§6 gap).
- **WebRTC `podip` mode.** The relay's ICE env is left empty, so Pion
  advertises plain host candidates — the webui pod IP — and the driver's
  aiortc peer does the same. Pod IPs are mutually routable in-cluster, so
  ICE/DTLS actually completes and the suite asserts decoded frames.
  Production uses `nodeport` mode (NAT1To1 node public IPs on the media
  NodePort) — covered by the live-preview workflow.
- **oauth2-proxy simulated via the identity header.** The driver reaches
  webui directly via the in-cluster Service — the same network position
  oauth2-proxy forwards from — and supplies the header the proxy would
  inject (`X-Auth-Request-User`, overridable via `FESSEL_AUTH_USER_HEADER` /
  `FESSEL_TEST_OPERATOR`). There is no JWT/JWKS/signed-URL machinery any
  more: header presence IS the authorisation on `/whep` and `/api/*`.
- **Camera**: `video`'s built-in `videotestsrc` (moving pattern), not a real
  webcam — the MJPG `v4l2src` preamble and the hardware encoder are §6 gaps.
- **Hardware encoder**: the test-Pi encodes with x264 (software); the Pi's
  `v4l2h264enc` behaviour is a §6 gap.

## The WHEP client is aiortc (no browser)

The driver builds a real recvonly `RTCPeerConnection` with
[aiortc](https://github.com/aiortc/aiortc), POSTs the offer to `/whep`,
applies the answer, and asserts ICE connects + a frame decodes. This
replaced the headed-Chrome/Xvfb/Playwright rig:

- The old blocker ("headless Chrome can't complete ICE in-cluster") was a
  media-server-config + Chrome-candidate artefact; with podip host candidates
  a plain Python peer connects fine.
- aiortc gives full control over candidates (no mDNS obfuscation, no STUN
  needed) and an H.264 decoder via PyAV — so codec negotiation AND decode are
  still exercised, without a 1GB browser image.

Note the relay answers the WHEP POST before/independently of the viewer's
ICE completing, and the activation/ingest metrics don't depend on the viewer
media path — so the metrics assertions and the viewer-media assertion are
independent witnesses.

## Live preview (manual verification of the production WebRTC path)

`.github/workflows/fessel-live-preview.yml` (`live-preview/`) deploys the
same system with **production WebRTC exposure** — `nodeport` mode (media on
node public IPs), `/whep` + the `/live` page behind HTTPS Ingresses — so a
real public browser can connect. Usage (Actions → "fessel live preview" →
Run workflow): `action=up` deploys and prints
`https://fessel-live.derfred.com/` (TTL then auto-teardown); `action=down`
tears down now. This covers the NodePort/public-IP ICE path the per-PR test
substitutes away.

## Debugging a failed run

**1. Read the CI log first — it is self-contained.**
On failure, `run-test.sh` dumps the driver log (with `[PASS]`/`[FAIL]`/
`[ice]`/`[media]` lines and the JUnit), then for **webui and pi**:
`describe`, current logs, and previous logs (if a pod restarted).
`deploy.sh` separately dumps diagnostics if any rollout fails or a pod is
unstable.
```
gh run list --workflow fessel-integration.yml -L 5
gh run view <run-id> --log
gh run view <run-id> --log | grep -E '\[PASS\]|\[FAIL\]|\[ice\]|\[media\]|INTEGRATION:|ERR|error'
gh run view <run-id> --log-failed
```

**2. Map the failure to a layer.**
- `whep_requires_auth` / 401 shape wrong → webui auth header config
  (`FESSEL_AUTH_USER_HEADER`).
- `happy_path` fails at the WHEP POST (503 `live_unavailable`) → the health
  gate fired: check webui's health monitor + supervisor reachability.
- `happy_path` gets 504 `live_timeout` → activation fired but WHIP ingest
  never came up: check **webui** logs (`live activation`, `WHIP offer
  received`, `ingest connection state`) and **pi** logs (`relay activate` in
  supervisor; `activate`/live-launch/state changes + whipclientsink errors
  in video). Supervisor `/state/live` history is the source of truth for
  `off→starting→running`.
- `happy_path` passes metrics but `wait_media` fails → viewer-leg ICE: check
  the `[ice]` candidate dumps in the driver log and webui's
  `ICE candidate pair selected` lines (podip: both should be pod IPs).
- `teardown_returns_off` fails → idle-timeout → deactivate path
  (webui `live teardown` logs, supervisor deactivate).
- `rapid_reconnect_no_leak` fails → viewer bookkeeping in the relay or the
  Pi live state machine (`make -C src/fessel test-video`).
- A pod CrashLoopBackOff before tests even run → `deploy.sh`'s rollout +
  stability diagnostics already printed the crash logs.

**3. Reproduce live in the fixed dev namespace.** The per-run `fessel-it-*`
namespace is auto-deleted, but you can render the exact stack into the
persistent `fessel-integration-test` namespace:
```
export KUBECONFIG=~/.kube/config-bugzoo-direct
cd src/fessel
TAG=<an image tag that exists, e.g. a recent run id or 'live-preview'>
tk eval deploy/jsonnet/envs/integration.jsonnet \
  -V ns=fessel-integration-test -V image_tag=$TAG \
  -V registry=ghcr.io/derfred \
  | kubectl apply -f -
kubectl rollout status deploy/webui -n fessel-integration-test
# then: kubectl logs / exec / get events, run a driver-probe pod, etc.
# clean up:  kubectl delete deploy,svc,cm,secret,job,pvc --all -n fessel-integration-test
```

**4. Watch a CI run live.** While the `integration` job runs, its
`fessel-it-<run-id>` namespace exists for ~5 min:
```
NS=fessel-it-<run-id>
kubectl get pods -n $NS -w
kubectl logs -n $NS -l app=webui -f    # relay: WHIP/WHEP sessions, ICE pairs, activation
kubectl logs -n $NS -l app=pi -f       # supervisor relays + video state machine
```

**5. Run the driver against a live deployment.** Launch the driver image as
a one-off pod to iterate on the test itself without a full CI cycle:
```
kubectl run driver-probe -n fessel-integration-test --restart=Never \
  --image=ghcr.io/derfred/fessel-test-driver:$TAG \
  --env=WEBUI=http://webui:8000 --env=WEBUI_INGEST=http://webui:8001 \
  --env=SUPERVISOR=http://supervisor:8443
kubectl logs -f driver-probe -n fessel-integration-test
```

**Knobs for more logging:**
- webui relay: per-candidate ICE detail is at `slog` debug level; the
  selected-pair + session lines are already on at info.
- supervisor/video: structured JSON to stdout; `/state/live` on supervisor
  exposes the live-state history over HTTP.
- driver: `happy_path` calls `connect(debug=True)` to dump both SDPs'
  candidates; `/metrics` on webui is scrapeable from any pod in the ns.

## Deploy library (single source of truth)

All cluster objects are rendered from the jsonnet library in
`src/fessel/deploy/jsonnet/` — shared by production and the tests
(architecture §5.4). One library, three shapes selected by config:

| Env file | Mode | Used by |
|---|---|---|
| `envs/integration.jsonnet` | `webrtc=podip`, in-cluster test-Pi, disk recordings backend, no ingress | per-PR integration test |
| `envs/live-preview.jsonnet` | `webrtc=nodeport`, public ingress, test-Pi | on-demand live preview |
| (production) | `webrtc=nodeport`, Tailscale Services + webui sidecar, real off-cluster Pi | `bugzoo-infrastructure` |

**Production consumes the same library via reverse import**: it's a
`jsonnet-bundler` git dependency in `bugzoo-infrastructure`
(`jsonnetfile.json` → `derfred/chuck_dreamer//src/fessel/deploy/jsonnet`),
wrapped by `lib/fessel/` there and merged into `environments/bugzoo` — bump
it by re-running `jb update` in the infra repo.
