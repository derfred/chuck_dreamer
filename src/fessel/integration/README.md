# Fessel integration test (Slice 1.5)

Deploys the full Fessel system into a fresh Kubernetes namespace and drives
the streaming start/stop chain end-to-end, then tears the namespace down.
Runs on every PR via `.github/workflows/fessel-integration.yml`:
build+push images (GitHub-hosted) → deploy + test on the self-hosted
`bugzoo` runner → publish JUnit as a separate "fessel integration" check.

## What it proves

| Assertion | What it validates |
|---|---|
| `happy_path_activation` | Browser fetches a signed WHEP URL from webui → mediamtx validates the JWT locally against webui's JWKS (no callback) → `runOnDemand` → supervisor (in-cluster DNS) → MQTT → `video` state machine `off→starting→running` → SRT publish to mediamtx (`is publishing … H264`). |
| `teardown_returns_off` | Disconnect → `runOnDemandCloseAfter` → deactivate → state machine returns to `off`. |
| `mint_requires_auth` | A WHEP-URL mint **without** the operator identity header is refused (401) — the Slice 2 auth gate. No token is issued, so there is no path to Pi activation. |
| `token_rejection_no_activation` | A tampered token is rejected at mediamtx/webui **before** any Pi activation (state stays `off`). The core security property. |
| `rapid_reconnect_no_leak` | N rapid connect/disconnect cycles leave the state machine in `off` with no leaked encoder — the regression the live state machine exists to prevent. |

These exercise the real installation assets: the Pi container installs the
built **dpkg**; mediamtx + webui are deployed from the built images.

## Tailnet substitutions (per the Slice 1.5 plan)

The two cluster↔Pi legs use in-cluster Service DNS instead of Tailscale:
- mediamtx → supervisor (`runOnDemand`): `http://supervisor:8443` (egress stand-in)
- video → mediamtx (SRT publish): `mediamtx-srt:8890` (ingress stand-in)

Both are config-only on the Fessel side, so the same dpkg/images run unmodified.

## Auth substitution (Slice 2)

oauth2-proxy is **not** in the integration loop (Slice 2 §6). The driver
reaches webui directly via the in-cluster Service — the same network
position oauth2-proxy forwards from in production. Slice 2 gates the
WHEP-URL mint on the identity header oauth2-proxy injects, so the driver
supplies that header itself (`X-Auth-Request-User`, overridable via
`FESSEL_AUTH_USER_HEADER` / `FESSEL_TEST_OPERATOR`) — a faithful stand-in
for the proxy. The in-cluster JWT-validation property is what this tier
covers; a full oauth2-proxy-fronted variant is a later-slice activity.

## Why there's no automated data-plane (video-bytes) assertion

The automated suite intentionally does **not** assert that video bytes reach
the browser. ICE between the in-cluster **headless Chrome** test client and
mediamtx never completes (`peer connection state: connecting → deadline
exceeded`) even though both pods have routable IPs and JWT auth + SDP
signaling + SRT publish all succeed — a containerized-Chrome / k8s UDP-ICE
limitation of the **test client**, not a Fessel defect. Such a test could
never pass here, so it isn't run.

The control plane still proves the chain up to the media plane: `happy_path`
drives activation → SRT publish, confirmed by mediamtx logging
`is publishing ... H264`. The remaining hop (media decoded in a browser) is
verified **manually** via the live-preview workflow below, in a real browser.

This is one of the Slice 1.5 §6 coverage gaps (alongside hardware encoder,
Tailscale Services, real cellular); a future physical-Pi / real-browser tier
can add an automated data-plane assertion there.

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

## Debugging a failed run

When the `fessel integration` check is red, work through these in order.

**1. Read the CI log first — it is self-contained.**
On failure, `run-test.sh` dumps a full post-mortem after the assertions:
the driver log (with `[PASS]`/`[FAIL]`/`[ice]` lines and the JUnit), then
for **mediamtx, webui, and pi**: `describe`, current logs, and previous
logs (if a pod restarted). `deploy.sh` separately dumps describe + logs if
any rollout fails or a pod is unstable. So most root causes are visible
without cluster access:
```
gh run list --workflow fessel-integration.yml -L 5
gh run view <run-id> --log                      # everything
gh run view <run-id> --log | grep -E '\[PASS\]|\[FAIL\]|\[ice\]|INTEGRATION:|ERR|error'
gh run view <run-id> --log-failed               # only failed steps
```

**2. Map the failure to a layer.** The assertions localize it:
- `happy_path_activation` fails at WHEP/auth → check **mediamtx** logs
  (look for `authentication failed`, `codecs`, `kid`, JWKS fetch errors)
  and that **webui** `/jwks` is reachable from mediamtx.
- `happy_path` reaches WHEP but never `running` → check **mediamtx**
  `runOnDemand` (did the `curl`/`wget` to supervisor fire?) and **pi** logs
  (`relay activate` in supervisor, `activate`/`live launch`/state changes in
  video). The supervisor `/state/live` history is the source of truth for
  the `off→starting→running` progression.
- `teardown_returns_off` fails → `runOnUnDemand` / deactivate path.
- `rapid_reconnect_no_leak` fails → the video state machine (run the unit
  tests: `make -C src/fessel test-video`).
- A pod CrashLoopBackOff before tests even run → `deploy.sh`'s rollout +
  stability diagnostics already printed the crash logs.

**3. Reproduce live in the fixed dev namespace.** The per-run `fessel-it-*`
namespace is auto-deleted, but you can render the exact stack from the
jsonnet library into the persistent `fessel-integration-test` namespace and
poke at it directly (this is how the original bring-up was debugged):
```
export KUBECONFIG=~/.kube/config-bugzoo-direct
cd src/fessel
TAG=<an image tag that exists, e.g. a recent run id or 'live-preview'>
tk eval deploy/jsonnet/envs/integration.jsonnet \
  -V ns=fessel-integration-test -V image_tag=$TAG \
  -V registry=ghcr.io/derfred -V whep_secret=devsecret \
  | kubectl apply -f -
kubectl rollout status deploy/mediamtx -n fessel-integration-test
# then: kubectl logs / exec / get events, run a driver-probe pod, etc.
# clean up:  kubectl delete deploy,svc,cm,secret,job --all -n fessel-integration-test
```
Bump mediamtx logging by piping the rendered output through
`sed 's/logLevel: info/logLevel: debug/'` before `kubectl apply` (or set it
in `deploy/jsonnet/mediamtx.libsonnet`) — reveals per-session ICE / DTLS /
auth detail.

**4. Watch a CI run live.** While the `integration` job runs, its
`fessel-it-<run-id>` namespace exists for ~5 min:
```
NS=fessel-it-<run-id>
kubectl get pods -n $NS -w
kubectl logs -n $NS -l app=mediamtx -f      # publish/read/runOnDemand/auth
kubectl logs -n $NS -l app=pi -f            # supervisor relays + video state
```

**5. Run the driver against a live deployment.** Launch the driver image as
a one-off pod (set `CONTROL_PLANE_ONLY=1` to skip nothing extra, or tweak
`RECONNECT_CYCLES`) to iterate on the test itself without a full CI cycle:
```
kubectl run driver-probe -n fessel-integration-test --restart=Never \
  --image=ghcr.io/derfred/fessel-test-driver:$TAG \
  --env=WEBUI=http://webui:8000 --env=MEDIA=http://mediamtx:8889 \
  --env=SUPERVISOR=http://supervisor:8443
kubectl logs -f driver-probe -n fessel-integration-test
```

**Knobs for more logging:**
- mediamtx: set `logLevel: debug` in `deploy/jsonnet/mediamtx.libsonnet`.
- supervisor/video: already structured JSON to stdout; `/state/live` on
  supervisor exposes the live-state history over HTTP.
- driver: `happy_path` already calls `connect(debug=True)` to dump ICE
  candidates; pass `debug=True` in other `connect()` calls to see their SDP.

## Deploy library (single source of truth)

All cluster objects are rendered from the jsonnet library in
`src/fessel/deploy/jsonnet/` — the **single source of truth** shared by
production and the tests (per architecture §5.4). One library, three shapes
selected by config:

| Env file | Mode | Used by |
|---|---|---|
| `envs/integration.jsonnet` | `webrtc=podip`, in-cluster test-Pi, no ingress | per-PR integration test |
| `envs/live-preview.jsonnet` | `webrtc=nodeport`, public ingress, test-Pi | on-demand live preview |
| (production) | `webrtc=nodeport`, Tailscale Services, real off-cluster Pi | `bugzoo-infrastructure` |

**Production consumes the same library via reverse import**: it's a
`jsonnet-bundler` git dependency in `bugzoo-infrastructure`
(`jsonnetfile.json` → `derfred/chuck_dreamer//src/fessel/deploy/jsonnet`),
wrapped by `lib/fessel/` there and merged into `environments/bugzoo`. So the
library is *owned* with the Fessel code and *consumed* by infra — bump it by
re-running `jb update` in the infra repo. This removes the prior drift where
test and shipped manifests were separate copies.

## Other deviations from the plan

- **Camera**: uses `video`'s built-in `videotestsrc` (moving pattern), not
  `v4l2loopback`. The loopback module isn't loaded on the bugzoo nodes and
  loading it means modifying shared production nodes. The MJPG `v4l2src`
  capture preamble is therefore not exercised here (already a §6 gap).
