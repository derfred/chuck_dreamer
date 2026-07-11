# fessel deploy library (jsonnet)

`src/fessel/deploy/jsonnet/` is the single source of truth for the
cluster-side deployment: the Go `webui` (backend + embedded frontend +
in-process Pion WHIP/WHEP relay), the Tailscale plumbing, and the test
envs. Production (bugzoo-infrastructure) consumes it via jsonnet-bundler:

```jsonnet
local fessel = import 'main.libsonnet';
{ items: fessel.objects(fessel.config + { ...overrides... }) }
```

Render an env locally (Tanka provides the evaluator; there is no vendor
dir — the library is dependency-free):

```sh
cd src/fessel/deploy/jsonnet
tk eval envs/integration.jsonnet -V ns=it -V image_tag=<tag> -V registry=ghcr.io/derfred
```

## Architecture (post-mediamtx)

mediamtx is **gone** (`mediamtx.libsonnet` deleted). The Go `webui`
image (built from `src/fessel/webui/deploy/go.Dockerfile`, image name
`fessel-webui` as before) replaces both the FastAPI backend and
mediamtx: one binary, one Deployment, `strategy: Recreate` **always**
(the single-replica relay holds per-session WebRTC state — architecture
§4.2).

### Relay exposure (production)

| Purpose | Protocol | Exposure |
|---|---|---|
| Frontend + `/api` + `/whep` signaling | TCP :8000 | public HTTPS Ingress, behind oauth2-proxy |
| WHEP viewer media | UDP | NodePort Service `webui-media` (`cfg.webrtc.udpNodePort`, default 31554), `externalTrafficPolicy: Local`; node public IPs advertised as ICE candidates (`FESSEL_VIEWER_PUBLIC_IPS`) |
| WHIP ingest (signaling + media), snapshot push, recording ingest | HTTP/UDP | the pod's **kernel-mode Tailscale sidecar** — the Pi dials `http://<sidecar-hostname>.<tailnet>.ts.net:8001` for all three HTTP routes, and sends WHIP media to the sidecar's tailnet IP on UDP `cfg.webrtc.ingestUdpPort` (default 31555). **No operator ingress Service** — an operator proxy DNATs, which breaks ICE address symmetry for WHIP (§5.1); the plain-HTTP routes have no such constraint but share the sidecar hostname rather than mint a second Tailscale identity |
| `/metrics` | TCP :8000 | ClusterIP only (scraped in-cluster) |

The ingest listener (`:8001`) carries **all** Pi→webui traffic
(recording uploads + snapshot push + WHIP signaling), all reached via the
one sidecar hostname; the public Ingress only ever backends `:8000`, so
the ingest surface is structurally unreachable from the public net.

### Tailscale sidecar (`cfg.webui.tailscaleSidecar`)

Kernel-mode `tailscale/tailscale` container in the webui pod
(`TS_USERSPACE=false`): puts a real `tailscale0` + the `100.64.0.0/10`
route inside the pod netns so the relay terminates the Pi's WHIP media
with symmetric ICE (`FESSEL_INGEST_PUBLIC_IP=auto` discovers the tailnet
address). Details:

- **State persists in a k8s Secret** (`TS_KUBE_SECRET` →
  `tailscaleSidecar.stateSecret`, default `fessel-webui-tsstate`), so the
  tailnet identity survives pod restarts. (The whip-relay prototype used
  an emptyDir — a new device per restart; deliberately fixed here.) The
  library renders the `webui` ServiceAccount plus a Role/RoleBinding
  granting `create` on secrets and `get`/`update`/`patch` on the state
  Secret.
- **Auth key** comes from a consumer-supplied Secret
  (`tailscaleSidecar.authkeySecret`, key `authkeyKey`) — referenced by
  name, never created by the library. Use a **reusable, non-ephemeral**
  key (state persists; an ephemeral key would be removed on disconnect).
- **Pod security:** the sidecar needs `NET_ADMIN` + `/dev/net/tun`. The
  namespace must carry `pod-security.kubernetes.io/enforce: privileged`
  (or the pod must be exempted) — the cluster-default `baseline` policy
  rejects it. The library's `namespace()` object does **not** set that
  label; it is the consumer's cluster-posture call. (The whip-relay
  prototype ran in its own privileged namespace for exactly this reason.)

### Media modes (`cfg.webrtc.mode`)

- `nodeport` (production / live-preview): `FESSEL_VIEWER_PUBLIC_IPS` =
  `cfg.webrtc.nodePublicIPs` (comma-joined), `FESSEL_VIEWER_UDP_PORT` =
  `cfg.webrtc.udpNodePort`, plus the `webui-media` NodePort UDP Service.
- `podip` (integration): viewer/ingest ICE env left **empty** — Pion
  gathers host candidates (the pod IP), directly reachable in-cluster.
  No NodePort Service, no sidecar.

## Breaking config changes (for bugzoo-infrastructure)

Migrating a consumer from the mediamtx-era library:

**Removed — delete from overrides:**

- `whepSecret` (and the rendered `fessel-whep-secret` Secret): no JWT /
  JWKS / WHEP tokens. Viewer auth is oauth2-proxy identity headers only.
- `hosts.media` (+ its Ingress/TLS/DNS): WHEP signaling is same-origin
  on `hosts.webui`.
- `images.mediamtx`, the whole mediamtx object set (ConfigMap,
  Deployment, `mediamtx`/`mediamtx-srt`/`mediamtx-webrtc` Services,
  media Ingress), and the `mediamtx-srt-ts` Tailscale ingress (SRT
  uplink retired).
- `webrtc.tcpNodePort` (relay ICE is UDP-only) and `webrtc.podIpPort`
  (podip mode no longer needs a fixed port).

**Added — set in production overrides:**

- `webui.tailscaleSidecar`: `{ enabled: true, hostname, authkeySecret,
  authkeyKey, stateSecret, image }`. Requires the auth-key Secret to
  exist and the namespace to be PodSecurity-privileged (above).
- `webrtc.ingestUdpPort` (default 31555): the WHIP ingest media UDP port
  (a plain container port on the tailnet, **not** a NodePort; it reuses
  the retired TCP-NodePort number).
- `live.activationTimeoutS` / `live.idleTimeoutS` (defaults 15 / 10) →
  `FESSEL_LIVE_ACTIVATION_TIMEOUT_S` / `FESSEL_LIVE_IDLE_TIMEOUT_S`.

**Changed semantics:**

- `webrtc.mode` now drives the **relay's** ICE env (see media modes
  above) instead of the mediamtx config; `webrtc.udpNodePort` (still
  31554) is now the relay's viewer media NodePort (Service
  `webui-media`).
- The webui Deployment is `strategy: Recreate` regardless of the
  recordings backend (was: Recreate only for `disk`).
- `includeTailscale` now renders only the `supervisor` egress Service (the
  SRT ingress and the `webui-recording-ingest` operator Service are both
  gone — recording-ingest now rides the tailscale sidecar alongside WHIP).
  Production wants `includeTailscale: true` **and**
  `webui.tailscaleSidecar.enabled: true`.

**Secrets delta:** drop `fessel-whep-secret`; add the Tailscale auth-key
Secret (`fessel-ts-auth` by default). The MinIO credentials Secret
(`fessel-minio-creds`, minio backend only) is unchanged; the
`fessel-webui-tsstate` state Secret is created/maintained by tailscaled
itself.

**Pi-side note:** the Pi's `video` config must point its WHIP endpoint
at the sidecar (`http://<sidecar-hostname>.<tailnet>.ts.net:8001/whip/ingest`).
That lives in the Pi's `/etc/fessel` config (dpkg asset), not in this
library. In the integration env the test-Pi uses
`http://webui:8001/whip/ingest` (in-cluster Service DNS).

See `src/fessel/webui/deploy/README.md` for the webui service's own
routing/auth topology, and architecture §4.2 / §5.1 / §5.5 for the
design.
