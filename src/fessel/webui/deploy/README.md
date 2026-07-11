# webui deployment topology

The cluster-side webui manifests live in the Tanka/jsonnet library under
`src/fessel/deploy/jsonnet/` (`webui.libsonnet`, `main.libsonnet`,
`config.libsonnet`, `tailscale.libsonnet`; see
`src/fessel/deploy/README.md` for the library API and the
bugzoo-infrastructure migration notes). This note documents the **auth
and network routing topology** so the next person doesn't have to
reverse-engineer which endpoints are guarded.

The webui is the Go binary from `deploy/go.Dockerfile`: backend + embedded
React frontend + in-process Pion WHIP/WHEP relay. There is no separate
media server and no JWT/JWKS/WHEP-token machinery — no streaming secret.

## Network exposure (production)

| Purpose | Protocol | Exposure |
|---|---|---|
| Frontend + `/api` + `/whep` signaling | TCP :8000 | public HTTPS Ingress behind oauth2-proxy |
| WHEP viewer media | UDP | `webui-media` NodePort (`externalTrafficPolicy: Local`), node public IPs as ICE candidates |
| WHIP ingest (signaling + media), snapshot push, recording ingest | HTTP/UDP | the pod's kernel-mode Tailscale **sidecar** — the Pi dials the sidecar's tailnet address directly for all three (no operator proxy: DNAT would break ICE symmetry for WHIP, architecture §5.1; the plain-HTTP routes share the same hostname rather than mint a second one) |
| `/metrics` | TCP :8000 | ClusterIP only, scraped by Prometheus |

Secrets: the Tailscale auth key (sidecar), MinIO credentials (minio
backend only), and the tailscaled-managed state Secret. The old
`fessel-whep-secret` no longer exists.

## Two endpoint classes

The backend's HTTP surface splits into two listeners in one process
(structural, not path-based):

### public (`:8000`) — behind oauth2-proxy (browser, interactive)

- `/` — the React app shell, `/live`
- `/api/...`
- `/whep` — WHEP signaling for the live view (the relay is in-process;
  viewer authorisation is entirely "the request carried valid
  oauth2-proxy identity headers")
- `/metrics`, `/healthz` — reached in-cluster only (the ingress host is
  proxied; Prometheus scrapes the ClusterIP)

An auth proxy (oauth2-proxy with GitHub OIDC today) runs in front of the
webui hostname and forwards authenticated requests with identity headers
(`X-Auth-Request-User`, `X-Auth-Request-Email`, optionally
`X-Auth-Request-Groups`). The backend trusts these headers when present
and 401s state-changing/identity-requiring endpoints without them. **The
auth proxy itself is out of scope for this library** — it is assumed
already deployed and configured to guard the webui hostname. The header
names are backend config (`FESSEL_AUTH_*_HEADER`); they must match what
the proxy emits.

### The app is auth-mechanism agnostic

Authentication and the flat in/out access decision are **infrastructure
concerns**; neither the frontend nor the backend knows or names a specific
auth mechanism:

- **Backend:** trusts infra-injected identity headers (names are config),
  401s without them. It runs no OIDC code. Any proxy that can inject the
  configured headers works — oauth2-proxy, a different forward-auth, a
  gateway — with no app change.
- **Frontend:** on a 401 it does **not** redirect to any login endpoint.
  It re-navigates (`window.location.reload()`, see `api.ts`
  `reauthenticate()`) so the proxy can re-authenticate the navigation and
  either restore the session or serve its own denial page. A one-shot
  guard prevents a reload loop if the proxy is misconfigured/absent.

What infra must therefore provide (the implicit contract): the proxy
**intercepts unauthenticated navigations** to the webui host and runs its
login (and in/out denial) flow itself.

### ingest (`:FESSEL_INGEST_PORT`, default 8001) — tailnet-only (Pi → backend)

Serves **all** Pi→webui traffic:

- `/recording-ingest/{id}/{name}` — the Pi-side `uploader` PUTs each
  recording file here, reached **directly at the pod's Tailscale sidecar**
  address (`http://<sidecar-hostname>.<tailnet>.ts.net:8001`) — the same
  hostname/port as WHIP and the snapshot push below, not a separate
  operator ingress.
- `/whip/ingest` — the Pi's `video` posts the WHIP offer here, reached at
  the same sidecar address; the WebRTC media then flows to the sidecar's
  tailnet IP on `FESSEL_INGEST_UDP_PORT`.

Auth is at the **network layer** (Tailscale identity) — no oauth2-proxy,
no token. The public Ingress never routes to the ingest port, so the
ingest surface is **structurally impossible** to reach through the public
ingress (the chosen "two listeners" model, settled over "one listener,
route-restrict the path"). If a future Pi→backend endpoint is added, put
it here too.

In the integration env the test-Pi reaches both routes via the in-cluster
`webui` Service DNS (`http://webui:8001`); neither the operator ingress
nor the sidecar is exercised there (a documented coverage gap).

## Tailscale sidecar (production)

The webui pod carries a kernel-mode `tailscale/tailscale` sidecar
(`cfg.webui.tailscaleSidecar`) so the pod netns has a real `tailscale0`
and the 100.64/10 route — the relay terminates the Pi's WHIP media with
symmetric ICE (`FESSEL_INGEST_PUBLIC_IP=auto`). Its tailnet state
persists in a k8s Secret (`TS_KUBE_SECRET`; ServiceAccount + Role
rendered by the library) so the identity survives restarts.

**Pod security:** the sidecar needs `NET_ADMIN` + `/dev/net/tun`, which
the default `baseline` PodSecurity level forbids. The namespace needs
`pod-security.kubernetes.io/enforce: privileged` (or a per-pod
exemption). The whip-relay prototype ran in its own privileged namespace;
production must make the same call for the fessel namespace.

## Recording storage backend

webui fronts the recording store via `recordingsStorage.backend`:

- `disk` — a directory mounted at `disk.path`; the backend serves playback
  as byte ranges itself. This is the default and the per-PR integration
  default. The directory is backed by one of (`disk.volume`):
  - `pvc` (default) — a `ReadWriteOnce` PVC (`fessel-recordings`).
    Portable; survives a pod reschedule onto another node.
  - `hostPath` — a node directory (`disk.hostPath.path`, mounted
    `DirectoryOrCreate`). Simpler — no StorageClass/provisioner — but the
    data lives on **one node**: if the pod reschedules elsewhere the
    directory is empty there. Fine for a single-node or node-pinned
    deployment; the operator accepts that trade-off. No PVC object is
    rendered in this mode.
- `minio` — an S3 bucket; the backend redirects playback to presigned
  URLs. Credentials come from a k8s Secret
  (`recordingsStorage.minio.secretName`), never inlined.

The Deployment is `strategy: Recreate` in **both** cases — the
single-replica relay holds live WebRTC session state, so there is never a
two-pod window (this also covers the RWO-PVC constraint that used to be
the only reason for Recreate).

The Pi holds **no** cluster-store credentials in either case; which store
is used is invisible to the Pi and to the frontend.

### NetworkPolicy (optional defence-in-depth)

The two-listener split already makes the ingest port unreachable through
the public ingress *structurally*. A `NetworkPolicy` admitting only
`oauth2-proxy → :8000` and the tailnet proxies → `:ingestPort` is
belt-and-suspenders — overkill for a single-operator system, deliberately
**not** rendered by the library today. Add it as a separate object if a
cluster's posture requires it; it does not change any app behaviour.

## TLS / ingress

- Public HTTPS ingress exists only in `nodeport` mode (production /
  live-preview). The `podip` integration env is ClusterIP-only and the
  test driver reaches services in-cluster, so the split is moot there
  (no public surface, no proxy).
- cert-manager issues the webui TLS cert (`cfg.clusterIssuer`).
