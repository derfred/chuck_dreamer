# webui deployment topology

The cluster-side webui manifests live in the Tanka/jsonnet library under
`src/fessel/deploy/jsonnet/` (`webui.libsonnet`, `mediamtx.libsonnet`,
`main.libsonnet`, `config.libsonnet`). This note documents the **auth
routing topology** so the next person doesn't have to reverse-engineer
which endpoints are guarded (Slice 2, I2.1).

## Two endpoint classes

The backend's HTTP surface splits into two buckets that the deployment
routes differently.

### Behind oauth2-proxy (browser, interactive)

- `/` — the React app shell
- `/live`
- `/api/me`, `/api/capabilities`, `/api/auth/whep-url` (and the rest of
  the `/api/...` surface in later slices)

An auth proxy (oauth2-proxy with GitHub OIDC today) runs in front of the
webui hostname and forwards authenticated requests with identity headers
(`X-Auth-Request-User`, `X-Auth-Request-Email`, optionally
`X-Auth-Request-Groups`). The backend trusts these headers when present
and 401s when absent (`require_identity`). **The auth proxy itself is out
of scope for this library** — it is assumed already deployed and
configured to guard the webui hostname. The header names are backend
config (`FESSEL_AUTH_*_HEADER`); they must match what the proxy emits.

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
login (and in/out denial) flow itself. Because the app re-navigates rather
than calling a login URL, there is **no requirement that any `/oauth2/*`
path be reachable** — that was the old app-aware coupling and it is gone.
A proxy that returns its own 302/login on an unauthenticated navigation is
all that's needed.

### Bypass oauth2-proxy (machine-to-machine)

- `/jwks` — fetched by **mediamtx**, not a browser. A login redirect would
  break the fetch. Served on the **public listener** (port 8000) but
  host-gated + identity-rejecting (see Enforcement below).

mediamtx fetches the JWK via the **in-cluster Service**
(`authJWTJWKS: http://webui:8000/jwks` in `mediamtx.libsonnet`), so the
fetch never traverses the public ingress or oauth2-proxy. It happens once
at mediamtx startup (refreshed on a long interval); there is no
per-request callback.

### Tailnet-only ingest listener (Pi → backend, Slice 5.5)

The backend runs **two listeners in one process** (`serve_both` in
`app/main.py`):

- **public** (`:8000`) — everything above: the browser surface behind
  oauth2-proxy plus the in-cluster `/jwks` fetch. The public HTTPS Ingress
  backends **only** this port.
- **ingest** (`:cfg.ingestPort`, default `8001`) — serves
  `/recording-ingest/{id}/{name}` **only**. The Pi-side `uploader` PUTs
  each recording file here. Auth is at the **network layer** (Tailscale
  identity) — no oauth2-proxy, no token.

`/recording-ingest/...` exists **only** on the ingest app, and the public
Ingress never routes to the ingest port, so it is **structurally
impossible** to reach the ingest endpoint through the public ingress
(B5.5.7 — the chosen "two listeners" model, settled over the alternative
"one listener, route-restrict the path"). **All Pi→backend traffic** is on
this listener; if a future Pi→backend endpoint is added, put it here too.

The `webui-recording-ingest` Tailscale ingress Service
(`tailscale.libsonnet`, production only) is the sole external path to the
ingest port; it targets `cfg.ingestPort` and is reached by the Pi at
`<cfg.ingestHostname>.<tailnet>.ts.net:8443`. In the integration env the
test-Pi reaches it via the in-cluster `webui` Service DNS
(`http://webui:8001`); the real Tailscale ingress is not exercised there
(a documented coverage gap, carried forward from Slice 1.5).

### Recording storage backend (Slice 5.5)

webui-backend fronts the recording store via `recordingsStorage.backend`:

- `disk` — a directory mounted at `disk.path`; the backend serves playback
  as byte ranges itself. The Deployment uses `strategy: Recreate` (no
  two-pod window over the volume). This is the default and the per-PR
  integration default. The directory is backed by one of (`disk.volume`):
  - `pvc` (default) — a `ReadWriteOnce` PVC (`fessel-recordings`).
    Portable; survives a pod reschedule onto another node.
  - `hostPath` — a node directory (`disk.hostPath.path`, mounted
    `DirectoryOrCreate`). Simpler — no StorageClass/provisioner — but the
    data lives on **one node**: if the pod reschedules elsewhere the
    directory is empty there. Fine for a single-node or node-pinned
    deployment; the operator accepts that trade-off. No PVC object is
    rendered in this mode.
- `minio` — an S3 bucket; the backend redirects playback to presigned
  URLs; the Deployment stays `RollingUpdate`. Credentials come from a k8s
  Secret (`recordingsStorage.minio.secretName`), never inlined.

The Pi holds **no** cluster-store credentials in either case; which store
is used is invisible to the Pi and to the frontend.

## Enforcement (two layers)

`/jwks` returns an `oct` JWK that **is** the WHEP signing secret. It must
never be reachable publicly, and a direct in-cluster caller must not be
able to forge an operator identity against it. Two layers enforce this:

1. **Public-host gate (application, driven by the deployment).** In nodeport
   mode `webui.libsonnet` sets `FESSEL_PUBLIC_WEBUI_HOST` on the Deployment to
   the public ingress host. The backend 404s `/jwks` for any request whose
   `Host` is that public host, so the public host never serves the JWK.
   mediamtx's in-cluster Service path (`Host: webui:8000`) is unaffected.
   (This replaces an earlier nginx `server-snippet: location = /jwks { return
   404; }` on the Ingress — moving the block into the app keeps it working on
   clusters whose ingress admission webhook disallows snippet annotations.)

2. **Application (`forbid_identity_headers`).** `/jwks` rejects (400) any
   request carrying identity headers. On the bypass path those headers are
   anomalous — only oauth2-proxy sets them — so their presence means a
   direct caller is forging an identity. This is the single-listener model
   from the Slice 2 plan (B2.1): one backend listener, bypass endpoints
   reject identity rather than running a second listener on another port.

If a future endpoint becomes supervisor-callable (mediamtx → backend),
apply the same split: in-cluster Service reachability, deny on the public
ingress, and `forbid_identity_headers` on the route.

### NetworkPolicy (optional defence-in-depth, I5.5.3)

The two-listener split already makes the ingest port unreachable through
the public ingress *structurally* (the public Ingress backends only port
8000). A `NetworkPolicy` admitting only `oauth2-proxy → :8000` and
`tailnet-ingress-proxy → :ingestPort` is belt-and-suspenders — overkill
for a single-operator system, deliberately **not** rendered by the library
today. Add it as a separate object if a cluster's posture requires it; it
does not change any app behaviour.

## TLS / ingress

- Public HTTPS ingress exists only in `nodeport` mode (production /
  live-preview). The `podip` integration env is ClusterIP-only and the
  test driver reaches services in-cluster, so the split is moot there
  (no public surface, no proxy).
- cert-manager issues the webui TLS cert (`cfg.clusterIssuer`).
