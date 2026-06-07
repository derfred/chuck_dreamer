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
  break the fetch.

mediamtx fetches the JWK via the **in-cluster Service**
(`authJWTJWKS: http://webui:8000/jwks` in `mediamtx.libsonnet`), so the
fetch never traverses the public ingress or oauth2-proxy. It happens once
at mediamtx startup (refreshed on a long interval); there is no
per-request callback.

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

## TLS / ingress

- Public HTTPS ingress exists only in `nodeport` mode (production /
  live-preview). The `podip` integration env is ClusterIP-only and the
  test driver reaches services in-cluster, so the split is moot there
  (no public surface, no proxy).
- cert-manager issues the webui TLS cert (`cfg.clusterIssuer`).
