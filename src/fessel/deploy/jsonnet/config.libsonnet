// Default configuration for the Fessel cluster-side deployment library.
// Consumers (the production env in bugzoo-infrastructure, and the
// integration / live-preview test envs in this repo) override these.
//
// One library, three shapes, selected by `webrtc.mode`:
//   - "nodeport" : production / live-preview. Viewer WebRTC media over a UDP
//                  NodePort on the node public IPs (browser is on the public
//                  net); the webui relay advertises those IPs as ICE
//                  candidates (FESSEL_VIEWER_PUBLIC_IPS).
//   - "podip"    : per-PR integration test. The viewer/ingest ICE env is left
//                  EMPTY so Pion gathers host candidates — the webui pod IP is
//                  directly reachable by the in-cluster test client. No
//                  NodePort Service is rendered.
{
  namespace: 'fessel',

  // Canonical release version, kept in sync with schemas/pyproject.toml by
  // src/fessel/release.sh. Committed (not generated at deploy time) because
  // `tk eval` renders this tree on every PR (integration / live-preview), and a
  // missing importstr target fails the render. importstr keeps the trailing
  // newline -> strip it.
  version: std.rstripChars(importstr 'VERSION', '\n'),

  images: {
    registry: 'ghcr.io/derfred',
    // Container imagePullPolicy; null = cluster default (IfNotPresent for
    // pinned tags). Environments that REUSE a stable tag across rebuilds
    // (live-preview) must set 'Always' or nodes run stale cached images.
    pullPolicy: null,
    // Default tag tracks the release version so production (which inherits these
    // defaults from bugzoo-infrastructure) pulls the released image. The test
    // envs still override images.tag via std.extVar('image_tag').
    tag: $.version,
    // The Go webui: single binary (API + WHIP/WHEP relay + recording store
    // front) + embedded React app, built from src/fessel/webui/deploy/go.Dockerfile.
    webui: $.images.registry + '/fessel-webui:' + $.images.tag,
    // test-pi image is only used by the test/live-preview envs.
    testPi: $.images.registry + '/fessel-test-pi:' + $.images.tag,
  },

  // Hostname for the public HTTPS ingress (nodeport/live-preview only). The
  // former hosts.media is GONE — WHEP signaling is same-origin on the webui
  // host (the relay is in-process; there is no separate media server).
  hosts: {
    webui: 'fessel.example.com',
  },
  // cert-manager ClusterIssuer for ingress TLS.
  clusterIssuer: 'letsencrypt-prod',
  ingressClass: 'nginx',

  // WebRTC media exposure for the in-process Pion relay (architecture §4.2).
  // Two independent UDP planes:
  //   - viewer plane (/whep, browsers on the public net): nodeport mode binds
  //     a fixed UDP port exposed as a NodePort (externalTrafficPolicy: Local)
  //     and advertises the node public IPs; podip mode leaves the ICE env
  //     empty (host candidates = pod IP, in-cluster reachable).
  //   - ingest plane (/whip/ingest, the Pi over the tailnet): a fixed UDP port
  //     on the pod's tailscale-sidecar interface. NOT a NodePort — the Pi
  //     dials the sidecar's tailnet IP directly (§5.1); only meaningful when
  //     webui.tailscaleSidecar.enabled.
  webrtc: {
    mode: 'nodeport',  // 'nodeport' | 'podip'
    // nodeport mode (viewer plane):
    nodePublicIPs: [],       // e.g. ['85.10.209.85','88.99.140.171']
    udpNodePort: 31554,      // viewer media UDP NodePort (= FESSEL_VIEWER_UDP_PORT)
    // ingest plane (tailscale sidecar mode): fixed UDP port the relay binds
    // for Pi->relay media. Reuses the old TCP-NodePort number (31555) since
    // that NodePort is retired; it is a plain container port here, not a
    // NodePort (tailnet traffic terminates in the pod netns).
    ingestUdpPort: 31555,    // = FESSEL_INGEST_UDP_PORT
  },

  // Relay live-session lifecycle (architecture §2.3/§4.2): first-viewer
  // activation timeout and zero-viewers idle deactivation timeout, in seconds.
  live: {
    activationTimeoutS: 15,  // FESSEL_LIVE_ACTIVATION_TIMEOUT_S
    idleTimeoutS: 10,        // FESSEL_LIVE_IDLE_TIMEOUT_S
  },

  // webui pod extras.
  webui: {
    // Kernel-mode Tailscale sidecar (production only, architecture §5.1): puts
    // a real tailscale0 + the 100.64/10 route INSIDE the webui pod netns so
    // the relay can terminate the Pi's WHIP media with symmetric ICE (the
    // sidecar's tailnet IP is the ingest candidate; FESSEL_INGEST_PUBLIC_IP=auto
    // discovers it). Needs NET_ADMIN + /dev/net/tun => the namespace must be
    // PodSecurity `privileged` (or the pod exempted); see deploy/README.md.
    tailscaleSidecar: {
      enabled: false,
      image: 'tailscale/tailscale:v1.98.4',
      // Tailnet device name; the Pi dials http://<hostname>.<tailnet>.ts.net:8001
      // for WHIP signaling, snapshot push, AND recording uploads (and the
      // pod's tailnet IP for WHIP media) — one hostname for all Pi->cluster
      // HTTP/media traffic.
      hostname: 'fessel-webui',
      // Secret holding the Tailscale auth key. Supplied by the consumer
      // (referenced by name here, never created by this library).
      authkeySecret: 'fessel-ts-auth',
      authkeyKey: 'TS_AUTHKEY',
      // Secret tailscaled persists its node state into (TS_KUBE_SECRET), so
      // the tailnet identity SURVIVES pod restarts (an emptyDir would mint a
      // new device each restart). The library renders the ServiceAccount +
      // Role/RoleBinding that let the pod get/update it.
      stateSecret: 'fessel-webui-tsstate',
    },
    // FESSEL_DEV_IDENTITY: auth bypass for PROXY-LESS THROWAWAY environments
    // (the live-preview harness has no oauth2-proxy, so a real browser could
    // never authenticate). null = strict header auth. NEVER set in production.
    devIdentity: null,
    // Pin the webui pod to specific nodes (kubernetes.io/hostname In [...]).
    // Needed when only SOME nodes own the advertised viewer public IPs
    // (externalTrafficPolicy: Local drops media on the others — a pod on a
    // no-public-IP node breaks viewer ICE silently). Empty = no affinity.
    nodeHostnames: [],
  },

  // How the relay/backend reaches supervisor. Production: Tailscale (egress
  // Service named 'supervisor'). Test: an in-cluster Service named
  // 'supervisor' backed by the test-pi pod.
  supervisorBase: 'http://supervisor:8443',

  // Recording storage backend (Slice 5.5, §4.3 / I5.5.4). webui fronts the
  // recording store; the Pi uploads to its tailnet-only ingest listener and
  // holds no cluster-store credentials. One of two backends, selected here:
  //   backend: 'disk'  -> a mounted directory at disk.path; backend serves
  //                       playback byte ranges. The directory is backed by
  //                       either a PVC or a node hostPath (disk.volume).
  //   backend: 'minio' -> an S3 bucket; backend redirects playback to presigned
  //                       URLs.
  // (The Deployment is strategy: Recreate in BOTH cases now — the single-
  // replica relay holds live WebRTC session state; see webui.libsonnet.)
  // The library validates the choice at apply time (webui.libsonnet).
  recordingsStorage: {
    backend: 'disk',  // 'disk' | 'minio'
    disk: {
      path: '/var/lib/fessel/recordings',
      // What backs the mounted directory:
      //   'pvc'      -> a PersistentVolumeClaim (disk.pvc). The portable choice.
      //   'hostPath' -> a node directory (disk.hostPath). Simpler (no
      //                 StorageClass/provisioner), but the data lives on ONE
      //                 node: if the pod reschedules elsewhere the directory is
      //                 empty there. Acceptable for a single-node / pinned
      //                 deployment; the operator accepts that trade-off.
      volume: 'pvc',  // 'pvc' | 'hostPath'
      pvc: {
        size: '100Gi',
        storageClass: null,  // null -> cluster default StorageClass
      },
      hostPath: {
        // Host directory mounted into the pod. Created if absent (DirectoryOrCreate).
        path: '/var/lib/fessel/recordings',
      },
    },
    minio: {
      endpoint: error 'recordingsStorage.minio.endpoint must be set for backend: minio',
      bucket: 'fessel',
      secure: true,
      // The access/secret keys are referenced from a k8s Secret (not inlined).
      // Consumers supply real values.
      secretName: 'fessel-minio-creds',
      accessKeyKey: 'access_key',
      secretKeyKey: 'secret_key',
    },
  },

  // The TCP port webui's tailnet-only ingest listener binds (B5.5.7). Serves
  // ALL Pi->webui traffic: recording uploads, snapshot push, AND /whip/ingest
  // signaling. The public Ingress never routes here. Production: the Pi
  // reaches this port directly on the webui pod's tailscale sidecar
  // (webui.tailscaleSidecar.hostname). Test: the in-cluster webui Service
  // targets it.
  ingestPort: 8001,

  // Whether to include the in-cluster test-Pi Deployment (test envs) or
  // expect a real Pi over Tailscale (production -> false).
  includeTestPi: false,

  // Whether to render the Slice 3 control-plane MOCKS (jetson-mock pod). Only
  // the integration env sets this true; it pairs with the test-Pi's
  // fessel.test.yaml `supervisor.control` block (wiz.driver: fake, jetson
  // base_url -> jetson-mock). Never set in production or live-preview.
  includeControlMocks: false,

  // Whether to render the Tailscale operator Services (production): the
  // supervisor egress (cluster -> Pi). ALL Pi -> cluster traffic (WHIP,
  // snapshot, recording-ingest) goes through the pod's tailscale sidecar
  // (webui.tailscaleSidecar) instead, not an operator Service; production
  // enables both this and the sidecar.
  includeTailscale: false,

  // Whether to render a throwaway in-namespace MinIO (integration env only),
  // for the OPTIONAL MinIO-backend round-trip variant (T5.5.2). The disk
  // backend is the per-PR default now (recordingsStorage.backend: 'disk'), so
  // this is normally false; set it true (with recordingsStorage.backend:
  // 'minio') only to exercise the presigned-redirect playback path in CI.
  // Never set in production (real MinIO exists) or live-preview.
  includeMinio: false,
}
