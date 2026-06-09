// Default configuration for the Fessel cluster-side deployment library.
// Consumers (the production env in bugzoo-infrastructure, and the
// integration / live-preview test envs in this repo) override these.
//
// One library, three shapes, selected by `webrtc.mode`:
//   - "nodeport" : production / live-preview. WebRTC media over a NodePort
//                  on the node public IPs (browser is on the public net).
//   - "podip"    : per-PR integration test. WebRTC media advertised on the
//                  mediamtx pod IP (the test client is in-cluster).
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
    // Default tag tracks the release version so production (which inherits these
    // defaults from bugzoo-infrastructure) pulls the released image. The test
    // envs still override images.tag via std.extVar('image_tag').
    tag: $.version,
    webui: $.images.registry + '/fessel-webui:' + $.images.tag,
    // test-pi image is only used by the test/live-preview envs.
    testPi: $.images.registry + '/fessel-test-pi:' + $.images.tag,
    // upstream mediamtx; >= 1.13.0 for authJWTExclude. -ffmpeg = Alpine
    // (has /bin/sh + wget for runOnDemand); plain image is scratch.
    mediamtx: 'bluenviron/mediamtx:1.18.2-ffmpeg',
  },

  // Shared HMAC/JWT secret (the value is supplied by the consumer; the
  // library only references the Secret it creates).
  whepSecret: error 'fessel.whepSecret must be set',

  // Hostnames for the public HTTPS ingresses (nodeport/live-preview only).
  hosts: {
    webui: 'fessel.example.com',
    media: 'media-fessel.example.com',
  },
  // cert-manager ClusterIssuer for ingress TLS.
  clusterIssuer: 'letsencrypt-prod',
  ingressClass: 'nginx',

  webrtc: {
    mode: 'nodeport',  // 'nodeport' | 'podip'
    // nodeport mode:
    nodePublicIPs: [],          // e.g. ['85.10.209.85','88.99.140.171']
    udpNodePort: 31554,
    tcpNodePort: 31555,
    // podip mode uses a fixed in-cluster port:
    podIpPort: 8189,
  },

  // How the Pi reaches supervisor and how mediamtx reaches supervisor.
  // Production: Tailscale (egress Service named 'supervisor'). Test: an
  // in-cluster Service named 'supervisor' backed by the test-pi pod.
  supervisorBase: 'http://supervisor:8443',

  // Recording storage backend (Slice 5.5, §4.3 / I5.5.4). webui-backend fronts
  // the recording store; the Pi uploads to its tailnet-only ingest listener and
  // holds no cluster-store credentials. One of two backends, selected here:
  //   backend: 'disk'  -> a mounted directory at disk.path; backend serves
  //                       playback byte ranges; Deployment strategy becomes
  //                       Recreate. The directory is backed by either a PVC or a
  //                       node hostPath (disk.volume).
  //   backend: 'minio' -> an S3 bucket; backend redirects playback to presigned
  //                       URLs; Deployment stays RollingUpdate.
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
      // Default to the existing WHEP secret's namespace convention: a Secret
      // named here with two keys. Consumers supply real values.
      secretName: 'fessel-minio-creds',
      accessKeyKey: 'access_key',
      secretKeyKey: 'secret_key',
    },
  },

  // The TCP port webui-backend's tailnet-only recording-ingest listener binds
  // (B5.5.7). The public Ingress never routes here; the webui-recording-ingest
  // Tailscale Service (and, in test, the in-cluster webui Service) target it.
  ingestPort: 8001,

  // Production only: the magicDNS hostname for the recording-ingest Tailscale
  // ingress Service. The Pi dials <name>.<tailnet>.ts.net:8443.
  ingestHostname: 'fessel-ingest',

  // Whether to include the in-cluster test-Pi Deployment (test envs) or
  // expect a real Pi over Tailscale (production -> false).
  includeTestPi: false,

  // Whether to render the Slice 3 control-plane MOCKS (jetson-mock pod). Only
  // the integration env sets this true; it pairs with the test-Pi's
  // fessel.test.yaml `supervisor.control` block (wiz.driver: fake, jetson
  // base_url -> jetson-mock). Never set in production or live-preview.
  includeControlMocks: false,

  // Whether to render the Tailscale ingress/egress Services (production).
  includeTailscale: false,

  // Whether to render a throwaway in-namespace MinIO (integration env only),
  // for the OPTIONAL MinIO-backend round-trip variant (T5.5.2). The disk
  // backend is the per-PR default now (recordingsStorage.backend: 'disk'), so
  // this is normally false; set it true (with recordingsStorage.backend:
  // 'minio') only to exercise the presigned-redirect playback path in CI.
  // Never set in production (real MinIO exists) or live-preview.
  includeMinio: false,
}
