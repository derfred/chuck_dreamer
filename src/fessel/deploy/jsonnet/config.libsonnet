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

  images: {
    registry: 'ghcr.io/derfred',
    tag: 'latest',
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
}
