// Integration-test environment: pod-IP WebRTC (in-cluster test client),
// in-cluster test-Pi, no ingress/Tailscale. Renders the infra objects
// (NOT the test Job — the workflow applies that separately after readiness).
//
// External vars (jsonnet --ext-str):
//   ns, image_tag, registry, whep_secret
local fessel = import '../main.libsonnet';

local cfg = fessel.config + {
  namespace: std.extVar('ns'),
  images+: { registry: std.extVar('registry'), tag: std.extVar('image_tag') },
  whepSecret: std.extVar('whep_secret'),
  webrtc+: { mode: 'podip' },
  includeTestPi: true,
  includeControlMocks: true,  // jetson-mock pod; WiZ is the in-process fake bulb
  includeTailscale: false,
};

// Emit a List so `kubectl apply -f -` consumes the whole stream.
{
  apiVersion: 'v1',
  kind: 'List',
  items: fessel.objects(cfg, withNamespace=true, withTestJob=false),
}
