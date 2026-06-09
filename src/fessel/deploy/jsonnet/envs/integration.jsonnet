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
  // Slice 5.5: the per-PR run uses the DISK backend (T5.5.1) — no MinIO needed.
  // The PVC is provisioned in the test namespace and cleaned up at teardown.
  // The disk-backed recording round-trip (T5.5.2) is the must-pass assertion.
  recordingsStorage+: {
    backend: 'disk',
    // A small PVC (the test writes a couple of tiny segments); use the cluster
    // default StorageClass.
    disk+: { pvc+: { size: '1Gi' } },
  },
  includeMinio: false,  // disk backend default; no throwaway MinIO in CI
  includeTailscale: false,
};

// Emit a List so `kubectl apply -f -` consumes the whole stream.
{
  apiVersion: 'v1',
  kind: 'List',
  items: fessel.objects(cfg, withNamespace=true, withTestJob=false),
}
