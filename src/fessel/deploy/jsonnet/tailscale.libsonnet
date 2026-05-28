// Tailscale Services for the cluster<->Pi legs (production only). Two
// distinct shapes (the footgun: they look similar but differ):
//   - ingress: cluster exposes mediamtx SRT TO the tailnet so the Pi pushes
//     to it (tailscale.com/expose).
//   - egress: cluster pods dial OUT to the Pi's supervisor over the tailnet
//     (type: ExternalName + tailscale.com/tailnet-fqdn). NOT the expose
//     annotation.
function(cfg) {
  local ns = cfg.namespace,

  // Pi -> cluster: SRT ingest exposed to the tailnet.
  srtIngress: {
    apiVersion: 'v1',
    kind: 'Service',
    metadata: {
      name: 'mediamtx-srt-ts',
      namespace: ns,
      annotations: {
        'tailscale.com/expose': 'true',
        'tailscale.com/hostname': 'mediamtx-srt',
      },
    },
    spec: {
      selector: { app: 'mediamtx' },
      ports: [{ port: 8890, targetPort: 8890, protocol: 'UDP' }],
    },
  },

  // cluster -> Pi: supervisor control plane reached over the tailnet.
  // ExternalName + tailnet-fqdn (the operator rewrites externalName).
  supervisorEgress: {
    apiVersion: 'v1',
    kind: 'Service',
    metadata: {
      name: 'supervisor',
      namespace: ns,
      annotations: { 'tailscale.com/tailnet-fqdn': cfg.tailnetPiFqdn },
    },
    spec: {
      type: 'ExternalName',
      externalName: 'placeholder',
      ports: [{ port: 8443, protocol: 'TCP' }],
    },
  },
}
