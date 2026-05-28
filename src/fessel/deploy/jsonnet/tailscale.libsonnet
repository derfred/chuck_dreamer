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
  // ExternalName + tailnet-fqdn: the Tailscale OPERATOR owns spec.externalName
  // (it provisions an egress proxy and fills the field with the proxy's
  // in-cluster DNS name). We must NOT set externalName here, or server-side
  // apply would fight the operator (manager `operator` owns that field) and
  // momentarily clobber it with a placeholder, breaking the tunnel.
  //
  // `externalName` is normally required for type:ExternalName, but the
  // operator sets it; on first create it may briefly be rejected until the
  // operator reconciles. If a first-create value is ever needed, set it to a
  // real resolvable name (the operator overwrites it) rather than
  // 'placeholder', and exclude it from Tanka's field management.
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
      // externalName intentionally omitted — owned by the Tailscale operator.
      ports: [{ port: 8443, protocol: 'TCP' }],
    },
  },
}
