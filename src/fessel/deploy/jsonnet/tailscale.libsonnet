// Tailscale OPERATOR Services for the cluster<->Pi legs (production only).
//   - egress: cluster pods dial OUT to the Pi's supervisor over the tailnet
//     (type: ExternalName + tailscale.com/tailnet-fqdn). NOT the expose
//     annotation.
//
// NOTE: ALL Pi -> cluster traffic (WHIP ingest, snapshot push, recording
// uploads) goes through the webui pod's tailscale SIDECAR (webui.libsonnet,
// cfg.webui.tailscaleSidecar), not an operator Service. An operator ingress
// proxy DNATs, which breaks ICE address symmetry for WHIP (architecture
// §5.1); the plain-HTTP routes (snapshot, recording-ingest) have no such
// constraint but reuse the sidecar's existing tailnet identity rather than
// mint a second one for no reason. The retired SRT-uplink ingress and the
// recording-ingest operator Service (fessel-ingest) are both retired.
function(cfg) {
  local ns = cfg.namespace,

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
