// Tailscale OPERATOR Services for the cluster<->Pi legs (production only).
// Two distinct shapes (the footgun: they look similar but differ):
//   - ingress: cluster exposes webui's recording-ingest listener TO the
//     tailnet so the Pi uploads to it (tailscale.com/expose).
//   - egress: cluster pods dial OUT to the Pi's supervisor over the tailnet
//     (type: ExternalName + tailscale.com/tailnet-fqdn). NOT the expose
//     annotation.
//
// NOTE: the WHIP ingest (live media) is deliberately NOT here. An operator
// ingress proxy DNATs, which breaks ICE address symmetry (architecture §5.1);
// the Pi instead dials the webui pod's tailscale SIDECAR (webui.libsonnet,
// cfg.webui.tailscaleSidecar) directly. The former mediamtx SRT ingress is
// retired with the SRT uplink.
function(cfg) {
  local ns = cfg.namespace,

  // Pi -> cluster: recording uploads exposed to the tailnet (I5.5.2). A SECOND,
  // distinct ingress in front of webui-backend's tailnet-only ingest listener
  // (cfg.ingestPort) — separate hostname from the public HTTPS ingress, same
  // Deployment selector, different target port. The Pi PUTs flagged recordings
  // to <ingestHostname>.<tailnet>.ts.net:8443. Auth is by Tailscale identity;
  // tailnet ACLs (architecture §5.1) admit the Pi and deny cluster-side
  // identities (which use the in-cluster Service DNS for the same backend).
  recordingIngress: {
    apiVersion: 'v1',
    kind: 'Service',
    metadata: {
      name: 'webui-recording-ingest',
      namespace: ns,
      annotations: {
        'tailscale.com/expose': 'true',
        'tailscale.com/hostname': cfg.ingestHostname,
      },
    },
    spec: {
      selector: { app: 'webui' },
      ports: [{ port: 8443, targetPort: cfg.ingestPort, protocol: 'TCP' }],
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
