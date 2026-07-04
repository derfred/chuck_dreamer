// Live-preview environment: production-style NodePort WebRTC (real browser
// over the public net), in-cluster test-Pi, public HTTPS ingress. No
//
// External vars: ns, image_tag, registry, base_domain,
//   node_public_ips (comma-separated), udp_nodeport
local fessel = import '../main.libsonnet';

local base = std.extVar('base_domain');
local ips = std.split(std.extVar('node_public_ips'), ',');

local cfg = fessel.config + {
  namespace: std.extVar('ns'),
  images+: { registry: std.extVar('registry'), tag: std.extVar('image_tag') },
  hosts: { webui: 'fessel-live.' + base },
  webrtc+: {
    mode: 'nodeport',
    nodePublicIPs: ips,
    udpNodePort: std.parseInt(std.extVar('udp_nodeport')),
  },
  includeTestPi: true,
  includeTailscale: false,
};

{
  apiVersion: 'v1',
  kind: 'List',
  items: fessel.objects(cfg, withNamespace=true, withTestJob=false),
}
