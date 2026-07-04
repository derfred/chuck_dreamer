// Live-preview environment: production-style NodePort WebRTC (real browser
// over the public net), in-cluster test-Pi, public HTTPS ingress. No
//
// External vars: ns, image_tag, registry, base_domain,
//   node_public_ips (comma-separated), node_hostnames (comma-separated,
//   may be empty), udp_nodeport, webui_host
local fessel = import '../main.libsonnet';

local base = std.extVar('base_domain');
local ips = std.split(std.extVar('node_public_ips'), ',');
local hostnamesRaw = std.extVar('node_hostnames');
local hostnames = if hostnamesRaw == '' then [] else std.split(hostnamesRaw, ',');

local cfg = fessel.config + {
  namespace: std.extVar('ns'),
  images+: { registry: std.extVar('registry'), tag: std.extVar('image_tag'), pullPolicy: 'Always' },
  hosts: { webui: std.extVar('webui_host') },
  webrtc+: {
    mode: 'nodeport',
    nodePublicIPs: ips,
    udpNodePort: std.parseInt(std.extVar('udp_nodeport')),
  },
  includeTestPi: true,
  // The control-plane assertions exercise the Jetson forwarders; without the
  // mock, pause/resume 503s against a Jetson that doesn't exist here.
  includeControlMocks: true,
  includeTailscale: false,
  webui+: {
    nodeHostnames: hostnames,
    // No oauth2-proxy in the preview: a real browser sends no identity
    // headers and would 401 everywhere. The preview is throwaway + TTL'd.
    devIdentity: 'live-preview-operator',
  },
};

{
  apiVersion: 'v1',
  kind: 'List',
  items: fessel.objects(cfg, withNamespace=true, withTestJob=false),
}
