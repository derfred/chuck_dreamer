// Fessel cluster-side deployment library — the single source of truth for
// mediamtx + webui (+ the test-Pi and Tailscale Services). Owned by the
// Fessel code (this repo); consumed by bugzoo-infrastructure via a jb git
// dependency, and by this repo's own integration / live-preview test envs.
//
// Usage:
//   local fessel = import 'main.libsonnet';
//   local objs = fessel.objects(fessel.config + { ...overrides... });
//   // objs is a flat array of k8s objects -> render to a YAML stream.
//
// One library, three shapes via config.webrtc.mode + includeTestPi +
// includeTailscale:
//   production       : nodeport WebRTC, Tailscale Services, real off-cluster Pi
//   live-preview     : nodeport WebRTC, in-cluster test-Pi, public ingress
//   integration test : podip WebRTC, in-cluster test-Pi, no ingress/tailscale

local mediamtxLib = import 'mediamtx.libsonnet';
local webuiLib = import 'webui.libsonnet';
local piLib = import 'pi.libsonnet';
local tailscaleLib = import 'tailscale.libsonnet';

{
  config:: import 'config.libsonnet',

  // Namespace object (consumers may skip it if the ns is managed elsewhere).
  namespace(cfg):: {
    apiVersion: 'v1',
    kind: 'Namespace',
    metadata: { name: cfg.namespace },
  },

  // The test driver Job (integration env only). Parameterised image tag.
  testJob(cfg):: {
    apiVersion: 'batch/v1',
    kind: 'Job',
    metadata: { name: 'fessel-test', namespace: cfg.namespace },
    spec: {
      backoffLimit: 0,
      template: {
        metadata: { labels: { app: 'fessel-test' } },
        spec: {
          restartPolicy: 'Never',
          containers: [{
            name: 'driver',
            image: cfg.images.registry + '/fessel-test-driver:' + cfg.images.tag,
            env: [
              { name: 'WEBUI', value: 'http://webui:8000' },
              { name: 'MEDIA', value: 'http://mediamtx:8889' },
              { name: 'SUPERVISOR', value: 'http://supervisor:8443' },
              { name: 'FESSEL_PATH', value: 'pi' },
              { name: 'FESSEL_MODE', value: '640x480@30@1000000' },
              { name: 'RECONNECT_CYCLES', value: '20' },
              { name: 'JUNIT_OUT', value: '/results/junit.xml' },
            ],
            volumeMounts: [{ name: 'results', mountPath: '/results' }],
            resources: { requests: { cpu: '500m', memory: '1Gi' } },
          }],
          volumes: [{ name: 'results', emptyDir: {} }],
        },
      },
    },
  },

  // Build the flat list of cluster objects for a config.
  // `withNamespace`: include the Namespace object (default true).
  // `withTestJob`: include the driver Job (integration env only).
  objects(cfg, withNamespace=true, withTestJob=false)::
    local mtx = mediamtxLib(cfg);
    local web = webuiLib(cfg);
    local pi = piLib(cfg);
    local ts = tailscaleLib(cfg);
    (if withNamespace then [$.namespace(cfg)] else [])
    + [web.secret]
    + [mtx.configMap, mtx.deployment, mtx.service, mtx.srtService]
    + (if std.objectHas(mtx, 'webrtcService') then [mtx.webrtcService] else [])
    + (if std.objectHas(mtx, 'ingress') then [mtx.ingress] else [])
    + [web.deployment, web.service]
    + (if std.objectHas(web, 'ingress') then [web.ingress] else [])
    + (if cfg.includeTestPi then [pi.deployment, pi.service] else [])
    + (if cfg.includeTailscale then [ts.srtIngress, ts.supervisorEgress] else [])
    + (if withTestJob then [$.testJob(cfg)] else []),
}
