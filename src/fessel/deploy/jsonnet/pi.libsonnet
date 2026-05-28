// Test-Pi Deployment + the `supervisor` Service. Only included when
// cfg.includeTestPi is true (integration / live-preview). In production the
// real Pi runs off-cluster and `supervisor` is a Tailscale egress Service
// (see tailscale.libsonnet).
function(cfg) {
  local ns = cfg.namespace,

  deployment: {
    apiVersion: 'apps/v1',
    kind: 'Deployment',
    metadata: { name: 'pi', namespace: ns, labels: { app: 'pi' } },
    spec: {
      replicas: 1,
      selector: { matchLabels: { app: 'pi' } },
      template: {
        metadata: { labels: { app: 'pi' } },
        spec: {
          containers: [{
            name: 'pi',
            image: cfg.images.testPi,
            ports: [{ name: 'control', containerPort: 8443, protocol: 'TCP' }],
            readinessProbe: {
              httpGet: { path: '/healthz', port: 8443 },
              initialDelaySeconds: 5,
              periodSeconds: 3,
              failureThreshold: 30,
            },
            resources: { requests: { cpu: '300m', memory: '384Mi' } },
          }],
        },
      },
    },
  },

  // Control-plane Service. mediamtx's runOnDemand resolves `supervisor`
  // (in-cluster DNS) here; the test driver reads /state/live here too.
  service: {
    apiVersion: 'v1',
    kind: 'Service',
    metadata: { name: 'supervisor', namespace: ns },
    spec: {
      selector: { app: 'pi' },
      ports: [{ name: 'control', port: 8443, targetPort: 8443, protocol: 'TCP' }],
    },
  },
}
