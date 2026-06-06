// Test-Pi Deployment + the `supervisor` Service. Only included when
// cfg.includeTestPi is true (integration / live-preview). In production the
// real Pi runs off-cluster and `supervisor` is a Tailscale egress Service
// (see tailscale.libsonnet).
function(cfg) {
  local ns = cfg.namespace,

  // When a throwaway MinIO is deployed (integration env, includeMinio), point
  // the test-Pi's uploader at it via env (FESSEL_MINIO_*), flipping it from the
  // baked-in `fake` store to the real minio driver so the recording E2E test
  // (X4.3) ships a flagged recording to actual object storage. `secure: false`
  // (plain HTTP, in-cluster). Empty otherwise (uploader stays `fake`).
  local minioEnv =
    if cfg.includeMinio then [
      { name: 'FESSEL_MINIO_DRIVER', value: 'minio' },
      { name: 'FESSEL_MINIO_ENDPOINT', value: 'minio:9000' },
      { name: 'FESSEL_MINIO_ACCESS_KEY', value: 'fesseltest' },
      { name: 'FESSEL_MINIO_SECRET_KEY', value: 'fesseltestsecret' },
      { name: 'FESSEL_MINIO_BUCKET', value: 'fessel' },
      { name: 'FESSEL_MINIO_SECURE', value: 'false' },
    ] else [],

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
            [if cfg.includeMinio then 'env']: minioEnv,
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

  // --- Slice 3 control-plane mocks (only when cfg.includeControlMocks) ------
  // The jetson-mock: a stdlib HTTP server (integration/mocks/jetson_mock.py)
  // mounted from a ConfigMap into a bare python:3-slim — no new image. The
  // test-Pi's fessel.test.yaml points supervisor.control.jetson.base_url at
  // the `jetson-mock` Service. The WiZ side needs no pod (the fake bulb is
  // in-process in supervisor via wiz.driver: fake).

  jetsonMockConfigMap: {
    apiVersion: 'v1',
    kind: 'ConfigMap',
    metadata: { name: 'jetson-mock-script', namespace: ns },
    data: { 'jetson_mock.py': importstr '../../integration/mocks/jetson_mock.py' },
  },

  jetsonMockDeployment: {
    apiVersion: 'apps/v1',
    kind: 'Deployment',
    metadata: { name: 'jetson-mock', namespace: ns, labels: { app: 'jetson-mock' } },
    spec: {
      replicas: 1,  // stateful per run, like the real device
      selector: { matchLabels: { app: 'jetson-mock' } },
      template: {
        metadata: { labels: { app: 'jetson-mock' } },
        spec: {
          containers: [{
            name: 'jetson-mock',
            // Pinned (not floating python:3-slim) for a deterministic test image.
            image: 'python:3.13-slim',
            command: ['python3', '/app/jetson_mock.py'],
            ports: [{ name: 'http', containerPort: 8080, protocol: 'TCP' }],
            readinessProbe: {
              httpGet: { path: '/healthz', port: 8080 },
              initialDelaySeconds: 2,
              periodSeconds: 2,
            },
            volumeMounts: [{ name: 'script', mountPath: '/app' }],
            resources: { requests: { cpu: '50m', memory: '64Mi' } },
          }],
          volumes: [{ name: 'script', configMap: { name: 'jetson-mock-script' } }],
        },
      },
    },
  },

  jetsonMockService: {
    apiVersion: 'v1',
    kind: 'Service',
    metadata: { name: 'jetson-mock', namespace: ns },
    spec: {
      selector: { app: 'jetson-mock' },
      ports: [{ name: 'http', port: 8080, targetPort: 8080, protocol: 'TCP' }],
    },
  },
}
