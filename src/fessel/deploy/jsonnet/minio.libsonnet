// Test-only MinIO instance for the integration env (recording end-to-end, X4.3).
//
// Production MinIO lives in the cluster already (architecture §4.1 "MinIO | New");
// this is a throwaway single-replica MinIO in the per-PR test namespace so the
// uploader's REAL minio driver ships a flagged recording end-to-end (the per-PR
// suite otherwise uses the in-memory `fake` store). emptyDir storage — the
// namespace is deleted after the run, so nothing persists.
//
// The bucket is created by a tiny one-shot Job (`mc mb`, idempotent) once the
// server is up; the uploader's MinioObjectStore does not create the bucket
// (a missing bucket is a PermanentError), so something must.

function(cfg) {
  local ns = cfg.namespace,
  local bucket = 'fessel',
  local accessKey = 'fesseltest',
  local secretKey = 'fesseltestsecret',

  // Exposed so the test-Pi env wiring + the driver can reference the same values.
  endpoint:: 'minio:9000',
  bucket:: bucket,
  accessKey:: accessKey,
  secretKey:: secretKey,

  deployment: {
    apiVersion: 'apps/v1',
    kind: 'Deployment',
    metadata: { name: 'minio', namespace: ns, labels: { app: 'minio' } },
    spec: {
      replicas: 1,
      strategy: { type: 'Recreate' },
      selector: { matchLabels: { app: 'minio' } },
      template: {
        metadata: { labels: { app: 'minio' } },
        spec: {
          containers: [{
            name: 'minio',
            image: 'minio/minio:latest',
            args: ['server', '/data', '--console-address', ':9001'],
            env: [
              { name: 'MINIO_ROOT_USER', value: accessKey },
              { name: 'MINIO_ROOT_PASSWORD', value: secretKey },
            ],
            ports: [
              { name: 'api', containerPort: 9000 },
              { name: 'console', containerPort: 9001 },
            ],
            readinessProbe: {
              httpGet: { path: '/minio/health/ready', port: 9000 },
              initialDelaySeconds: 3,
              periodSeconds: 3,
            },
            volumeMounts: [{ name: 'data', mountPath: '/data' }],
            resources: { requests: { cpu: '100m', memory: '256Mi' } },
          }],
          volumes: [{ name: 'data', emptyDir: {} }],
        },
      },
    },
  },

  service: {
    apiVersion: 'v1',
    kind: 'Service',
    metadata: { name: 'minio', namespace: ns },
    spec: {
      selector: { app: 'minio' },
      ports: [
        { name: 'api', port: 9000, targetPort: 9000 },
        { name: 'console', port: 9001, targetPort: 9001 },
      ],
    },
  },

  // One-shot bucket creator: waits for MinIO, sets an mc alias, makes the
  // bucket (idempotent). Uses the minio/mc image (has /bin/sh + mc).
  bucketJob: {
    apiVersion: 'batch/v1',
    kind: 'Job',
    metadata: { name: 'minio-mkbucket', namespace: ns },
    spec: {
      backoffLimit: 10,  // retry until MinIO is reachable
      template: {
        metadata: { labels: { app: 'minio-mkbucket' } },
        spec: {
          restartPolicy: 'OnFailure',
          containers: [{
            name: 'mc',
            image: 'minio/mc:latest',
            command: ['/bin/sh', '-c'],
            args: [
              'until mc alias set m http://minio:9000 %s %s; do echo waiting for minio; sleep 2; done; '
              % [accessKey, secretKey]
              + 'mc mb --ignore-existing m/%s && echo bucket-ready' % bucket,
            ],
            resources: { requests: { cpu: '50m', memory: '64Mi' } },
          }],
        },
      },
    },
  },
}
