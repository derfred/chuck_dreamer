// webui (backend+frontend) Deployment + Service, the shared WHEP Secret, and
// — Slice 5.5 — the recording storage wiring (disk PVC or MinIO) plus the
// tailnet-only recording-ingest listener.
//
// In nodeport mode it's exposed via HTTPS Ingress and mints WHEP URLs at the
// public media host; in podip (test) mode it's ClusterIP only and the test
// client reaches mediamtx in-cluster.
//
// Two listeners (B5.5.7): the public surface on :8000 (behind oauth2-proxy via
// the public Ingress) and the tailnet-only recording-ingest listener on
// cfg.ingestPort. The public Ingress NEVER routes to the ingest port; the
// webui Service exposes it for the tailnet ingress (prod) / the in-cluster
// test-Pi (integration) to reach.
function(cfg)
  local ns = cfg.namespace;
  local nodeport = cfg.webrtc.mode == 'nodeport';
  local rs = cfg.recordingsStorage;
  local disk = rs.backend == 'disk';
  // The disk backend's directory is backed by either a PVC or a node hostPath.
  local hostPath = disk && rs.disk.volume == 'hostPath';
  local pvc = disk && !hostPath;  // 'pvc' (default) when disk and not hostPath
  local mediaBase = if nodeport then 'https://' + cfg.hosts.media else 'http://mediamtx:8889';

  // Validate the backend choice at apply time (I5.5.4).
  assert rs.backend == 'disk' || rs.backend == 'minio' :
         "recordingsStorage.backend must be 'disk' or 'minio'";
  assert !disk || rs.disk.volume == 'pvc' || rs.disk.volume == 'hostPath' :
         "recordingsStorage.disk.volume must be 'pvc' or 'hostPath'";

  // recordings_storage env (the factory in app/storage/factory.py reads these).
  local storageEnv =
    if disk then [
      { name: 'FESSEL_RECORDINGS_BACKEND', value: 'disk' },
      { name: 'FESSEL_RECORDINGS_DISK_PATH', value: rs.disk.path },
    ] else [
      { name: 'FESSEL_RECORDINGS_BACKEND', value: 'minio' },
      { name: 'FESSEL_MINIO_ENDPOINT', value: rs.minio.endpoint },
      { name: 'FESSEL_MINIO_BUCKET', value: rs.minio.bucket },
      { name: 'FESSEL_MINIO_SECURE', value: if rs.minio.secure then 'true' else 'false' },
      {
        name: 'FESSEL_MINIO_ACCESS_KEY',
        valueFrom: { secretKeyRef: { name: rs.minio.secretName, key: rs.minio.accessKeyKey } },
      },
      {
        name: 'FESSEL_MINIO_SECRET_KEY',
        valueFrom: { secretKeyRef: { name: rs.minio.secretName, key: rs.minio.secretKeyKey } },
      },
    ];

  // The disk backend mounts a directory at rs.disk.path, backed by either a PVC
  // (the portable default) or a node hostPath (single-node / pinned). Either
  // way the Deployment uses strategy: Recreate (I5.5.1) — a RWO PVC can't span a
  // rolling update's two-pod window, and a hostPath shouldn't have two writers.
  local diskVolume =
    if pvc then [{ name: 'recordings', persistentVolumeClaim: { claimName: 'fessel-recordings' } }]
    else if hostPath then [{
      name: 'recordings',
      // DirectoryOrCreate: the node creates the dir if it's absent. Data lives
      // on whatever node the pod lands on; if it reschedules, the new node's
      // dir is empty (an accepted trade-off for the hostPath choice).
      hostPath: { path: rs.disk.hostPath.path, type: 'DirectoryOrCreate' },
    }]
    else [];
  local diskMount = if disk then [{ name: 'recordings', mountPath: rs.disk.path }] else [];

  {
  secret: {
    apiVersion: 'v1',
    kind: 'Secret',
    metadata: { name: 'fessel-whep-secret', namespace: ns },
    stringData: { secret: cfg.whepSecret },
  },

  // I5.5.1: the PVC for the disk backend. Only rendered when the disk backend
  // uses the PVC volume (not hostPath, and not the MinIO backend).
  [if pvc then 'recordingsPvc']: {
    apiVersion: 'v1',
    kind: 'PersistentVolumeClaim',
    metadata: { name: 'fessel-recordings', namespace: ns },
    spec: {
      accessModes: ['ReadWriteOnce'],
      resources: { requests: { storage: rs.disk.pvc.size } },
      [if rs.disk.pvc.storageClass != null then 'storageClassName']: rs.disk.pvc.storageClass,
    },
  },

  deployment: {
    apiVersion: 'apps/v1',
    kind: 'Deployment',
    metadata: { name: 'webui', namespace: ns, labels: { app: 'webui' } },
    spec: {
      replicas: 1,
      // Recreate when the disk backend's ReadWriteOnce PVC is mounted (a
      // transient two-pod rolling update can't both attach it); otherwise the
      // backend is stateless again, so RollingUpdate is fine (I5.5.1).
      strategy: { type: if disk then 'Recreate' else 'RollingUpdate' },
      selector: { matchLabels: { app: 'webui' } },
      template: {
        metadata: { labels: { app: 'webui' } },
        spec: {
          containers: [{
            name: 'webui',
            image: cfg.images.webui,
            env: [
              { name: 'FESSEL_WHEP_SECRET', valueFrom: { secretKeyRef: { name: 'fessel-whep-secret', key: 'secret' } } },
              { name: 'FESSEL_MEDIA_BASE', value: mediaBase },
              { name: 'FESSEL_WHEP_TTL_S', value: if nodeport then '300' else '60' },
              { name: 'FESSEL_INGEST_PORT', value: std.toString(cfg.ingestPort) },
            ]
            // Public-host gate for /jwks (I2.1): in nodeport mode the backend
            // 404s /jwks for requests on the public ingress host.
            + (if nodeport then [{ name: 'FESSEL_PUBLIC_WEBUI_HOST', value: cfg.hosts.webui }] else [])
            + storageEnv,
            ports: [
              { name: 'public', containerPort: 8000 },
              // Tailnet-only ingest listener (B5.5.7). NOT routed by the public
              // Ingress; targeted by the webui Service's `ingest` port.
              { name: 'ingest', containerPort: cfg.ingestPort },
            ],
            volumeMounts: diskMount,
            readinessProbe: {
              httpGet: { path: '/healthz', port: 8000 },
              initialDelaySeconds: 2,
              periodSeconds: 3,
            },
            resources: { requests: { cpu: '100m', memory: '128Mi' } },
          }],
          volumes: diskVolume,
        },
      },
    },
  },

  service: {
    apiVersion: 'v1',
    kind: 'Service',
    metadata: { name: 'webui', namespace: ns },
    spec: {
      selector: { app: 'webui' },
      ports: [
        { name: 'public', port: 8000, targetPort: 8000 },
        // The ingest port is exposed on the ClusterIP Service so the tailnet
        // ingress (prod) and the in-cluster test-Pi (integration) can reach it.
        // The public Ingress below still only routes to the public port.
        { name: 'ingest', port: cfg.ingestPort, targetPort: cfg.ingestPort },
      ],
    },
  },

  [if nodeport then 'ingress']: {
    apiVersion: 'networking.k8s.io/v1',
    kind: 'Ingress',
    metadata: {
      name: 'webui',
      namespace: ns,
      annotations: {
        'cert-manager.io/cluster-issuer': cfg.clusterIssuer,
        // Proxy-bypass split (I2.1): /jwks must NEVER be reachable on the
        // public ingress (the `oct` JWK is the signing secret); the backend's
        // FESSEL_PUBLIC_WEBUI_HOST host gate enforces that. The recording-
        // ingest listener (cfg.ingestPort) is a SEPARATE port this Ingress
        // never targets (it only backends the public :8000), so
        // /recording-ingest/... is structurally unreachable through the public
        // ingress (B5.5.7). The webui-recording-ingest Tailscale Service
        // (tailscale.libsonnet) is the only path to the ingest port.
      },
    },
    spec: {
      ingressClassName: cfg.ingressClass,
      tls: [{ hosts: [cfg.hosts.webui], secretName: 'fessel-webui-tls' }],
      rules: [{
        host: cfg.hosts.webui,
        http: { paths: [{
          path: '/',
          pathType: 'Prefix',
          backend: { service: { name: 'webui', port: { number: 8000 } } },
        }] },
      }],
    },
  },
}
