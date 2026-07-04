// webui (Go binary: backend + embedded frontend + Pion WHIP/WHEP relay)
// Deployment + Services, the recording storage wiring (disk PVC or MinIO),
// the tailnet-only ingest listener, and — production — the kernel-mode
// Tailscale sidecar for WHIP ingest (architecture §4.2/§5.1).
//
// One container, two HTTP
// listeners —
//   - public (:8000): frontend + /api + /whep signaling + /metrics; behind
//     oauth2-proxy via the public Ingress (nodeport mode).
//   - ingest (:cfg.ingestPort): ALL Pi->webui traffic — recording uploads +
//     /whip/ingest signaling. The public Ingress NEVER routes here; the
//     tailnet reaches it via the webui-recording-ingest operator Service
//     (recordings) and the pod's tailscale sidecar (WHIP).
// — plus two UDP media planes (viewer + ingest, see below).
//
// Media exposure by cfg.webrtc.mode:
//   - nodeport: FESSEL_VIEWER_PUBLIC_IPS = node public IPs, viewer media on a
//     UDP NodePort (externalTrafficPolicy: Local).
//   - podip: viewer/ingest ICE env left EMPTY -> Pion gathers host candidates
//     (the pod IP), which the in-cluster test client reaches directly. No
//     NodePort Service.
function(cfg)
  local ns = cfg.namespace;
  local nodeport = cfg.webrtc.mode == 'nodeport';
  local ts = cfg.webui.tailscaleSidecar;
  local rs = cfg.recordingsStorage;
  local disk = rs.backend == 'disk';
  // The disk backend's directory is backed by either a PVC or a node hostPath.
  local hostPath = disk && rs.disk.volume == 'hostPath';
  local pvc = disk && !hostPath;  // 'pvc' (default) when disk and not hostPath

  // Validate the backend choice at apply time (I5.5.4).
  assert rs.backend == 'disk' || rs.backend == 'minio' :
         "recordingsStorage.backend must be 'disk' or 'minio'";
  assert !disk || rs.disk.volume == 'pvc' || rs.disk.volume == 'hostPath' :
         "recordingsStorage.disk.volume must be 'pvc' or 'hostPath'";

  // recordings_storage env (internal/config/config.go reads these).
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

  // ICE advertisement, per plane (relay.ICEConfig; see the mode note above).
  // podip mode sets NEITHER plane: empty env = Pion host candidates = pod IP.
  local viewerEnv =
    if nodeport then [
      { name: 'FESSEL_VIEWER_PUBLIC_IPS', value: std.join(',', cfg.webrtc.nodePublicIPs) },
      { name: 'FESSEL_VIEWER_UDP_PORT', value: std.toString(cfg.webrtc.udpNodePort) },
    ] else [];
  local ingestIceEnv =
    if ts.enabled then [
      // 'auto': the relay discovers the sidecar's 100.x tailnet address and
      // advertises it as the ingest ICE candidate — symmetric with what the
      // Pi dials, so media rides WireGuard end to end (§5.1).
      { name: 'FESSEL_INGEST_PUBLIC_IP', value: 'auto' },
      { name: 'FESSEL_INGEST_UDP_PORT', value: std.toString(cfg.webrtc.ingestUdpPort) },
    ] else [];

  // The disk backend mounts a directory at rs.disk.path, backed by either a PVC
  // (the portable default) or a node hostPath (single-node / pinned).
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

  // Kernel-mode Tailscale sidecar (production, §5.1): joins THIS pod to the
  // tailnet so the pod netns gets a real tailscale0 (100.x) interface + the
  // 100.64/10 route; the relay sources WHIP ingest media from it with
  // symmetric ICE. Kernel mode specifically (TS_USERSPACE=false): only kernel
  // mode installs a route other in-netns processes (the relay) can use.
  //
  // POD SECURITY: needs NET_ADMIN + /dev/net/tun — the namespace must carry
  // `pod-security.kubernetes.io/enforce: privileged` (or exempt this pod);
  // the cluster-default `baseline` policy rejects it. See deploy/README.md.
  local tailscaleContainer = {
    name: 'tailscale',
    image: ts.image,
    env: [
      { name: 'TS_AUTHKEY', valueFrom: { secretKeyRef: { name: ts.authkeySecret, key: ts.authkeyKey } } },
      { name: 'TS_HOSTNAME', value: ts.hostname },
      { name: 'TS_USERSPACE', value: 'false' },  // kernel mode: real tailscale0 in the netns
      // PERSISTENT node state in a k8s Secret: the tailnet identity (device +
      // node key) survives pod restarts. An emptyDir would mint a brand-new
      // device on every restart (the whip-relay prototype's flaw). Requires
      // the Role/RoleBinding below (get/update on the state Secret).
      { name: 'TS_KUBE_SECRET', value: ts.stateSecret },
      { name: 'TS_EXTRA_ARGS', value: '--accept-dns=false' },
    ],
    securityContext: { capabilities: { add: ['NET_ADMIN'] } },
    volumeMounts: [{ name: 'dev-net-tun', mountPath: '/dev/net/tun' }],
    resources: { requests: { cpu: '50m', memory: '64Mi' } },
  };

  {
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

  // --- Tailscale sidecar state RBAC (only when the sidecar is enabled) ------
  // tailscaled (containerboot) reads/writes its node state into the
  // ts.stateSecret Secret via the pod's ServiceAccount. `create` cannot be
  // restricted by resourceName, hence the separate unscoped rule (first boot
  // creates the Secret if the consumer didn't pre-create it).
  [if ts.enabled then 'serviceAccount']: {
    apiVersion: 'v1',
    kind: 'ServiceAccount',
    metadata: { name: 'webui', namespace: ns },
  },

  [if ts.enabled then 'tsStateRole']: {
    apiVersion: 'rbac.authorization.k8s.io/v1',
    kind: 'Role',
    metadata: { name: 'webui-ts-state', namespace: ns },
    rules: [
      { apiGroups: [''], resources: ['secrets'], verbs: ['create'] },
      {
        apiGroups: [''],
        resources: ['secrets'],
        resourceNames: [ts.stateSecret],
        verbs: ['get', 'update', 'patch'],
      },
    ],
  },

  [if ts.enabled then 'tsStateRoleBinding']: {
    apiVersion: 'rbac.authorization.k8s.io/v1',
    kind: 'RoleBinding',
    metadata: { name: 'webui-ts-state', namespace: ns },
    roleRef: { apiGroup: 'rbac.authorization.k8s.io', kind: 'Role', name: 'webui-ts-state' },
    subjects: [{ kind: 'ServiceAccount', name: 'webui', namespace: ns }],
  },

  deployment: {
    apiVersion: 'apps/v1',
    kind: 'Deployment',
    metadata: { name: 'webui', namespace: ns, labels: { app: 'webui' } },
    spec: {
      replicas: 1,
      // Recreate ALWAYS: the single-replica relay holds per-session WebRTC
      // state (and, disk backend, a ReadWriteOnce volume) — a rolling update's
      // transient two-pod window would split sessions/volume (§4.2).
      strategy: { type: 'Recreate' },
      selector: { matchLabels: { app: 'webui' } },
      template: {
        metadata: { labels: { app: 'webui' } },
        spec: {
          [if ts.enabled then 'serviceAccountName']: 'webui',
          [if std.length(cfg.webui.nodeHostnames) > 0 then 'affinity']: {
            nodeAffinity: {
              requiredDuringSchedulingIgnoredDuringExecution: {
                nodeSelectorTerms: [{
                  matchExpressions: [{
                    key: 'kubernetes.io/hostname',
                    operator: 'In',
                    values: cfg.webui.nodeHostnames,
                  }],
                }],
              },
            },
          },
          // The webui image runs as non-root (uid 1000, user `webui`); fsGroup
          // makes the recordings volume group-writable for it. Applies to
          // PVC/emptyDir volumes — for the hostPath variant the kubelet does
          // NOT chown, so the operator must ensure the host dir is writable
          // by gid 1000.
          securityContext: { fsGroup: 1000 },
          containers: [{
            name: 'webui',
            image: cfg.images.webui,
            env: [
              { name: 'FESSEL_INGEST_PORT', value: std.toString(cfg.ingestPort) },
              { name: 'FESSEL_SUPERVISOR_BASE', value: cfg.supervisorBase },
              { name: 'FESSEL_LIVE_ACTIVATION_TIMEOUT_S', value: std.toString(cfg.live.activationTimeoutS) },
              { name: 'FESSEL_LIVE_IDLE_TIMEOUT_S', value: std.toString(cfg.live.idleTimeoutS) },
            ]
            + viewerEnv
            + ingestIceEnv
            + storageEnv,
            ports: [
              { name: 'public', containerPort: 8000, protocol: 'TCP' },
              // Tailnet-only ingest listener (B5.5.7): recording uploads +
              // /whip/ingest signaling. NOT routed by the public Ingress;
              // targeted by the webui Service's `ingest` port (recordings)
              // and dialled at the sidecar's tailnet IP (WHIP).
              { name: 'ingest', containerPort: cfg.ingestPort, protocol: 'TCP' },
            ]
            // Viewer media UDP (nodeport mode; the NodePort Service targets it).
            + (if nodeport then [
              { name: 'media-view', containerPort: cfg.webrtc.udpNodePort, protocol: 'UDP' },
            ] else [])
            // Ingest media UDP (sidecar mode; tailnet traffic terminates in-pod).
            + (if ts.enabled then [
              { name: 'media-ing', containerPort: cfg.webrtc.ingestUdpPort, protocol: 'UDP' },
            ] else []),
            volumeMounts: diskMount,
            readinessProbe: {
              httpGet: { path: '/healthz', port: 8000 },
              initialDelaySeconds: 2,
              periodSeconds: 3,
            },
            resources: { requests: { cpu: '100m', memory: '128Mi' } },
          }]
          + (if ts.enabled then [tailscaleContainer] else []),
          volumes: diskVolume
          + (if ts.enabled then [
            { name: 'dev-net-tun', hostPath: { path: '/dev/net/tun', type: 'CharDevice' } },
          ] else []),
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
        // recording ingress (prod) and the in-cluster test-Pi (integration)
        // can reach it. The public Ingress below still only routes to the
        // public port.
        { name: 'ingest', port: cfg.ingestPort, targetPort: cfg.ingestPort },
      ],
    },
  },

  // nodeport mode only: viewer WebRTC media on the node public IPs. Local
  // traffic policy keeps the source address symmetric with the advertised
  // node candidate (no SNAT hop).
  [if nodeport then 'viewerMediaService']: {
    apiVersion: 'v1',
    kind: 'Service',
    metadata: { name: 'webui-media', namespace: ns },
    spec: {
      type: 'NodePort',
      externalTrafficPolicy: 'Local',
      selector: { app: 'webui' },
      ports: [{
        name: 'media-view',
        port: cfg.webrtc.udpNodePort,
        targetPort: cfg.webrtc.udpNodePort,
        nodePort: cfg.webrtc.udpNodePort,
        protocol: 'UDP',
      }],
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
        // The recording-ingest/WHIP listener (cfg.ingestPort) is a SEPARATE
        // port this Ingress never targets (it only backends the public
        // :8000), so the ingest surface is structurally unreachable through
        // the public ingress (B5.5.7). The tailnet (operator Service +
        // sidecar) is the only path to the ingest port.
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
