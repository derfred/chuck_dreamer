// mediamtx objects: ConfigMap (rendered mediamtx.yml), Deployment, and
// Services. WebRTC exposure shape is selected by cfg.webrtc.mode.
// Locals are hoisted to function scope (not object `local`s) so they're
// visible in computed field names like [if nodeport then 'webrtcService'].
function(cfg)
  local ns = cfg.namespace;
  local w = cfg.webrtc;
  local nodeport = w.mode == 'nodeport';
  // The WebRTC ICE port: a NodePort in production, or a fixed pod-IP port
  // in the in-cluster test.
  local udpPort = if nodeport then w.udpNodePort else w.podIpPort;
  local tcpPort = if nodeport then w.tcpNodePort else w.podIpPort;

  // The mediamtx.yml document, assembled as a string (mediamtx reads YAML).
  local mediamtxYml =
    |||
      logLevel: info
      logDestinations: [stdout]
      readTimeout: 10s
      writeTimeout: 10s
      rtsp: no
      rtmp: no
      hls: no
      srt: yes
      webrtc: yes
      api: yes
      metrics: yes
      srtAddress: :8890
      webrtcLocalUDPAddress: :%(udp)d
      webrtcLocalTCPAddress: :%(tcp)d
      webrtcICEServers2: []
      %(additionalHosts)s
      # JWT auth, validated locally against the webui JWKS (NO callback).
      # authJWTExclude lets the Pi's trusted SRT publish through without a
      # token (a JWT can't be carried over SRT: 512-char streamid cap).
      # Requires mediamtx >= 1.13.0.
      authMethod: jwt
      authJWTJWKS: %(jwks)s
      authJWTClaimKey: mediamtx_permissions
      authJWTInHTTPQuery: true
      authJWTExclude:
        - action: publish
        - action: api
        - action: metrics
        - action: pprof
      pathDefaults:
        source: publisher
      paths:
        pi:
          source: publisher
          runOnDemand: >
            wget -q -O- --header="Content-Type: application/json"
            --post-data="{\"path\":\"$MTX_PATH\",\"query\":\"$MTX_QUERY\"}"
            %(supervisor)s/control/live/activate
          runOnUnDemand: >
            wget -q -O- --header="Content-Type: application/json"
            --post-data="{\"path\":\"$MTX_PATH\"}"
            %(supervisor)s/control/live/deactivate
          runOnDemandStartTimeout: 15s
          runOnDemandCloseAfter: 10s
          runOnDemandRestart: no
    ||| % {
      udp: udpPort,
      tcp: tcpPort,
      jwks: 'http://webui:8000/jwks',
      supervisor: cfg.supervisorBase,
      // In nodeport mode advertise the node public IPs; in podip mode the
      // pod IP is injected at runtime via MTX_WEBRTCADDITIONALHOSTS env.
      additionalHosts:
        if nodeport
        then 'webrtcAdditionalHosts: [%s]' % std.join(', ', ['"%s"' % ip for ip in w.nodePublicIPs])
        else '# webrtcAdditionalHosts injected via MTX_WEBRTCADDITIONALHOSTS (pod IP)',
    };

  {
  configMap: {
    apiVersion: 'v1',
    kind: 'ConfigMap',
    metadata: { name: 'mediamtx-config', namespace: ns },
    data: { 'mediamtx.yml': mediamtxYml },
  },

  deployment: {
    apiVersion: 'apps/v1',
    kind: 'Deployment',
    metadata: { name: 'mediamtx', namespace: ns, labels: { app: 'mediamtx' } },
    spec: {
      replicas: 1,
      strategy: { type: 'Recreate' },
      selector: { matchLabels: { app: 'mediamtx' } },
      template: {
        metadata: { labels: { app: 'mediamtx' } },
        spec: {
          // Non-root: avoids per-UID inotify-instance exhaustion that
          // crashes root mediamtx at startup on shared rke2 nodes.
          securityContext: { runAsUser: 1000, runAsGroup: 1000 },
          containers: [{
            name: 'mediamtx',
            image: cfg.images.mediamtx,
            args: ['/config/mediamtx.yml'],
            [if !nodeport then 'env']: [
              { name: 'POD_IP', valueFrom: { fieldRef: { fieldPath: 'status.podIP' } } },
              { name: 'MTX_WEBRTCADDITIONALHOSTS', value: '$(POD_IP)' },
            ],
            ports: [
              { name: 'whep', containerPort: 8889, protocol: 'TCP' },
              { name: 'srt', containerPort: 8890, protocol: 'UDP' },
              { name: 'wrtc-udp', containerPort: udpPort, protocol: 'UDP' },
              { name: 'wrtc-tcp', containerPort: tcpPort, protocol: 'TCP' },
              { name: 'api', containerPort: 9997, protocol: 'TCP' },
            ],
            resources: { requests: { cpu: '200m', memory: '256Mi' } },
            volumeMounts: [{ name: 'config', mountPath: '/config' }],
          }],
          volumes: [{ name: 'config', configMap: { name: 'mediamtx-config' } }],
        },
      },
    },
  },

  // WHEP signaling Service (browser/test entry).
  service: {
    apiVersion: 'v1',
    kind: 'Service',
    metadata: { name: 'mediamtx', namespace: ns },
    spec: {
      selector: { app: 'mediamtx' },
      ports:
        [{ name: 'whep', port: 8889, targetPort: 8889, protocol: 'TCP' },
         { name: 'api', port: 9997, targetPort: 9997, protocol: 'TCP' }]
        // podip mode: the in-cluster test client reaches WebRTC via this
        // ClusterIP too.
        + (if !nodeport then [
          { name: 'wrtc-udp', port: udpPort, targetPort: udpPort, protocol: 'UDP' },
          { name: 'wrtc-tcp', port: tcpPort, targetPort: tcpPort, protocol: 'TCP' },
        ] else []),
    },
  },

  // SRT ingest Service (the Pi publishes here).
  srtService: {
    apiVersion: 'v1',
    kind: 'Service',
    metadata: { name: 'mediamtx-srt', namespace: ns },
    spec: {
      selector: { app: 'mediamtx' },
      ports: [{ name: 'srt', port: 8890, targetPort: 8890, protocol: 'UDP' }],
    },
  },

  // nodeport mode only: WebRTC media NodePort on the node public IPs.
  [if nodeport then 'webrtcService']: {
    apiVersion: 'v1',
    kind: 'Service',
    metadata: { name: 'mediamtx-webrtc', namespace: ns },
    spec: {
      type: 'NodePort',
      externalTrafficPolicy: 'Local',
      selector: { app: 'mediamtx' },
      ports: [
        { name: 'wrtc-udp', port: udpPort, targetPort: udpPort, nodePort: udpPort, protocol: 'UDP' },
        { name: 'wrtc-tcp', port: tcpPort, targetPort: tcpPort, nodePort: tcpPort, protocol: 'TCP' },
      ],
    },
  },

  // nodeport mode only: HTTPS Ingress for WHEP signaling, CORS for the webui.
  [if nodeport then 'ingress']: {
    apiVersion: 'networking.k8s.io/v1',
    kind: 'Ingress',
    metadata: {
      name: 'mediamtx',
      namespace: ns,
      annotations: {
        'cert-manager.io/cluster-issuer': cfg.clusterIssuer,
        'nginx.ingress.kubernetes.io/enable-cors': 'true',
        'nginx.ingress.kubernetes.io/cors-allow-origin': 'https://' + cfg.hosts.webui,
        'nginx.ingress.kubernetes.io/cors-allow-methods': 'GET, POST, PATCH, DELETE, OPTIONS',
        'nginx.ingress.kubernetes.io/cors-allow-headers': 'Content-Type, Authorization',
      },
    },
    spec: {
      ingressClassName: cfg.ingressClass,
      tls: [{ hosts: [cfg.hosts.media], secretName: 'fessel-mediamtx-tls' }],
      rules: [{
        host: cfg.hosts.media,
        http: { paths: [{
          path: '/',
          pathType: 'Prefix',
          backend: { service: { name: 'mediamtx', port: { number: 8889 } } },
        }] },
      }],
    },
  },
}
