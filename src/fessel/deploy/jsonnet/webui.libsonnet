// webui (backend+frontend) Deployment + Service, and the shared WHEP Secret.
// In nodeport mode it's exposed via HTTPS Ingress and mints WHEP URLs at the
// public media host; in podip (test) mode it's ClusterIP only and the test
// client reaches mediamtx in-cluster.
function(cfg)
  local ns = cfg.namespace;
  local nodeport = cfg.webrtc.mode == 'nodeport';
  // Minted WHEP URL base: public media host in nodeport mode, in-cluster
  // mediamtx Service in podip mode.
  local mediaBase = if nodeport then 'https://' + cfg.hosts.media else 'http://mediamtx:8889';

  {
  secret: {
    apiVersion: 'v1',
    kind: 'Secret',
    metadata: { name: 'fessel-whep-secret', namespace: ns },
    stringData: { secret: cfg.whepSecret },
  },

  deployment: {
    apiVersion: 'apps/v1',
    kind: 'Deployment',
    metadata: { name: 'webui', namespace: ns, labels: { app: 'webui' } },
    spec: {
      replicas: 1,
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
            ],
            ports: [{ containerPort: 8000 }],
            readinessProbe: {
              httpGet: { path: '/healthz', port: 8000 },
              initialDelaySeconds: 2,
              periodSeconds: 3,
            },
            resources: { requests: { cpu: '100m', memory: '128Mi' } },
          }],
        },
      },
    },
  },

  service: {
    apiVersion: 'v1',
    kind: 'Service',
    metadata: { name: 'webui', namespace: ns },
    spec: { selector: { app: 'webui' }, ports: [{ port: 8000, targetPort: 8000 }] },
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
        // public ingress — an `oct` JWK *is* the signing secret. mediamtx
        // fetches it via the in-cluster Service (http://webui:8000/jwks),
        // not this host, so denying it publicly costs nothing. This is the
        // deployment-level half of the split; the backend's
        // forbid_identity_headers guard is the in-cluster half.
        'nginx.ingress.kubernetes.io/server-snippet': 'location = /jwks { return 404; }\n',
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
