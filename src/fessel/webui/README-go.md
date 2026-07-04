# fessel webui (Go)

The single cluster-side service from `docs/fessel/architecture.md` §4: it
serves the React frontend + `/api` surface, terminates the WebRTC live path
(WHIP ingest from the Pi, WHEP fan-out to browsers), and fronts the recording
store.

## Layout

    cmd/webui/        entrypoint: both listeners, wiring
    relay/            Pion two-plane relay + activation controller + metrics
    internal/server/  public listener (API, /whep, static) + ingest listener
                      (PUT /recording-ingest, POST /whip/ingest)
    internal/schemas/ GENERATED Go structs from the shared pydantic models
                      (tools/generate-types.sh; CI check-types gates drift)
    internal/storage/ recording store: disk (PVC) + MinIO + fake
    internal/health/  Pi-health facts, snapshot, /whep gate
    internal/supervisor/  pass-through client over the Tailscale egress
    frontend/         React app (built + served as static assets; excluded
                      from the Go build via the go.mod ignore directive)

## API contract

`src/fessel/schemas` (pydantic) is the single source of truth. `make types`
generates three checked-in artifacts — TS types (`shared/schemas.ts`), Zod
validators (`frontend/src/generated/validators.ts`), and Go structs
(`internal/schemas/schemas_gen.go`) — and CI's `check-types` fails on stale
output. The Go structs cover the bodies this service AUTHORS (capabilities,
live-view errors); supervisor bodies are passed through verbatim by design,
so they stay untyped here and are validated at their producer (supervisor's
pydantic models) and consumer (the frontend's Zod validators).

## Configuration (env)

    FESSEL_PORT / FESSEL_INGEST_PORT          8000 / 8001
    FESSEL_AUTH_{USER,EMAIL,GROUPS}_HEADER    oauth2-proxy identity headers
    FESSEL_SUPERVISOR_BASE / _TIMEOUT_S       Tailscale egress to the Pi
    FESSEL_RECORDINGS_BACKEND                 disk | minio (+ FESSEL_RECORDINGS_DISK_PATH / FESSEL_MINIO_*)
    FESSEL_STATIC_DIR                         built frontend (default /app/static)
    FESSEL_HEALTH_{REFRESH,FRESH,STALE}_S     health thresholds
    FESSEL_LIVE_ACTIVATION_TIMEOUT_S          default 15 (blocks /whep on first viewer)
    FESSEL_LIVE_IDLE_TIMEOUT_S                default 10 (deactivate debounce)
    FESSEL_VIEWER_PUBLIC_IPS / _UDP_PORT      viewer-plane ICE (node public IPs + NodePort)
    FESSEL_INGEST_PUBLIC_IP / _UDP_PORT       ingest-plane ICE ("auto" = tailnet discovery)
    FESSEL_H264_FMTP                          override if the Pi encoder profile changes

## Build & test

    go build ./...
    go test ./...        # includes a real Pion loopback end-to-end test
    docker build -f deploy/go.Dockerfile --build-arg FESSEL_VERSION=<ver> .
