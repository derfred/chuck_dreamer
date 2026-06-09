# webui image (architecture §5.4): React frontend + FastAPI backend in one
# image. Build context is src/fessel/ so it can see schemas/, webui/, etc.
#   docker build -f webui/deploy/Dockerfile -t <tag> .

# ---- stage 1: build the React frontend ----
FROM node:20-bookworm-slim AS frontend
WORKDIR /build
# shared generated types live one level up from frontend in the repo layout.
COPY webui/shared /build/webui/shared
COPY webui/frontend /build/webui/frontend
WORKDIR /build/webui/frontend
RUN npm ci && npm run build

# ---- stage 2: backend runtime serving the built frontend ----
FROM python:3.12-slim-bookworm
WORKDIR /app

# Install backend + its local deps (schemas).
COPY schemas /app/schemas
COPY webui/backend /app/backend
RUN pip install --no-cache-dir /app/schemas /app/backend

# Built static assets served at / by the backend.
COPY --from=frontend /build/webui/frontend/dist /app/static
ENV FESSEL_STATIC_DIR=/app/static

# Release version stamp. The release workflow passes --build-arg
# FESSEL_VERSION=<version> (the same string used for the image tag and the Pi
# dpkg), and the backend surfaces it on /healthz so the running cluster can be
# checked for drift against the Pi. Defaults to "unknown" for local builds.
ARG FESSEL_VERSION=unknown
ENV FESSEL_VERSION=${FESSEL_VERSION}

# Two listeners (Slice 5.5, B5.5.7): the public browser surface on :8000 (behind
# oauth2-proxy via the public Ingress) and the tailnet-only recording-ingest
# listener on :8001 (targeted by the webui-recording-ingest Tailscale Service,
# never by the public Ingress). serve_both binds both in one process.
EXPOSE 8000 8001
CMD ["fessel-backend"]
