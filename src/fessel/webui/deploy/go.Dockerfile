# fessel webui (Go): single image with the Go binary (API + WHIP/WHEP relay +
# recording store front) and the built React app served as static assets.
# Replaces the FastAPI image (deploy/Dockerfile) AND the mediamtx deployment at
# cutover; until then both Dockerfiles coexist.
#
# Build from the repo's src/fessel/webui directory:
#   docker build -f deploy/go.Dockerfile --build-arg FESSEL_VERSION=<ver> .

# --- stage 1: frontend build ---------------------------------------------------
FROM node:24-bookworm-slim AS frontend
WORKDIR /build
# shared generated types live one level up from frontend (tsconfig ../shared).
COPY shared /build/shared
COPY frontend /build/frontend
WORKDIR /build/frontend
RUN npm ci && npm run build

# --- stage 2: Go build ----------------------------------------------------------
FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd/ cmd/
COPY internal/ internal/
COPY relay/ relay/
ARG FESSEL_VERSION=unknown
RUN CGO_ENABLED=0 go build -ldflags "-s -w -X github.com/derfred/fessel/webui/internal/version.Stamp=${FESSEL_VERSION}" -o /webui ./cmd/webui

# --- stage 3: runtime -------------------------------------------------------------
FROM alpine:3.20
RUN adduser -D -u 1000 webui
USER webui
COPY --from=build /webui /usr/local/bin/webui
COPY --from=frontend /build/frontend/dist /app/static
ARG FESSEL_VERSION=unknown
ENV FESSEL_VERSION=${FESSEL_VERSION} \
    FESSEL_STATIC_DIR=/app/static
# public listener, ingest listener; media UDP ports are configured per-env
# (FESSEL_VIEWER_UDP_PORT / FESSEL_INGEST_UDP_PORT) and exposed by the pod spec.
EXPOSE 8000 8001
ENTRYPOINT ["/usr/local/bin/webui"]
