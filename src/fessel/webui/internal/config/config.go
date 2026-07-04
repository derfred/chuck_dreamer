// Package config reads the webui service configuration from environment
// variables. The Tanka library wires these at deploy time; defaults suit the
// in-cluster dev/integration environments.
//
// The FESSEL_WHEP_SECRET / FESSEL_MEDIA_BASE / FESSEL_WHEP_TTL_S /
// FESSEL_MEDIAMTX_BASE knobs of the FastAPI backend are GONE: the relay is
// in-process (no mediamtx, no JWT/JWKS).
package config

import (
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	// Listeners. Public serves browsers (behind oauth2-proxy) + /metrics;
	// ingest serves ALL Pi->webui traffic (recording uploads + WHIP signaling)
	// and is exposed tailnet-only.
	PublicPort int // FESSEL_PORT, default 8000
	IngestPort int // FESSEL_INGEST_PORT, default 8001

	// oauth2-proxy identity header names.
	AuthUserHeader   string // FESSEL_AUTH_USER_HEADER, default X-Auth-Request-User
	AuthEmailHeader  string // FESSEL_AUTH_EMAIL_HEADER, default X-Auth-Request-Email
	AuthGroupsHeader string // FESSEL_AUTH_GROUPS_HEADER, default X-Auth-Request-Groups

	// Supervisor over the cluster->Pi Tailscale egress.
	SupervisorBase    string        // FESSEL_SUPERVISOR_BASE, default http://supervisor:8443
	SupervisorTimeout time.Duration // FESSEL_SUPERVISOR_TIMEOUT_S, default 10s

	// Recording storage.
	RecordingsBackend string // FESSEL_RECORDINGS_BACKEND: "minio" | "disk"
	DiskPath          string // FESSEL_RECORDINGS_DISK_PATH
	MinioEndpoint     string // FESSEL_MINIO_ENDPOINT
	MinioBucket       string // FESSEL_MINIO_BUCKET
	MinioAccessKey    string // FESSEL_MINIO_ACCESS_KEY
	MinioSecretKey    string // FESSEL_MINIO_SECRET_KEY
	MinioSecure       bool   // FESSEL_MINIO_SECURE, default true

	// Static frontend assets (built React app inside the image).
	StaticDir string // FESSEL_STATIC_DIR, default /app/static

	// Health monitor thresholds.
	HealthRefresh time.Duration // FESSEL_HEALTH_REFRESH_S, default 5s
	HealthFresh   time.Duration // FESSEL_HEALTH_FRESH_S, default 10s
	HealthStale   time.Duration // FESSEL_HEALTH_STALE_S, default 60s

	// Live relay (architecture §4.2/§2.3).
	LiveActivationTimeout time.Duration // FESSEL_LIVE_ACTIVATION_TIMEOUT_S, default 15s
	LiveIdleTimeout       time.Duration // FESSEL_LIVE_IDLE_TIMEOUT_S, default 10s
	// Transitional: the current supervisor ActivateRequest still requires a
	// `path`; carried until the parameterless activate lands on the Pi side.
	LivePath string // FESSEL_LIVE_PATH, default "pi"

	// ICE advertisement, per plane (see relay.ICEConfig).
	ViewerPublicIPs []string // FESSEL_VIEWER_PUBLIC_IPS (comma-separated node public IPs)
	ViewerUDPPort   int      // FESSEL_VIEWER_UDP_PORT (the NodePort)
	IngestPublicIP  string   // FESSEL_INGEST_PUBLIC_IP ("auto" = discover tailnet 100.x addr)
	IngestUDPPort   int      // FESSEL_INGEST_UDP_PORT

	// Optional WHEP-client uplink (validated escape hatch; empty = WHIP push).
	PullFrom string // FESSEL_PULL_FROM
}

func FromEnv() Config {
	return Config{
		PublicPort:            envInt("FESSEL_PORT", 8000),
		IngestPort:            envInt("FESSEL_INGEST_PORT", 8001),
		AuthUserHeader:        envStr("FESSEL_AUTH_USER_HEADER", "X-Auth-Request-User"),
		AuthEmailHeader:       envStr("FESSEL_AUTH_EMAIL_HEADER", "X-Auth-Request-Email"),
		AuthGroupsHeader:      envStr("FESSEL_AUTH_GROUPS_HEADER", "X-Auth-Request-Groups"),
		SupervisorBase:        strings.TrimRight(envStr("FESSEL_SUPERVISOR_BASE", "http://supervisor:8443"), "/"),
		SupervisorTimeout:     envSeconds("FESSEL_SUPERVISOR_TIMEOUT_S", 10*time.Second),
		RecordingsBackend:     strings.ToLower(strings.TrimSpace(os.Getenv("FESSEL_RECORDINGS_BACKEND"))),
		DiskPath:              os.Getenv("FESSEL_RECORDINGS_DISK_PATH"),
		MinioEndpoint:         os.Getenv("FESSEL_MINIO_ENDPOINT"),
		MinioBucket:           os.Getenv("FESSEL_MINIO_BUCKET"),
		MinioAccessKey:        os.Getenv("FESSEL_MINIO_ACCESS_KEY"),
		MinioSecretKey:        os.Getenv("FESSEL_MINIO_SECRET_KEY"),
		MinioSecure:           !strings.EqualFold(os.Getenv("FESSEL_MINIO_SECURE"), "false"),
		StaticDir:             envStr("FESSEL_STATIC_DIR", "/app/static"),
		HealthRefresh:         envSeconds("FESSEL_HEALTH_REFRESH_S", 5*time.Second),
		HealthFresh:           envSeconds("FESSEL_HEALTH_FRESH_S", 10*time.Second),
		HealthStale:           envSeconds("FESSEL_HEALTH_STALE_S", 60*time.Second),
		LiveActivationTimeout: envSeconds("FESSEL_LIVE_ACTIVATION_TIMEOUT_S", 15*time.Second),
		LiveIdleTimeout:       envSeconds("FESSEL_LIVE_IDLE_TIMEOUT_S", 10*time.Second),
		LivePath:              envStr("FESSEL_LIVE_PATH", "pi"),
		ViewerPublicIPs:       splitCSV(os.Getenv("FESSEL_VIEWER_PUBLIC_IPS")),
		ViewerUDPPort:         envInt("FESSEL_VIEWER_UDP_PORT", 0),
		IngestPublicIP:        strings.TrimSpace(os.Getenv("FESSEL_INGEST_PUBLIC_IP")),
		IngestUDPPort:         envInt("FESSEL_INGEST_UDP_PORT", 0),
		PullFrom:              strings.TrimSpace(os.Getenv("FESSEL_PULL_FROM")),
	}
}

func envStr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

func envSeconds(key string, def time.Duration) time.Duration {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return def
	}
	return time.Duration(f * float64(time.Second))
}

func splitCSV(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
