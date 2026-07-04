// The fessel webui: a single Go service that serves the frontend + API,
// terminates the WebRTC live path (WHIP ingest from the Pi, WHEP fan-out to
// browsers), and fronts the recording store. Replaces the former
// FastAPI-backend + mediamtx pair (architecture §4).
//
// Two listeners in one process (B5.5.7): the public app on FESSEL_PORT and
// the tailnet-only ingest app on FESSEL_INGEST_PORT. Both share the storage
// backend so an uploaded-then-played round trip is consistent in-process.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/derfred/fessel/webui/internal/auth"
	"github.com/derfred/fessel/webui/internal/config"
	"github.com/derfred/fessel/webui/internal/health"
	"github.com/derfred/fessel/webui/internal/server"
	"github.com/derfred/fessel/webui/internal/storage"
	"github.com/derfred/fessel/webui/internal/supervisor"
	"github.com/derfred/fessel/webui/relay"
)

func main() {
	setupLogging()
	cfg := config.FromEnv()

	sup := supervisor.New(cfg.SupervisorBase, cfg.SupervisorTimeout)

	// Storage may legitimately be unconfigured in a control-plane-only dev run;
	// the recordings/ingest endpoints then fail per-request rather than the
	// whole process failing at boot.
	store, err := storage.Build(cfg)
	if err != nil {
		slog.Warn("recordings storage not configured; recordings/ingest endpoints will error", "err", err)
		store = storage.NewFakeBackend() // empty in-memory stand-in for listing
	}

	// The relay: two planes with different reachability (§5.1). The ingest
	// plane advertises the Tailscale sidecar's tailnet address (auto-discovered
	// from the pod netns); the viewer plane advertises the node public IPs on
	// the media NodePort. Empty config = default host-candidate gathering
	// (integration / local dev).
	ingestIPs := []string{}
	if cfg.IngestPublicIP == "auto" {
		ip := relay.WaitForTailnetIP(30 * time.Second)
		if ip == "" {
			slog.Error("FESSEL_INGEST_PUBLIC_IP=auto but no tailnet (100.64/10) address appeared; is the tailscale sidecar up?")
			os.Exit(1)
		}
		slog.Info("discovered tailnet IP for ingest", "ip", ip)
		ingestIPs = []string{ip}
	} else if cfg.IngestPublicIP != "" {
		ingestIPs = []string{cfg.IngestPublicIP}
	}
	metrics := relay.NewMetrics(prometheus.DefaultRegisterer)
	rly, err := relay.New(
		relay.ICEConfig{NAT1To1IPs: ingestIPs, UDPMuxPort: cfg.IngestUDPPort},
		relay.ICEConfig{NAT1To1IPs: cfg.ViewerPublicIPs, UDPMuxPort: cfg.ViewerUDPPort},
		metrics,
	)
	if err != nil {
		slog.Error("init relay", "err", err)
		os.Exit(1)
	}
	if cfg.PullFrom != "" {
		slog.Info("uplink mode: PULL (WHEP client)", "origin", cfg.PullFrom)
		rly.StartPull(cfg.PullFrom)
	}

	mon := health.NewMonitor(sup, rly, health.Config{
		RefreshInterval: cfg.HealthRefresh,
		FreshThreshold:  cfg.HealthFresh,
		StaleThreshold:  cfg.HealthStale,
	})
	mon.Start()
	defer mon.Stop()

	// Activation ownership (§2.3): the controller drives supervisor's
	// parameterless live activate/deactivate. The `path` field is transitional
	// — the current supervisor ActivateRequest still requires it.
	ctrl := &relay.Controller{
		Gate: mon.GateLiveView,
		Activate: func() error {
			res := sup.Post("/control/live/activate", map[string]any{"path": cfg.LivePath})
			if res.StatusCode < 200 || res.StatusCode >= 300 {
				return fmt.Errorf("supervisor activate returned %d: %v", res.StatusCode, res.Body)
			}
			return nil
		},
		Deactivate: func() error {
			res := sup.Post("/control/live/deactivate", map[string]any{"path": cfg.LivePath})
			if res.StatusCode < 200 || res.StatusCode >= 300 {
				return fmt.Errorf("supervisor deactivate returned %d: %v", res.StatusCode, res.Body)
			}
			return nil
		},
		IngestLive:        rly.IngestLive,
		WaitIngest:        rly.WaitIngest,
		ActivationTimeout: cfg.LiveActivationTimeout,
		IdleTimeout:       cfg.LiveIdleTimeout,
		Metrics:           metrics,
	}
	rly.SetViewerCountHook(ctrl.ViewerCountChanged)

	headers := auth.Headers{User: cfg.AuthUserHeader, Email: cfg.AuthEmailHeader, Groups: cfg.AuthGroupsHeader}
	public := &server.Public{
		Auth:       headers,
		Supervisor: sup,
		Storage:    store,
		Health:     mon,
		Relay:      rly,
		Controller: ctrl,
		StaticDir:  cfg.StaticDir,
	}
	ingest := &server.Ingest{Storage: store, Relay: rly}

	publicSrv := &http.Server{Addr: fmt.Sprintf(":%d", cfg.PublicPort), Handler: public.Handler()}
	ingestSrv := &http.Server{Addr: fmt.Sprintf(":%d", cfg.IngestPort), Handler: ingest.Handler()}

	errCh := make(chan error, 2)
	go func() {
		slog.Info("public listener up", "addr", publicSrv.Addr)
		errCh <- publicSrv.ListenAndServe()
	}()
	go func() {
		slog.Info("ingest listener up", "addr", ingestSrv.Addr)
		errCh <- ingestSrv.ListenAndServe()
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	select {
	case sig := <-stop:
		slog.Info("shutting down", "signal", sig.String())
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("listener failed", "err", err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = publicSrv.Shutdown(ctx)
	_ = ingestSrv.Shutdown(ctx)
}

// setupLogging installs a JSON slog handler (promtail/Loki-friendly, matching
// the FastAPI backend's structured audit lines). LOG_LEVEL selects
// debug|info|warn|error (default info).
func setupLogging() {
	lvl := slog.LevelInfo
	switch strings.ToLower(strings.TrimSpace(os.Getenv("LOG_LEVEL"))) {
	case "debug":
		lvl = slog.LevelDebug
	case "warn", "warning":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	}
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: lvl})))
}
