// Package server wires the two HTTP listeners (B5.5.7):
//
//   - PUBLIC (public.go)  -> browser surface behind oauth2-proxy: the /api
//     forwarders, WHEP viewer signaling, the static frontend, /metrics.
//   - INGEST (ingest.go)  -> tailnet-only: ALL Pi->webui traffic (recording
//     uploads + WHIP ingest signaling). The public Ingress never routes to the
//     ingest port, so those endpoints are structurally unreachable through it.
//
// There is no /jwks and no /api/auth/whep-url: with the relay in-process,
// viewer authorisation is "the /whep request carried valid oauth2-proxy
// identity headers" (architecture §4.5) — the JWT/JWKS machinery is deleted.
package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/derfred/fessel/webui/internal/auth"
	"github.com/derfred/fessel/webui/internal/schemas"
	"github.com/derfred/fessel/webui/internal/storage"
	"github.com/derfred/fessel/webui/internal/supervisor"
	"github.com/derfred/fessel/webui/internal/version"
	"github.com/derfred/fessel/webui/relay"
)

// controlActions maps operator control actions to the supervisor path each
// forwards to (B3.1). One source of truth so the endpoints and the audit log
// agree on names.
var controlActions = map[string]string{
	"pause":           "/control/pause",
	"stop":            "/control/stop",
	"resume":          "/control/resume",
	"shutdown/arm":    "/control/shutdown/arm",
	"shutdown/jetson": "/control/shutdown/jetson",
	"poweron/arm":     "/control/poweron/arm",
	"poweron/jetson":  "/control/poweron/jetson",
}

// SupervisorAPI is the slice of the supervisor client the handlers use.
type SupervisorAPI interface {
	Post(path string, jsonBody any) supervisor.ForwardResult
	Get(path string) supervisor.ForwardResult
	GetBytes(path string, headers map[string]string) supervisor.ProxyResult
}

// HealthAPI serves /api/health/pi from the monitor's cached snapshot.
type HealthAPI interface {
	Snapshot() map[string]any
}

// ViewerRelay is the slice of the relay the WHEP endpoints use.
type ViewerRelay interface {
	HandleViewerOffer(offer, peer string) (answer string, id int64, err error)
	CloseViewer(id int64)
}

type Public struct {
	Auth       auth.Headers
	Supervisor SupervisorAPI
	Storage    storage.Backend
	Health     HealthAPI
	Relay      ViewerRelay       // nil -> /whep returns 503 "relay disabled"
	Controller *relay.Controller // nil -> no gate/activation (tests)
	StaticDir  string
}

func (p *Public) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		// `version` is the release stamp (image tag == dpkg version). A
		// deployed cluster webui and Pi supervisor should report the same
		// string; a mismatch means the two halves of a release drifted.
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "version": version.Version()})
	})

	mux.HandleFunc("GET /api/me", func(w http.ResponseWriter, r *http.Request) {
		identity := p.requireIdentity(w, r)
		if identity == nil {
			return
		}
		groups := identity.Groups
		if groups == nil {
			groups = []string{}
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"user": identity.User, "email": nullable(identity.Email), "groups": groups,
		})
	})

	mux.HandleFunc("GET /api/capabilities", func(w http.ResponseWriter, r *http.Request) {
		if p.requireIdentity(w, r) == nil {
			return
		}
		// Static dev set until the supervisor /capabilities proxy lands.
		// RecordingMode is the deploy's configured recording operating point (the
		// UI no longer picks resolution — recording resolution is a deploy
		// setting); MaxLookbackSeconds bounds how far the record dialog can reach
		// back into the ring buffer.
		writeJSON(w, http.StatusOK, schemas.Capabilities{
			RecordingMode:      schemas.ModeTriplet{Resolution: "1280x720", Fps: 30, BitrateBps: 2_500_000},
			MaxLookbackSeconds: 120,
		})
	})

	// --- control plane (B3.1–B3.3) ------------------------------------------
	for action, supPath := range controlActions {
		action, supPath := action, supPath
		mux.HandleFunc("POST /api/control/"+action, func(w http.ResponseWriter, r *http.Request) {
			// Read identity in-handler so a missing-auth rejection is audited
			// as `auth_missing` (B3.3) before the 401.
			t0 := time.Now()
			identity := p.Auth.Read(r)
			if identity == nil {
				auditControl(action, "unknown", "auth_missing", t0)
				writeDetail(w, http.StatusUnauthorized, "authentication required")
				return
			}
			result := p.Supervisor.Post(supPath, nil)
			auditControl(action, identity.User, auditOutcome(result.StatusCode), t0)
			// Pass supervisor's status + body straight through to the frontend.
			writeJSON(w, result.StatusCode, result.Body)
		})
	}

	mux.HandleFunc("GET /api/state", func(w http.ResponseWriter, r *http.Request) {
		if p.requireIdentity(w, r) == nil {
			return
		}
		result := p.Supervisor.Get("/state")
		writeJSON(w, result.StatusCode, result.Body)
	})

	mux.HandleFunc("GET /api/health/pi", func(w http.ResponseWriter, r *http.Request) {
		if p.requireIdentity(w, r) == nil {
			return
		}
		// Read-only projection of the cached health picture; never probes
		// synchronously, so the frontend poll rate is decoupled from
		// supervisor load.
		writeJSON(w, http.StatusOK, p.Health.Snapshot())
	})

	mux.HandleFunc("GET /api/anomalies", func(w http.ResponseWriter, r *http.Request) {
		if p.requireIdentity(w, r) == nil {
			return
		}
		result := p.Supervisor.Get("/anomalies")
		writeJSON(w, result.StatusCode, result.Body)
	})

	// --- recording control + recordings/ring API (B4.1–B4.6, B5.5.4/5) -------
	mux.HandleFunc("POST /api/recording/start", p.forwardRecording("recording/start", "/recording/start"))
	mux.HandleFunc("POST /api/recording/stop", p.forwardRecording("recording/stop", "/recording/stop"))
	mux.HandleFunc("POST /api/recording/flag-upload", p.forwardRecording("recording/flag-upload", "/recording/flag-upload"))

	mux.HandleFunc("GET /api/recordings", func(w http.ResponseWriter, r *http.Request) {
		if p.requireIdentity(w, r) == nil {
			return
		}
		writeJSON(w, http.StatusOK, p.mergedRecordings())
	})

	mux.HandleFunc("GET /api/recordings/{id}/playlist", func(w http.ResponseWriter, r *http.Request) {
		if p.requireIdentity(w, r) == nil {
			return
		}
		p.serveRecordingFile(w, r, r.PathValue("id"), storage.PlaylistFilename)
	})

	mux.HandleFunc("GET /api/recordings/{id}/segment/{name}", func(w http.ResponseWriter, r *http.Request) {
		if p.requireIdentity(w, r) == nil {
			return
		}
		p.serveRecordingFile(w, r, r.PathValue("id"), r.PathValue("name"))
	})

	mux.HandleFunc("GET /api/recordings/{id}/upload", func(w http.ResponseWriter, r *http.Request) {
		if p.requireIdentity(w, r) == nil {
			return
		}
		result := p.Supervisor.Get("/recordings/" + r.PathValue("id") + "/upload")
		writeJSON(w, result.StatusCode, result.Body)
	})

	mux.HandleFunc("GET /api/ring/playlist", func(w http.ResponseWriter, r *http.Request) {
		if p.requireIdentity(w, r) == nil {
			return
		}
		// The ring is always local — no store path. Range support is
		// end-to-end (frontend -> here -> supervisor -> file).
		p.proxyBinary(w, r, "/ring/index.m3u8")
	})

	mux.HandleFunc("GET /api/ring/segment/{name}", func(w http.ResponseWriter, r *http.Request) {
		if p.requireIdentity(w, r) == nil {
			return
		}
		p.proxyBinary(w, r, "/ring/"+r.PathValue("name"))
	})

	// --- live WHEP viewer signaling (architecture §2.3/§4.2) ------------------
	mux.HandleFunc("POST /whep", p.handleWHEP)
	mux.HandleFunc("DELETE /whep/{id}", func(w http.ResponseWriter, r *http.Request) {
		if p.Relay == nil {
			w.WriteHeader(http.StatusOK)
			return
		}
		var id int64
		fmt.Sscanf(r.PathValue("id"), "%d", &id)
		p.Relay.CloseViewer(id)
		w.WriteHeader(http.StatusOK)
	})

	mux.Handle("GET /metrics", promhttp.Handler())

	// Serve the built React app at / when present (single-image deploy). The
	// API routes above are more specific, so they take precedence. Unknown paths
	// fall back to index.html so the client-side router owns them — a hard
	// refresh or deep-link on /footage (or oauth2-proxy redirecting back to it)
	// must serve the SPA, not 404.
	if st, err := os.Stat(p.StaticDir); err == nil && st.IsDir() {
		mux.Handle("/", spaFileServer(p.StaticDir))
	}

	return mux
}

// spaFileServer serves static assets from dir, falling back to index.html for
// any path that doesn't map to an existing file. This makes client-side routes
// (/footage) deep-linkable and refresh-safe. It only sees non-API paths: the
// /api, /whep and /metrics routes are registered with more specific patterns,
// so Go's mux dispatches those before this catch-all "/" handler.
func spaFileServer(dir string) http.Handler {
	fs := http.FileServer(http.Dir(dir))
	index := filepath.Join(dir, "index.html")
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Map the request path to an on-disk file. A real asset (index.html,
		// /assets/*.js, favicon) is served as-is; anything else is a client
		// route and gets index.html so the SPA boots and routes in the browser.
		clean := filepath.Clean(r.URL.Path)
		full := filepath.Join(dir, filepath.FromSlash(clean))
		if info, err := os.Stat(full); err == nil && !info.IsDir() {
			fs.ServeHTTP(w, r)
			return
		}
		http.ServeFile(w, r, index)
	})
}

// handleWHEP is the viewer signaling endpoint. It sits behind oauth2-proxy:
// the presence of valid identity headers IS the authorisation (no token). It
// runs the health gate + first-viewer activation, blocking up to
// live_activation_timeout before answering, so camera-down surfaces as a WHEP
// error rather than black frames.
func (p *Public) handleWHEP(w http.ResponseWriter, r *http.Request) {
	identity := p.Auth.Read(r)
	if identity == nil {
		writeDetail(w, http.StatusUnauthorized, "authentication required")
		return
	}
	if p.Relay == nil {
		writeJSON(w, http.StatusServiceUnavailable, schemas.LiveViewError{
			Error: schemas.LiveViewErrorCodeLiveUnavailable, Reason: "relay disabled",
		})
		return
	}
	offer, err := io.ReadAll(r.Body)
	if err != nil || len(offer) == 0 {
		writeDetail(w, http.StatusBadRequest, "missing SDP offer")
		return
	}

	if p.Controller != nil {
		if err := p.Controller.ViewerJoining(r.Context()); err != nil {
			var gate *relay.GateRejectedError
			var actFail *relay.ActivationFailedError
			switch {
			case errors.As(err, &gate):
				writeJSON(w, http.StatusServiceUnavailable, schemas.LiveViewError{
					Error: schemas.LiveViewErrorCodeLiveUnavailable, Reason: gate.Reason,
				})
			case errors.Is(err, relay.ErrActivationTimeout):
				writeJSON(w, http.StatusGatewayTimeout, schemas.LiveViewError{
					Error: schemas.LiveViewErrorCodeLiveTimeout, Reason: "timed out starting stream",
				})
			case errors.As(err, &actFail):
				writeJSON(w, http.StatusBadGateway, schemas.LiveViewError{
					Error: schemas.LiveViewErrorCodeLiveActivationFailed, Reason: actFail.Detail,
				})
			default:
				writeJSON(w, http.StatusInternalServerError, schemas.LiveViewError{
					Error: schemas.LiveViewErrorCodeLiveError, Reason: err.Error(),
				})
			}
			slog.Info("whep rejected", "operator", identity.User, "err", err)
			return
		}
	}

	answer, id, err := p.Relay.HandleViewerOffer(string(offer), r.RemoteAddr)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, schemas.LiveViewError{
			Error: schemas.LiveViewErrorCodeWhepNegotiateFailed, Reason: err.Error(),
		})
		return
	}
	// WHEP requires a Location header naming the session resource.
	w.Header().Set("Content-Type", "application/sdp")
	w.Header().Set("Location", fmt.Sprintf("/whep/%d", id))
	w.WriteHeader(http.StatusCreated)
	_, _ = io.WriteString(w, answer)
}

// --- recordings ---------------------------------------------------------------

// mergedRecordings implements B5.5.5: a unified list from supervisor (Pi-local)
// + the storage backend (uploaded), deduplicated by recording_id, each tagged
// available_local/available_remote.
func (p *Public) mergedRecordings() []map[string]any {
	sup := p.Supervisor.Get("/recordings")
	local, _ := sup.Body.([]any)
	remote := map[string]storage.RecordingView{}
	for _, v := range p.Storage.List() {
		remote[v.RecordingID] = v
	}

	merged := map[string]map[string]any{}
	var order []string
	for _, item := range local {
		obj, ok := item.(map[string]any)
		if !ok {
			continue
		}
		rid, _ := obj["recording_id"].(string)
		if rid == "" {
			continue
		}
		_, isRemote := remote[rid]
		merged[rid] = map[string]any{
			"recording_id":       rid,
			"type":               defaultStr(obj["type"], "explicit"),
			"started_at":         obj["started_at"],
			"ended_at":           obj["ended_at"],
			"duration_seconds":   obj["duration_seconds"],
			"operator":           obj["operator"],
			"flagged_for_upload": defaultBool(obj["flagged_for_upload"], false),
			"upload_state":       defaultStr(obj["upload_state"], "none"),
			"available_local":    true,
			"available_remote":   isRemote,
		}
		order = append(order, rid)
	}
	// Recordings present only in the store (Pi-side copy removed). metadata.json
	// is the uploader's last file, so a recording missing it is still "in
	// progress": surface upload_state accordingly.
	for rid, view := range remote {
		if _, ok := merged[rid]; ok {
			continue
		}
		meta := view.Metadata
		uploadState := "uploading"
		if meta != nil {
			uploadState = "uploaded"
		}
		if meta == nil {
			meta = map[string]any{}
		}
		merged[rid] = map[string]any{
			"recording_id":       rid,
			"type":               view.RecType,
			"started_at":         meta["started_at"],
			"ended_at":           meta["ended_at"],
			"duration_seconds":   meta["duration_seconds"],
			"operator":           meta["operator"],
			"flagged_for_upload": true,
			"upload_state":       uploadState,
			"available_local":    false,
			"available_remote":   true,
		}
		order = append(order, rid)
	}
	out := make([]map[string]any, 0, len(merged))
	for _, rid := range order {
		out = append(out, merged[rid])
	}
	// Newest first; missing started_at sorts last.
	sortByStartedAtDesc(out)
	return out
}

// serveRecordingFile is the B5.5.4 uniform front with backend-specific
// behaviour: PresignedURL -> 302 redirect (MinIO); ServeLocally -> 200/206
// byte stream (disk, Range honoured); absent from the store -> proxied from
// supervisor (a recording present only on the Pi).
func (p *Public) serveRecordingFile(w http.ResponseWriter, r *http.Request, recordingID, fileName string) {
	target, err := p.Storage.PlaybackURL(recordingID, fileName)
	if err != nil {
		writeDetail(w, http.StatusBadRequest, "invalid recording path")
		return
	}
	switch t := target.(type) {
	case storage.PresignedURL:
		http.Redirect(w, r, t.URL, http.StatusFound)
	case storage.ServeLocally:
		p.serveFromDiskBackend(w, r, recordingID, fileName)
	default:
		p.proxyBinary(w, r, "/recordings/"+recordingID+"/"+fileName)
	}
}

func (p *Public) serveFromDiskBackend(w http.ResponseWriter, r *http.Request, recordingID, fileName string) {
	result, err := p.Storage.Read(recordingID, fileName, r.Header.Get("Range"))
	if err != nil {
		if errors.Is(err, storage.ErrRangeNotSatisfiable) {
			w.Header().Set("Accept-Ranges", "bytes")
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return
		}
		writeDetail(w, http.StatusInternalServerError, err.Error())
		return
	}
	if result == nil {
		writeDetail(w, http.StatusNotFound, "not found")
		return
	}
	defer result.Body.Close()
	w.Header().Set("Content-Type", result.ContentType)
	w.Header().Set("Content-Length", fmt.Sprintf("%d", result.ContentLength))
	w.Header().Set("Accept-Ranges", "bytes")
	status := http.StatusOK
	if result.ByteRange != nil {
		w.Header().Set("Content-Range",
			fmt.Sprintf("bytes %d-%d/%d", result.ByteRange.Start, result.ByteRange.End, result.ByteRange.Total))
		status = http.StatusPartialContent
	}
	w.WriteHeader(status)
	_, _ = io.Copy(w, result.Body)
}

// proxyBinary passes the Range header through so HLS scrubbing works, and
// forwards supervisor's status (200/206/404) + the HLS-relevant response
// headers verbatim.
func (p *Public) proxyBinary(w http.ResponseWriter, r *http.Request, supPath string) {
	fwd := map[string]string{}
	if rng := r.Header.Get("Range"); rng != "" {
		fwd["Range"] = rng
	}
	result := p.Supervisor.GetBytes(supPath, fwd)
	if result.Err != "" {
		writeJSON(w, http.StatusBadGateway, map[string]any{
			"error": "supervisor_unreachable", "message": result.Err,
		})
		return
	}
	for k, v := range result.Headers {
		w.Header().Set(k, v)
	}
	w.WriteHeader(result.StatusCode)
	_, _ = w.Write(result.Body)
}

// forwardRecording forwards a recording control POST to supervisor, folding
// the trusted operator identity into start (so metadata records who started
// it) and emitting a B3.3-style audit line (auth_missing on rejection).
func (p *Public) forwardRecording(action, supPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		identity := p.Auth.Read(r)
		if identity == nil {
			auditRecording(action, "unknown", "auth_missing", nil, t0)
			writeDetail(w, http.StatusUnauthorized, "authentication required")
			return
		}
		body := safeJSONBody(r)
		if action == "recording/start" {
			body["operator"] = identity.User
		}
		var payload any
		if len(body) > 0 {
			payload = body
		}
		result := p.Supervisor.Post(supPath, payload)
		var recID any
		if obj, ok := result.Body.(map[string]any); ok {
			recID = obj["recording_id"]
		}
		if recID == nil {
			recID = body["recording_id"]
		}
		auditRecording(action, identity.User, auditOutcome(result.StatusCode), recID, t0)
		writeJSON(w, result.StatusCode, result.Body)
	}
}

// --- helpers ------------------------------------------------------------------

func (p *Public) requireIdentity(w http.ResponseWriter, r *http.Request) *auth.Identity {
	identity := p.Auth.Read(r)
	if identity == nil {
		writeDetail(w, http.StatusUnauthorized, "authentication required")
	}
	return identity
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeDetail(w http.ResponseWriter, status int, detail string) {
	writeJSON(w, status, map[string]any{"detail": detail})
}

func nullable(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func defaultStr(v any, def string) string {
	if s, ok := v.(string); ok && s != "" {
		return s
	}
	return def
}

func defaultBool(v any, def bool) bool {
	if b, ok := v.(bool); ok {
		return b
	}
	return def
}

// safeJSONBody tolerates an empty or non-JSON body as {} (recording POSTs may
// carry a JSON body — flag-upload: recording_id — or none: stop).
func safeJSONBody(r *http.Request) map[string]any {
	raw, err := io.ReadAll(r.Body)
	if err != nil || len(raw) == 0 {
		return map[string]any{}
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil || body == nil {
		return map[string]any{}
	}
	return body
}

func sortByStartedAtDesc(items []map[string]any) {
	key := func(m map[string]any) string {
		if s, ok := m["started_at"].(string); ok {
			return s
		}
		return ""
	}
	// Stable insertion-friendly sort (small lists).
	for i := 1; i < len(items); i++ {
		for j := i; j > 0 && key(items[j-1]) < key(items[j]); j-- {
			items[j-1], items[j] = items[j], items[j-1]
		}
	}
}

// auditOutcome maps a forwarded status to an audit outcome tag: 2xx ->
// success; our 502 -> unreachable; else the supervisor status.
func auditOutcome(status int) string {
	switch {
	case status >= 200 && status < 300:
		return "success"
	case status == 502:
		return "supervisor_unreachable"
	default:
		return fmt.Sprintf("supervisor_%d", status)
	}
}

func auditControl(action, operator, outcome string, t0 time.Time) {
	slog.Info("audit",
		"event", "control", "action", action, "operator", operator,
		"outcome", outcome, "latency_ms", time.Since(t0).Milliseconds(),
		"timestamp", time.Now().UTC().Format(time.RFC3339Nano))
}

func auditRecording(action, operator, outcome string, recordingID any, t0 time.Time) {
	slog.Info("audit",
		"event", "recording", "action", action, "operator", operator,
		"recording_id", recordingID, "outcome", outcome,
		"latency_ms", time.Since(t0).Milliseconds(),
		"timestamp", time.Now().UTC().Format(time.RFC3339Nano))
}
