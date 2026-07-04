// The INGEST listener (B5.5.6/B5.5.7): a SEPARATE handler bound to a SEPARATE
// port (FESSEL_INGEST_PORT). The `webui-recording-ingest` Tailscale Service
// targets this port; the public Ingress never does. Because these routes exist
// ONLY on this handler, they are structurally impossible to reach through the
// public listener. ALL Pi->webui traffic lives here — recording uploads AND
// the live WHIP ingest signaling (the Pi dials the pod's tailnet address).
//
// Auth is at the network layer (Tailscale identity); there is no oauth2-proxy
// in front, no application-layer token.
package server

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/derfred/fessel/webui/internal/storage"
	"github.com/derfred/fessel/webui/internal/version"
)

// IngestRelay is the slice of the relay the WHIP endpoints use.
type IngestRelay interface {
	HandleIngestOffer(offer, peer string) (answer string, id int64, err error)
	CloseIngest()
}

type Ingest struct {
	Storage storage.Backend
	Relay   IngestRelay // nil -> /whip/ingest returns 503
}

func (in *Ingest) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"status": "ok", "version": version.Version(), "listener": "ingest",
		})
	})

	mux.HandleFunc("PUT /recording-ingest/{id}/{name}", func(w http.ResponseWriter, r *http.Request) {
		// The Pi uploader PUTs each recording file; the body streams straight
		// into the storage backend (both backends consume an io.Reader
		// incrementally, so memory stays bounded for a large segment). Per-file
		// PUT matches the uploader's per-file retry: a repeated PUT overwrites
		// cleanly (Store is idempotent).
		t0 := time.Now()
		recordingID, fileName := r.PathValue("id"), r.PathValue("name")
		counted := &countingReader{r: r.Body}
		err := in.Storage.Store(recordingID, fileName, counted)
		switch {
		case err == nil:
			ingestLog(recordingID, fileName, counted.n, "stored", t0)
			w.WriteHeader(http.StatusCreated)
		case errors.Is(err, storage.ErrInvalidPath):
			// Path-safety / bad id or name -> 400 (malformed request).
			ingestLog(recordingID, fileName, counted.n, "rejected", t0)
			writeDetail(w, http.StatusBadRequest, err.Error())
		default:
			// Backend store failure -> 5xx, the Pi retries.
			ingestLog(recordingID, fileName, counted.n, "store_error", t0)
			writeDetail(w, http.StatusBadGateway, fmt.Sprintf("store failed: %v", err))
		}
	})

	// Live WHIP ingest from the Pi (tailnet, not browser-facing, not behind
	// oauth2-proxy). Trust is by network position: the Pi dials the tsnet
	// sidecar's tailnet address, which terminates here.
	mux.HandleFunc("POST /whip/ingest", func(w http.ResponseWriter, r *http.Request) {
		if in.Relay == nil {
			writeDetail(w, http.StatusServiceUnavailable, "relay disabled")
			return
		}
		offer, err := io.ReadAll(r.Body)
		if err != nil || len(offer) == 0 {
			writeDetail(w, http.StatusBadRequest, "missing SDP offer")
			return
		}
		answer, id, err := in.Relay.HandleIngestOffer(string(offer), r.RemoteAddr)
		if err != nil {
			writeDetail(w, http.StatusInternalServerError, err.Error())
			return
		}
		w.Header().Set("Content-Type", "application/sdp")
		w.Header().Set("Location", fmt.Sprintf("/whip/ingest/%d", id))
		w.WriteHeader(http.StatusCreated)
		_, _ = io.WriteString(w, answer)
	})

	// WHIP teardown. Single-stream: closing the current ingest is enough; the
	// id is accepted for spec compliance but ignored.
	closeIngest := func(w http.ResponseWriter, r *http.Request) {
		if in.Relay != nil {
			in.Relay.CloseIngest()
		}
		w.WriteHeader(http.StatusOK)
	}
	mux.HandleFunc("DELETE /whip/ingest", closeIngest)
	mux.HandleFunc("DELETE /whip/ingest/{id}", closeIngest)

	return mux
}

type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

// ingestLog emits one structured line per PUT — the Loki audit trail for
// uploads (B5.5.6).
func ingestLog(recordingID, fileName string, size int64, outcome string, t0 time.Time) {
	slog.Info("audit",
		"event", "recording_ingest", "recording_id", recordingID, "file", fileName,
		"size_bytes", size, "outcome", outcome,
		"latency_ms", time.Since(t0).Milliseconds(),
		"timestamp", time.Now().UTC().Format(time.RFC3339Nano))
}
