package server

import (
	"errors"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/derfred/fessel/webui/internal/storage"
)

func TestIngestPutStoresFile(t *testing.T) {
	store := storage.NewFakeBackend()
	h := (&Ingest{Storage: store}).Handler()
	w := do(t, h, "PUT", "/recording-ingest/r1/seg-00001.ts", "segment-bytes", false)
	if w.Code != 201 {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	raw, ok := store.Bytes("r1", "seg-00001.ts")
	if !ok || string(raw) != "segment-bytes" {
		t.Fatalf("stored: %q %v", raw, ok)
	}
}

func TestIngestPutIsIdempotent(t *testing.T) {
	store := storage.NewFakeBackend()
	h := (&Ingest{Storage: store}).Handler()
	for _, body := range []string{"one", "two"} {
		if w := do(t, h, "PUT", "/recording-ingest/r1/index.m3u8", body, false); w.Code != 201 {
			t.Fatalf("%d", w.Code)
		}
	}
	raw, _ := store.Bytes("r1", "index.m3u8")
	if string(raw) != "two" {
		t.Fatalf("%q", raw)
	}
}

func TestIngestRejectsBadPath(t *testing.T) {
	store := storage.NewFakeBackend()
	h := (&Ingest{Storage: store}).Handler()
	// Traversal-shaped ids: a single ".." path component (an encoded slash
	// inside a component is normalised away by the mux; the backend's
	// isPlainComponent is the second line of defence, exercised directly in
	// the storage tests).
	w := do(t, h, "PUT", "/recording-ingest/%2E%2E/index.m3u8", "x", false)
	if w.Code != 400 {
		t.Fatalf("want 400, got %d", w.Code)
	}
}

func TestIngestStoreFailureIs502(t *testing.T) {
	store := storage.NewFakeBackend()
	store.StoreErr = errors.New("bucket down")
	h := (&Ingest{Storage: store}).Handler()
	w := do(t, h, "PUT", "/recording-ingest/r1/seg-00001.ts", "x", false)
	if w.Code != 502 {
		t.Fatalf("%d", w.Code)
	}
}

func TestIngestHealthzNamesListener(t *testing.T) {
	h := (&Ingest{Storage: storage.NewFakeBackend()}).Handler()
	w := do(t, h, "GET", "/healthz", "", false)
	if w.Code != 200 || decode(t, w)["listener"] != "ingest" {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
}

func TestIngestListenerHasNoAPIRoutes(t *testing.T) {
	h := (&Ingest{Storage: storage.NewFakeBackend()}).Handler()
	for _, p := range []string{"/api/state", "/api/recordings", "/whep"} {
		req := httptest.NewRequest("GET", p, nil)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != 404 && rec.Code != 405 {
			t.Fatalf("%s reachable on ingest listener: %d", p, rec.Code)
		}
	}
}

// fakeIngestRelay stands in for the Pion relay on the WHIP route.
type fakeIngestRelay struct {
	offers []string
	closed int
}

func (f *fakeIngestRelay) HandleIngestOffer(offer, peer string) (string, int64, error) {
	f.offers = append(f.offers, offer)
	return "v=0 answer", 7, nil
}
func (f *fakeIngestRelay) CloseIngest() { f.closed++ }

func TestWhipIngestAnswersOffer(t *testing.T) {
	rly := &fakeIngestRelay{}
	h := (&Ingest{Storage: storage.NewFakeBackend(), Relay: rly}).Handler()
	w := do(t, h, "POST", "/whip/ingest", "v=0 offer", false)
	if w.Code != 201 || w.Body.String() != "v=0 answer" {
		t.Fatalf("%d %q", w.Code, w.Body.String())
	}
	if w.Header().Get("Location") != "/whip/ingest/7" || w.Header().Get("Content-Type") != "application/sdp" {
		t.Fatalf("headers: %v", w.Header())
	}
	// Teardown.
	w = do(t, h, "DELETE", "/whip/ingest/7", "", false)
	if w.Code != 200 || rly.closed != 1 {
		t.Fatalf("%d closed=%d", w.Code, rly.closed)
	}
}

func TestWhipIngestWithoutRelayIs503(t *testing.T) {
	h := (&Ingest{Storage: storage.NewFakeBackend()}).Handler()
	w := do(t, h, "POST", "/whip/ingest", "v=0", false)
	if w.Code != 503 {
		t.Fatalf("%d", w.Code)
	}
}

func TestWhipIngestRejectsEmptyOffer(t *testing.T) {
	h := (&Ingest{Storage: storage.NewFakeBackend(), Relay: &fakeIngestRelay{}}).Handler()
	w := do(t, h, "POST", "/whip/ingest", "", false)
	if w.Code != 400 {
		t.Fatalf("%d", w.Code)
	}
}

func TestSnapshotPutStoresIntoBackend(t *testing.T) {
	store := storage.NewFakeBackend()
	h := (&Ingest{Storage: store}).Handler()
	w := do(t, h, "PUT", "/snapshot", "jpeg-bytes", false)
	if w.Code != 201 {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	data, _, ok := store.ReadSnapshot()
	if !ok || string(data) != "jpeg-bytes" {
		t.Fatalf("stored: %q %v", data, ok)
	}
}

func TestSnapshotPutRejectsEmptyBody(t *testing.T) {
	store := storage.NewFakeBackend()
	h := (&Ingest{Storage: store}).Handler()
	w := do(t, h, "PUT", "/snapshot", "", false)
	if w.Code != 400 {
		t.Fatalf("want 400, got %d", w.Code)
	}
}

// backendWithoutSnapshot is a storage.Backend that does NOT implement
// storage.MonitorSnapshot, exercising the ingest handler's type-assertion
// fallback (a hypothetical future backend that hasn't added snapshot support).
type backendWithoutSnapshot struct{ storage.Backend }

func TestSnapshotPutWithUnsupportedBackendIs503(t *testing.T) {
	h := (&Ingest{Storage: backendWithoutSnapshot{storage.NewFakeBackend()}}).Handler()
	w := do(t, h, "PUT", "/snapshot", "jpeg-bytes", false)
	if w.Code != 503 {
		t.Fatalf("%d", w.Code)
	}
}

var _ = strings.NewReader // keep strings import if unused by edits
