package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/derfred/fessel/webui/internal/auth"
	"github.com/derfred/fessel/webui/internal/storage"
	"github.com/derfred/fessel/webui/internal/supervisor"
)

// fakeSupervisor records calls and returns canned results per path.
type fakeSupervisor struct {
	posts    []string
	bodies   []any
	results  map[string]supervisor.ForwardResult
	proxies  map[string]supervisor.ProxyResult
	lastHdrs map[string]string
}

func newFakeSupervisor() *fakeSupervisor {
	return &fakeSupervisor{
		results: map[string]supervisor.ForwardResult{},
		proxies: map[string]supervisor.ProxyResult{},
	}
}

func (f *fakeSupervisor) Post(path string, body any) supervisor.ForwardResult {
	f.posts = append(f.posts, path)
	f.bodies = append(f.bodies, body)
	if r, ok := f.results[path]; ok {
		return r
	}
	return supervisor.ForwardResult{StatusCode: 200, Body: map[string]any{"ok": true}}
}

func (f *fakeSupervisor) Get(path string) supervisor.ForwardResult {
	if r, ok := f.results[path]; ok {
		return r
	}
	return supervisor.ForwardResult{StatusCode: 200, Body: map[string]any{}}
}

func (f *fakeSupervisor) GetBytes(path string, headers map[string]string) supervisor.ProxyResult {
	f.lastHdrs = headers
	if r, ok := f.proxies[path]; ok {
		return r
	}
	return supervisor.ProxyResult{StatusCode: 404, Body: nil, Headers: map[string]string{}}
}

type fakeHealth struct{ snap map[string]any }

func (f *fakeHealth) Snapshot() map[string]any { return f.snap }

func testHeaders() auth.Headers {
	return auth.Headers{User: "X-Auth-Request-User", Email: "X-Auth-Request-Email", Groups: "X-Auth-Request-Groups"}
}

func newPublic(sup SupervisorAPI, store storage.Backend) *Public {
	if store == nil {
		store = storage.NewFakeBackend()
	}
	return &Public{
		Auth:       testHeaders(),
		Supervisor: sup,
		Storage:    store,
		Health:     &fakeHealth{snap: map[string]any{"light": "green"}},
		StaticDir:  "/nonexistent",
	}
}

func do(t *testing.T, h http.Handler, method, path string, body string, authd bool) *httptest.ResponseRecorder {
	t.Helper()
	var rdr io.Reader
	if body != "" {
		rdr = strings.NewReader(body)
	}
	req := httptest.NewRequest(method, path, rdr)
	if authd {
		req.Header.Set("X-Auth-Request-User", "alice")
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	return w
}

func decode(t *testing.T, w *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode %q: %v", w.Body.String(), err)
	}
	return out
}

// --- healthz / me / auth -------------------------------------------------------

func TestHealthzReportsVersion(t *testing.T) {
	t.Setenv("FESSEL_VERSION", "9.9.9")
	h := newPublic(newFakeSupervisor(), nil).Handler()
	w := do(t, h, "GET", "/healthz", "", false)
	out := decode(t, w)
	if w.Code != 200 || out["version"] != "9.9.9" {
		t.Fatalf("%d %v", w.Code, out)
	}
}

func TestMeRequiresIdentity(t *testing.T) {
	h := newPublic(newFakeSupervisor(), nil).Handler()
	if w := do(t, h, "GET", "/api/me", "", false); w.Code != 401 {
		t.Fatalf("unauth: %d", w.Code)
	}
	w := do(t, h, "GET", "/api/me", "", true)
	if w.Code != 200 || decode(t, w)["user"] != "alice" {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
}

func TestCapabilitiesRequiresIdentityAndListsModes(t *testing.T) {
	h := newPublic(newFakeSupervisor(), nil).Handler()
	if w := do(t, h, "GET", "/api/capabilities", "", false); w.Code != 401 {
		t.Fatalf("unauth: %d", w.Code)
	}
	w := do(t, h, "GET", "/api/capabilities", "", true)
	modes := decode(t, w)["modes"].([]any)
	if len(modes) != 3 {
		t.Fatalf("modes: %v", modes)
	}
}

// --- control plane ----------------------------------------------------------------

func TestControlForwardsAndPassesStatusThrough(t *testing.T) {
	sup := newFakeSupervisor()
	sup.results["/control/shutdown/arm"] = supervisor.ForwardResult{
		StatusCode: 503, Body: map[string]any{"error": "plug_unverified"},
	}
	h := newPublic(sup, nil).Handler()
	w := do(t, h, "POST", "/api/control/shutdown/arm", "", true)
	if w.Code != 503 || decode(t, w)["error"] != "plug_unverified" {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	if len(sup.posts) != 1 || sup.posts[0] != "/control/shutdown/arm" {
		t.Fatalf("posts: %v", sup.posts)
	}
}

func TestControlRejectsUnauthenticatedWithoutForwarding(t *testing.T) {
	sup := newFakeSupervisor()
	h := newPublic(sup, nil).Handler()
	w := do(t, h, "POST", "/api/control/pause", "", false)
	if w.Code != 401 {
		t.Fatalf("%d", w.Code)
	}
	if len(sup.posts) != 0 {
		t.Fatalf("forwarded despite missing auth: %v", sup.posts)
	}
}

func TestControlUnreachableSupervisorIs502(t *testing.T) {
	sup := newFakeSupervisor()
	sup.results["/control/pause"] = supervisor.ForwardResult{
		StatusCode: 502, Body: map[string]any{"error": "supervisor_unreachable", "message": "dial tcp: timeout"},
	}
	h := newPublic(sup, nil).Handler()
	w := do(t, h, "POST", "/api/control/pause", "", true)
	if w.Code != 502 || decode(t, w)["error"] != "supervisor_unreachable" {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
}

func TestAllControlActionsAreRouted(t *testing.T) {
	sup := newFakeSupervisor()
	h := newPublic(sup, nil).Handler()
	for action := range controlActions {
		w := do(t, h, "POST", "/api/control/"+action, "", true)
		if w.Code != 200 {
			t.Fatalf("%s: %d", action, w.Code)
		}
	}
	if len(sup.posts) != len(controlActions) {
		t.Fatalf("posts %v", sup.posts)
	}
}

// --- state / anomalies / health ------------------------------------------------------

func TestStatePassthrough(t *testing.T) {
	sup := newFakeSupervisor()
	sup.results["/state"] = supervisor.ForwardResult{StatusCode: 200, Body: map[string]any{"safety_state": "IDLE"}}
	h := newPublic(sup, nil).Handler()
	if w := do(t, h, "GET", "/api/state", "", false); w.Code != 401 {
		t.Fatalf("unauth %d", w.Code)
	}
	w := do(t, h, "GET", "/api/state", "", true)
	if decode(t, w)["safety_state"] != "IDLE" {
		t.Fatalf("%s", w.Body.String())
	}
}

func TestHealthPiServesSnapshot(t *testing.T) {
	p := newPublic(newFakeSupervisor(), nil)
	p.Health = &fakeHealth{snap: map[string]any{"light": "yellow"}}
	w := do(t, p.Handler(), "GET", "/api/health/pi", "", true)
	if decode(t, w)["light"] != "yellow" {
		t.Fatalf("%s", w.Body.String())
	}
}

// --- recording control ---------------------------------------------------------------

func TestRecordingStartFoldsOperator(t *testing.T) {
	sup := newFakeSupervisor()
	sup.results["/recording/start"] = supervisor.ForwardResult{
		StatusCode: 200, Body: map[string]any{"recording_id": "r1"},
	}
	h := newPublic(sup, nil).Handler()
	w := do(t, h, "POST", "/api/recording/start", `{"mode":"640x480@30@1000000"}`, true)
	if w.Code != 200 {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	body := sup.bodies[0].(map[string]any)
	if body["operator"] != "alice" || body["mode"] != "640x480@30@1000000" {
		t.Fatalf("body: %v", body)
	}
}

func TestRecordingStopToleratesEmptyBody(t *testing.T) {
	sup := newFakeSupervisor()
	h := newPublic(sup, nil).Handler()
	w := do(t, h, "POST", "/api/recording/stop", "", true)
	if w.Code != 200 {
		t.Fatalf("%d", w.Code)
	}
	if sup.bodies[0] != nil {
		t.Fatalf("stop should forward no body, got %v", sup.bodies[0])
	}
}

func TestRecordingControlRequiresAuth(t *testing.T) {
	sup := newFakeSupervisor()
	h := newPublic(sup, nil).Handler()
	for _, p := range []string{"/api/recording/start", "/api/recording/stop", "/api/recording/flag-upload"} {
		if w := do(t, h, "POST", p, "", false); w.Code != 401 {
			t.Fatalf("%s: %d", p, w.Code)
		}
	}
	if len(sup.posts) != 0 {
		t.Fatalf("forwarded despite missing auth: %v", sup.posts)
	}
}

// --- recordings list + playback -------------------------------------------------------

func TestRecordingsMergeLocalAndRemote(t *testing.T) {
	sup := newFakeSupervisor()
	sup.results["/recordings"] = supervisor.ForwardResult{StatusCode: 200, Body: []any{
		map[string]any{"recording_id": "both", "started_at": "2026-06-02T00:00:00Z", "upload_state": "uploaded"},
		map[string]any{"recording_id": "local-only", "started_at": "2026-06-03T00:00:00Z"},
	}}
	store := storage.NewFakeBackend()
	_ = store.Store("both", "metadata.json", strings.NewReader(`{"started_at":"2026-06-02T00:00:00Z"}`))
	_ = store.Store("remote-only", "metadata.json", strings.NewReader(`{"started_at":"2026-06-01T00:00:00Z","operator":"bob"}`))
	_ = store.Store("partial", "seg-00001.ts", strings.NewReader("x"))

	h := newPublic(sup, store).Handler()
	w := do(t, h, "GET", "/api/recordings", "", true)
	var out []map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	byID := map[string]map[string]any{}
	for _, r := range out {
		byID[r["recording_id"].(string)] = r
	}
	if len(out) != 4 {
		t.Fatalf("want 4 rows, got %d: %v", len(out), out)
	}
	if r := byID["both"]; r["available_local"] != true || r["available_remote"] != true {
		t.Fatalf("both: %v", r)
	}
	if r := byID["local-only"]; r["available_local"] != true || r["available_remote"] != false {
		t.Fatalf("local-only: %v", r)
	}
	if r := byID["remote-only"]; r["available_local"] != false || r["available_remote"] != true ||
		r["upload_state"] != "uploaded" || r["operator"] != "bob" {
		t.Fatalf("remote-only: %v", r)
	}
	// A remote recording without metadata.json is still uploading.
	if r := byID["partial"]; r["upload_state"] != "uploading" {
		t.Fatalf("partial: %v", r)
	}
	// Newest first.
	if out[0]["recording_id"] != "local-only" {
		t.Fatalf("order: %v", out)
	}
}

func TestPlaylistRedirectsForPresignedBackend(t *testing.T) {
	store := storage.NewFakeBackend()
	store.PresignBase = "http://minio:9000/rec"
	_ = store.Store("r1", "index.m3u8", strings.NewReader("#EXTM3U"))
	h := newPublic(newFakeSupervisor(), store).Handler()
	w := do(t, h, "GET", "/api/recordings/r1/playlist", "", true)
	if w.Code != 302 || !strings.Contains(w.Header().Get("Location"), "http://minio:9000/rec/r1/index.m3u8") {
		t.Fatalf("%d %q", w.Code, w.Header().Get("Location"))
	}
}

func TestSegmentServedLocallyWithRange(t *testing.T) {
	store := storage.NewFakeBackend()
	_ = store.Store("r1", "seg-00001.ts", strings.NewReader("0123456789"))
	h := newPublic(newFakeSupervisor(), store).Handler()

	// Full read.
	w := do(t, h, "GET", "/api/recordings/r1/segment/seg-00001.ts", "", true)
	if w.Code != 200 || w.Body.String() != "0123456789" || w.Header().Get("Content-Type") != "video/mp2t" {
		t.Fatalf("%d %q %q", w.Code, w.Body.String(), w.Header().Get("Content-Type"))
	}

	// Ranged read -> 206 + Content-Range.
	req := httptest.NewRequest("GET", "/api/recordings/r1/segment/seg-00001.ts", nil)
	req.Header.Set("X-Auth-Request-User", "alice")
	req.Header.Set("Range", "bytes=2-5")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 206 || rec.Body.String() != "2345" || rec.Header().Get("Content-Range") != "bytes 2-5/10" {
		t.Fatalf("%d %q %q", rec.Code, rec.Body.String(), rec.Header().Get("Content-Range"))
	}

	// Unsatisfiable range -> 416.
	req = httptest.NewRequest("GET", "/api/recordings/r1/segment/seg-00001.ts", nil)
	req.Header.Set("X-Auth-Request-User", "alice")
	req.Header.Set("Range", "bytes=50-60")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 416 {
		t.Fatalf("%d", rec.Code)
	}
}

func TestRecordingAbsentFromStoreProxiesSupervisor(t *testing.T) {
	sup := newFakeSupervisor()
	sup.proxies["/recordings/pi-only/index.m3u8"] = supervisor.ProxyResult{
		StatusCode: 200, Body: []byte("#EXTM3U"),
		Headers: map[string]string{"Content-Type": "application/vnd.apple.mpegurl"},
	}
	h := newPublic(sup, storage.NewFakeBackend()).Handler()
	w := do(t, h, "GET", "/api/recordings/pi-only/playlist", "", true)
	if w.Code != 200 || w.Body.String() != "#EXTM3U" {
		t.Fatalf("%d %q", w.Code, w.Body.String())
	}
}

func TestPlaybackRequiresAuth(t *testing.T) {
	h := newPublic(newFakeSupervisor(), nil).Handler()
	for _, p := range []string{"/api/recordings", "/api/recordings/r1/playlist", "/api/ring/playlist"} {
		if w := do(t, h, "GET", p, "", false); w.Code != 401 {
			t.Fatalf("%s: %d", p, w.Code)
		}
	}
}

// --- ring proxy ------------------------------------------------------------------------

func TestRingProxyForwardsRangeAndHeaders(t *testing.T) {
	sup := newFakeSupervisor()
	sup.proxies["/ring/index.m3u8"] = supervisor.ProxyResult{
		StatusCode: 206, Body: []byte("part"),
		Headers: map[string]string{"Content-Range": "bytes 0-3/100", "Content-Type": "application/vnd.apple.mpegurl"},
	}
	h := newPublic(sup, nil).Handler()
	req := httptest.NewRequest("GET", "/api/ring/playlist", nil)
	req.Header.Set("X-Auth-Request-User", "alice")
	req.Header.Set("Range", "bytes=0-3")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 206 || rec.Header().Get("Content-Range") != "bytes 0-3/100" {
		t.Fatalf("%d %v", rec.Code, rec.Header())
	}
	if sup.lastHdrs["Range"] != "bytes=0-3" {
		t.Fatalf("range not forwarded: %v", sup.lastHdrs)
	}
}

func TestRingProxyUnreachableIs502(t *testing.T) {
	sup := newFakeSupervisor()
	sup.proxies["/ring/index.m3u8"] = supervisor.ProxyResult{StatusCode: 502, Err: "dial timeout"}
	h := newPublic(sup, nil).Handler()
	w := do(t, h, "GET", "/api/ring/playlist", "", true)
	if w.Code != 502 || decode(t, w)["error"] != "supervisor_unreachable" {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
}

// --- listener separation (T5.5.3) --------------------------------------------------------

func TestPublicListenerHasNoIngestRoute(t *testing.T) {
	h := newPublic(newFakeSupervisor(), nil).Handler()
	w := do(t, h, "PUT", "/recording-ingest/r1/index.m3u8", "data", true)
	if w.Code != 404 && w.Code != 405 {
		t.Fatalf("ingest route reachable on public listener: %d", w.Code)
	}
	// And no WHIP ingest on the public listener either.
	w = do(t, h, "POST", "/whip/ingest", "v=0", true)
	if w.Code != 404 && w.Code != 405 {
		t.Fatalf("whip ingest reachable on public listener: %d", w.Code)
	}
}

// --- SPA static serving + client-route fallback --------------------------------

// The frontend is a client-side-routed SPA (Monitor `/`, Footage `/footage`).
// The static handler must serve real assets as-is but fall back to index.html
// for unknown paths, so a hard refresh or deep-link on a client route (or
// oauth2-proxy redirecting back to it) boots the SPA instead of 404ing.
func TestSPAStaticFallback(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "index.html"), []byte("<!doctype html>SPA"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "assets"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "assets", "app.js"), []byte("console.log(1)"), 0o644); err != nil {
		t.Fatal(err)
	}

	p := newPublic(newFakeSupervisor(), nil)
	p.StaticDir = dir
	h := p.Handler()

	// A client route that has no file on disk serves index.html (200), not 404.
	w := do(t, h, "GET", "/footage", "", true)
	if w.Code != 200 || !strings.Contains(w.Body.String(), "SPA") {
		t.Fatalf("client route fallback: %d %q", w.Code, w.Body.String())
	}

	// The root serves index.html.
	if w := do(t, h, "GET", "/", "", true); w.Code != 200 || !strings.Contains(w.Body.String(), "SPA") {
		t.Fatalf("root: %d %q", w.Code, w.Body.String())
	}

	// A real asset is served as itself, not the index fallback.
	w = do(t, h, "GET", "/assets/app.js", "", true)
	if w.Code != 200 || !strings.Contains(w.Body.String(), "console.log") {
		t.Fatalf("asset: %d %q", w.Code, w.Body.String())
	}

	// An unauthenticated API path still 401s — the SPA fallback must not swallow
	// API routes (they are registered as more specific patterns and win).
	if w := do(t, h, "GET", "/api/me", "", false); w.Code != 401 {
		t.Fatalf("api route swallowed by SPA fallback: %d", w.Code)
	}
}
