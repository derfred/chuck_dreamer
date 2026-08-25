package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/derfred/fessel/webui/internal/auth"
	"github.com/derfred/fessel/webui/internal/storage"
	"github.com/derfred/fessel/webui/internal/supervisor"
)

// fakeSupervisor records calls and returns canned results per path.
type fakeSupervisor struct {
	posts     []string
	bodies    []any
	results   map[string]supervisor.ForwardResult
	proxies   map[string]supervisor.ProxyResult
	lastHdrs  map[string]string
	connCheck supervisor.ConnectionCheckResult
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

func (f *fakeSupervisor) ConnectionCheck() supervisor.ConnectionCheckResult {
	return f.connCheck
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

func TestCapabilitiesRequiresIdentityAndReportsRecordingModeAndLookback(t *testing.T) {
	h := newPublic(newFakeSupervisor(), nil).Handler()
	if w := do(t, h, "GET", "/api/capabilities", "", false); w.Code != 401 {
		t.Fatalf("unauth: %d", w.Code)
	}
	w := do(t, h, "GET", "/api/capabilities", "", true)
	body := decode(t, w)
	// No mode list any more (the UI doesn't pick a mode); the deploy's recording
	// mode + the look-back bound are what the record dialog reads.
	if _, ok := body["modes"]; ok {
		t.Fatalf("capabilities must not carry a modes list: %v", body)
	}
	rec, ok := body["recording_mode"].(map[string]any)
	if !ok || rec["resolution"] != "1280x720" {
		t.Fatalf("recording_mode: %v", body["recording_mode"])
	}
	if body["max_lookback_seconds"].(float64) != 120 {
		t.Fatalf("max_lookback_seconds: %v", body["max_lookback_seconds"])
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

func TestConnectionCheckRequiresAuth(t *testing.T) {
	h := newPublic(newFakeSupervisor(), nil).Handler()
	if w := do(t, h, "POST", "/api/connection-check", "", false); w.Code != 401 {
		t.Fatalf("unauth %d", w.Code)
	}
}

func TestConnectionCheckReportsLatencyAndThroughput(t *testing.T) {
	sup := newFakeSupervisor()
	sup.connCheck = supervisor.ConnectionCheckResult{
		LatencyMs: 42.5, ThroughputMbps: 12.3, PayloadBytes: 1_000_000,
	}
	h := newPublic(sup, nil).Handler()
	w := do(t, h, "POST", "/api/connection-check", "", true)
	if w.Code != 200 {
		t.Fatalf("%d: %s", w.Code, w.Body.String())
	}
	body := decode(t, w)
	if body["latency_ms"] != 42.5 || body["throughput_mbps"] != 12.3 {
		t.Fatalf("%v", body)
	}
}

func TestConnectionCheckUnreachableIs502(t *testing.T) {
	sup := newFakeSupervisor()
	sup.connCheck = supervisor.ConnectionCheckResult{Err: "dial tcp: timeout"}
	h := newPublic(sup, nil).Handler()
	w := do(t, h, "POST", "/api/connection-check", "", true)
	if w.Code != 502 {
		t.Fatalf("%d: %s", w.Code, w.Body.String())
	}
}

// --- Monitor freeze-frame ------------------------------------------------------------
// Persisted through the same storage.Backend recordings use (storage.MonitorSnapshot),
// so these tests exercise it via the FakeBackend's snapshot slot, not a separate holder.

func TestSnapshotRequiresAuth(t *testing.T) {
	h := newPublic(newFakeSupervisor(), nil).Handler()
	if w := do(t, h, "GET", "/api/snapshot", "", false); w.Code != 401 {
		t.Fatalf("unauth: %d", w.Code)
	}
	if w := do(t, h, "GET", "/api/snapshot/meta", "", false); w.Code != 401 {
		t.Fatalf("unauth meta: %d", w.Code)
	}
}

func TestSnapshotNotYetAvailable(t *testing.T) {
	h := newPublic(newFakeSupervisor(), nil).Handler()

	w := do(t, h, "GET", "/api/snapshot", "", true)
	if w.Code != 404 {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	w = do(t, h, "GET", "/api/snapshot/meta", "", true)
	if w.Code != 200 || decode(t, w)["available"] != false {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
}

func TestSnapshotServesLatestJPEGAndMeta(t *testing.T) {
	store := storage.NewFakeBackend()
	capturedAt := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	_ = store.StoreSnapshot([]byte("jpeg-bytes"), capturedAt)
	h := newPublic(newFakeSupervisor(), store).Handler()

	w := do(t, h, "GET", "/api/snapshot", "", true)
	if w.Code != 200 || w.Body.String() != "jpeg-bytes" || w.Header().Get("Content-Type") != "image/jpeg" {
		t.Fatalf("%d %q %q", w.Code, w.Body.String(), w.Header().Get("Content-Type"))
	}

	// meta reports the CAPTURE time the store carries (what the "Xs ago" label
	// is about), not a fresh now-timestamp minted at read time.
	w = do(t, h, "GET", "/api/snapshot/meta", "", true)
	body := decode(t, w)
	if w.Code != 200 || body["available"] != true {
		t.Fatalf("%d %v", w.Code, body)
	}
	got, err := time.Parse(time.RFC3339Nano, body["captured_at"].(string))
	if err != nil || !got.Equal(capturedAt) {
		t.Fatalf("captured_at %v (%v), want %v", body["captured_at"], err, capturedAt)
	}
}

func TestSnapshotWithUnsupportedBackendIsNotFound(t *testing.T) {
	h := newPublic(newFakeSupervisor(), backendWithoutSnapshot{storage.NewFakeBackend()}).Handler()
	w := do(t, h, "GET", "/api/snapshot", "", true)
	if w.Code != 404 {
		t.Fatalf("%d", w.Code)
	}
	w = do(t, h, "GET", "/api/snapshot/meta", "", true)
	if w.Code != 200 || decode(t, w)["available"] != false {
		t.Fatalf("%d %v", w.Code, decode(t, w))
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
	for _, p := range []string{"/api/recordings", "/api/recordings/r1/playlist"} {
		if w := do(t, h, "GET", p, "", false); w.Code != 401 {
			t.Fatalf("%s: %d", p, w.Code)
		}
	}
}

// (No ring-proxy tests: the ring is never streamed back for viewing — see the
// note where the /api/ring routes used to be registered. Recording playback's
// range/header proxying is covered by the recordings-playback tests above.)

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

// --- playlist URI rewriting (the F5.5.1 playback regression) ------------------

// resolveAgainst mimics how an HLS player resolves a playlist's relative URIs:
// against the playlist's OWN url (RFC 8216 §4), not against a path the caller
// reconstructs. Reconstructing the segment URL is exactly what let the original
// bug through CI — both endpoints worked in isolation while no test ever walked
// the playlist the way hls.js does.
func resolveAgainst(playlistURL, uri string) string {
	base, err := url.Parse(playlistURL)
	if err != nil {
		return uri
	}
	ref, err := url.Parse(uri)
	if err != nil {
		return uri
	}
	return base.ResolveReference(ref).Path
}

// firstSegmentURI returns the first non-tag, non-blank line of a playlist.
func firstSegmentURI(playlist string) string {
	for _, line := range strings.Split(playlist, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		return line
	}
	return ""
}

const canonicalPlaylist = "#EXTM3U\n" +
	"#EXT-X-VERSION:3\n" +
	"#EXT-X-MEDIA-SEQUENCE:0\n" +
	"#EXT-X-TARGETDURATION:2\n" +
	"#EXT-X-PLAYLIST-TYPE:VOD\n" +
	"#EXTINF:2.0,\n" +
	"seg-00000.ts\n" +
	"#EXTINF:2.0,\n" +
	"seg-00001.ts\n" +
	"#EXT-X-ENDLIST\n"

// The end-to-end contract: fetch the playlist, resolve its first segment URI
// the way a player does, and that URL must actually serve the segment bytes.
func TestPlaylistSegmentURIsResolveToServableSegments(t *testing.T) {
	store := storage.NewFakeBackend()
	_ = store.Store("r1", "index.m3u8", strings.NewReader(canonicalPlaylist))
	_ = store.Store("r1", "seg-00000.ts", strings.NewReader("SEGMENT-ZERO"))
	h := newPublic(newFakeSupervisor(), store).Handler()

	const playlistURL = "/api/recordings/r1/playlist"
	w := do(t, h, "GET", playlistURL, "", true)
	if w.Code != 200 {
		t.Fatalf("playlist status %d", w.Code)
	}
	uri := firstSegmentURI(w.Body.String())
	if uri == "" {
		t.Fatalf("no segment URI in playlist: %q", w.Body.String())
	}

	resolved := resolveAgainst(playlistURL, uri)
	seg := do(t, h, "GET", resolved, "", true)
	if seg.Code != 200 {
		t.Fatalf("segment %q (resolved from %q) -> %d; a player would fail here",
			resolved, uri, seg.Code)
	}
	if seg.Body.String() != "SEGMENT-ZERO" {
		t.Fatalf("segment body %q", seg.Body.String())
	}
	if ct := seg.Header().Get("Content-Type"); ct != "video/mp2t" {
		t.Fatalf("segment content-type %q", ct)
	}
}

// Tags must survive the rewrite untouched, and Content-Length must describe the
// rewritten body (not the stored one) or the response is truncated.
func TestPlaylistRewritePreservesTagsAndLength(t *testing.T) {
	store := storage.NewFakeBackend()
	_ = store.Store("r1", "index.m3u8", strings.NewReader(canonicalPlaylist))
	h := newPublic(newFakeSupervisor(), store).Handler()

	w := do(t, h, "GET", "/api/recordings/r1/playlist", "", true)
	body := w.Body.String()
	for _, tag := range []string{"#EXTM3U", "#EXT-X-PLAYLIST-TYPE:VOD", "#EXTINF:2.0,", "#EXT-X-ENDLIST"} {
		if !strings.Contains(body, tag) {
			t.Fatalf("tag %q lost: %q", tag, body)
		}
	}
	if !strings.Contains(body, "segment/seg-00000.ts") || !strings.Contains(body, "segment/seg-00001.ts") {
		t.Fatalf("segment URIs not rewritten: %q", body)
	}
	if cl := w.Header().Get("Content-Length"); cl != fmt.Sprint(len(body)) {
		t.Fatalf("Content-Length %q != body length %d", cl, len(body))
	}
	if ct := w.Header().Get("Content-Type"); ct != "application/vnd.apple.mpegurl" {
		t.Fatalf("content-type %q", ct)
	}
}

// A recording still only on the Pi is proxied from supervisor — same URL, so it
// needs the same rewrite.
func TestProxiedPlaylistIsAlsoRewritten(t *testing.T) {
	sup := newFakeSupervisor()
	sup.proxies["/recordings/pi-only/index.m3u8"] = supervisor.ProxyResult{
		StatusCode: 200, Body: []byte(canonicalPlaylist),
		Headers: map[string]string{"Content-Type": "application/vnd.apple.mpegurl"},
	}
	h := newPublic(sup, storage.NewFakeBackend()).Handler()

	const playlistURL = "/api/recordings/pi-only/playlist"
	w := do(t, h, "GET", playlistURL, "", true)
	if w.Code != 200 {
		t.Fatalf("status %d", w.Code)
	}
	resolved := resolveAgainst(playlistURL, firstSegmentURI(w.Body.String()))
	if resolved != "/api/recordings/pi-only/segment/seg-00000.ts" {
		t.Fatalf("resolved to %q", resolved)
	}
}

// The presigned (MinIO) branch redirects the browser to the object store, where
// segments ARE siblings of index.m3u8. Rewriting there would corrupt playback,
// so that branch must stay a plain 302 with the body untouched.
func TestPresignedPlaylistIsNotRewritten(t *testing.T) {
	store := storage.NewFakeBackend()
	store.PresignBase = "http://minio:9000/rec"
	_ = store.Store("r1", "index.m3u8", strings.NewReader(canonicalPlaylist))
	h := newPublic(newFakeSupervisor(), store).Handler()

	w := do(t, h, "GET", "/api/recordings/r1/playlist", "", true)
	if w.Code != 302 {
		t.Fatalf("expected redirect, got %d", w.Code)
	}
	if strings.Contains(w.Body.String(), "segment/") {
		t.Fatalf("presigned branch rewrote the playlist: %q", w.Body.String())
	}
}

// --- delete (cluster-store copy only) ----------------------------------------

func TestDeleteRemovesFromStoreOnly(t *testing.T) {
	store := storage.NewFakeBackend()
	_ = store.Store("r1", "seg-00000.ts", strings.NewReader("bytes"))
	_ = store.Store("r1", "index.m3u8", strings.NewReader(canonicalPlaylist))
	sup := newFakeSupervisor()
	h := newPublic(sup, store).Handler()

	w := do(t, h, "DELETE", "/api/recordings/r1", "", true)
	if w.Code != 200 {
		t.Fatalf("delete status %d (%s)", w.Code, w.Body.String())
	}
	if store.Exists("r1") {
		t.Fatal("still in the store after delete")
	}
	// The Pi copy is a separate lifecycle — nothing is forwarded to supervisor.
	if len(sup.posts) != 0 {
		t.Fatalf("delete forwarded to supervisor: %v", sup.posts)
	}
}

func TestDeleteIsIdempotentWith404(t *testing.T) {
	h := newPublic(newFakeSupervisor(), storage.NewFakeBackend()).Handler()
	w := do(t, h, "DELETE", "/api/recordings/nope", "", true)
	if w.Code != 404 {
		t.Fatalf("status %d", w.Code)
	}
}

func TestDeleteRequiresAuth(t *testing.T) {
	store := storage.NewFakeBackend()
	_ = store.Store("r1", "index.m3u8", strings.NewReader("#EXTM3U"))
	h := newPublic(newFakeSupervisor(), store).Handler()
	w := do(t, h, "DELETE", "/api/recordings/r1", "", false)
	if w.Code != 401 {
		t.Fatalf("status %d", w.Code)
	}
	if !store.Exists("r1") {
		t.Fatal("unauthenticated DELETE removed the recording")
	}
}
