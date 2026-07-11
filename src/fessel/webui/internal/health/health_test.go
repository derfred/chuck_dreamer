package health

import (
	"testing"
	"time"

	"github.com/derfred/fessel/webui/internal/supervisor"
)

type fakeSup struct {
	result supervisor.ForwardResult
}

func (f *fakeSup) Get(path string) supervisor.ForwardResult { return f.result }

type fakeRelay struct {
	live    bool
	viewers int
}

func (f *fakeRelay) IngestLive() bool { return f.live }
func (f *fakeRelay) ViewerCount() int { return f.viewers }

func healthyState() map[string]any {
	return map[string]any{
		"vision":         map[string]any{"healthy": true},
		"camera":         map[string]any{"up": true},
		"upload_backlog": map[string]any{"count": float64(0)},
		"version":        map[string]any{"component": "1.4.2"},
	}
}

func newTestMonitor(sup SupervisorAPI, relay RelayStatus) *Monitor {
	m := NewMonitor(sup, relay, Config{
		RefreshInterval: time.Second,
		FreshThreshold:  10 * time.Second,
		StaleThreshold:  60 * time.Second,
	})
	m.ClusterVersion = "1.4.0"
	return m
}

func factByID(snap map[string]any, id string) map[string]any {
	for _, f := range snap["facts"].([]any) {
		fm := f.(map[string]any)
		if fm["id"] == id {
			return fm
		}
	}
	return nil
}

func TestSnapshotBeforeFirstRefreshIsAllUnknown(t *testing.T) {
	m := newTestMonitor(&fakeSup{}, nil)
	snap := m.Snapshot()
	if snap["light"] != "unknown" {
		t.Fatalf("light %v", snap["light"])
	}
	if len(snap["facts"].([]any)) != 6 {
		t.Fatalf("want 6 facts")
	}
}

func TestAllGreen(t *testing.T) {
	m := newTestMonitor(&fakeSup{supervisor.ForwardResult{StatusCode: 200, Body: healthyState()}}, &fakeRelay{live: true, viewers: 2})
	snap := m.RefreshOnce()
	if snap["light"] != "green" {
		t.Fatalf("light %v (facts %v)", snap["light"], snap["facts"])
	}
	if f := factByID(snap, "relay"); f["state"] != "green" {
		t.Fatalf("relay %v", f)
	}
	if f := factByID(snap, "version_sync"); f["state"] != "green" {
		t.Fatalf("version_sync %v", f)
	}
}

func TestMaskingOnUnreachablePi(t *testing.T) {
	m := newTestMonitor(&fakeSup{supervisor.ForwardResult{StatusCode: 502, Body: map[string]any{"error": "supervisor_unreachable"}}}, &fakeRelay{})
	snap := m.RefreshOnce()
	if snap["light"] != "red" {
		t.Fatalf("light %v", snap["light"])
	}
	if f := factByID(snap, "pi_control"); f["state"] != "red" {
		t.Fatalf("pi_control %v", f)
	}
	// Facts observed through supervisor become unknown, not red.
	for _, id := range []string{"video", "camera", "uploader", "version_sync"} {
		if f := factByID(snap, id); f["state"] != "unknown" {
			t.Fatalf("%s should be masked unknown: %v", id, f)
		}
	}
	// The in-process relay is never masked.
	if f := factByID(snap, "relay"); f["state"] != "green" {
		t.Fatalf("relay %v", f)
	}
}

func TestVersionSyncStates(t *testing.T) {
	cases := []struct {
		name    string
		version any
		want    string
	}{
		{"match", map[string]any{"component": "1.4.9"}, "green"},
		{"mismatch", map[string]any{"component": "1.5.0"}, "red"},
		{"missing", nil, "yellow"},
		{"unparseable", map[string]any{"component": "unknown"}, "yellow"},
	}
	for _, tc := range cases {
		body := healthyState()
		if tc.version == nil {
			delete(body, "version")
		} else {
			body["version"] = tc.version
		}
		m := newTestMonitor(&fakeSup{supervisor.ForwardResult{StatusCode: 200, Body: body}}, &fakeRelay{})
		snap := m.RefreshOnce()
		if f := factByID(snap, "version_sync"); f["state"] != tc.want {
			t.Fatalf("%s: version_sync %v want %s", tc.name, f, tc.want)
		}
	}
}

func TestUploaderStates(t *testing.T) {
	cases := []struct {
		name    string
		backlog any
		want    string
	}{
		{"not reported", nil, "unknown"},
		{"empty", map[string]any{"count": float64(0)}, "green"},
		{"queued recent", map[string]any{"count": float64(2), "oldest_pending_seconds": float64(60)}, "yellow"},
		{"queued stuck", map[string]any{"count": float64(2), "oldest_pending_seconds": float64(4 * 3600)}, "red"},
		// healthy=false overrides an empty queue: a crash-looping uploader that
		// never gets far enough to publish anything also reports count=0, so
		// queue depth alone can't tell it apart from a genuinely idle uploader.
		{"crash-looping (empty queue, no heartbeat)", map[string]any{"count": float64(0), "healthy": false}, "red"},
		{"healthy with queue", map[string]any{"count": float64(2), "oldest_pending_seconds": float64(60), "healthy": true}, "yellow"},
		// healthy omitted (nil in Python == not present in the JSON map) is NOT
		// treated as unhealthy: a freshly-started, working uploader also has no
		// heartbeat yet until its first ~30s publish.
		{"healthy unreported, empty queue", map[string]any{"count": float64(0)}, "green"},
	}
	for _, tc := range cases {
		body := healthyState()
		if tc.backlog == nil {
			delete(body, "upload_backlog")
		} else {
			body["upload_backlog"] = tc.backlog
		}
		m := newTestMonitor(&fakeSup{supervisor.ForwardResult{StatusCode: 200, Body: body}}, &fakeRelay{})
		snap := m.RefreshOnce()
		if f := factByID(snap, "uploader"); f["state"] != tc.want {
			t.Fatalf("%s: uploader %v want %s", tc.name, f, tc.want)
		}
	}
}

func TestCameraUnreportedIsUnknownAndGateProceeds(t *testing.T) {
	// video has never published arm/video/state/camera (up: null / absent):
	// the fact is unknown, NOT red — and unknown must not fast-reject /whep
	// (the activation timeout arbitrates ambiguity).
	for _, body := range []map[string]any{
		{"vision": map[string]any{"healthy": true}, "version": map[string]any{"component": "1.4.0"}},
		{"vision": map[string]any{"healthy": true}, "camera": map[string]any{"up": nil},
			"version": map[string]any{"component": "1.4.0"}},
	} {
		m := newTestMonitor(&fakeSup{supervisor.ForwardResult{StatusCode: 200, Body: body}}, &fakeRelay{})
		snap := m.RefreshOnce()
		if f := factByID(snap, "camera"); f["state"] != "unknown" {
			t.Fatalf("camera %v want unknown", f)
		}
		if ok, reason := m.GateLiveView(); !ok {
			t.Fatalf("gate must proceed on unreported camera, rejected: %q", reason)
		}
	}
}

func TestCameraAndVideoRed(t *testing.T) {
	body := healthyState()
	body["camera"] = map[string]any{"up": false}
	body["vision"] = map[string]any{"healthy": false}
	m := newTestMonitor(&fakeSup{supervisor.ForwardResult{StatusCode: 200, Body: body}}, &fakeRelay{})
	snap := m.RefreshOnce()
	if f := factByID(snap, "camera"); f["state"] != "red" {
		t.Fatalf("camera %v", f)
	}
	if f := factByID(snap, "video"); f["state"] != "red" {
		t.Fatalf("video %v", f)
	}
	if snap["light"] != "red" {
		t.Fatalf("light %v", snap["light"])
	}
}

func TestStaleTransitionsYellowThenRed(t *testing.T) {
	sup := &fakeSup{supervisor.ForwardResult{StatusCode: 200, Body: healthyState()}}
	m := newTestMonitor(sup, &fakeRelay{})
	base := time.Now()
	m.Now = func() time.Time { return base }
	m.RefreshOnce()

	// A later refresh that fails: success age inside the stale window -> yellow.
	sup.result = supervisor.ForwardResult{StatusCode: 502, Body: map[string]any{"error": "x"}}
	m.Now = func() time.Time { return base.Add(30 * time.Second) }
	snap := m.RefreshOnce()
	if f := factByID(snap, "pi_control"); f["state"] != "yellow" {
		t.Fatalf("pi_control %v want yellow", f)
	}

	// Past the stale floor -> red.
	m.Now = func() time.Time { return base.Add(120 * time.Second) }
	snap = m.RefreshOnce()
	if f := factByID(snap, "pi_control"); f["state"] != "red" {
		t.Fatalf("pi_control %v want red", f)
	}
}

func TestGateLiveView(t *testing.T) {
	// No snapshot yet -> proceed (activation timeout arbitrates).
	m := newTestMonitor(&fakeSup{}, &fakeRelay{})
	if ok, _ := m.GateLiveView(); !ok {
		t.Fatal("no-snapshot should proceed")
	}

	// Healthy -> proceed.
	m = newTestMonitor(&fakeSup{supervisor.ForwardResult{StatusCode: 200, Body: healthyState()}}, &fakeRelay{})
	m.RefreshOnce()
	if ok, _ := m.GateLiveView(); !ok {
		t.Fatal("healthy should proceed")
	}

	// Pi unreachable -> fast reject.
	m = newTestMonitor(&fakeSup{supervisor.ForwardResult{StatusCode: 502, Body: map[string]any{}}}, &fakeRelay{})
	m.RefreshOnce()
	if ok, reason := m.GateLiveView(); ok || reason != "Pi unreachable" {
		t.Fatalf("gate: %v %q", ok, reason)
	}

	// Camera down with fresh pi_control -> differentiated reject.
	body := healthyState()
	body["camera"] = map[string]any{"up": false}
	m = newTestMonitor(&fakeSup{supervisor.ForwardResult{StatusCode: 200, Body: body}}, &fakeRelay{})
	m.RefreshOnce()
	if ok, reason := m.GateLiveView(); ok || reason != "camera not detected" {
		t.Fatalf("gate: %v %q", ok, reason)
	}

	// Video down -> differentiated reject.
	body = healthyState()
	body["vision"] = map[string]any{"healthy": false}
	m = newTestMonitor(&fakeSup{supervisor.ForwardResult{StatusCode: 200, Body: body}}, &fakeRelay{})
	m.RefreshOnce()
	if ok, reason := m.GateLiveView(); ok || reason != "video component down" {
		t.Fatalf("gate: %v %q", ok, reason)
	}

	// Stale (yellow) pi_control -> do NOT fast-reject even with camera red.
	sup := &fakeSup{supervisor.ForwardResult{StatusCode: 200, Body: body}}
	m = newTestMonitor(sup, &fakeRelay{})
	base := time.Now()
	m.Now = func() time.Time { return base }
	m.RefreshOnce()
	sup.result = supervisor.ForwardResult{StatusCode: 502, Body: map[string]any{}}
	m.Now = func() time.Time { return base.Add(30 * time.Second) }
	m.RefreshOnce()
	if ok, _ := m.GateLiveView(); !ok {
		t.Fatal("yellow pi_control must fall through to the activation timeout")
	}
}
