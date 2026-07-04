// Package health maintains the composite Pi-connection health picture
// (health-check spec §3), refreshed on a timer; GET /api/health/pi serves the
// latest snapshot. The corner light in the frontend is a *rollup* — the worst
// state across a set of independently observable facts.
//
// The five facts (root-cause-first display order):
//
//	pi_control    reachability of supervisor over the cluster->Pi control path
//	relay         the in-process live relay (nothing to reach over the
//	              network — the fact reports ingest/viewer state)
//	video         video process liveness, read from /state.vision.healthy
//	camera        camera presence, read from /state.camera.up
//	version_sync  cluster vs Pi MAJOR.MINOR must-match (spec §3.4)
//
// Masking (spec §1.5): facts observed *through* supervisor (video, camera,
// version_sync) become `unknown` when pi_control is red — painting them red
// would manufacture a cascade obscuring the single root cause. relay is
// in-process and is never masked.
//
// The snapshot is also the relay's fast-reject gate on /whep (architecture
// §4.3): GateLiveView applies the confidently-bad table; ambiguous states fall
// through to the activation timeout, which stays the authoritative check.
package health

import (
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/derfred/fessel/webui/internal/supervisor"
	"github.com/derfred/fessel/webui/internal/version"
)

type State = string // "green" | "yellow" | "red" | "unknown"

// FactOrder is the fixed display order — reachability first so a red
// pi_control is seen first.
var FactOrder = []string{"pi_control", "relay", "video", "camera", "version_sync"}

var factLabels = map[string]string{
	"pi_control":   "Pi control",
	"relay":        "live relay",
	"video":        "video process",
	"camera":       "camera",
	"version_sync": "Component versions",
}

type Fact struct {
	ID         string
	State      State
	Detail     string
	ObservedAt time.Time
}

func (f Fact) toDict() map[string]any {
	return map[string]any{
		"id":          f.ID,
		"label":       factLabels[f.ID],
		"state":       f.State,
		"detail":      f.Detail,
		"observed_at": f.ObservedAt.UTC().Format(time.RFC3339Nano),
	}
}

// rollup is the light's colour: worst state across all facts, after masking.
// red > yellow > unknown > green: `unknown` outranks green because a green you
// can't fully determine is not a confident green.
func rollup(facts []Fact) State {
	worst := "green"
	rank := map[State]int{"green": 0, "unknown": 1, "yellow": 2, "red": 3}
	for _, f := range facts {
		if rank[f.State] > rank[worst] {
			worst = f.State
		}
	}
	return worst
}

func ageStr(d time.Duration) string {
	s := int(d.Seconds())
	switch {
	case s < 60:
		return fmt.Sprintf("%ds", s)
	case s < 3600:
		return fmt.Sprintf("%dm", s/60)
	default:
		return fmt.Sprintf("%dh", s/3600)
	}
}

// RelayStatus is what the in-process relay exposes to health (implemented by
// relay.Relay; nil when the relay is disabled).
type RelayStatus interface {
	IngestLive() bool
	ViewerCount() int
}

type Config struct {
	RefreshInterval time.Duration
	FreshThreshold  time.Duration
	StaleThreshold  time.Duration
}

// SupervisorAPI is the slice of the supervisor client health needs (a seam
// for tests).
type SupervisorAPI interface {
	Get(path string) supervisor.ForwardResult
}

type Monitor struct {
	Supervisor     SupervisorAPI
	Relay          RelayStatus
	Config         Config
	ClusterVersion string
	Now            func() time.Time // test seam; defaults to time.Now

	mu            sync.Mutex
	lastSuccessAt *time.Time
	lastAttemptAt *time.Time
	snapshot      map[string]any
	facts         []Fact
	stop          chan struct{}
	stopped       sync.WaitGroup
}

func NewMonitor(sup SupervisorAPI, relay RelayStatus, cfg Config) *Monitor {
	return &Monitor{
		Supervisor:     sup,
		Relay:          relay,
		Config:         cfg,
		ClusterVersion: version.Version(),
		Now:            time.Now,
	}
}

// --- fact computation ---------------------------------------------------------

func (m *Monitor) piControlFact(ok bool, now time.Time) Fact {
	if m.lastSuccessAt == nil {
		// No successful contact this session. red if we've tried and failed,
		// unknown only before the first attempt completes.
		if m.lastAttemptAt == nil {
			return Fact{"pi_control", "unknown", "checking…", now}
		}
		return Fact{"pi_control", "red", "unreachable — no contact this session", now}
	}
	// The state is a function of the AGE of the last success against the
	// fresh/stale thresholds (spec §3.6: FRESH_S is the green ceiling, STALE_S
	// the red floor, both defined on the age of last success). A single failed
	// probe with a recent success degrades to yellow, not straight to red —
	// red is reserved for a sustained outage, which is what the /whep fast-
	// reject gate keys on. (The FastAPI implementation went red on any failed
	// probe, making its yellow branch unreachable; this port follows the
	// documented threshold semantics instead.)
	age := now.Sub(*m.lastSuccessAt)
	switch {
	case age < m.Config.FreshThreshold:
		if ok {
			return Fact{"pi_control", "green", fmt.Sprintf("reachable — %s ago", ageStr(age)), now}
		}
		return Fact{"pi_control", "yellow", fmt.Sprintf("probe failed — last contact %s ago", ageStr(age)), now}
	case age < m.Config.StaleThreshold:
		return Fact{"pi_control", "yellow", fmt.Sprintf("reachable but stale — %s ago", ageStr(age)), now}
	default:
		return Fact{"pi_control", "red", fmt.Sprintf("unreachable — last contact %s ago", ageStr(age)), now}
	}
}

func (m *Monitor) relayFact(now time.Time) Fact {
	if m.Relay == nil {
		return Fact{"relay", "unknown", "relay not running", now}
	}
	if m.Relay.IngestLive() {
		return Fact{"relay", "green", fmt.Sprintf("ingest live, %d viewer(s)", m.Relay.ViewerCount()), now}
	}
	return Fact{"relay", "green", "idle (no ingest)", now}
}

func videoFact(body map[string]any, now time.Time) Fact {
	if vision, ok := body["vision"].(map[string]any); ok {
		if healthy, ok := vision["healthy"].(bool); ok && healthy {
			return Fact{"video", "green", "running", now}
		}
	}
	return Fact{"video", "red", "not running — heartbeat stale", now}
}

func cameraFact(body map[string]any, now time.Time) Fact {
	if camera, ok := body["camera"].(map[string]any); ok {
		if up, ok := camera["up"].(bool); ok && up {
			return Fact{"camera", "green", "present", now}
		}
	}
	return Fact{"camera", "red", "not detected", now}
}

func (m *Monitor) versionSyncFact(body map[string]any, now time.Time) Fact {
	versionObj, _ := body["version"].(map[string]any)
	component, _ := versionObj["component"].(string)
	if component == "" {
		// /state responded but carries no version -> an old Pi that predates
		// the field. A distinct, honest "can't determine" yellow, not a
		// mismatch red.
		return Fact{"version_sync", "yellow", "Pi did not report a version", now}
	}
	piMaj, piMin, piOK := version.Minor(component)
	clMaj, clMin, clOK := version.Minor(m.ClusterVersion)
	if !piOK || !clOK {
		return Fact{"version_sync", "yellow",
			fmt.Sprintf("can't compare versions (pi %s / cluster %s)", component, m.ClusterVersion), now}
	}
	if piMaj == clMaj && piMin == clMin {
		return Fact{"version_sync", "green",
			fmt.Sprintf("in sync (pi %s / cluster %s)", component, m.ClusterVersion), now}
	}
	return Fact{"version_sync", "red",
		fmt.Sprintf("version mismatch — pi %s, cluster %s; features may fail", component, m.ClusterVersion), now}
}

// compute builds all five facts, applies masking + rollup. Caller holds m.mu.
func (m *Monitor) compute(stateOK bool, stateBody map[string]any) ([]Fact, map[string]any) {
	now := m.Now()
	piControl := m.piControlFact(stateOK, now)
	relay := m.relayFact(now)

	var video, camera, versionSync Fact
	if piControl.State == "red" {
		// Masking (spec §1.5): everything observed *through* supervisor is
		// unknown, not red. The relay is in-process and stays as observed.
		video = Fact{"video", "unknown", "unknown — Pi unreachable", now}
		camera = Fact{"camera", "unknown", "unknown — Pi unreachable", now}
		versionSync = Fact{"version_sync", "unknown", "unknown — can't read Pi version", now}
	} else {
		video = videoFact(stateBody, now)
		camera = cameraFact(stateBody, now)
		versionSync = m.versionSyncFact(stateBody, now)
	}

	facts := []Fact{piControl, relay, video, camera, versionSync}
	dicts := make([]any, 0, len(facts))
	for _, f := range facts {
		dicts = append(dicts, f.toDict())
	}
	return facts, map[string]any{
		"light":        rollup(facts),
		"generated_at": now.UTC().Format(time.RFC3339Nano),
		"facts":        dicts,
	}
}

// --- refresh --------------------------------------------------------------------

// RefreshOnce runs one refresh cycle synchronously and stores the snapshot.
// Reuses supervisor /state (the same call /api/state makes).
func (m *Monitor) RefreshOnce() map[string]any {
	attempt := m.Now()
	result := m.Supervisor.Get("/state")
	stateBody, isMap := result.Body.(map[string]any)
	stateOK := result.StatusCode >= 200 && result.StatusCode < 300 && isMap
	if !isMap {
		stateBody = map[string]any{}
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.lastAttemptAt = &attempt
	if stateOK {
		m.lastSuccessAt = &attempt
	}
	m.facts, m.snapshot = m.compute(stateOK, stateBody)
	return m.snapshot
}

// Snapshot returns the latest snapshot, or a before-first-refresh all-unknown
// picture.
func (m *Monitor) Snapshot() map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.snapshot != nil {
		return m.snapshot
	}
	now := m.Now()
	dicts := make([]any, 0, len(FactOrder))
	for _, fid := range FactOrder {
		dicts = append(dicts, Fact{fid, "unknown", "checking…", now}.toDict())
	}
	return map[string]any{
		"light":        "unknown",
		"generated_at": now.UTC().Format(time.RFC3339Nano),
		"facts":        dicts,
	}
}

// GateLiveView is the /whep fast-reject gate (architecture §4.3):
//
//	pi_control red                          -> reject "Pi unreachable"
//	pi_control green && camera red          -> reject "camera not detected"
//	pi_control green && video red           -> reject "video component down"
//	anything ambiguous (yellow/stale/unknown) -> proceed; the activation
//	  timeout arbitrates.
//
// The gate is an optimisation for *sustained, confident* outages only, never
// load-bearing for correctness.
func (m *Monitor) GateLiveView() (ok bool, reason string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	byID := map[string]Fact{}
	for _, f := range m.facts {
		byID[f.ID] = f
	}
	pi, have := byID["pi_control"]
	if !have {
		return true, "" // no snapshot yet -> proceed, activation arbitrates
	}
	if pi.State == "red" {
		return false, "Pi unreachable"
	}
	if pi.State == "green" {
		if byID["camera"].State == "red" {
			return false, "camera not detected"
		}
		if byID["video"].State == "red" {
			return false, "video component down"
		}
	}
	return true, ""
}

// Start launches the refresh loop; Stop terminates it.
func (m *Monitor) Start() {
	m.mu.Lock()
	if m.stop != nil {
		m.mu.Unlock()
		return
	}
	stop := make(chan struct{})
	m.stop = stop
	m.mu.Unlock()

	m.stopped.Add(1)
	go func() {
		defer m.stopped.Done()
		for {
			func() {
				defer func() {
					// A wedged refresh must not kill the loop; keep serving the
					// last snapshot.
					if r := recover(); r != nil {
						slog.Error("health refresh cycle panicked", "err", r)
					}
				}()
				m.RefreshOnce()
			}()
			select {
			case <-stop:
				return
			case <-time.After(m.Config.RefreshInterval):
			}
		}
	}()
}

func (m *Monitor) Stop() {
	m.mu.Lock()
	stop := m.stop
	m.stop = nil
	m.mu.Unlock()
	if stop != nil {
		close(stop)
		m.stopped.Wait()
	}
}
