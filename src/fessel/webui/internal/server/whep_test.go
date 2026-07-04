package server

import (
	"context"
	"testing"
	"time"

	"github.com/derfred/fessel/webui/relay"
)

// fakeViewerRelay stands in for the Pion relay on the WHEP route.
type fakeViewerRelay struct {
	closed []int64
}

func (f *fakeViewerRelay) HandleViewerOffer(offer, peer string) (string, int64, error) {
	return "v=0 answer", 42, nil
}
func (f *fakeViewerRelay) CloseViewer(id int64) { f.closed = append(f.closed, id) }

func newLivePublic(gateOK bool, gateReason string, live bool) (*Public, *fakeViewerRelay, *countingActivator) {
	rly := &fakeViewerRelay{}
	act := &countingActivator{}
	ctrl := &relay.Controller{
		Gate:       func() (bool, string) { return gateOK, gateReason },
		Activate:   act.activate,
		Deactivate: act.deactivate,
		IngestLive: func() bool { return live },
		WaitIngest: func(ctx context.Context) bool {
			act.waited = true
			return act.ingestComesUp
		},
		ActivationTimeout: 50 * time.Millisecond,
		IdleTimeout:       20 * time.Millisecond,
	}
	p := newPublic(newFakeSupervisor(), nil)
	p.Relay = rly
	p.Controller = ctrl
	return p, rly, act
}

type countingActivator struct {
	activated     int
	deactivated   int
	waited        bool
	ingestComesUp bool
	activateErr   error
}

func (a *countingActivator) activate() error {
	a.activated++
	return a.activateErr
}

func (a *countingActivator) deactivate() error {
	a.deactivated++
	return nil
}

func TestWhepRequiresIdentity(t *testing.T) {
	p, _, _ := newLivePublic(true, "", true)
	w := do(t, p.Handler(), "POST", "/whep", "v=0 offer", false)
	if w.Code != 401 {
		t.Fatalf("%d", w.Code)
	}
}

func TestWhepAnswersWhenIngestLive(t *testing.T) {
	p, _, act := newLivePublic(true, "", true)
	w := do(t, p.Handler(), "POST", "/whep", "v=0 offer", true)
	if w.Code != 201 || w.Body.String() != "v=0 answer" {
		t.Fatalf("%d %q", w.Code, w.Body.String())
	}
	if w.Header().Get("Location") != "/whep/42" {
		t.Fatalf("location %q", w.Header().Get("Location"))
	}
	if act.activated != 0 {
		t.Fatalf("activated with live ingest: %d", act.activated)
	}
}

func TestWhepGateFastRejects(t *testing.T) {
	p, _, act := newLivePublic(false, "camera not detected", false)
	w := do(t, p.Handler(), "POST", "/whep", "v=0 offer", true)
	if w.Code != 503 || decode(t, w)["reason"] != "camera not detected" {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	if act.activated != 0 {
		t.Fatal("gate reject must not activate")
	}
}

func TestWhepFirstViewerActivatesAndBlocks(t *testing.T) {
	p, _, act := newLivePublic(true, "", false)
	act.ingestComesUp = true
	w := do(t, p.Handler(), "POST", "/whep", "v=0 offer", true)
	if w.Code != 201 {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	if act.activated != 1 || !act.waited {
		t.Fatalf("activated=%d waited=%v", act.activated, act.waited)
	}
}

func TestWhepActivationTimeoutIs504(t *testing.T) {
	p, _, act := newLivePublic(true, "", false)
	act.ingestComesUp = false // WaitIngest reports timeout
	w := do(t, p.Handler(), "POST", "/whep", "v=0 offer", true)
	if w.Code != 504 || decode(t, w)["error"] != "live_timeout" {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
}

func TestWhepActivateFailureIs502(t *testing.T) {
	p, _, act := newLivePublic(true, "", false)
	act.activateErr = context.DeadlineExceeded
	w := do(t, p.Handler(), "POST", "/whep", "v=0 offer", true)
	if w.Code != 502 || decode(t, w)["error"] != "live_activation_failed" {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
}

func TestWhepDeleteClosesViewer(t *testing.T) {
	p, rly, _ := newLivePublic(true, "", true)
	w := do(t, p.Handler(), "DELETE", "/whep/42", "", true)
	if w.Code != 200 || len(rly.closed) != 1 || rly.closed[0] != 42 {
		t.Fatalf("%d %v", w.Code, rly.closed)
	}
}
