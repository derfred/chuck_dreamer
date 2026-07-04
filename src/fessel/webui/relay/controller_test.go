package relay

import (
	"context"
	"sync"
	"testing"
	"time"
)

type fakeLifecycle struct {
	mu          sync.Mutex
	activates   int
	deactivates int
	live        bool
	comesUp     bool
}

func (f *fakeLifecycle) activate() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.activates++
	return nil
}

func (f *fakeLifecycle) deactivate() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.deactivates++
	return nil
}

func (f *fakeLifecycle) ingestLive() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.live
}

func (f *fakeLifecycle) waitIngest(ctx context.Context) bool {
	f.mu.Lock()
	up := f.comesUp
	f.mu.Unlock()
	if up {
		return true
	}
	<-ctx.Done()
	return false
}

func (f *fakeLifecycle) counts() (int, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.activates, f.deactivates
}

func newController(f *fakeLifecycle, gateOK bool, reason string) *Controller {
	return &Controller{
		Gate:              func() (bool, string) { return gateOK, reason },
		Activate:          f.activate,
		Deactivate:        f.deactivate,
		IngestLive:        f.ingestLive,
		WaitIngest:        f.waitIngest,
		ActivationTimeout: 30 * time.Millisecond,
		IdleTimeout:       25 * time.Millisecond,
	}
}

func TestControllerGateReject(t *testing.T) {
	f := &fakeLifecycle{}
	c := newController(f, false, "Pi unreachable")
	err := c.ViewerJoining(context.Background())
	gate, ok := err.(*GateRejectedError)
	if !ok || gate.Reason != "Pi unreachable" {
		t.Fatalf("err: %v", err)
	}
	if a, _ := f.counts(); a != 0 {
		t.Fatal("gate reject must not activate")
	}
}

func TestControllerSkipsActivationWhenLive(t *testing.T) {
	f := &fakeLifecycle{live: true}
	c := newController(f, true, "")
	if err := c.ViewerJoining(context.Background()); err != nil {
		t.Fatal(err)
	}
	if a, _ := f.counts(); a != 0 {
		t.Fatalf("activated: %d", a)
	}
}

func TestControllerActivatesFirstViewer(t *testing.T) {
	f := &fakeLifecycle{comesUp: true}
	c := newController(f, true, "")
	if err := c.ViewerJoining(context.Background()); err != nil {
		t.Fatal(err)
	}
	if a, _ := f.counts(); a != 1 {
		t.Fatalf("activated: %d", a)
	}
}

func TestControllerActivationTimeout(t *testing.T) {
	f := &fakeLifecycle{comesUp: false}
	c := newController(f, true, "")
	start := time.Now()
	err := c.ViewerJoining(context.Background())
	if err == nil || time.Since(start) < 25*time.Millisecond {
		t.Fatalf("err=%v elapsed=%s", err, time.Since(start))
	}
}

func TestControllerIdleTimeoutDeactivates(t *testing.T) {
	f := &fakeLifecycle{}
	c := newController(f, true, "")
	c.ViewerCountChanged(0)
	time.Sleep(60 * time.Millisecond)
	if _, d := f.counts(); d != 1 {
		t.Fatalf("deactivates: %d", d)
	}
}

func TestControllerViewerWithinIdleWindowCancelsDeactivate(t *testing.T) {
	f := &fakeLifecycle{live: true}
	c := newController(f, true, "")
	c.ViewerCountChanged(0)
	time.Sleep(5 * time.Millisecond)
	// A reconnecting viewer within the window must debounce the teardown
	// (§7.11: the Pi pipeline doesn't thrash).
	if err := c.ViewerJoining(context.Background()); err != nil {
		t.Fatal(err)
	}
	time.Sleep(60 * time.Millisecond)
	if _, d := f.counts(); d != 0 {
		t.Fatalf("deactivated despite rejoin: %d", d)
	}
}

func TestControllerNonZeroCountCancelsIdleTimer(t *testing.T) {
	f := &fakeLifecycle{}
	c := newController(f, true, "")
	c.ViewerCountChanged(0)
	time.Sleep(5 * time.Millisecond)
	c.ViewerCountChanged(1)
	time.Sleep(60 * time.Millisecond)
	if _, d := f.counts(); d != 0 {
		t.Fatalf("deactivated despite viewer: %d", d)
	}
}

func TestControllerRepeatedZeroDoesNotStackTimers(t *testing.T) {
	f := &fakeLifecycle{}
	c := newController(f, true, "")
	c.ViewerCountChanged(0)
	c.ViewerCountChanged(0)
	c.ViewerCountChanged(0)
	time.Sleep(60 * time.Millisecond)
	if _, d := f.counts(); d != 1 {
		t.Fatalf("deactivates: %d (timers stacked?)", d)
	}
}
