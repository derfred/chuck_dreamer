// The activation controller owns the on-demand live lifecycle, in-process
// (architecture §2.3/§4.2):
//
//   - First viewer on /whep with no live ingest -> call
//     supervisor:/control/live/activate; block until the WHIP ingest is up or
//     live_activation_timeout expires.
//   - Viewer count reaches zero -> start live_idle_timeout (debounces a
//     reconnecting viewer so the Pi pipeline doesn't thrash); on expiry call
//     supervisor:/control/live/deactivate.
//   - A cached Pi-health gate short-circuits /whep on confidently-bad health
//     with a differentiated error; ambiguous health falls through and the
//     activation timeout stays the authoritative check.
//
// The controller is pure lifecycle logic over small function seams so it is
// unit-testable without Pion or a supervisor.
package relay

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// GateRejectedError is a health-gate fast reject; the handler maps it to a
// differentiated 503.
type GateRejectedError struct{ Reason string }

func (e *GateRejectedError) Error() string { return "live view rejected: " + e.Reason }

// ErrActivationTimeout: the WHIP ingest did not establish within
// live_activation_timeout; the handler maps it to 504 "timed out starting
// stream".
var ErrActivationTimeout = errors.New("timed out starting stream")

// ActivationFailedError: supervisor rejected or couldn't be reached for the
// activate call.
type ActivationFailedError struct{ Detail string }

func (e *ActivationFailedError) Error() string { return "live activation failed: " + e.Detail }

type Controller struct {
	// Gate is the cached-health fast-reject (health.Monitor.GateLiveView).
	Gate func() (ok bool, reason string)
	// Activate/Deactivate call supervisor's /control/live/* endpoints; the
	// error carries the supervisor diagnostic.
	Activate   func() error
	Deactivate func() error
	// IngestLive/WaitIngest observe the relay's ingest plane.
	IngestLive func() bool
	WaitIngest func(ctx context.Context) bool

	ActivationTimeout time.Duration
	IdleTimeout       time.Duration

	Metrics *Metrics

	mu        sync.Mutex
	idleTimer *time.Timer
}

// ViewerJoining runs the pre-answer sequence for a WHEP request: health gate,
// idle-timer cancel, first-viewer activation, block-until-ingest. Returns nil
// when the ingest is live and the viewer may be answered.
func (c *Controller) ViewerJoining(ctx context.Context) error {
	if c.Gate != nil {
		if ok, reason := c.Gate(); !ok {
			c.metrics().ActivationOutcome("gate_rejected")
			return &GateRejectedError{Reason: reason}
		}
	}

	// A viewer arriving within the idle window must cancel the pending
	// deactivate (§7.11: a reconnecting viewer never thrashes the Pi pipeline).
	c.cancelIdleTimer()

	if c.IngestLive() {
		return nil
	}

	// First viewer with no live ingest: drive the Pi to attach its sender.
	slog.Info("live activation: first viewer, activating Pi sender")
	if err := c.Activate(); err != nil {
		c.metrics().ActivationOutcome("activate_failed")
		return &ActivationFailedError{Detail: err.Error()}
	}
	waitCtx, cancel := context.WithTimeout(ctx, c.ActivationTimeout)
	defer cancel()
	if !c.WaitIngest(waitCtx) {
		c.metrics().ActivationOutcome("timeout")
		return fmt.Errorf("%w after %s", ErrActivationTimeout, c.ActivationTimeout)
	}
	c.metrics().ActivationOutcome("activated")
	return nil
}

// ViewerCountChanged is the relay's viewer-count hook: when the count reaches
// zero, start the idle timer; on expiry, deactivate. A non-zero count cancels
// any pending timer (covers the join path too, belt-and-braces with
// ViewerJoining's cancel).
func (c *Controller) ViewerCountChanged(count int) {
	if count > 0 {
		c.cancelIdleTimer()
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.idleTimer != nil {
		return // already pending
	}
	slog.Info("live teardown: no viewers, starting idle timeout", "idle_timeout", c.IdleTimeout.String())
	c.idleTimer = time.AfterFunc(c.IdleTimeout, func() {
		c.mu.Lock()
		c.idleTimer = nil
		c.mu.Unlock()
		slog.Info("live teardown: idle timeout expired, deactivating Pi sender")
		if err := c.Deactivate(); err != nil {
			// Deactivate failure is not fatal: the Pi-side state machine also
			// recovers on the next activate, and a dangling sender only costs
			// uplink bandwidth until then.
			slog.Warn("live deactivate failed", "err", err)
		}
	})
}

func (c *Controller) cancelIdleTimer() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.idleTimer != nil {
		c.idleTimer.Stop()
		c.idleTimer = nil
		slog.Info("live teardown cancelled: viewer arrived within idle window")
	}
}

func (c *Controller) metrics() *Metrics {
	if c.Metrics == nil {
		c.Metrics = NopMetrics()
	}
	return c.Metrics
}
