// Prometheus metrics for the relay (architecture §2.6: "relay state — ingest
// up, viewer count, reconnects" — the alerting surface the log-only prototype
// lacked). Registered on the default registry; scraped from /metrics on the
// public listener.
package relay

import (
	"github.com/prometheus/client_golang/prometheus"
)

type Metrics struct {
	ingestLive     prometheus.Gauge
	viewers        prometheus.Gauge
	ingestBytes    prometheus.Counter
	ingestPackets  prometheus.Counter
	ingestSessions prometheus.Counter
	viewerSessions prometheus.Counter
	activations    *prometheus.CounterVec
}

// NewMetrics builds and registers the relay metrics on reg (use
// prometheus.DefaultRegisterer in production, a private registry in tests).
func NewMetrics(reg prometheus.Registerer) *Metrics {
	m := &Metrics{
		ingestLive: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "fessel_relay_ingest_live",
			Help: "1 while a WHIP ingest session is feeding the shared track.",
		}),
		viewers: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "fessel_relay_viewers",
			Help: "Number of connected WHEP viewers.",
		}),
		ingestBytes: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "fessel_relay_ingest_bytes_total",
			Help: "RTP payload bytes received from the Pi ingest.",
		}),
		ingestPackets: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "fessel_relay_ingest_packets_total",
			Help: "RTP packets received from the Pi ingest.",
		}),
		ingestSessions: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "fessel_relay_ingest_sessions_total",
			Help: "WHIP ingest sessions accepted (Pi reconnects show up here).",
		}),
		viewerSessions: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "fessel_relay_viewer_sessions_total",
			Help: "WHEP viewer sessions answered.",
		}),
		activations: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "fessel_relay_activations_total",
			Help: "Live activation attempts by outcome (activated, timeout, activate_failed, gate_rejected).",
		}, []string{"outcome"}),
	}
	reg.MustRegister(m.ingestLive, m.viewers, m.ingestBytes, m.ingestPackets,
		m.ingestSessions, m.viewerSessions, m.activations)
	return m
}

// NopMetrics returns an unregistered instance (tests / disabled relay).
func NopMetrics() *Metrics {
	return NewMetrics(prometheus.NewRegistry())
}

func (m *Metrics) SetIngestLive(live bool) {
	if live {
		m.ingestLive.Set(1)
	} else {
		m.ingestLive.Set(0)
	}
}
func (m *Metrics) SetViewers(n int) { m.viewers.Set(float64(n)) }
func (m *Metrics) IngestRTP(n int) {
	m.ingestPackets.Inc()
	m.ingestBytes.Add(float64(n))
}
func (m *Metrics) IngestSessionStarted()      { m.ingestSessions.Inc() }
func (m *Metrics) ViewerSessionStarted()      { m.viewerSessions.Inc() }
func (m *Metrics) ActivationOutcome(o string) { m.activations.WithLabelValues(o).Inc() }
