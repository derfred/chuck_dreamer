// Package relay is the in-process WHIP->WHEP WebRTC relay (architecture §4.2).
//
// Ingest: the Pi's whipclientsink POSTs a WHIP offer to /whip/ingest on the
// tailnet-only listener; Pion terminates the session and, on OnTrack, copies
// incoming RTP into a single PERMANENT shared TrackLocalStaticRTP. Viewer: a
// browser POSTs a WHEP offer to /whep behind oauth2-proxy; the shared track is
// added to a per-viewer PeerConnection. Single active ingest, fan-out to N
// viewers.
//
// Two separate webrtc.APIs because the two legs have different reachability
// (§5.1): the ingest plane advertises the pod's tailnet address (the
// Tailscale sidecar's 100.x IP), the viewer plane advertises the node public
// IPs on the media NodePort.
package relay

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pion/ice/v4"
	"github.com/pion/interceptor"
	"github.com/pion/rtcp"
	"github.com/pion/webrtc/v4"
)

// h264Codec is registered on both ingest and viewer MediaEngines. The fmtp
// matches whipclientsink's default H.264 output (constrained-baseline,
// non-interleaved). See the whip-relay prototype notes on profile-level-id:
// in PULL mode the relay is the offerer and GStreamer's whepserversink is
// STRICT about profile matching — offer the profile the Pi actually encodes.
var h264Codec = webrtc.RTPCodecParameters{
	RTPCodecCapability: webrtc.RTPCodecCapability{
		MimeType:     webrtc.MimeTypeH264,
		ClockRate:    90000,
		SDPFmtpLine:  envOr("FESSEL_H264_FMTP", "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f"),
		RTCPFeedback: []webrtc.RTCPFeedback{{Type: "nack"}, {Type: "nack", Parameter: "pli"}, {Type: "goog-remb"}},
	},
	PayloadType: 102,
}

// stunConfig is shared by ingest and viewer.
var stunConfig = webrtc.Configuration{
	ICEServers: []webrtc.ICEServer{{URLs: []string{"stun:stun.l.google.com:19302"}}},
}

// ICEConfig captures how a plane advertises itself:
//   - viewer: the browser reaches WebRTC media via a NodePort on the node's
//     public IP. Pion must advertise those public IPs (NAT1To1) on the fixed
//     NodePort (UDPMuxPort), not the unroutable pod IP.
//   - ingest: the Pi reaches the relay over the tailnet; Pion advertises the
//     pod's tailnet address on a fixed port.
//
// A zero ICEConfig means "default gathering" (host candidates from all
// interfaces) — correct for localhost / integration-cluster (podip) use.
type ICEConfig struct {
	NAT1To1IPs []string
	UDPMuxPort int
}

type viewer struct {
	pc *webrtc.PeerConnection
}

type Relay struct {
	mu         sync.RWMutex
	ingestPC   *webrtc.PeerConnection
	ingestSSRC uint32
	live       bool
	liveWait   []chan struct{}

	viewers map[int64]*viewer

	// track is PERMANENT: created once at startup and never swapped. Every
	// ingest OnTrack writes its RTP into this same track, so viewers stay bound
	// across Pi reconnects — a viewer's PeerConnection survives the Pi dropping
	// and coming back. A fresh track per ingest would silently orphan every
	// already-connected viewer.
	track *webrtc.TrackLocalStaticRTP

	sessions atomic.Int64

	ingestAPI *webrtc.API
	viewerAPI *webrtc.API

	metrics *Metrics

	// onViewerCount is invoked (outside the lock) whenever the viewer count
	// changes; the activation controller hooks the idle-timeout logic here.
	onViewerCount atomic.Pointer[func(count int)]
}

func New(ingestICE, viewerICE ICEConfig, metrics *Metrics) (*Relay, error) {
	ingestAPI, err := buildAPI(ingestICE)
	if err != nil {
		return nil, fmt.Errorf("ingest api: %w", err)
	}
	viewerAPI, err := buildAPI(viewerICE)
	if err != nil {
		return nil, fmt.Errorf("viewer api: %w", err)
	}
	track, err := webrtc.NewTrackLocalStaticRTP(h264Codec.RTPCodecCapability, "video", "fessel")
	if err != nil {
		return nil, fmt.Errorf("shared track: %w", err)
	}
	if metrics == nil {
		metrics = NopMetrics()
	}
	return &Relay{
		ingestAPI: ingestAPI,
		viewerAPI: viewerAPI,
		track:     track,
		viewers:   map[int64]*viewer{},
		metrics:   metrics,
	}, nil
}

// SetViewerCountHook registers the callback the controller uses for the idle
// timeout. Must be set before serving.
func (r *Relay) SetViewerCountHook(fn func(count int)) { r.onViewerCount.Store(&fn) }

func buildAPI(c ICEConfig) (*webrtc.API, error) {
	var mux ice.UDPMux
	if c.UDPMuxPort != 0 {
		conn, err := net.ListenUDP("udp", &net.UDPAddr{Port: c.UDPMuxPort})
		if err != nil {
			return nil, fmt.Errorf("bind udp mux :%d: %w", c.UDPMuxPort, err)
		}
		mux = ice.NewUDPMuxDefault(ice.UDPMuxParams{UDPConn: conn})
		slog.Info("ICE UDP mux bound", "port", c.UDPMuxPort, "nat1to1", c.NAT1To1IPs)
	}
	var se webrtc.SettingEngine
	if len(c.NAT1To1IPs) > 0 {
		// Advertise every candidate IP (e.g. all worker public IPs behind the
		// NodePort). The browser tries all; whichever node the pod runs on is
		// the one that answers.
		se.SetNAT1To1IPs(c.NAT1To1IPs, webrtc.ICECandidateTypeHost)
	}
	if mux != nil {
		se.SetICEUDPMux(mux)
	}
	return newAPI(se)
}

// newAPI builds a webrtc.API with H.264 (+ Opus, see below) registered and the
// default interceptors (NACK / RTCP reports / TWCC — the TWCC receiver
// feedback is what the Pi's rtpgccbwe needs to rate-adapt on a lossy link).
func newAPI(se webrtc.SettingEngine) (*webrtc.API, error) {
	m := &webrtc.MediaEngine{}
	if err := m.RegisterCodec(h264Codec, webrtc.RTPCodecTypeVideo); err != nil {
		return nil, err
	}
	// Register Opus even though no audio is relayed yet. Spec-compliant WHEP
	// clients offer an audio m-line; with no audio codec Pion answers it as a
	// port-0 rejected section with no ice-ufrag, which GStreamer's webrtcbin
	// refuses. Registering Opus makes Pion emit a well-formed (inactive) audio
	// m-line instead.
	if err := m.RegisterCodec(webrtc.RTPCodecParameters{
		RTPCodecCapability: webrtc.RTPCodecCapability{
			MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2,
			SDPFmtpLine: "minptime=10;useinbandfec=1",
		},
		PayloadType: 111,
	}, webrtc.RTPCodecTypeAudio); err != nil {
		return nil, err
	}
	i := &interceptor.Registry{}
	if err := webrtc.RegisterDefaultInterceptors(m, i); err != nil {
		return nil, err
	}
	return webrtc.NewAPI(webrtc.WithMediaEngine(m), webrtc.WithInterceptorRegistry(i)), nil
}

// --- observability ------------------------------------------------------------

func (r *Relay) IngestLive() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.live
}

func (r *Relay) ViewerCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.viewers)
}

// WaitIngest blocks until the ingest is live or ctx is done; reports whether
// the ingest became live. This is what lets the /whep handler block until the
// WHIP ingest is established or live_activation_timeout expires (§2.3).
func (r *Relay) WaitIngest(ctx context.Context) bool {
	r.mu.Lock()
	if r.live {
		r.mu.Unlock()
		return true
	}
	ch := make(chan struct{})
	r.liveWait = append(r.liveWait, ch)
	r.mu.Unlock()
	select {
	case <-ch:
		return true
	case <-ctx.Done():
		return false
	}
}

// setLive updates liveness and wakes WaitIngest callers on a rising edge.
// Caller must hold r.mu.
func (r *Relay) setLiveLocked(live bool) {
	r.live = live
	r.metrics.SetIngestLive(live)
	if live {
		for _, ch := range r.liveWait {
			close(ch)
		}
		r.liveWait = nil
	}
}

func (r *Relay) notifyViewerCount() {
	count := r.ViewerCount()
	r.metrics.SetViewers(count)
	if fn := r.onViewerCount.Load(); fn != nil {
		(*fn)(count)
	}
}

// --- ingest ---------------------------------------------------------------------

// HandleIngestOffer terminates an ingest WHIP session: it answers the SDP
// offer and wires OnTrack into the permanent shared track. Returns the SDP
// answer and a session id for the WHIP Location header.
func (r *Relay) HandleIngestOffer(offer, peer string) (answer string, id int64, err error) {
	id = r.sessions.Add(1)
	l := slog.With("dir", "ingest", "session", id, "peer", peer)
	l.Info("WHIP offer received", "bytes", len(offer))

	pc, err := r.ingestAPI.NewPeerConnection(stunConfig)
	if err != nil {
		return "", 0, fmt.Errorf("peerconnection: %w", err)
	}
	logCandidates(l, pc)

	// whipclientsink sends media; the relay only receives.
	if _, err := pc.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo,
		webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly}); err != nil {
		_ = pc.Close()
		return "", 0, fmt.Errorf("transceiver: %w", err)
	}

	r.attachIngest(pc, l)

	answer, err = setRemoteAndAnswer(pc, offer)
	if err != nil {
		_ = pc.Close()
		return "", 0, fmt.Errorf("negotiate: %w", err)
	}
	r.metrics.IngestSessionStarted()
	l.Info("WHIP answered")
	return answer, id, nil
}

// CloseIngest is the WHIP teardown (DELETE). Single-stream: closing the
// current ingest is enough.
func (r *Relay) CloseIngest() {
	r.mu.Lock()
	if r.ingestPC != nil {
		_ = r.ingestPC.Close()
		r.ingestPC = nil
		r.setLiveLocked(false)
	}
	r.mu.Unlock()
}

// attachIngest wires an ingest PeerConnection (from either the WHIP server or
// the WHEP-pull client) to the shared track: OnTrack copies RTP into the
// permanent track and logs a throughput heartbeat; the connection-state
// handler clears liveness on drop. It also registers pc as the current
// ingest, closing any prior one (single-stream).
func (r *Relay) attachIngest(pc *webrtc.PeerConnection, l *slog.Logger) {
	pc.OnTrack(func(remote *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) {
		l.Info("ingest track started", "codec", remote.Codec().MimeType,
			"ssrc", uint32(remote.SSRC()), "pt", remote.PayloadType())

		r.mu.Lock()
		r.ingestSSRC = uint32(remote.SSRC())
		r.setLiveLocked(true)
		r.mu.Unlock()

		// Drain RTCP on the receiver so interceptors keep working, and send a
		// periodic PLI as a keyframe safety net in addition to the per-viewer
		// PLI on join.
		go r.drainAndKeepAlive(pc, receiver, uint32(remote.SSRC()))

		// Copy incoming RTP into the PERMANENT shared track. The SSRC changes
		// each time the Pi reconnects; TrackLocalStaticRTP rewrites the SSRC
		// downstream per viewer, so viewers keep decoding across reconnects.
		// A 5s throughput line is the heartbeat that says media is actually
		// moving (vs connected-but-silent) — the key signal on a flaky uplink.
		var pkts, bytes uint64
		lastLog := time.Now()
		buf := make([]byte, 1500)
		for {
			n, _, readErr := remote.Read(buf)
			if readErr != nil {
				l.Info("ingest track ended", "err", readErr, "packets", pkts, "bytes", bytes)
				r.mu.Lock()
				if r.ingestPC == pc {
					r.setLiveLocked(false)
				}
				r.mu.Unlock()
				return
			}
			pkts++
			bytes += uint64(n)
			r.metrics.IngestRTP(n)
			if now := time.Now(); now.Sub(lastLog) >= 5*time.Second {
				secs := now.Sub(lastLog).Seconds()
				l.Info("ingest media", "kbps", int(float64(bytes)*8/1000/secs), "pps", int(float64(pkts)/secs))
				pkts, bytes, lastLog = 0, 0, now
			}
			if _, writeErr := r.track.Write(buf[:n]); writeErr != nil && writeErr != io.ErrClosedPipe {
				l.Error("shared track write", "err", writeErr)
				return
			}
		}
	})

	pc.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
		l.Info("ingest connection state", "state", s.String())
		if s == webrtc.PeerConnectionStateFailed || s == webrtc.PeerConnectionStateClosed {
			r.mu.Lock()
			if r.ingestPC == pc {
				r.ingestPC = nil
				r.setLiveLocked(false)
			}
			r.mu.Unlock()
		}
	})

	// Replace any prior ingest (single-stream assumption).
	r.mu.Lock()
	if r.ingestPC != nil {
		l.Warn("replacing existing ingest (single-stream)")
		_ = r.ingestPC.Close()
	}
	r.ingestPC = pc
	r.mu.Unlock()
}

// drainAndKeepAlive reads (and discards) RTCP from the ingest receiver and, as
// a keyframe safety net, periodically sends a PLI on the ingest PC.
func (r *Relay) drainAndKeepAlive(pc *webrtc.PeerConnection, receiver *webrtc.RTPReceiver, ssrc uint32) {
	go func() {
		ticker := time.NewTicker(3 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			if err := pc.WriteRTCP([]rtcp.Packet{&rtcp.PictureLossIndication{MediaSSRC: ssrc}}); err != nil {
				return // PC closed
			}
		}
	}()
	rtcpBuf := make([]byte, 1500)
	for {
		if _, _, err := receiver.Read(rtcpBuf); err != nil {
			return
		}
	}
}

// RequestKeyframe asks the current ingest to produce a fresh keyframe (called
// on viewer join so a late viewer doesn't wait for the next natural IDR; §4.2
// "PLI on viewer-join").
func (r *Relay) RequestKeyframe() {
	r.mu.RLock()
	pc, ssrc := r.ingestPC, r.ingestSSRC
	r.mu.RUnlock()
	if pc == nil {
		return
	}
	if err := pc.WriteRTCP([]rtcp.Packet{&rtcp.PictureLossIndication{MediaSSRC: ssrc}}); err != nil {
		slog.Debug("PLI send failed (ingest gone?)", "err", err)
	}
}

// --- viewers --------------------------------------------------------------------

// HandleViewerOffer serves a viewer: it adds the permanent shared track to a
// new per-viewer PeerConnection and answers the browser's offer. The caller
// (server) has already run the health gate + activation; by the time this is
// called the ingest is expected live, but a viewer connecting before ingest is
// harmless — the track is permanent.
func (r *Relay) HandleViewerOffer(offer, peer string) (answer string, id int64, err error) {
	id = r.sessions.Add(1)
	l := slog.With("dir", "viewer", "session", id, "peer", peer)
	l.Info("WHEP offer received", "bytes", len(offer), "ingest_live", r.IngestLive())

	pc, err := r.viewerAPI.NewPeerConnection(stunConfig)
	if err != nil {
		return "", 0, fmt.Errorf("peerconnection: %w", err)
	}
	logCandidates(l, pc)

	rtpSender, err := pc.AddTrack(r.track)
	if err != nil {
		_ = pc.Close()
		return "", 0, fmt.Errorf("add track: %w", err)
	}
	// Drain the viewer's RTCP (RRs, PLIs) so interceptors keep processing.
	// When the browser signals loss, forward the PLI upstream to the ingest.
	go func() {
		for {
			pkts, _, err := rtpSender.ReadRTCP()
			if err != nil {
				return
			}
			for _, p := range pkts {
				if _, ok := p.(*rtcp.PictureLossIndication); ok {
					r.RequestKeyframe()
				}
			}
		}
	}()

	pc.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
		l.Info("viewer connection state", "state", s.String())
		if s == webrtc.PeerConnectionStateFailed || s == webrtc.PeerConnectionStateClosed {
			r.removeViewer(id, pc)
		}
	})

	answer, err = setRemoteAndAnswer(pc, offer)
	if err != nil {
		_ = pc.Close()
		return "", 0, fmt.Errorf("negotiate: %w", err)
	}

	r.mu.Lock()
	r.viewers[id] = &viewer{pc: pc}
	r.mu.Unlock()
	r.metrics.ViewerSessionStarted()
	r.notifyViewerCount()

	// Kick a keyframe so the viewer paints immediately.
	r.RequestKeyframe()
	l.Info("WHEP answered")
	return answer, id, nil
}

// CloseViewer is the WHEP DELETE teardown for a well-behaved client; PC state
// changes cover misbehaving ones.
func (r *Relay) CloseViewer(id int64) {
	r.mu.Lock()
	v, ok := r.viewers[id]
	r.mu.Unlock()
	if ok {
		_ = v.pc.Close()
		r.removeViewer(id, v.pc)
	}
}

func (r *Relay) removeViewer(id int64, pc *webrtc.PeerConnection) {
	r.mu.Lock()
	v, ok := r.viewers[id]
	if !ok || v.pc != pc {
		r.mu.Unlock()
		return
	}
	delete(r.viewers, id)
	r.mu.Unlock()
	_ = pc.Close()
	r.notifyViewerCount()
}

// --- helpers --------------------------------------------------------------------

// setRemoteAndAnswer performs the standard non-trickle exchange: set the
// remote offer, create an answer, set it local, and block on ICE gathering so
// the returned SDP is complete.
func setRemoteAndAnswer(pc *webrtc.PeerConnection, offer string) (string, error) {
	if err := pc.SetRemoteDescription(webrtc.SessionDescription{
		Type: webrtc.SDPTypeOffer, SDP: offer,
	}); err != nil {
		return "", err
	}
	answer, err := pc.CreateAnswer(nil)
	if err != nil {
		return "", err
	}
	gatherComplete := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(answer); err != nil {
		return "", err
	}
	<-gatherComplete
	return pc.LocalDescription().SDP, nil
}

// logCandidates records ICE diagnostics: every local candidate gathered (so
// you can confirm the expected nat1To1 address was offered), and the pair Pion
// selects once connected — the single most useful line for "did media take the
// tunnel / the NodePort?".
func logCandidates(l *slog.Logger, pc *webrtc.PeerConnection) {
	pc.OnICECandidate(func(c *webrtc.ICECandidate) {
		if c == nil {
			l.Debug("ICE gathering complete")
			return
		}
		l.Debug("local ICE candidate",
			"type", c.Typ.String(), "addr", fmt.Sprintf("%s:%d", c.Address, c.Port), "proto", c.Protocol.String())
	})
	it := pc.SCTP().Transport().ICETransport()
	it.OnSelectedCandidatePairChange(func(p *webrtc.ICECandidatePair) {
		if p == nil {
			return
		}
		l.Info("ICE candidate pair selected",
			"local", fmt.Sprintf("%s:%d/%s", p.Local.Address, p.Local.Port, p.Local.Typ),
			"remote", fmt.Sprintf("%s:%d/%s", p.Remote.Address, p.Remote.Port, p.Remote.Typ))
	})
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// --- tailnet discovery ------------------------------------------------------------

// tailnetCGNAT is the 100.64.0.0/10 range Tailscale assigns node addresses from.
var tailnetCGNAT = func() *net.IPNet { _, n, _ := net.ParseCIDR("100.64.0.0/10"); return n }()

func findTailnetIP() string {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return ""
	}
	for _, a := range addrs {
		ipn, ok := a.(*net.IPNet)
		if !ok {
			continue
		}
		if ip4 := ipn.IP.To4(); ip4 != nil && tailnetCGNAT.Contains(ip4) {
			return ip4.String()
		}
	}
	return ""
}

// WaitForTailnetIP polls for the tailnet address until it appears or timeout.
// The sidecar's tailscaled takes a few seconds to authenticate and bring up
// tailscale0, so the relay may start before the address exists.
func WaitForTailnetIP(timeout time.Duration) string {
	deadline := time.Now().Add(timeout)
	for {
		if ip := findTailnetIP(); ip != "" {
			return ip
		}
		if time.Now().After(deadline) {
			return ""
		}
		slog.Info("waiting for tailnet interface (tailscale sidecar starting)…")
		time.Sleep(time.Second)
	}
}

// --- pull-mode uplink (validated escape hatch) --------------------------------------

// StartPull switches the uplink to WHEP-CLIENT mode: the relay dials OUT to
// the origin's (the Pi's) WHEP endpoint and pulls the stream, instead of
// waiting for a WHIP push. This inverts who initiates and sidesteps the
// ICE-address-symmetry / privileged-sidecar problem; kept as the validated
// fallback from the whip-relay prototype (deploy/PULL-VALIDATION.md).
func (r *Relay) StartPull(whepURL string) {
	go func() {
		backoff := time.Second
		for {
			l := slog.With("dir", "ingest", "mode", "pull", "origin", whepURL)
			if err := r.pullOnce(whepURL, l); err != nil {
				l.Warn("pull failed; will retry", "err", err, "in", backoff.String())
				time.Sleep(backoff)
				if backoff < 15*time.Second {
					backoff *= 2
				}
				continue
			}
			backoff = time.Second
			l.Info("pull ended; reconnecting")
			time.Sleep(time.Second)
		}
	}()
}

func (r *Relay) pullOnce(whepURL string, l *slog.Logger) error {
	pc, err := r.ingestAPI.NewPeerConnection(stunConfig)
	if err != nil {
		return fmt.Errorf("new peer connection: %w", err)
	}
	logCandidates(l, pc)

	if _, err := pc.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo,
		webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly}); err != nil {
		_ = pc.Close()
		return fmt.Errorf("add transceiver: %w", err)
	}

	done := make(chan struct{})
	var once sync.Once
	closeDone := func() { once.Do(func() { close(done) }) }

	r.attachIngest(pc, l)
	pc.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
		if s == webrtc.PeerConnectionStateFailed || s == webrtc.PeerConnectionStateClosed || s == webrtc.PeerConnectionStateDisconnected {
			closeDone()
		}
	})

	offer, err := pc.CreateOffer(nil)
	if err != nil {
		_ = pc.Close()
		return fmt.Errorf("create offer: %w", err)
	}
	gatherComplete := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(offer); err != nil {
		_ = pc.Close()
		return fmt.Errorf("set local: %w", err)
	}
	<-gatherComplete

	l.Info("dialing origin WHEP")
	answer, err := postSDP(whepURL, pc.LocalDescription().SDP)
	if err != nil {
		_ = pc.Close()
		return fmt.Errorf("POST offer: %w", err)
	}
	if err := pc.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: answer}); err != nil {
		_ = pc.Close()
		return fmt.Errorf("set remote answer: %w", err)
	}

	<-done
	_ = pc.Close()
	return nil
}

func postSDP(url, offer string) (string, error) {
	req, err := http.NewRequest(http.MethodPost, url, strings.NewReader(offer))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/sdp")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("origin returned %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return string(body), nil
}
