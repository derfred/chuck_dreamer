package relay

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
)

// newLoopbackRelay builds a relay with default host-candidate gathering (no
// NAT1To1, ephemeral ports) — media flows over loopback.
func newLoopbackRelay(t *testing.T) *Relay {
	t.Helper()
	r, err := New(ICEConfig{}, ICEConfig{}, NopMetrics())
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// dialIngest stands in for the Pi's whipclientsink: a sendonly PC publishing
// an H.264 track through HandleIngestOffer.
func dialIngest(t *testing.T, r *Relay) (*webrtc.PeerConnection, *webrtc.TrackLocalStaticSample) {
	t.Helper()
	pc, err := webrtc.NewPeerConnection(stunConfig)
	if err != nil {
		t.Fatal(err)
	}
	track, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeH264}, "video", "pi")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pc.AddTrack(track); err != nil {
		t.Fatal(err)
	}
	offer, err := pc.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	gather := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-gather

	answer, _, err := r.HandleIngestOffer(pc.LocalDescription().SDP, "test-pi")
	if err != nil {
		t.Fatal(err)
	}
	if err := pc.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: answer}); err != nil {
		t.Fatal(err)
	}
	return pc, track
}

// dialViewer stands in for the browser's WHEP client.
func dialViewer(t *testing.T, r *Relay) (*webrtc.PeerConnection, chan struct{}, int64) {
	t.Helper()
	pc, err := webrtc.NewPeerConnection(stunConfig)
	if err != nil {
		t.Fatal(err)
	}
	gotMedia := make(chan struct{})
	pc.OnTrack(func(remote *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		buf := make([]byte, 1500)
		if _, _, err := remote.Read(buf); err == nil {
			close(gotMedia)
		}
	})
	if _, err := pc.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo,
		webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly}); err != nil {
		t.Fatal(err)
	}
	offer, err := pc.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	gather := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-gather

	answer, id, err := r.HandleViewerOffer(pc.LocalDescription().SDP, "test-browser")
	if err != nil {
		t.Fatal(err)
	}
	if err := pc.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: answer}); err != nil {
		t.Fatal(err)
	}
	return pc, gotMedia, id
}

// TestRelayEndToEnd is the Pion loopback smoke test: a publisher pushes H.264
// samples through the ingest plane; a viewer on the viewer plane receives RTP
// from the permanent shared track.
func TestRelayEndToEnd(t *testing.T) {
	r := newLoopbackRelay(t)

	ingestPC, track := dialIngest(t, r)
	defer ingestPC.Close()

	// Feed samples until the ingest goes live (OnTrack fires on first packet).
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		ticker := time.NewTicker(20 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				// A minimal NAL-ish payload; content doesn't matter for RTP flow.
				_ = track.WriteSample(media.Sample{Data: []byte{0x65, 0x01, 0x02, 0x03}, Duration: 20 * time.Millisecond})
			}
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if !r.WaitIngest(ctx) {
		t.Fatal("ingest never went live")
	}
	if !r.IngestLive() {
		t.Fatal("IngestLive false after WaitIngest")
	}

	viewerPC, gotMedia, id := dialViewer(t, r)
	defer viewerPC.Close()
	if r.ViewerCount() != 1 {
		t.Fatalf("viewer count %d", r.ViewerCount())
	}

	select {
	case <-gotMedia:
	case <-time.After(10 * time.Second):
		t.Fatal("viewer received no media")
	}

	// Teardown: closing the viewer drops the count (via CloseViewer, the
	// well-behaved WHEP DELETE path).
	r.CloseViewer(id)
	deadline := time.Now().Add(5 * time.Second)
	for r.ViewerCount() != 0 && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if r.ViewerCount() != 0 {
		t.Fatalf("viewer count %d after close", r.ViewerCount())
	}
}

func TestViewerCountHookFires(t *testing.T) {
	r := newLoopbackRelay(t)
	counts := make(chan int, 8)
	r.SetViewerCountHook(func(c int) { counts <- c })

	ingestPC, track := dialIngest(t, r)
	defer ingestPC.Close()
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		ticker := time.NewTicker(20 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				_ = track.WriteSample(media.Sample{Data: []byte{0x65, 0x00}, Duration: 20 * time.Millisecond})
			}
		}
	}()

	viewerPC, _, id := dialViewer(t, r)
	defer viewerPC.Close()

	select {
	case c := <-counts:
		if c != 1 {
			t.Fatalf("first hook count %d", c)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("hook never fired on join")
	}

	r.CloseViewer(id)
	select {
	case c := <-counts:
		if c != 0 {
			t.Fatalf("hook count after close %d", c)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("hook never fired on close")
	}
}

func TestWaitIngestTimesOutWithoutIngest(t *testing.T) {
	r := newLoopbackRelay(t)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if r.WaitIngest(ctx) {
		t.Fatal("WaitIngest reported live with no ingest")
	}
}

// TestViewerAnswerAdvertisesConfiguredCandidates guards the SettingEngine
// wiring (WithSettingEngine): with a NAT1To1 IP + UDP mux configured on the
// viewer plane, the WHEP answer must advertise exactly that IP on exactly the
// mux port as a host candidate — NOT the default-gathered interface addresses
// on ephemeral ports. (The prototype dropped the SettingEngine silently; every
// in-cluster path still worked, only real off-cluster browsers broke.)
func TestViewerAnswerAdvertisesConfiguredCandidates(t *testing.T) {
	r, err := New(ICEConfig{}, ICEConfig{NAT1To1IPs: []string{"203.0.113.7"}, UDPMuxPort: 39877}, NopMetrics())
	if err != nil {
		t.Fatal(err)
	}
	pc, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	defer pc.Close()
	if _, err := pc.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo,
		webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly}); err != nil {
		t.Fatal(err)
	}
	offer, err := pc.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	gather := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-gather

	answer, id, err := r.HandleViewerOffer(pc.LocalDescription().SDP, "test")
	if err != nil {
		t.Fatal(err)
	}
	defer r.CloseViewer(id)
	if !strings.Contains(answer, "203.0.113.7 39877 typ host") {
		t.Fatalf("answer lacks the advertised host candidate on the mux port:\n%s", answer)
	}
	// SetNAT1To1IPs replaces IPv4 host candidates only; IPv6 hosts may remain
	// (harmless extras). Assert no OTHER IPv4 host candidate leaks and that
	// every candidate sits on the mux port.
	for _, line := range strings.Split(answer, "\n") {
		if !strings.HasPrefix(line, "a=candidate") {
			continue
		}
		if strings.Contains(line, "typ host") && strings.Count(line, ".") >= 3 &&
			!strings.Contains(line, "203.0.113.7") {
			t.Fatalf("answer leaks a non-advertised IPv4 host candidate: %s", line)
		}
		if !strings.Contains(line, " 39877 typ") {
			t.Fatalf("candidate not on the mux port: %s", line)
		}
	}
}
