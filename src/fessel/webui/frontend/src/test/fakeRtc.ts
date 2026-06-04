// A controllable RTCPeerConnection stub. jsdom has no WebRTC, so the live
// session tests install this as the global `RTCPeerConnection` and drive media
// flow by hand: bytesReceived and iceConnectionState are mutable, so a test can
// simulate a silent media stall (bytes frozen, ICE still "connected") or an ICE
// drop, which are the two conditions the state machine exists to catch (F2.3).

export interface FakeStat {
  type: string;
  kind?: string;
  bytesReceived?: number;
}

// Registry of every PC the code under test created, in order. Lets a test grab
// "the current PC" after a reconnect built a fresh one (F2.4: dead PC is never
// reused — each attempt is a new instance here).
export const createdPcs: FakeRTCPeerConnection[] = [];

export function resetFakeRtc(): void {
  createdPcs.length = 0;
}

export class FakeRTCPeerConnection {
  // Driven by the test. Reads exactly like the real getStats inbound-rtp video.
  bytesReceived = 0;
  iceConnectionState: RTCIceConnectionState = "connected";
  closed = false;
  ontrack: ((ev: { streams: MediaStream[] }) => void) | null = null;

  private localSdp = "v=0\r\n";

  constructor() {
    createdPcs.push(this);
  }

  addTransceiver(_kind: string, _init?: RTCRtpTransceiverInit): void {
    /* no-op: receive-only direction is irrelevant to the fake */
  }

  async createOffer(): Promise<RTCSessionDescriptionInit> {
    return { type: "offer", sdp: this.localSdp };
  }

  async setLocalDescription(_d: RTCSessionDescriptionInit): Promise<void> {
    /* no-op */
  }

  async setRemoteDescription(_d: RTCSessionDescriptionInit): Promise<void> {
    // The track "arrives" once the answer is applied; deliver a stream so the
    // caller wires up <video>.srcObject exactly as in production.
    this.ontrack?.({ streams: [{} as MediaStream] });
  }

  getStats(): Promise<Map<string, FakeStat>> {
    const m = new Map<string, FakeStat>();
    m.set("ir", { type: "inbound-rtp", kind: "video", bytesReceived: this.bytesReceived });
    return Promise.resolve(m);
  }

  close(): void {
    this.closed = true;
  }
}

// Install/uninstall the fake as the global RTCPeerConnection.
export function installFakeRtc(): void {
  resetFakeRtc();
  (globalThis as unknown as { RTCPeerConnection: unknown }).RTCPeerConnection =
    FakeRTCPeerConnection;
}
