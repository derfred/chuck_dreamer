// WHEP signaling primitive. Reconnect, stall detection, and re-auth live in
// the /live state machine (useLiveSession); this is just the one-shot SDP
// exchange against a signed WHEP URL.

// Perform the WHEP handshake against a signed URL and attach the inbound
// track to the given <video> element. Returns the peer connection so the
// caller (the state machine) can poll its stats and close it on teardown.
//
// The caller builds a FRESH RTCPeerConnection per attempt: a dead connection
// is never reused (F2.4).
export async function startWhep(signedUrl: string, video: HTMLVideoElement): Promise<RTCPeerConnection> {
  const pc = new RTCPeerConnection();
  // We are a receiver only.
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" });

  pc.ontrack = (ev) => {
    if (ev.streams[0]) {
      video.srcObject = ev.streams[0];
    }
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const res = await fetch(signedUrl, {
    method: "POST",
    headers: { "Content-Type": "application/sdp" },
    body: offer.sdp ?? "",
  });
  if (!res.ok) {
    pc.close();
    throw new Error(`WHEP POST failed: ${res.status}`);
  }
  const answerSdp = await res.text();
  await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
  return pc;
}
