// Minimal WHEP client (F1.1). No Stalled/auto-reconnect logic yet — that
// is Slice 2. A dropped stream simply appears frozen in this slice.

import type { ModeTriplet } from "../../shared/schemas";

// Canonical mode string "<W>x<H>@<fps>@<bitrate_bps>" (X0.2). Must match
// the Python serialiser (fessel_schemas.mode_to_canonical) exactly.
export function modeToCanonical(m: ModeTriplet): string {
  return `${m.resolution}@${m.fps}@${m.bitrate_bps}`;
}

interface WhepUrlResponse {
  url: string;
}

// Fetch a signed WHEP URL from the backend for the chosen path + mode.
export async function fetchWhepUrl(path: string, mode: ModeTriplet): Promise<string> {
  const params = new URLSearchParams({ path, mode: modeToCanonical(mode) });
  const res = await fetch(`/api/auth/whep-url?${params.toString()}`);
  if (!res.ok) {
    throw new Error(`whep-url failed: ${res.status}`);
  }
  const body = (await res.json()) as WhepUrlResponse;
  return body.url;
}

// Perform the WHEP handshake against a signed URL and attach the inbound
// track to the given <video> element. Returns the peer connection so the
// caller can close it on teardown.
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
