// WHEP signaling primitive. Reconnect, stall detection, and re-auth live in
// the /live state machine (useLiveSession); this is just the one-shot SDP
// exchange against the same-origin /whep endpoint behind the auth proxy
// (architecture §4.5: the identity headers ARE the authorisation — no signed
// URL, no token, no mode selection; the live profile is fixed).

import { AuthError } from "./api";
import { LiveViewErrorSchema } from "./generated/validators";

// A non-201 from /whep. The backend answers structured JSON errors
// ({error, reason}) for the health-gate/activation outcomes:
//   503 live_unavailable        — Pi unreachable / camera not detected / video down
//   504 live_timeout            — timed out starting stream
//   502 live_activation_failed  — activation ran but failed
// `reason` is operator-facing and the message carries it for display.
export class WhepError extends Error {
  readonly status: number;
  readonly reason: string;
  constructor(status: number, reason: string) {
    super(reason);
    this.name = "WhepError";
    this.status = status;
    this.reason = reason;
  }
}

// A live WHEP viewer session: the peer connection (the caller polls its stats
// and closes it on teardown) plus the resource the server minted in the
// Location header, so the session can be DELETEd for clean teardown.
export interface WhepSession {
  pc: RTCPeerConnection;
  // e.g. "/whep/42"; null if the server sent no Location header.
  location: string | null;
}

// Perform the WHEP handshake against same-origin /whep and attach the inbound
// track to the given <video> element.
//
// NOTE: the POST BLOCKS up to the server's activation timeout (~15s) on the
// first viewer — the caller's "Connecting…/Establishing connection…" states
// mask that wait; do not put a short client-side timeout on it.
//
// The caller builds a FRESH RTCPeerConnection per attempt: a dead connection
// is never reused (F2.4). `onSignaling` (optional) fires just before the POST
// goes out, so the state machine can flip Requesting ("Connecting…") →
// Signaling ("Establishing connection…") exactly when the blocking wait starts.
export async function startWhep(
  video: HTMLVideoElement,
  onSignaling?: () => void,
): Promise<WhepSession> {
  const pc = new RTCPeerConnection();
  // We are a receiver only.
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" });

  pc.ontrack = (ev) => {
    // ontrack fires once per m-line — including the relay's inactive Opus
    // section (present for WHEP-client compat; no audio is ever relayed).
    // Only the video track may claim the element: srcObject is a single-track
    // stream, so an unconditional assignment lets whichever track fires LAST
    // win — when that's the audio track the element holds an audio-only
    // stream and shows black (videoWidth 0, currentTime frozen). The firing
    // order is unspecified, which made this a per-browser/per-run race.
    if (ev.track.kind !== "video") return;
    // Wrap the receiver's track in a LOCALLY-constructed MediaStream instead
    // of using the remote-announced ev.streams[0]: Firefox decodes the track
    // either way (getStats framesDecoded advances) but never paints frames
    // from the remote stream object — videoWidth stays 0 and the element
    // shows black. Verified live: the same track renders once re-wrapped.
    // Safari/Chrome render both forms identically.
    video.srcObject = new MediaStream([ev.track]);
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  onSignaling?.();
  let res: Response;
  try {
    res = await fetch("/whep", {
      method: "POST",
      headers: { "Content-Type": "application/sdp" },
      body: offer.sdp ?? "",
    });
  } catch (e) {
    pc.close();
    throw e;
  }
  if (res.status === 401) {
    // Session expired at the auth proxy — the caller escalates to re-auth.
    pc.close();
    throw new AuthError();
  }
  if (res.status !== 201) {
    pc.close();
    throw new WhepError(res.status, await whepReason(res));
  }
  const location = res.headers.get("Location");
  const answerSdp = await res.text();
  await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
  return { pc, location };
}

// Release a WHEP session resource server-side. Fire-and-forget: teardown must
// never block or fail on this — the server also reaps dead viewers via ICE.
export function deleteWhepSession(location: string): void {
  void fetch(location, { method: "DELETE" }).catch(() => {
    /* best-effort teardown; the server garbage-collects regardless */
  });
}

// Pull the operator-facing `reason` out of a /whep error body. The body is
// validated against the generated LiveViewError schema (the same definition
// the Go relay's generated struct serialises), so both ends of the live-view
// error contract derive from the one pydantic source.
async function whepReason(res: Response): Promise<string> {
  try {
    const parsed = LiveViewErrorSchema.safeParse(await res.json());
    if (parsed.success) return parsed.data.reason;
  } catch {
    // non-JSON error body; fall through to status-only message.
  }
  return `WHEP POST failed: ${res.status}`;
}
