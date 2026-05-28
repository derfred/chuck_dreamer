// Minimal /live page (F1.1). Full UX-state machine + media-liveness
// detection are Slice 2; here we just fetch capabilities, let the operator
// pick a mode, mint a signed WHEP URL, and run the handshake.

import { useEffect, useRef, useState } from "react";
import type { Capabilities, ModeTriplet } from "../../shared/schemas";
import { fetchWhepUrl, modeToCanonical, startWhep } from "./whep";

const LIVE_PATH = "pi";

export function Live() {
  const [modes, setModes] = useState<ModeTriplet[]>([]);
  const [selected, setSelected] = useState<number>(0);
  const [status, setStatus] = useState<string>("idle");
  const videoRef = useRef<HTMLVideoElement>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);

  useEffect(() => {
    fetch("/api/capabilities")
      .then((r) => r.json() as Promise<Capabilities>)
      .then((c) => setModes(c.modes))
      .catch((e) => setStatus(`capabilities error: ${e}`));
  }, []);

  async function start() {
    const mode = modes[selected];
    if (!mode || !videoRef.current) return;
    try {
      setStatus("requesting signed URL…");
      const url = await fetchWhepUrl(LIVE_PATH, mode);
      setStatus("signaling…");
      pcRef.current = await startWhep(url, videoRef.current);
      setStatus("playing");
    } catch (e) {
      setStatus(`error: ${e}`);
    }
  }

  function stop() {
    pcRef.current?.close();
    pcRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus("idle");
  }

  return (
    <div style={{ fontFamily: "sans-serif", padding: 16 }}>
      <h1>Fessel — Live</h1>
      <div>
        <label>
          Mode:{" "}
          <select value={selected} onChange={(e) => setSelected(Number(e.target.value))}>
            {modes.map((m, i) => (
              <option key={i} value={i}>
                {modeToCanonical(m)}
              </option>
            ))}
          </select>
        </label>{" "}
        <button onClick={start} disabled={modes.length === 0}>
          Start live view
        </button>{" "}
        <button onClick={stop}>Stop</button>
      </div>
      <p>Status: {status}</p>
      <video ref={videoRef} autoPlay playsInline muted style={{ width: 640, background: "#000" }} />
    </div>
  );
}
