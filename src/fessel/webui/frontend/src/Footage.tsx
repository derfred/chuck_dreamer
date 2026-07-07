// Footage page (`/footage`): the explicit + anomaly recordings on disk, played
// back on demand over HLS. There is NO ring-buffer viewer here: the ring lives
// on the Pi behind the cellular link, so it is never streamed back — its only
// role is retroactive capture (the record dialog's look-back on Monitor). What
// you review here are finalised recordings that were already captured.

import { Recordings } from "./Recordings";

export function Footage() {
  return (
    <div style={{ padding: 16, maxWidth: 1080, margin: "0 auto" }}>
      <h1 style={{ fontSize: 22, letterSpacing: "-0.02em", margin: "10px 0 18px" }}>Footage</h1>
      <Recordings embedded />
    </div>
  );
}
