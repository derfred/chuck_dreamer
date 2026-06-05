// Shared HLS playback hook (F4.1, F4.3). Attaches hls.js to a <video> element,
// or uses the browser's native HLS when available (Safari / iOS, the
// iPhone-primary access pattern). Tearing down is clean: the hls.js instance is
// destroyed on URL change / unmount so a closed player frees its buffers.
//
// `src` null means "no source" (player idle). When `live` is true the player
// seeks to the live edge as new segments arrive (the ring is rolling, F4.1);
// for a finite recording it is false so the operator can scrub from the start.

import { useEffect } from "react";
import Hls from "hls.js";

export function useHls(
  videoRef: React.RefObject<HTMLVideoElement>,
  src: string | null,
  opts: { live?: boolean } = {},
): void {
  const live = opts.live ?? false;
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !src) return;

    // Native HLS (Safari/iOS): point the element straight at the playlist.
    if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = src;
      return () => {
        video.removeAttribute("src");
        video.load();
      };
    }

    if (!Hls.isSupported()) {
      // No MSE and no native HLS — nothing we can do; leave the element blank.
      return;
    }

    const hls = new Hls({
      // For the live ring, keep the player near the edge; for a finite
      // recording these are inert.
      liveSyncDurationCount: 3,
      lowLatencyMode: false,
    });
    hls.loadSource(src);
    hls.attachMedia(video);
    if (live) {
      hls.on(Hls.Events.LEVEL_LOADED, () => {
        // Nudge toward the live edge when fresh segments land (F4.1).
        if (hls.liveSyncPosition != null && video.currentTime < hls.liveSyncPosition - 5) {
          video.currentTime = hls.liveSyncPosition;
        }
      });
    }
    return () => {
      hls.destroy();
    };
  }, [videoRef, src, live]);
}
