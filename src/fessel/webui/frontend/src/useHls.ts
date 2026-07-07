// Shared HLS playback hook (F4.3). Attaches hls.js to a <video> element, or uses
// the browser's native HLS when available (Safari / iOS, the iPhone-primary
// access pattern). Tearing down is clean: the hls.js instance is destroyed on
// URL change / unmount so a closed player frees its buffers.
//
// `src` null means "no source" (player idle). Playback is for finite recordings
// (the recordings playback modal) — the operator scrubs from the start. The ring
// buffer is never played back in the UI (it lives on the Pi behind cellular), so
// there is no live-edge-following mode here.

import { useEffect } from "react";
import Hls from "hls.js";

export function useHls(videoRef: React.RefObject<HTMLVideoElement>, src: string | null): void {
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

    const hls = new Hls({ lowLatencyMode: false });
    hls.loadSource(src);
    hls.attachMedia(video);
    return () => {
      hls.destroy();
    };
  }, [videoRef, src]);
}
