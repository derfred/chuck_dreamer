#!/usr/bin/env python3
"""Safe-box install helper (Slice 5 X5.2).

The `vision.safe_box` rectangle has no good default — the right rectangle is
camera-position-specific and must be set during install. This script captures a
single frame from the camera at the 640x360 vision-analysis resolution, overlays
the configured safe box, and writes a PNG so the operator can confirm the box
visually before committing the config.

It is a side artifact, NOT a UI: run it on the Pi after positioning the camera,
look at the output PNG, adjust `video.vision.safe_box` in fessel.yaml, repeat.

Usage:
  safe_box_preview.py [--config /etc/fessel/fessel.yaml] [--out /tmp/safe_box.png]
                      [--device /dev/video0] [--test-source]

Requires the same gi + OpenCV stack the vision branch uses (system packages on
the Pi). On a dev box without a camera, pass --test-source.
"""

from __future__ import annotations

import argparse
import os
import sys

from fessel_shared import load_component_config

# These imports come from the system site-packages (like the vision branch).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video.pipeline import (  # noqa: E402
  VISION_APPSINK_NAME,
  VISION_HEIGHT,
  VISION_WIDTH,
  build_pipeline,
  build_vision_launch,
)


def _capture_frame(device: str, use_test_source: bool):  # noqa: ANN202
  """Pull one 640x360 I420 frame from the vision branch and return it as a BGR
  numpy array. Tears the pipeline down immediately after."""
  import gi

  gi.require_version("Gst", "1.0")
  from gi.repository import Gst

  import cv2
  import numpy as np

  from fessel_schemas import ModeTriplet

  mode = ModeTriplet(resolution="1280x720", fps=30, bitrate_bps=2_000_000)
  launch = build_vision_launch(mode=mode, device=device, use_test_source=use_test_source)
  pipeline = build_pipeline(launch)
  appsink = pipeline.get_by_name(VISION_APPSINK_NAME)
  pipeline.set_state(Gst.State.PLAYING)
  try:
    sample = appsink.emit("try-pull-sample", 5 * Gst.SECOND)
    if sample is None:
      raise RuntimeError("no frame captured within 5s (camera up?)")
    buf = sample.get_buffer()
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
      raise RuntimeError("failed to map captured buffer")
    try:
      y_plane = np.frombuffer(mapinfo.data[: VISION_WIDTH * VISION_HEIGHT], dtype=np.uint8).reshape(
        (VISION_HEIGHT, VISION_WIDTH)
      )
    finally:
      buf.unmap(mapinfo)
    return cv2.cvtColor(y_plane, cv2.COLOR_GRAY2BGR)
  finally:
    pipeline.set_state(Gst.State.NULL)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--config", default=os.environ.get("FESSEL_CONFIG", "/etc/fessel/fessel.yaml"))
  ap.add_argument("--out", default="/tmp/safe_box_preview.png")
  ap.add_argument("--device", default=None, help="override camera device")
  ap.add_argument("--test-source", action="store_true", help="use videotestsrc (dev)")
  args = ap.parse_args()

  cfg = load_component_config(args.config, "video")
  cam = cfg.get("camera") or {}
  device = args.device or cam.get("device", "/dev/video0")
  use_test = args.test_source or bool(cam.get("use_test_source", False))
  vision_cfg = cfg.get("vision") or {}
  box = vision_cfg.get("safe_box")
  if not isinstance(box, dict):
    print("no vision.safe_box configured — nothing to preview", file=sys.stderr)
    sys.exit(2)

  import cv2

  frame = _capture_frame(device, use_test)
  x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
  cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
  cv2.putText(
    frame,
    f"safe_box x={x} y={y} w={w} h={h}",
    (8, 20),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 0, 255),
    1,
    cv2.LINE_AA,
  )
  cv2.imwrite(args.out, frame)
  print(f"wrote {args.out} — open it and confirm the red box bounds the arm workspace")


if __name__ == "__main__":
  main()
