"""OpenCV camera capture wrapped for the real-arm run loop.

The camera is opened by :class:`Camera` and exposes a single ``read()``
that returns the latest BGR frame (numpy ``uint8`` of shape ``(H, W, 3)``)
along with its RGB sibling for downstream policies. A separate ``show()``
helper draws the BGR frame into a named cv2 window so the operator can
verify what the policy is seeing.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2  # type: ignore[import-not-found]
import numpy as np


def _parse_source(raw: str | int) -> int | str:
  """Turn ``"0"`` / ``"/dev/video0"`` into the type ``cv2.VideoCapture`` wants.

  Integer-looking strings become ints so the V4L2 backend treats them as
  indices; non-numeric strings (device paths, RTSP URLs) pass through
  unchanged.
  """
  if isinstance(raw, int):
    return raw
  s = str(raw).strip()
  try:
    return int(s)
  except ValueError:
    return s


@dataclass
class CameraConfig:
  source: int | str
  width:  int = 640
  height: int = 480
  fps:    int = 30
  flip:   bool = False


class Camera:
  """Thin OpenCV camera wrapper.

  Use it as a context manager so the underlying ``VideoCapture`` is
  released even if the run loop crashes. ``read()`` returns
  ``(bgr, rgb)``; the BGR frame goes to the cv2 window, the RGB one is
  what policies expect.
  """

  def __init__(self, config: CameraConfig) -> None:
    self.config  = config
    self._cap: cv2.VideoCapture | None = None

  def open(self) -> "Camera":
    source = _parse_source(self.config.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
      raise RuntimeError(f"could not open camera source={source!r}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
    cap.set(cv2.CAP_PROP_FPS,          self.config.fps)
    self._cap = cap
    return self

  def __enter__(self) -> "Camera":
    return self.open()

  def __exit__(self, *_exc) -> None:
    self.close()

  def close(self) -> None:
    if self._cap is not None:
      self._cap.release()
      self._cap = None

  def read(self) -> tuple[np.ndarray, np.ndarray]:
    if self._cap is None:
      raise RuntimeError("Camera.read() called before open()")
    ok, frame_bgr = self._cap.read()
    if not ok or frame_bgr is None:
      raise RuntimeError("camera returned no frame")
    if self.config.flip:
      frame_bgr = cv2.flip(frame_bgr, 1)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_bgr, frame_rgb
