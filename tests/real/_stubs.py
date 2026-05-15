"""Shared test stubs for phase-A interactive code paths."""

from __future__ import annotations

from collections import deque
from typing import Callable

import cv2
import numpy as np


class StubFrameSource:
  """Yields scripted frames; ``read()`` cycles through a generator.

  Frames are RGB ``(H, W, 3)`` uint8. ``frame_factory(i)`` is called for
  every read with a monotonically increasing index so tests can encode
  time-varying content (e.g. blob present for some frames, absent for
  others).
  """

  def __init__(
    self,
    name: str,
    *,
    fps: int,
    image_size: tuple[int, int],
    frame_factory: Callable[[int], np.ndarray],
  ) -> None:
    self.name = name
    self.fps = fps
    self.image_size = image_size
    self._factory = frame_factory
    self._i = 0
    self.disconnected = False

  def read(self) -> np.ndarray:
    img = self._factory(self._i)
    self._i += 1
    return img

  def disconnect(self) -> None:
    self.disconnected = True


class StubUi:
  """Replays a scripted sequence of keypresses + prompt responses.

  ``key_script`` is consumed left-to-right by ``poll_key``. To make
  missing-script bugs fail loudly instead of hanging, ``poll_key``
  raises ``RuntimeError`` after ``max_empty_polls`` consecutive empty
  returns. Tests that intentionally rely on "keep showing" can pass
  ``max_empty_polls=0`` to disable the watchdog.
  """

  def __init__(
    self,
    *,
    key_script: list[str] | None = None,
    prompt_script: list[str] | None = None,
    max_empty_polls: int = 2000,
  ) -> None:
    self.keys: deque[str] = deque(key_script or [])
    self.prompts: deque[str] = deque(prompt_script or [])
    self.shown: list[tuple[str, np.ndarray]] = []
    self.closed: list[str] = []
    self.poll_count = 0
    self.pump_count = 0
    self.max_empty_polls = max_empty_polls
    self._empty_streak = 0

  def show(self, window: str, image_bgr: np.ndarray) -> None:
    self.shown.append((window, image_bgr))

  def poll_key(self, timeout_ms: int = 1) -> str:
    self.poll_count += 1
    if self.keys:
      self._empty_streak = 0
      return self.keys.popleft()
    self._empty_streak += 1
    if self.max_empty_polls and self._empty_streak > self.max_empty_polls:
      raise RuntimeError(
        f"StubUi: {self._empty_streak} consecutive empty polls — "
        f"missing scripted key?")
    return ""

  def pump(self, timeout_ms: int = 1) -> None:
    # Window refresh: never advances ``keys`` so clip-capture loops
    # cannot accidentally drain the script.
    self.pump_count += 1

  def prompt(self, message: str) -> str:
    if self.prompts:
      return self.prompts.popleft()
    return ""

  def close(self, window: str) -> None:
    self.closed.append(window)


def make_disk_frame_factory(
  *,
  H: int = 240,
  W: int = 320,
  cx: int = 160,
  cy: int = 120,
  r: int = 30,
  blob_predicate: Callable[[int], bool] = lambda i: True,
) -> Callable[[int], np.ndarray]:
  """Return a factory: index → RGB frame with optional white disk."""
  def factory(i: int) -> np.ndarray:
    bg = np.full((H, W, 3), 30, dtype=np.uint8)
    if blob_predicate(i):
      cv2.circle(bg, (cx, cy), r, (240, 240, 240), thickness=-1)
    return bg                              # RGB == BGR for this content
  return factory
