"""Tests for marker-clip capture in :mod:`chuck_dreamer.real.scene_registration`.

These exercise the per-slot validation logic, retry path on rejection,
and ESC-abort behaviour without touching ``cv2.imshow`` or a real
camera.
"""

from __future__ import annotations

import numpy as np
import pytest

from chuck_dreamer.real.scene_registration import (
  MarkerCaptureSpec,
  SceneRegistrationAborted,
  register_markers,
)

from ._stubs import StubFrameSource, StubUi, make_disk_frame_factory


def _spec(locations, duration_s=0.2, min_frac=0.8):
  # fps × duration_s = number of frames captured per clip; use small values
  # so tests stay snappy.
  return MarkerCaptureSpec(
    locations=tuple(locations),
    clip_duration_s=duration_s,
    min_blob_fraction=min_frac,
  )


def test_blob_slot_accepted_when_marker_visible():
  src = StubFrameSource(
    name="top", fps=10, image_size=(320, 240),
    frame_factory=make_disk_frame_factory(blob_predicate=lambda i: True))
  ui = StubUi(key_script=["space"])
  clips = register_markers(src, _spec(["start"]), ui=ui)
  assert len(clips) == 1
  c = clips[0]
  assert c.loc_name == "start"
  assert c.frames.shape == (2, 240, 320, 3)
  assert c.detected_fraction >= 0.8


def test_blob_slot_rejected_and_retried():
  # The first clip is all-blank → rejected; the second clip has the marker
  # → accepted. UI delivers SPACE twice (one for the failed slot, one for
  # the retry).
  attempts = {"n": 0}

  def factory(i: int):
    import cv2
    H, W = 240, 320
    img = np.full((H, W, 3), 30, dtype=np.uint8)
    if attempts["n"] >= 1:
      cv2.circle(img, (160, 120), 30, (240, 240, 240), -1)
    return img

  src = StubFrameSource(name="top", fps=10, image_size=(320, 240),
                         frame_factory=factory)
  ui = StubUi(key_script=["space", "space"])

  # Patch register_markers? We need a way to flip the attempt counter
  # between captures. Instead, observe ``_capture_clip`` via the read
  # generator — bump the counter when the first clip finishes (read count
  # crosses fps*duration_s).
  orig_read = src.read
  fps_dur = int(round(10 * 0.2))

  def counting_read():
    img = orig_read()
    if src._i == fps_dur:
      attempts["n"] = 1
    return img
  src.read = counting_read  # type: ignore[method-assign]

  clips = register_markers(src, _spec(["start"]), ui=ui)
  assert len(clips) == 1
  assert clips[0].detected_fraction >= 0.8


def test_off_screen_slot_accepted_when_no_blob():
  src = StubFrameSource(
    name="top", fps=10, image_size=(320, 240),
    frame_factory=make_disk_frame_factory(blob_predicate=lambda i: False))
  ui = StubUi(key_script=["space"])
  clips = register_markers(src, _spec(["off_screen"]), ui=ui)
  assert len(clips) == 1
  assert clips[0].loc_name == "off_screen"
  assert clips[0].detected_fraction == 0.0


def test_off_screen_slot_rejected_when_blob_present():
  # Marker is fully visible → off_screen validation should fail.
  visible = {"v": True}

  def factory(i: int):
    import cv2
    H, W = 240, 320
    img = np.full((H, W, 3), 30, dtype=np.uint8)
    if visible["v"]:
      cv2.circle(img, (160, 120), 30, (240, 240, 240), -1)
    return img

  src = StubFrameSource(name="top", fps=10, image_size=(320, 240),
                         frame_factory=factory)
  # First clip rejected, then user removes the marker (we toggle) and retries.
  ui = StubUi(key_script=["space", "space"])
  fps_dur = int(round(10 * 0.2))
  orig_read = src.read

  def counting_read():
    img = orig_read()
    if src._i == fps_dur:
      visible["v"] = False
    return img
  src.read = counting_read  # type: ignore[method-assign]

  clips = register_markers(src, _spec(["off_screen"]), ui=ui)
  assert len(clips) == 1
  assert clips[0].detected_fraction < 0.2


def test_esc_during_preview_raises():
  src = StubFrameSource(
    name="top", fps=10, image_size=(320, 240),
    frame_factory=make_disk_frame_factory())
  ui = StubUi(key_script=["esc"])
  with pytest.raises(SceneRegistrationAborted):
    register_markers(src, _spec(["start"]), ui=ui)


def test_multiple_locations_captured_in_order():
  # off_screen first (no blob), then start (blob present). Flip on the
  # second slot's first preview-poll. SPACE for each.
  visible = {"v": False}

  def factory(i: int):
    import cv2
    H, W = 240, 320
    img = np.full((H, W, 3), 30, dtype=np.uint8)
    if visible["v"]:
      cv2.circle(img, (160, 120), 30, (240, 240, 240), -1)
    return img

  src = StubFrameSource(name="top", fps=10, image_size=(320, 240),
                         frame_factory=factory)
  fps_dur = int(round(10 * 0.2))

  class FlipUi(StubUi):
    """Flip ``visible`` to True right before delivering SPACE for slot 2."""

    def __init__(self, *a, **kw):
      super().__init__(*a, **kw)
      self.space_count = 0

    def poll_key(self, timeout_ms=1):
      k = super().poll_key(timeout_ms=timeout_ms)
      if k == "space":
        self.space_count += 1
        if self.space_count == 2:
          visible["v"] = True
      return k

  ui = FlipUi(key_script=["space", "space"])
  clips = register_markers(src, _spec(["off_screen", "start"]), ui=ui)
  assert [c.loc_name for c in clips] == ["off_screen", "start"]
  assert clips[0].detected_fraction < 0.2
  assert clips[1].detected_fraction >= 0.8
  _ = fps_dur  # avoid lint
