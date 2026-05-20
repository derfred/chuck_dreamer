"""Interactive cv2 picker for checkerboard calibration frames.

Loops over one dataset at a time. The user navigates the
checkerboard episode with the keyboard, presses ``s`` to draw a
bounding box around the checkerboard via ``cv2.selectROI`` (a
built-in OpenCV widget), and presses Enter to add (frame, bbox) to
the dataset's picks. Picks survive a relaunch via picks.json so
incremental selection is possible.

The ROI drawing is delegated to ``cv2.selectROI`` because the
homemade in-canvas mouse-drag handler had platform-dependent issues
on macOS where MOUSEMOVE events weren't consistently delivered.

Controls (visible in the banner):

  ←  / →        frame -1 / +1
  n  / p        frame +10 / -10
  Home / End    jump to episode start / end
  s             draw bbox via cv2.selectROI (drag a rectangle, then Enter)
  c             clear the current bbox without committing
  Enter         add (current frame, current bbox) as a pick
  Backspace     remove pick at current frame, if any
  Tab           accept dataset and advance to next
  q / Esc       quit the picker (saves picks for the current dataset)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..dataset import episode_bounds, episode_bounds_from_meta, get_frame
from .picks import Pick

logger = logging.getLogger(__name__)

_BANNER_H = 64
_FOOTER_H = 24
_MAX_W    = 1280
_MAX_H    = 720
_WIN_NAME = "checkerboard picker"


@dataclass
class PickerResult:
  picks: list[Pick]
  advance: bool   # True if user pressed Tab/Enter-then-Tab; False on q/Esc


def pick_frames(
  dataset_id: str, camera_key: str, episode_idx: int,
  initial_picks: list[Pick] | None = None,
  start_at: int | None = None,
) -> PickerResult:
  """Open the picker on one dataset; return the user's pick set.

  The expensive ``LeRobotDataset(dataset_id)`` is built lazily on the
  first frame decode — episode bounds and image dimensions come from
  ``LeRobotDatasetMetadata``, which only reads the cached parquet
  sidecars (seconds, not minutes). The cv2 window therefore comes up
  with a "decoding first frame" placeholder while PyAV builds its
  video index in the background of the first ``get_frame`` call.
  """
  import cv2

  from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore

  meta = LeRobotDatasetMetadata(dataset_id)
  try:
    row = meta.episodes[episode_idx]
    fr = int(row["dataset_from_index"])
    to = int(row["dataset_to_index"])
  except (KeyError, IndexError):
    logger.warning("[picker] %s episode %d missing from metadata.",
                   dataset_id, episode_idx)
    return PickerResult(picks=list(initial_picks or []), advance=True)
  if to <= fr:
    logger.warning("[picker] %s episode %d is empty.", dataset_id, episode_idx)
    return PickerResult(picks=list(initial_picks or []), advance=True)

  shape = meta.shapes.get(camera_key)
  if not shape or len(shape) < 2:
    raise RuntimeError(
      f"{dataset_id}: camera {camera_key!r} not in metadata.shapes; "
      f"available: {list(meta.shapes)}")
  Himg, Wimg = int(shape[0]), int(shape[1])

  scale = min(_MAX_W / Wimg, _MAX_H / Himg, 1.0)
  win_w = int(Wimg * scale)
  canvas_h = int(Himg * scale)
  win_h = _BANNER_H + canvas_h + _FOOTER_H

  picks_by_frame: dict[int, Pick] = {p.frame_idx: p for p in (initial_picks or [])}
  cursor = max(fr, min(to - 1, int(start_at if start_at is not None else fr)))

  # Lazy LeRobotDataset construction: the first frame decode pays the
  # PyAV index-build cost (and a download if anything is missing). The
  # picker draws a placeholder until that completes.
  ds_holder: dict = {"ds": None, "loading": False}
  frame_cache: dict[int, np.ndarray] = {}

  def _frame(idx: int) -> np.ndarray | None:
    """Return the frame if available; None if still decoding the first one."""
    if idx in frame_cache:
      return frame_cache[idx]
    if ds_holder["ds"] is None:
      if not ds_holder["loading"]:
        ds_holder["loading"] = True
        logger.info("[picker] %s: opening LeRobotDataset (first decode may take "
                    "a minute while PyAV builds the video index)...", dataset_id)
        import time as _time
        t0 = _time.perf_counter()
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        ds_holder["ds"] = LeRobotDataset(dataset_id)
        logger.info("[picker] %s:   LeRobotDataset built in %.1fs",
                    dataset_id, _time.perf_counter() - t0)
      else:
        return None
    img = get_frame(ds_holder["ds"], idx, camera_key)
    frame_cache[idx] = img
    if len(frame_cache) > 32:
      frame_cache.pop(next(iter(frame_cache)))
    return img

  state: dict = {
    "bbox":    None,            # (x0, y0, x1, y1) in image coords
    "dirty":   True,
    "quit":    False,
    "advance": False,
  }

  def _img_to_win(u: int, v: int) -> tuple[int, int]:
    return int(round(u * scale)), int(round(v * scale + _BANNER_H))

  def _select_roi() -> tuple[int, int, int, int] | None:
    """Open cv2.selectROI on the current frame (full-resolution) and
    return ``(x0, y0, x1, y1)`` in image coords, or None on cancel.

    selectROI opens its own sub-window. It blocks until the user
    presses Enter/Space (accept) or 'c' (cancel) or closes the window.
    """
    img = _frame(cursor)
    if img is None:
      return None
    # Display in BGR so colors look right. selectROI returns (x, y, w, h).
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # showCrosshair=True puts cross-hairs in the box; fromCenter=False
    # makes the first click anchor a corner (more intuitive than
    # centroid-based).
    roi = cv2.selectROI("select bbox (drag, then Enter / Space to accept, c to cancel)",
                        bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("select bbox (drag, then Enter / Space to accept, c to cancel)")
    cv2.waitKey(1)
    x, y, w, h = (int(v) for v in roi)
    if w < 8 or h < 8:
      return None
    return (x, y, x + w, y + h)

  cv2.namedWindow(_WIN_NAME, cv2.WINDOW_AUTOSIZE)

  # First-frame decode pump: draw a "loading" placeholder, force the
  # window to show, then block on the (slow) first decode. After that
  # frame is in the cache, subsequent navigation is instant.
  placeholder = _render(
    frame=None, Wimg=Wimg, Himg=Himg, scale=scale,
    win_w=win_w, canvas_h=canvas_h, win_h=win_h,
    dataset_id=dataset_id, frame_idx=cursor, fr=fr, to=to,
    picks_by_frame=picks_by_frame, state=state, img_to_win=_img_to_win,
    loading=True,
  )
  cv2.imshow(_WIN_NAME, placeholder)
  cv2.waitKey(1)
  _frame(cursor)  # blocks: opens LeRobotDataset + decodes first frame
  state["dirty"] = True

  try:
    while True:
      if state["dirty"]:
        canvas = _render(
          frame=_frame(cursor), Wimg=Wimg, Himg=Himg, scale=scale,
          win_w=win_w, canvas_h=canvas_h, win_h=win_h,
          dataset_id=dataset_id, frame_idx=cursor, fr=fr, to=to,
          picks_by_frame=picks_by_frame, state=state, img_to_win=_img_to_win,
        )
        cv2.imshow(_WIN_NAME, canvas)
        state["dirty"] = False

      key = cv2.waitKey(16) & 0xFFFF
      if key == 0xFFFF:
        continue
      # ROI selection is handled here (not in _handle_key) because it
      # needs access to the current-frame decoder.
      if key in (ord('s'), ord('S')):
        roi = _select_roi()
        if roi is not None:
          state["bbox"] = roi
        state["dirty"] = True
        continue
      handled, new_cursor = _handle_key(key, cursor, fr, to, picks_by_frame, state)
      if not handled:
        continue
      cursor = new_cursor
      state["dirty"] = True
      if state["quit"] or state["advance"]:
        break
  finally:
    cv2.destroyWindow(_WIN_NAME)
    cv2.waitKey(1)

  return PickerResult(picks=list(picks_by_frame.values()),
                      advance=state["advance"])


def _handle_key(key: int, cursor: int, fr: int, to: int,
                picks: dict[int, Pick], state: dict
                ) -> tuple[bool, int]:
  """Returns (handled, new_cursor)."""
  # Arrow keys vary by platform; OpenCV gives us either special codes
  # or fall-through ASCII for the alphabetic aliases below.
  ARROW_LEFT  = (81, 0xBB, 0x250000, 2424832)  # Linux, mac, win variants
  ARROW_RIGHT = (83, 0xBD, 0x270000, 2555904)
  ARROW_UP    = (82, 0xBE, 0x260000, 2490368)
  ARROW_DOWN  = (84, 0xBF, 0x280000, 2621440)
  HOME        = (80, 2359296)
  END         = (87, 2293760)

  if key in (ord('q'), ord('Q'), 27):
    state["quit"] = True
    return True, cursor
  if key == ord('\t'):  # Tab
    state["advance"] = True
    return True, cursor
  if key in (13, 10):  # Enter
    if state["bbox"] is None:
      logger.warning("[picker] no bbox drawn; pick rejected. Drag a box around the board first.")
      return True, cursor
    picks[cursor] = Pick(frame_idx=cursor, bbox=state["bbox"])
    return True, cursor
  if key in (8, 127):  # Backspace / Delete: remove pick at cursor
    picks.pop(cursor, None)
    return True, cursor
  if key in (ord('c'), ord('C')):
    state["bbox"] = None
    return True, cursor

  # Navigation
  if key in (ord('h'),) or key in ARROW_LEFT:
    return True, max(fr, cursor - 1)
  if key in (ord('l'),) or key in ARROW_RIGHT:
    return True, min(to - 1, cursor + 1)
  if key == ord('p'):
    return True, max(fr, cursor - 10)
  if key == ord('n'):
    return True, min(to - 1, cursor + 10)
  if key in (ord('P'),):
    return True, max(fr, cursor - 100)
  if key in (ord('N'),):
    return True, min(to - 1, cursor + 100)
  if key in HOME or key == ord('0'):
    return True, fr
  if key in END or key == ord('$'):
    return True, to - 1

  return False, cursor


def _render(*, frame: np.ndarray | None, Wimg: int, Himg: int, scale: float,
            win_w: int, canvas_h: int, win_h: int,
            dataset_id: str, frame_idx: int, fr: int, to: int,
            picks_by_frame: dict[int, Pick], state: dict, img_to_win,
            loading: bool = False) -> np.ndarray:
  import cv2

  canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
  canvas[:_BANNER_H, :] = (32, 32, 32)
  canvas[_BANNER_H + canvas_h:, :] = (48, 48, 48)

  if frame is None:
    canvas[_BANNER_H:_BANNER_H + canvas_h, :, :] = (16, 16, 16)
    msg = "decoding first frame — PyAV is building the video index, this can take ~1 min"
    cv2.putText(canvas, msg, (20, _BANNER_H + canvas_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)
  else:
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if scale != 1.0:
      bgr = cv2.resize(bgr, (win_w, canvas_h), interpolation=cv2.INTER_AREA)
    canvas[_BANNER_H:_BANNER_H + canvas_h, :, :] = bgr

  # Persisted pick bbox at this frame
  pick_here = picks_by_frame.get(frame_idx)
  if pick_here and pick_here.bbox:
    x0, y0 = img_to_win(pick_here.bbox[0], pick_here.bbox[1])
    x1, y1 = img_to_win(pick_here.bbox[2], pick_here.bbox[3])
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 200, 0), 2)
    cv2.putText(canvas, "PICK", (x0 + 4, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2, cv2.LINE_AA)

  # Pending bbox (drawn via cv2.selectROI, awaiting Enter)
  if state["bbox"] is not None:
    x0, y0 = img_to_win(state["bbox"][0], state["bbox"][1])
    x1, y1 = img_to_win(state["bbox"][2], state["bbox"][3])
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 200, 255), 2)
    cv2.putText(canvas, "PENDING (press Enter)", (x0 + 4, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)

  # Banner
  n_picks = len(picks_by_frame)
  title = f"{dataset_id}  |  frame {frame_idx}  ({frame_idx - fr + 1}/{to - fr})  |  picks: {n_picks}"
  cv2.putText(canvas, title, (10, 24),
              cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
  status = ("PICK SAVED HERE" if pick_here else
            ("BBOX READY (press Enter to commit)" if state["bbox"] else
             "press 's' to draw a bbox, then Enter to commit"))
  cv2.putText(canvas, status, (10, 50),
              cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 200, 255), 1, cv2.LINE_AA)

  # Footer / controls
  controls = "<-/->  +/-1   n/p  +/-10   s draw-bbox   Enter add   Bksp remove   c clear bbox   Tab next dataset   q quit"
  cv2.putText(canvas, controls, (10, win_h - 8),
              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

  return canvas
