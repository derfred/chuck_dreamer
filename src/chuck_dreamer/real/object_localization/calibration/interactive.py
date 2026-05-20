"""Interactive review/edit UIs used by the calibration pipeline.

Everything is OpenCV (one UI stack). Four entry points:

* :func:`review_mat_detection` - auto-then-edit. Drag any colored
  control point to fine-tune the auto detection in place; click
  buttons (or use hotkeys) to Accept, Invalidate + re-annotate,
  Re-run auto, or Quit. The re-annotate path runs
  :func:`annotate_mat_cv2` (9-click flow).
* :func:`annotate_mat_cv2` - 9-click manual annotation.
* :func:`show_overlay` - read-only "press any key to continue" popup
  used by the verify-overlay sign-off step.
* :func:`click_to_seed_point` - single-click popup for the object-
  highlight click-to-seed fallback.

All interactive entry points are no-ops when ``interactive=False`` is
threaded through.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from .mat_detection import (
  MatDetection,
  _ellipse_samples,
  detect_mat_markings,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Shared window / abort state
# --------------------------------------------------------------------------- #


_CV2_WINDOW = "calibration"
_CV2_QUIT = {"v": False}
_BANNER_H = 36
_BUTTON_BAR_H = 44


def reset_popup_state() -> None:
  """Reset the abort flag - call once at the start of a calibration run."""
  _CV2_QUIT["v"] = False


def close_popup_window() -> None:
  """Tear down the shared cv2 window at the end of a run."""
  import cv2
  try:
    cv2.destroyAllWindows()
  except Exception:
    pass
  _CV2_QUIT["v"] = False


# --------------------------------------------------------------------------- #
# Shared layout helpers
# --------------------------------------------------------------------------- #


def _fit_to_window(rgb: np.ndarray, max_w: int = 1280, max_h: int = 760) -> float:
  H, W = rgb.shape[:2]
  return min(max_w / W, max_h / H, 1.0)


def _win_to_image(x: int, y: int, scale: float) -> tuple[int, int]:
  """Convert window coords (banner above the image) to image coords."""
  ix = int(round(x / scale)) if scale < 1.0 else x
  iy_raw = y - _BANNER_H
  iy = int(round(iy_raw / scale)) if scale < 1.0 else iy_raw
  return ix, iy


def _image_to_win(ix: float, iy: float, scale: float) -> tuple[int, int]:
  wx = int(round(ix * scale)) if scale < 1.0 else int(round(ix))
  wy = int(round(iy * scale)) if scale < 1.0 else int(round(iy))
  return wx, wy + _BANNER_H


def _make_button_bar(
  W_win: int, y_top: int,
  specs: list[tuple[str, tuple[int, int, int], str]],
) -> dict:
  """Lay out buttons left-to-right inside a strip of width W_win."""
  margin = 10
  total_w = W_win - 2 * margin
  n = len(specs)
  gap = 6
  bw = (total_w - gap * (n - 1)) // n
  bh = _BUTTON_BAR_H - 12
  y0 = y_top + 6
  y1 = y0 + bh
  buttons: dict = {}
  for i, (label, color, action) in enumerate(specs):
    x0 = margin + i * (bw + gap)
    x1 = x0 + bw
    buttons[label] = (x0, y0, x1, y1, color, action)
  return buttons


def _draw_banner(canvas: np.ndarray, text: str) -> None:
  import cv2
  W = canvas.shape[1]
  cv2.rectangle(canvas, (0, 0), (W, _BANNER_H), (0, 0, 0), -1)
  cv2.putText(canvas, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
              0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_buttons(canvas: np.ndarray, buttons: dict) -> None:
  import cv2
  for label, (x0, y0, x1, y1, color, _action) in buttons.items():
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 40, 40), 1)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    tx = x0 + (x1 - x0 - text_size[0]) // 2
    ty = y0 + (y1 - y0 + text_size[1]) // 2
    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (20, 20, 20), 1, cv2.LINE_AA)


def _hit_test_buttons(buttons: dict, x: int, y: int) -> str | None:
  for label, (x0, y0, x1, y1, _color, _action) in buttons.items():
    if x0 <= x <= x1 and y0 <= y <= y1:
      return label
  return None


# --------------------------------------------------------------------------- #
# Read-only popups (show_overlay, click_to_seed_point)
# --------------------------------------------------------------------------- #


def show_overlay(
  rgb: np.ndarray, title: str, max_w: int = 1280, max_h: int = 760,
) -> Literal["next", "abort"]:
  """Pop a cv2 window with ``rgb``; advance on any key, Esc/q to abort."""
  import cv2
  if _CV2_QUIT["v"]:
    return "abort"
  H, W = rgb.shape[:2]
  scale = min(max_w / W, max_h / H, 1.0)
  disp = cv2.resize(rgb, (int(W * scale), int(H * scale)),
                    interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb
  bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
  Hd, Wd = bgr.shape[:2]
  canvas = np.zeros((_BANNER_H + Hd, Wd, 3), dtype=np.uint8)
  canvas[_BANNER_H:_BANNER_H + Hd, :] = bgr
  _draw_banner(canvas, f"{title}    [any key: next | Esc/q: abort]")
  cv2.namedWindow(_CV2_WINDOW, cv2.WINDOW_AUTOSIZE)
  cv2.imshow(_CV2_WINDOW, canvas)
  key = cv2.waitKey(0) & 0xFF
  if key in (27, ord("q"), ord("Q")):
    _CV2_QUIT["v"] = True
    cv2.destroyAllWindows()
    return "abort"
  return "next"


def click_to_seed_point(
  rgb: np.ndarray, title: str,
) -> tuple[int, int] | None:
  """Pop a cv2 window; return ``(u, v)`` of the user's click, or None on cancel."""
  import cv2
  if _CV2_QUIT["v"]:
    return None
  scale = _fit_to_window(rgb)
  H, W = rgb.shape[:2]
  Wd = int(W * scale) if scale < 1.0 else W
  Hd = int(H * scale) if scale < 1.0 else H

  picked: dict[str, tuple[int, int] | None] = {"v": None}
  cancelled = {"v": False}

  def on_mouse(event, x, y, _flags, _userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
      ix, iy = _win_to_image(x, y, scale)
      if 0 <= ix < W and 0 <= iy < H:
        picked["v"] = (ix, iy)

  cv2.namedWindow(_CV2_WINDOW, cv2.WINDOW_AUTOSIZE)
  cv2.setMouseCallback(_CV2_WINDOW, on_mouse)
  while True:
    disp = cv2.resize(rgb, (Wd, Hd), interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb
    bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
    canvas = np.zeros((_BANNER_H + Hd, Wd, 3), dtype=np.uint8)
    canvas[_BANNER_H:_BANNER_H + Hd, :] = bgr
    _draw_banner(canvas, f"{title}   [left-click the object | Esc/q to skip]")
    cv2.imshow(_CV2_WINDOW, canvas)
    key = cv2.waitKey(20) & 0xFF
    if picked["v"] is not None:
      return picked["v"]
    if key in (27, ord("q"), ord("Q")):
      cancelled["v"] = True
      return None


# --------------------------------------------------------------------------- #
# Overlay renderers (used in both interactive and non-interactive modes)
# --------------------------------------------------------------------------- #


def render_checkerboard_overlay(
  rgb: np.ndarray, corners: np.ndarray, cols: int, rows: int,
) -> np.ndarray:
  """Return ``rgb`` with the detected checkerboard corners drawn on top."""
  import cv2
  bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
  cv2.drawChessboardCorners(bgr, (cols, rows), corners, True)
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_object_highlight_overlay(
  rgb:           np.ndarray,
  bbox_xyxy:     tuple[int, int, int, int] | None,
  centroid_uv:   tuple[int, int] | None,
  world_xy_mm:   tuple[float, float] | None,
  label:         str,
) -> np.ndarray:
  """Draw a bbox + centroid + world-frame (x, y) label on ``rgb``."""
  import cv2
  out = rgb.copy()
  bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
  if bbox_xyxy is not None:
    x0, y0, x1, y1 = bbox_xyxy
    cv2.rectangle(bgr, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 3)
  if centroid_uv is not None:
    cx, cy = centroid_uv
    cv2.circle(bgr, (int(cx), int(cy)), 8, (0, 0, 255), -1)
    if world_xy_mm is not None:
      cv2.putText(
        bgr,
        f"{label}: x={world_xy_mm[0]:+.1f}mm  y={world_xy_mm[1]:+.1f}mm",
        (int(cx) + 12, int(cy) - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
      )
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# OpenCV mat editor + 9-click annotator
# --------------------------------------------------------------------------- #


_MAT_POINT_LABELS = ("line1_near", "line1_far", "line2_near", "line2_far", "circle_center")
_MAT_POINT_COLORS = {
  "line1_near":    (0, 255, 0),
  "line1_far":     (0, 255, 0),
  "line2_near":    (0, 0, 255),
  "line2_far":     (0, 0, 255),
  "circle_center": (0, 165, 255),
}


def _render_mat_canvas(
  rgb: np.ndarray, scale: float, det: MatDetection | None,
  hovered_point: str | None, drag_label: str | None,
  banner_text: str, buttons: dict,
) -> np.ndarray:
  """Compose the editor canvas: banner + image with overlays + button bar."""
  import cv2
  H, W = rgb.shape[:2]
  Wd = int(W * scale) if scale < 1.0 else W
  Hd = int(H * scale) if scale < 1.0 else H
  disp = cv2.resize(rgb, (Wd, Hd), interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb.copy()
  bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
  canvas = np.zeros((_BANNER_H + Hd + _BUTTON_BAR_H, Wd, 3), dtype=np.uint8)
  canvas[_BANNER_H:_BANNER_H + Hd, :] = bgr
  _draw_banner(canvas, banner_text)

  if det is not None:
    p1n = _image_to_win(det.line1_near[0], det.line1_near[1], scale)
    p1f = _image_to_win(det.line1_far[0],  det.line1_far[1],  scale)
    p2n = _image_to_win(det.line2_near[0], det.line2_near[1], scale)
    p2f = _image_to_win(det.line2_far[0],  det.line2_far[1],  scale)
    cv2.line(canvas, p1n, p1f, (0, 255, 0), 2)
    cv2.line(canvas, p2n, p2f, (0, 0, 255), 2)
    samples = _ellipse_samples(det.circle_center, det.circle_axes,
                               det.circle_angle, n=64)
    pts = np.array(
      [_image_to_win(float(s[0]), float(s[1]), scale) for s in samples],
      dtype=np.int32,
    )
    cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    for label in _MAT_POINT_LABELS:
      pt = getattr(det, label) if label != "circle_center" else det.circle_center
      wx, wy = _image_to_win(float(pt[0]), float(pt[1]), scale)
      highlighted = (label == hovered_point or label == drag_label)
      r = 11 if highlighted else 7
      cv2.circle(canvas, (wx, wy), r + 2, (255, 255, 255), 2)
      cv2.circle(canvas, (wx, wy), r, _MAT_POINT_COLORS[label], -1)
      cv2.putText(canvas, label, (wx + 12, wy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                  0.45, _MAT_POINT_COLORS[label], 1, cv2.LINE_AA)

  _draw_buttons(canvas, buttons)
  return canvas


def annotate_mat_cv2(rgb: np.ndarray) -> MatDetection | None:
  """Pure-cv2 9-click annotator. Esc/q cancels, Enter finishes (>=9 clicks)."""
  import cv2
  scale = _fit_to_window(rgb)
  H, W = rgb.shape[:2]
  Wd = int(W * scale) if scale < 1.0 else W
  Hd = int(H * scale) if scale < 1.0 else H

  state: dict = {"clicks": [], "decision": None}

  prompts = [
    "Click 1/9: Line 1 (bottom) FAR endpoint",
    "Click 2/9: Line 1 (bottom) NEAR endpoint",
    "Click 3/9: Line 2 (top) NEAR endpoint",
    "Click 4/9: Line 2 (top) FAR endpoint",
    "Click 5+/9: >=5 points on the circle, then Enter to finish",
  ]

  buttons = _make_button_bar(
    Wd, _BANNER_H + Hd,
    [
      ("[Enter] Finish", (200, 240, 200), "finish"),
      ("[U]ndo last",    (220, 220, 220), "undo"),
      ("[Q]uit",         (180, 180, 240), "quit"),
    ],
  )

  def on_mouse(event, x, y, _flags, _userdata):
    if event != cv2.EVENT_LBUTTONDOWN:
      return
    hit = _hit_test_buttons(buttons, x, y)
    if hit:
      action = buttons[hit][5]
      if action == "finish" and len(state["clicks"]) >= 9:
        state["decision"] = "accept"
      elif action == "undo" and state["clicks"]:
        state["clicks"].pop()
      elif action == "quit":
        state["decision"] = "abort"
      return
    ix, iy = _win_to_image(x, y, scale)
    if 0 <= ix < W and 0 <= iy < H:
      state["clicks"].append((ix, iy))

  def render():
    n = len(state["clicks"])
    prompt = prompts[min(n, len(prompts) - 1)]
    banner = f"Re-annotate mat - {prompt}    [{n}/9 clicks  |  U: undo  |  Q: quit]"
    disp = cv2.resize(rgb, (Wd, Hd), interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb.copy()
    bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
    canvas = np.zeros((_BANNER_H + Hd + _BUTTON_BAR_H, Wd, 3), dtype=np.uint8)
    canvas[_BANNER_H:_BANNER_H + Hd, :] = bgr
    _draw_banner(canvas, banner)

    clicks = state["clicks"]
    # ----- Progressive geometric feedback ---------------------------------- #
    # After click 2: draw line 1 (green) connecting clicks 1 (FAR) and 2 (NEAR).
    if n >= 2:
      p_l1f = _image_to_win(clicks[0][0], clicks[0][1], scale)
      p_l1n = _image_to_win(clicks[1][0], clicks[1][1], scale)
      cv2.line(canvas, p_l1f, p_l1n, (0, 255, 0), 2)
    # After click 4: draw line 2 (red) and a predicted circle center
    # (midpoint of the two NEAR endpoints — clicks 2 and 3).
    if n >= 4:
      p_l2n = _image_to_win(clicks[2][0], clicks[2][1], scale)
      p_l2f = _image_to_win(clicks[3][0], clicks[3][1], scale)
      cv2.line(canvas, p_l2n, p_l2f, (0, 0, 255), 2)
      mid_x = (clicks[1][0] + clicks[2][0]) / 2.0
      mid_y = (clicks[1][1] + clicks[2][1]) / 2.0
      pm = _image_to_win(mid_x, mid_y, scale)
      cv2.drawMarker(canvas, pm, (255, 255, 0), cv2.MARKER_DIAMOND, 18, 2)
      cv2.putText(canvas, "expected circle center",
                  (pm[0] + 14, pm[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                  0.45, (255, 255, 0), 1, cv2.LINE_AA)
    # After click 5+: draw the fitted ellipse through circle points.
    if n >= 4 + 5:
      circle_pts = np.array(clicks[4:], dtype=np.float32).reshape(-1, 1, 2)
      try:
        (ecx, ecy), (eax_a, eax_b), eang = cv2.fitEllipse(circle_pts)
        samples = _ellipse_samples(
          np.array([ecx, ecy]), (eax_a, eax_b), eang, n=64,
        )
        pts_win = np.array(
          [_image_to_win(float(s[0]), float(s[1]), scale) for s in samples],
          dtype=np.int32,
        )
        cv2.polylines(canvas, [pts_win], isClosed=True, color=(0, 165, 255), thickness=2)
        pec = _image_to_win(ecx, ecy, scale)
        cv2.circle(canvas, pec, 4, (0, 165, 255), -1)
      except cv2.error:
        pass
    # ----- Click markers --------------------------------------------------- #
    for i, (ix, iy) in enumerate(clicks):
      wx, wy = _image_to_win(ix, iy, scale)
      if i < 4:
        cv2.line(canvas, (wx - 6, wy - 6), (wx + 6, wy + 6), (0, 255, 255), 2)
        cv2.line(canvas, (wx - 6, wy + 6), (wx + 6, wy - 6), (0, 255, 255), 2)
        cv2.putText(canvas, f"{i + 1}", (wx + 8, wy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
      else:
        cv2.circle(canvas, (wx, wy), 5, (0, 165, 255), -1)
    _draw_buttons(canvas, buttons)
    return canvas

  cv2.namedWindow(_CV2_WINDOW, cv2.WINDOW_AUTOSIZE)
  cv2.setMouseCallback(_CV2_WINDOW, on_mouse)
  while True:
    cv2.imshow(_CV2_WINDOW, render())
    key = cv2.waitKey(16) & 0xFF
    if key in (13, 10) and len(state["clicks"]) >= 9:
      state["decision"] = "accept"
    elif key in (ord("u"), ord("U")) and state["clicks"]:
      state["clicks"].pop()
    elif key in (27, ord("q"), ord("Q")):
      state["decision"] = "abort"
    if state["decision"] is not None:
      break

  if state["decision"] != "accept" or len(state["clicks"]) < 9:
    return None

  pts = np.asarray(state["clicks"], dtype=np.float64)
  line1_far, line1_near, line2_near, line2_far = pts[:4]
  circle_pts = pts[4:]
  ellipse = cv2.fitEllipse(circle_pts.astype(np.float32).reshape(-1, 1, 2))
  (ecx, ecy), (eax_a, eax_b), eang = ellipse
  ccenter = np.array([ecx, ecy], dtype=np.float64)
  return MatDetection(
    line1_near=line1_near, line1_far=line1_far,
    line2_near=line2_near, line2_far=line2_far,
    circle_center=ccenter, circle_axes=(eax_a, eax_b), circle_angle=eang,
    circle_samples_uv=_ellipse_samples(ccenter, (eax_a, eax_b), eang, n=64),
  )


def review_mat_detection(
  rgb:        np.ndarray,
  detection:  MatDetection | None,
) -> tuple[MatDetection | None, str]:
  """Interactive mat-detection review/edit (pure OpenCV).

  Returns ``(detection, source)`` where ``source`` is one of
  ``"auto"``, ``"edited"``, ``"manual"``, or ``"aborted"``.

  Mouse: drag any colored control point to move it. Click a bottom
  button or press the matching hotkey:
    [A]ccept, [N] Invalidate + re-annotate, [R] Re-run auto, [Q]uit.
  """
  import cv2
  scale = _fit_to_window(rgb)
  H, W = rgb.shape[:2]
  Wd = int(W * scale) if scale < 1.0 else W
  Hd = int(H * scale) if scale < 1.0 else H
  hit_radius_sq = (20 / max(scale, 1e-3)) ** 2

  buttons = _make_button_bar(
    Wd, _BANNER_H + Hd,
    [
      ("[A]ccept",                   (200, 240, 200), "accept"),
      ("[N] Invalidate+re-annotate", (0, 220, 245),   "reannotate"),
      ("[R]e-run auto",              (240, 220, 200), "rerun"),
      ("[Q]uit",                     (180, 180, 240), "quit"),
    ],
  )

  state: dict = {
    "detection": detection,
    "source":    "auto" if detection is not None else "missing",
    "decision":  None,
    "drag":      None,
    "hover":     None,
  }

  def find_nearest_point(ix: int, iy: int) -> str | None:
    det = state["detection"]
    if det is None:
      return None
    best_label, best_d = None, float("inf")
    for label in _MAT_POINT_LABELS:
      pt = getattr(det, label) if label != "circle_center" else det.circle_center
      d = (pt[0] - ix) ** 2 + (pt[1] - iy) ** 2
      if d < best_d:
        best_d = d
        best_label = label
    if best_d < hit_radius_sq:
      return best_label
    return None

  def do_accept():
    if state["detection"] is None:
      return
    state["decision"] = "accept"

  def do_reannotate():
    new = annotate_mat_cv2(rgb)
    if new is not None:
      state["detection"] = new
      state["source"]    = "manual"
      state["decision"]  = "accept"
    # If cancelled, stay in the editor.

  def do_rerun():
    state["detection"] = detect_mat_markings(rgb)
    state["source"]    = "auto" if state["detection"] is not None else "missing"

  def do_abort():
    state["decision"] = "abort"

  actions = {
    "accept": do_accept, "reannotate": do_reannotate,
    "rerun": do_rerun, "quit": do_abort,
  }

  def on_mouse(event, x, y, flags, _userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
      hit = _hit_test_buttons(buttons, x, y)
      if hit:
        actions[buttons[hit][5]]()
        return
      ix, iy = _win_to_image(x, y, scale)
      state["drag"] = find_nearest_point(ix, iy)
    elif event == cv2.EVENT_MOUSEMOVE:
      ix, iy = _win_to_image(x, y, scale)
      if state["drag"] is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
        det = state["detection"]
        if det is None:
          return
        new = np.array([float(ix), float(iy)])
        if state["drag"] == "circle_center":
          det.circle_center = new
          det.circle_samples_uv = _ellipse_samples(
            det.circle_center, det.circle_axes, det.circle_angle, n=64)
        else:
          setattr(det, state["drag"], new)
        if state["source"] == "auto":
          state["source"] = "edited"
      else:
        state["hover"] = find_nearest_point(ix, iy)
    elif event == cv2.EVENT_LBUTTONUP:
      state["drag"] = None

  def render():
    if state["detection"] is None:
      banner = ("Auto-detect FAILED - click [N] Invalidate + re-annotate "
                "or [R] Re-run auto. [Q] to abort.")
    else:
      banner = (f"Mat detection ({state['source']}) - drag a colored dot to "
                f"fine-tune, or click a button below.")
    return _render_mat_canvas(
      rgb, scale, state["detection"], state["hover"], state["drag"],
      banner, buttons,
    )

  cv2.namedWindow(_CV2_WINDOW, cv2.WINDOW_AUTOSIZE)
  cv2.setMouseCallback(_CV2_WINDOW, on_mouse)
  while True:
    cv2.imshow(_CV2_WINDOW, render())
    key = cv2.waitKey(16) & 0xFF
    if key in (ord("a"), ord("A")):
      do_accept()
    elif key in (ord("n"), ord("N")):
      do_reannotate()
    elif key in (ord("r"), ord("R")):
      do_rerun()
    elif key in (27, ord("q"), ord("Q")):
      do_abort()
    if state["decision"] is not None:
      break

  if state["decision"] != "accept":
    return None, "aborted"
  return state["detection"], state["source"]


# --------------------------------------------------------------------------- #
# Verify-overlay sign-off
# --------------------------------------------------------------------------- #


def review_verify_overlays(overlays: list[tuple[str, np.ndarray]]) -> bool:
  """Walk the verify overlays in cv2 windows; return True if user accepts.

  Each overlay is shown via :func:`show_overlay` ("any key" advances,
  Esc/q aborts). A final summary popup asks for explicit accept/reject.
  """
  import cv2
  if not overlays:
    return True
  for title, img in overlays:
    if show_overlay(img, f"verify: {title}") == "abort":
      return False
  last_title, last_img = overlays[-1]
  H, W = last_img.shape[:2]
  scale = min(1280 / W, 720 / H, 1.0)
  disp = cv2.resize(last_img, (int(W * scale), int(H * scale)),
                    interpolation=cv2.INTER_AREA) if scale < 1.0 else last_img
  bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
  Hd, Wd = bgr.shape[:2]
  canvas = np.zeros((_BANNER_H + Hd, Wd, 3), dtype=np.uint8)
  canvas[_BANNER_H:_BANNER_H + Hd, :] = bgr
  _draw_banner(canvas,
               "Verify mat geometry - [A] accept & write calibration   "
               "[R] reject (abort)")
  cv2.namedWindow(_CV2_WINDOW, cv2.WINDOW_AUTOSIZE)
  cv2.imshow(_CV2_WINDOW, canvas)
  while True:
    key = cv2.waitKey(0) & 0xFF
    if key in (ord("a"), ord("A")):
      return True
    if key in (ord("r"), ord("R"), 27, ord("q"), ord("Q")):
      return False
