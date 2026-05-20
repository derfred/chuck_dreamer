"""OpenCV click-driven UI for annotating the mat fiducials in one frame.

Reference for the contract is ``object_localization_spec.md`` (Mat
annotation UI section). The UI is intentionally dumb — it collects 9+
clicks and runs ``cv2.fitEllipse`` on the circle points. Downstream
geometry (homography / PnP) lives in ``annotate-mat`` so the UI is
reusable across calibration variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..runtime import active
from ..types import MatDetection

_CV2_WINDOW = "calibration"
_BANNER_H   = 36
_BUTTONS_H  = 44
_MAX_W      = 1280
_MAX_H      = 760


def reset_popup_state() -> None:
  """Tear down the shared cv2 window before a new interactive flow."""
  import cv2
  try:
    cv2.destroyWindow(_CV2_WINDOW)
  except cv2.error:
    pass


def close_popup_window() -> None:
  """Tear down the shared cv2 window after an interactive flow."""
  import cv2
  try:
    cv2.destroyWindow(_CV2_WINDOW)
    cv2.waitKey(1)
  except cv2.error:
    pass


@dataclass
class _ButtonRect:
  label: str
  x0: int
  y0: int
  x1: int
  y1: int

  def hit(self, x: int, y: int) -> bool:
    return self.x0 <= x < self.x1 and self.y0 <= y < self.y1


def _fit_to_window(W: int, H: int) -> float:
  return float(min(_MAX_W / W, _MAX_H / H, 1.0))


def _win_to_image(x: int, y: int, scale: float) -> tuple[float, float]:
  u = (x) / scale
  v = (y - _BANNER_H) / scale
  return float(u), float(v)


def _buttons(window_w: int) -> list[_ButtonRect]:
  bar_y0 = _BANNER_H + int(_canvas_h() * 1)  # set after canvas height known
  return []


def _canvas_h() -> int:
  # Filled in dynamically; this helper exists only to make _buttons readable.
  return 0


def annotate_mat_cv2(rgb: np.ndarray) -> Optional[MatDetection]:
  """Run the interactive 9-click annotation flow on ``rgb`` (HxWx3 uint8).

  Returns a populated ``MatDetection`` on accept (Enter / Finish), or
  ``None`` on any cancel path. See the spec for the full state diagram.
  """
  import cv2

  if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
    raise ValueError(f"annotate_mat_cv2 expects HxWx3 uint8 RGB, got "
                     f"shape={rgb.shape} dtype={rgb.dtype}")

  try:
    n_circle_samples = active().mat_circle_samples
  except RuntimeError:
    n_circle_samples = 64

  H_img, W_img = rgb.shape[:2]
  scale = _fit_to_window(W_img, H_img)
  win_w = int(W_img * scale)
  canvas_h = int(H_img * scale)
  win_h = _BANNER_H + canvas_h + _BUTTONS_H

  buttons = [
    _ButtonRect("Finish", 8,                              _BANNER_H + canvas_h + 6, 108, win_h - 6),
    _ButtonRect("Undo",   116,                            _BANNER_H + canvas_h + 6, 196, win_h - 6),
    _ButtonRect("Quit",   204,                            _BANNER_H + canvas_h + 6, 284, win_h - 6),
  ]

  clicks: list[tuple[float, float]] = []
  accepted = {"value": False}
  cancelled = {"value": False}

  def on_mouse(event: int, x: int, y: int, _flags: int, _ud) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
      return
    for b in buttons:
      if b.hit(x, y):
        if b.label == "Finish" and len(clicks) >= 9:
          accepted["value"] = True
        elif b.label == "Undo" and clicks:
          clicks.pop()
        elif b.label == "Quit":
          cancelled["value"] = True
        return
    if y < _BANNER_H or y >= _BANNER_H + canvas_h:
      return
    u, v = _win_to_image(x, y, scale)
    if not (0 <= u < W_img and 0 <= v < H_img):
      return
    clicks.append((u, v))

  cv2.namedWindow(_CV2_WINDOW, cv2.WINDOW_AUTOSIZE)
  cv2.setMouseCallback(_CV2_WINDOW, on_mouse)

  while True:
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    canvas[:_BANNER_H, :] = (40, 40, 40)
    canvas[_BANNER_H + canvas_h:, :] = (60, 60, 60)

    img_disp = cv2.resize(rgb, (win_w, canvas_h),
                          interpolation=cv2.INTER_AREA) if scale != 1.0 else rgb
    img_bgr = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)
    canvas[_BANNER_H:_BANNER_H + canvas_h, :, :] = img_bgr

    _draw_overlays(canvas, clicks, scale, n_circle_samples)
    _draw_banner(canvas, clicks)
    _draw_buttons(canvas, buttons, len(clicks))

    cv2.imshow(_CV2_WINDOW, canvas)
    key = cv2.waitKey(16) & 0xFF
    if key in (13, 10) and len(clicks) >= 9:
      accepted["value"] = True
    elif key in (ord("u"), ord("U")) and clicks:
      clicks.pop()
    elif key in (27, ord("q"), ord("Q")):
      cancelled["value"] = True

    if accepted["value"]:
      break
    if cancelled["value"]:
      return None

  return _finalize(clicks, n_circle_samples)


def _finalize(clicks: list[tuple[float, float]], n_samples: int) -> Optional[MatDetection]:
  import cv2

  if len(clicks) < 9:
    return None
  line_pts = np.asarray(clicks[:4], dtype=np.float64)
  circle_clicks = np.asarray(clicks[4:], dtype=np.float64)
  try:
    (cx, cy), (ax_w, ax_h), angle_deg = cv2.fitEllipse(circle_clicks.astype(np.float32))
  except cv2.error:
    return None

  samples = _sample_ellipse((cx, cy), (ax_w, ax_h), float(angle_deg), n_samples)

  return MatDetection(
    line1_near        = line_pts[1],
    line1_far         = line_pts[0],
    line2_near        = line_pts[2],
    line2_far         = line_pts[3],
    circle_center     = np.array([cx, cy], dtype=np.float64),
    circle_axes       = (float(ax_w), float(ax_h)),
    circle_angle      = float(angle_deg),
    circle_samples_uv = samples,
    circle_clicks     = circle_clicks,
  )


def _sample_ellipse(center: tuple[float, float],
                    axes_full: tuple[float, float],
                    angle_deg: float, n: int) -> np.ndarray:
  """Sample ``n`` evenly-spaced points along the ellipse, starting from
  cv2's major-axis convention.
  """
  cx, cy = center
  ax, ay = axes_full[0] / 2.0, axes_full[1] / 2.0
  theta = np.deg2rad(angle_deg)
  cos_t, sin_t = np.cos(theta), np.sin(theta)
  phis = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
  x_local = ax * np.cos(phis)
  y_local = ay * np.sin(phis)
  u = cx + cos_t * x_local - sin_t * y_local
  v = cy + sin_t * x_local + cos_t * y_local
  return np.stack([u, v], axis=1)


def _draw_overlays(canvas: np.ndarray, clicks: list[tuple[float, float]],
                   scale: float, n_samples: int) -> None:
  import cv2

  def to_win(u: float, v: float) -> tuple[int, int]:
    return int(round(u * scale)), int(round(v * scale + _BANNER_H))

  n = len(clicks)
  for i, (u, v) in enumerate(clicks):
    x, y = to_win(u, v)
    if i < 4:
      cv2.drawMarker(canvas, (x, y), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
      cv2.putText(canvas, str(i + 1), (x + 6, y - 6),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    else:
      cv2.circle(canvas, (x, y), 4, (0, 165, 255), -1)

  if n >= 2:
    cv2.line(canvas, to_win(*clicks[0]), to_win(*clicks[1]), (0, 255, 0), 2)
  if n >= 4:
    cv2.line(canvas, to_win(*clicks[2]), to_win(*clicks[3]), (0, 0, 255), 2)
    mx = (clicks[1][0] + clicks[2][0]) / 2.0
    my = (clicks[1][1] + clicks[2][1]) / 2.0
    mxw, myw = to_win(mx, my)
    cv2.drawMarker(canvas, (mxw, myw), (0, 255, 255), cv2.MARKER_DIAMOND, 14, 2)
    cv2.putText(canvas, "expected circle center", (mxw + 8, myw - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

  if n >= 9:
    try:
      circle_clicks = np.asarray(clicks[4:], dtype=np.float32)
      (cx, cy), (ax_w, ax_h), angle_deg = cv2.fitEllipse(circle_clicks)
      poly = _sample_ellipse((cx, cy), (ax_w, ax_h), float(angle_deg), n_samples)
      pts_win = np.array([to_win(u, v) for u, v in poly], dtype=np.int32)
      cv2.polylines(canvas, [pts_win], isClosed=True, color=(0, 165, 255), thickness=2)
      cwx, cwy = to_win(cx, cy)
      cv2.circle(canvas, (cwx, cwy), 4, (0, 165, 255), -1)
    except cv2.error:
      pass


def _draw_banner(canvas: np.ndarray, clicks: list[tuple[float, float]]) -> None:
  import cv2
  n = len(clicks)
  prompts = [
    "Click 1/9: Line 1 (bottom) FAR endpoint",
    "Click 2/9: Line 1 (bottom) NEAR endpoint",
    "Click 3/9: Line 2 (top) NEAR endpoint",
    "Click 4/9: Line 2 (top) FAR endpoint",
  ]
  if n < 4:
    msg = prompts[n]
  elif n < 9:
    msg = f"Click {n+1}/9: at least 5 points on the circle, then Enter"
  else:
    msg = f"Have {n} clicks. Enter / Finish to accept, Undo to revise, Quit to cancel."
  cv2.putText(canvas, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
              0.6, (220, 220, 220), 1, cv2.LINE_AA)


def _draw_buttons(canvas: np.ndarray, buttons: list[_ButtonRect], n_clicks: int) -> None:
  import cv2
  for b in buttons:
    if b.label == "Finish":
      enabled = n_clicks >= 9
    elif b.label == "Undo":
      enabled = n_clicks >= 1
    else:
      enabled = True
    color = (90, 140, 90) if enabled else (60, 60, 60)
    cv2.rectangle(canvas, (b.x0, b.y0), (b.x1, b.y1), color, -1)
    cv2.rectangle(canvas, (b.x0, b.y0), (b.x1, b.y1), (220, 220, 220), 1)
    text_color = (255, 255, 255) if enabled else (160, 160, 160)
    cv2.putText(canvas, b.label, (b.x0 + 10, b.y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
