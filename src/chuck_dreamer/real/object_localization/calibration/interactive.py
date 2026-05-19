"""Interactive review/edit UIs used by the calibration pipeline.

Three review steps, all matplotlib-based so they share an event loop and
can run on a remote machine over X11:

* :func:`review_checkerboard_frame` — accept / reject / click-fallback
  per checkerboard frame.
* :func:`review_mat_detection` — auto-then-edit: show the detector
  output, let the user accept, drag individual control points,
  re-annotate from scratch, or rerun a different detector.
* :func:`review_verify_overlays` — final sign-off step: shows the
  projected-mat overlays on the empty + object frames; the calibration
  is only written if the user accepts.

All three are no-ops when ``interactive=False`` is threaded through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .mat_detection import (
  MatDetection,
  _ellipse_samples,
  detect_mat_markings,
)

logger = logging.getLogger(__name__)


CheckerboardDecision = Literal["accept", "reject"]


@dataclass
class CheckerboardReviewResult:
  decision: CheckerboardDecision
  corners:  np.ndarray | None    # (rows*cols, 1, 2) when accepted, else None


# --------------------------------------------------------------------------- #
# Checkerboard review
# --------------------------------------------------------------------------- #


def _infer_grid_from_outer_corners(
  outer: np.ndarray,
  cols:  int,
  rows:  int,
) -> np.ndarray:
  """Given the 4 outer board corners, infer the inner-corner grid.

  ``outer`` is shape (4, 2) in this order: top-left, top-right,
  bottom-right, bottom-left (matching the click order in the
  fallback UI). The returned grid is shape (rows*cols, 1, 2)
  matching cv2's findChessboardCorners convention.
  """
  tl, tr, br, bl = outer
  pts = np.zeros((rows * cols, 2), dtype=np.float32)
  for r in range(rows):
    fr = r / max(rows - 1, 1)
    left  = tl + fr * (bl - tl)
    right = tr + fr * (br - tr)
    for c in range(cols):
      fc = c / max(cols - 1, 1)
      pts[r * cols + c] = left + fc * (right - left)
  return pts.reshape(-1, 1, 2)


def _click_chessboard_outer_corners(
  rgb:   np.ndarray,
  cols:  int,
  rows:  int,
) -> np.ndarray | None:
  """Pop a window and have the user click the 4 outer board corners.

  Returns the (rows*cols, 1, 2) interpolated grid, or ``None`` if the
  user closes the window or clicks fewer than 4 points.
  """
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(figsize=(12, 7))
  ax.imshow(rgb)
  title = ax.set_title("Click the 4 OUTER board corners: top-left, top-right, bottom-right, bottom-left. ESC to cancel.")
  clicks: list[tuple[float, float]] = []
  cancelled = {"v": False}

  def on_click(event):
    if event.xdata is None or event.ydata is None:
      return
    clicks.append((float(event.xdata), float(event.ydata)))
    ax.plot(event.xdata, event.ydata, "yx", markersize=12, markeredgewidth=2)
    if len(clicks) == 4:
      title.set_text("Got 4 corners — close the window to apply.")
      plt.close(fig)
    else:
      title.set_text(f"Click corner {len(clicks) + 1}/4 (top-L, top-R, bot-R, bot-L).")
    fig.canvas.draw_idle()

  def on_key(event):
    if event.key == "escape":
      cancelled["v"] = True
      plt.close(fig)

  fig.canvas.mpl_connect("button_press_event", on_click)
  fig.canvas.mpl_connect("key_press_event", on_key)
  plt.show()

  if cancelled["v"] or len(clicks) < 4:
    return None
  outer = np.asarray(clicks[:4], dtype=np.float32)
  return _infer_grid_from_outer_corners(outer, cols, rows)


def review_checkerboard_frame(
  rgb:     np.ndarray,
  corners: np.ndarray | None,
  cols:    int,
  rows:    int,
  frame_label: str,
) -> CheckerboardReviewResult:
  """Show one checkerboard frame; let the user accept/reject/edit.

  Buttons:
    a — accept current (auto) detection
    r — reject this frame (drop from the intrinsic fit)
    e — re-edit: click the 4 outer corners; grid is interpolated
    Esc — same as reject
  """
  import matplotlib.pyplot as plt
  import cv2

  fig, ax = plt.subplots(figsize=(12, 7))
  ax.imshow(rgb)
  status = ax.set_title("")

  def draw_corners(c):
    overlay = rgb.copy()
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawChessboardCorners(bgr, (cols, rows), c, True)
    ax.images[0].set_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

  if corners is not None:
    draw_corners(corners)
    status.set_text(f"{frame_label}: auto-detected. [A]ccept  [R]eject  [E]dit (click outer corners)")
  else:
    status.set_text(f"{frame_label}: NO auto-detection. [E]dit (click outer corners)  [R]eject")

  decision: dict[str, CheckerboardDecision | None] = {"v": None}
  out_corners: dict[str, np.ndarray | None] = {"v": corners}

  def on_key(event):
    if event.key in ("a", "A"):
      if out_corners["v"] is None:
        status.set_text("No corners to accept. [E]dit or [R]eject.")
        fig.canvas.draw_idle()
        return
      decision["v"] = "accept"
      plt.close(fig)
    elif event.key in ("r", "R", "escape"):
      decision["v"] = "reject"
      plt.close(fig)
    elif event.key in ("e", "E"):
      plt.close(fig)
      grid = _click_chessboard_outer_corners(rgb, cols, rows)
      if grid is not None:
        out_corners["v"] = grid
        decision["v"] = "accept"
      else:
        decision["v"] = "reject"

  fig.canvas.mpl_connect("key_press_event", on_key)
  plt.show()
  return CheckerboardReviewResult(
    decision=decision["v"] or "reject",
    corners=out_corners["v"] if decision["v"] == "accept" else None,
  )


# --------------------------------------------------------------------------- #
# Mat detection review
# --------------------------------------------------------------------------- #


_MAT_POINT_LABELS = ("line1_near", "line1_far", "line2_near", "line2_far", "circle_center")


def _draw_mat(ax, rgb: np.ndarray, det: MatDetection, draggable_artists: dict):
  ax.clear()
  ax.imshow(rgb)
  # Lines (lower-image line green, upper red)
  l1, = ax.plot([det.line1_near[0], det.line1_far[0]],
                [det.line1_near[1], det.line1_far[1]], "g-", lw=2)
  l2, = ax.plot([det.line2_near[0], det.line2_far[0]],
                [det.line2_near[1], det.line2_far[1]], "r-", lw=2)
  # Circle (sampled ellipse)
  samples = _ellipse_samples(det.circle_center, det.circle_axes,
                             det.circle_angle, n=64)
  circ, = ax.plot(np.concatenate([samples[:, 0], samples[:1, 0]]),
                  np.concatenate([samples[:, 1], samples[:1, 1]]),
                  "y-", lw=2)
  # Control points (draggable).
  pts = {
    "line1_near":    det.line1_near,
    "line1_far":     det.line1_far,
    "line2_near":    det.line2_near,
    "line2_far":     det.line2_far,
    "circle_center": det.circle_center,
  }
  scatter_artists = {}
  for label, pt in pts.items():
    color = "green" if label.startswith("line1") else ("red" if label.startswith("line2") else "orange")
    sc = ax.scatter([pt[0]], [pt[1]], c=color, s=120, marker="o",
                    edgecolors="white", linewidths=1.5, zorder=5, picker=True)
    ax.annotate(label, (pt[0] + 8, pt[1] - 8), color=color, fontsize=8)
    scatter_artists[label] = sc
  draggable_artists.clear()
  draggable_artists.update({
    "line1": l1, "line2": l2, "circle": circ,
    "points": scatter_artists,
  })


def review_mat_detection(
  rgb:         np.ndarray,
  detection:   MatDetection | None,
) -> tuple[MatDetection | None, str]:
  """Interactive mat-detection review/edit.

  Returns ``(detection, source)`` where ``source`` is one of
  ``"auto"``, ``"edited"``, ``"manual"``, or ``"aborted"`` (the
  returned detection is ``None`` when aborted).

  Keys (printed in the title):
    a — Accept current
    e — Edit (click + drag a control point to move it)
    n — Re-aNnotate from scratch (9-click flow from annotate.py)
    r — Re-run the auto detector
    Esc — Abort (no calibration written)
  """
  import matplotlib.pyplot as plt
  from .annotate import annotate_mat

  state: dict = {
    "detection": detection,
    "source":    "auto" if detection is not None else "missing",
    "decision":  None,
  }

  fig, ax = plt.subplots(figsize=(13, 8))
  artists: dict = {}
  drag: dict = {"label": None}

  def redraw():
    if state["detection"] is None:
      ax.clear()
      ax.imshow(rgb)
      ax.set_title("Auto-detect FAILED. "
                   "[N] re-annotate  [R] re-run auto detect  [Esc] abort")
    else:
      _draw_mat(ax, rgb, state["detection"], artists)
      ax.set_title(
        f"[source={state['source']}]  "
        f"[A]ccept  [E]dit (drag points)  [N] re-annotate  [R] re-run auto  [Esc] abort"
      )
    fig.canvas.draw_idle()

  redraw()

  def find_nearest_point(event):
    if state["detection"] is None or event.xdata is None:
      return None
    det = state["detection"]
    best_label, best_d = None, float("inf")
    for label in _MAT_POINT_LABELS:
      pt = getattr(det, label) if label != "circle_center" else det.circle_center
      d = (pt[0] - event.xdata) ** 2 + (pt[1] - event.ydata) ** 2
      if d < best_d:
        best_d = d
        best_label = label
    if best_d < 30 ** 2:
      return best_label
    return None

  def on_press(event):
    drag["label"] = find_nearest_point(event)

  def on_motion(event):
    if drag["label"] is None or event.xdata is None or state["detection"] is None:
      return
    det = state["detection"]
    new = np.array([float(event.xdata), float(event.ydata)])
    if drag["label"] == "circle_center":
      det.circle_center = new
      det.circle_samples_uv = _ellipse_samples(
        det.circle_center, det.circle_axes, det.circle_angle, n=64)
    else:
      setattr(det, drag["label"], new)
    state["source"] = "edited"
    redraw()

  def on_release(_event):
    drag["label"] = None

  def on_key(event):
    if event.key in ("a", "A"):
      if state["detection"] is None:
        return
      state["decision"] = "accept"
      plt.close(fig)
    elif event.key in ("e", "E"):
      # Edit is the default mouse mode; just print a hint.
      ax.set_title(ax.get_title() + "   [drag a control point]")
      fig.canvas.draw_idle()
    elif event.key in ("n", "N"):
      plt.close(fig)
      try:
        state["detection"] = annotate_mat(rgb)
        state["source"] = "manual"
      except Exception as e:
        logger.warning("re-annotation cancelled: %s", e)
      state["decision"] = "accept" if state["detection"] is not None else None
    elif event.key in ("r", "R"):
      plt.close(fig)
      new_det = detect_mat_markings(rgb)
      state["detection"] = new_det
      state["source"]    = "auto" if new_det is not None else "missing"
      nd, ns = review_mat_detection(rgb, new_det)
      state["detection"] = nd
      state["source"]    = ns
      state["decision"]  = "accept" if nd is not None else "abort"
    elif event.key == "escape":
      state["decision"] = "abort"
      plt.close(fig)

  fig.canvas.mpl_connect("button_press_event", on_press)
  fig.canvas.mpl_connect("motion_notify_event", on_motion)
  fig.canvas.mpl_connect("button_release_event", on_release)
  fig.canvas.mpl_connect("key_press_event", on_key)
  plt.show()

  if state["decision"] in ("abort", None) and state["source"] != "manual":
    return None, "aborted"
  return state["detection"], state["source"]


# --------------------------------------------------------------------------- #
# Verify-overlay sign-off
# --------------------------------------------------------------------------- #


def review_verify_overlays(overlays: list[tuple[str, np.ndarray]]) -> bool:
  """Show all verify overlays on one figure; return True if user accepts."""
  import matplotlib.pyplot as plt
  n = len(overlays)
  if n == 0:
    return True
  cols = min(n, 3)
  rows = (n + cols - 1) // cols
  fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
  axes_flat = axes.flatten() if n > 1 else [axes]
  for ax, (title, img) in zip(axes_flat, overlays):
    ax.imshow(img)
    ax.set_title(title)
    ax.set_axis_off()
  for ax in axes_flat[n:]:
    ax.set_axis_off()
  fig.suptitle("Verify projected mat geometry. [A] accept and write calibration  [R] reject (abort)",
               fontsize=12)
  decision: dict = {"v": None}
  def on_key(event):
    if event.key in ("a", "A"):
      decision["v"] = True
      plt.close(fig)
    elif event.key in ("r", "R", "escape"):
      decision["v"] = False
      plt.close(fig)
  fig.canvas.mpl_connect("key_press_event", on_key)
  plt.show()
  return bool(decision["v"])
