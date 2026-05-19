"""Interactive matplotlib annotation for mat markings (fallback)."""

from __future__ import annotations

import logging

import numpy as np

from .mat_detection import MatDetection, _ellipse_samples

logger = logging.getLogger(__name__)


def annotate_mat(rgb: np.ndarray) -> MatDetection:
  """Pop a matplotlib window asking the user to click in a fixed order.

  Click order:
    1. Line 1 (bottom-of-image) far endpoint
    2. Line 1 near endpoint
    3. Line 2 (top-of-image) near endpoint
    4. Line 2 far endpoint
    5..N. >=5 points along the circle boundary

  The user presses Enter when done with circle points.
  """
  import matplotlib.pyplot as plt

  prompts = [
    "Click: Line 1 (bottom) FAR endpoint",
    "Click: Line 1 (bottom) NEAR endpoint",
    "Click: Line 2 (top) NEAR endpoint",
    "Click: Line 2 (top) FAR endpoint",
    "Click: >=5 points along the circle, then press Enter",
  ]

  fig, ax = plt.subplots(figsize=(12, 7))
  ax.imshow(rgb)
  title = ax.set_title(prompts[0])

  clicks: list[tuple[float, float]] = []
  finished = {"done": False}

  def on_click(event):
    if event.xdata is None or event.ydata is None:
      return
    clicks.append((float(event.xdata), float(event.ydata)))
    ax.plot(event.xdata, event.ydata, "rx", markersize=8)
    next_idx = len(clicks)
    if next_idx < 4:
      title.set_text(prompts[next_idx])
    elif next_idx == 4:
      title.set_text(prompts[4])
    fig.canvas.draw_idle()

  def on_key(event):
    if event.key in ("enter", "return") and len(clicks) >= 4 + 5:
      finished["done"] = True
      plt.close(fig)

  fig.canvas.mpl_connect("button_press_event", on_click)
  fig.canvas.mpl_connect("key_press_event", on_key)
  plt.show()

  if not finished["done"] or len(clicks) < 9:
    raise RuntimeError(
      f"annotation aborted (need 4 line endpoints + >=5 circle points, got {len(clicks)})")

  pts = np.asarray(clicks, dtype=np.float64)
  line1_far, line1_near, line2_near, line2_far = pts[:4]
  circle_pts = pts[4:]

  import cv2
  ellipse = cv2.fitEllipse(circle_pts.astype(np.float32).reshape(-1, 1, 2))
  (ecx, ecy), (eax_a, eax_b), eang = ellipse
  ccenter = np.array([ecx, ecy], dtype=np.float64)

  return MatDetection(
    line1_near=line1_near,
    line1_far=line1_far,
    line2_near=line2_near,
    line2_far=line2_far,
    circle_center=ccenter,
    circle_axes=(eax_a, eax_b),
    circle_angle=eang,
    circle_samples_uv=_ellipse_samples(ccenter, (eax_a, eax_b), eang, n=64),
  )
