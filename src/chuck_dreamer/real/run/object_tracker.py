"""Async SAM2-based object tracker for the live run loop.

Holds a :class:`sam2.sam2_image_predictor.SAM2ImagePredictor`, takes a
single click prompt at startup, and re-segments every new frame on a
worker thread. The main control loop reads the most recent ``(u, v)``
centroid via :meth:`latest_uv`, so segmentation latency (~150-250 ms
per call for ``base-plus`` on MPS) is hidden — control runs at full
cadence and the obs lags by 1-2 worker iterations.

SAM2's video predictor isn't used here: it expects a fixed frame list
at init, not a streaming source. The image predictor re-runs the ViT
each frame, which is what costs the latency, but the async worker
makes that latency a one-way obs delay instead of a control-rate floor.

Convention matches sim ``object_uv``: raw ``(u, v)`` pixel coordinates
(col, row), ``NaN`` per axis when no mask is available.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _largest_component_centroid(mask: np.ndarray) -> tuple[float, float] | None:
  """Return the ``(u, v)`` centroid of the mask's largest component."""
  import cv2
  m = mask.astype(np.uint8)
  if m.sum() == 0:
    return None
  num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
  if num <= 1:
    return None
  areas = stats[1:, cv2.CC_STAT_AREA]
  best = int(np.argmax(areas)) + 1
  ys, xs = np.where(labels == best)
  if xs.size == 0:
    return None
  return float(xs.mean()), float(ys.mean())


class ObjectTracker:
  """SAM2 image-predictor wrapped in a worker thread.

  Lifecycle:
    * ``__init__`` — load the SAM2 image predictor (once).
    * ``start(initial_rgb, click_uv)`` — set the prompt, kick off the worker.
    * ``submit_frame(rgb)`` — drop the latest frame in (overwrites any
      pending one; we don't queue — we always want freshest).
    * ``latest_uv()`` — read the most recent centroid; ``[nan, nan]``
      before the first inference completes or when the mask was lost.
    * ``stop()`` — join the worker.

  The "drop-on-arrival" queue (single-slot, overwrite-on-write) is on
  purpose: stale frames are useless for control, and queueing them
  would just inflate latency without improving information freshness.
  """

  def __init__(self, checkpoint: str, device: str) -> None:
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

    self._predictor = SAM2ImagePredictor.from_pretrained(checkpoint, device=device)
    self._device    = device
    self._click_uv: tuple[float, float] | None = None

    self._frame_lock = threading.Lock()
    self._pending_frame: np.ndarray | None = None
    self._frame_event = threading.Event()

    self._uv_lock = threading.Lock()
    self._latest_uv: np.ndarray = np.array([np.nan, np.nan], dtype=np.float32)
    self._inference_count = 0
    self._lost_count = 0

    self._stop_event = threading.Event()
    self._thread: threading.Thread | None = None

  def start(self, initial_rgb: np.ndarray, click_uv: tuple[float, float]) -> None:
    """Latch the prompt and start the worker.

    A first inference is run synchronously on ``initial_rgb`` so
    :meth:`latest_uv` returns a real value on the first control tick
    instead of NaN. After that, the worker takes over.
    """
    if self._thread is not None:
      raise RuntimeError("ObjectTracker already started")
    self._click_uv = (float(click_uv[0]), float(click_uv[1]))
    logger.info("ObjectTracker prompt = (u=%.1f, v=%.1f)", *self._click_uv)

    # Synchronous warm inference so the first control tick has data.
    self._infer_and_store(initial_rgb)

    self._thread = threading.Thread(
      target=self._worker, name="ObjectTracker", daemon=True,
    )
    self._thread.start()

  def submit_frame(self, rgb: np.ndarray) -> None:
    """Hand the worker a new frame (overwrites any pending one)."""
    with self._frame_lock:
      self._pending_frame = rgb
    self._frame_event.set()

  def latest_uv(self) -> np.ndarray:
    """Return the most recent ``(u, v)`` centroid (copy). NaN when unset/lost."""
    with self._uv_lock:
      return self._latest_uv.copy()

  def is_started(self) -> bool:
    """True once :meth:`start` has been called (prompt latched + worker live)."""
    return self._thread is not None

  def stop(self) -> None:
    self._stop_event.set()
    self._frame_event.set()
    if self._thread is not None:
      self._thread.join(timeout=2.0)
      if self._thread.is_alive():
        logger.warning("ObjectTracker worker did not stop within 2s")
      self._thread = None

  def _worker(self) -> None:
    while not self._stop_event.is_set():
      self._frame_event.wait()
      if self._stop_event.is_set():
        break
      with self._frame_lock:
        frame = self._pending_frame
        self._pending_frame = None
        self._frame_event.clear()
      if frame is None:
        continue
      try:
        self._infer_and_store(frame)
      except Exception as e:  # noqa: BLE001 — surface and keep going
        logger.warning("ObjectTracker inference failed: %s (holding last UV)", e)

  def _infer_and_store(self, rgb: np.ndarray) -> None:
    """Run SAM2 on ``rgb`` and update :attr:`_latest_uv`.

    On mask loss we hold the previous value (don't overwrite with NaN
    just because of a transient occlusion) and bump :attr:`_lost_count`
    so the runner can surface the situation.
    """
    assert self._click_uv is not None
    pts = np.array([[self._click_uv[0], self._click_uv[1]]], dtype=np.float32)
    labels = np.ones((1,), dtype=np.int32)
    t0 = time.monotonic()
    self._predictor.set_image(rgb)
    masks, _, _ = self._predictor.predict(
      point_coords=pts, point_labels=labels, multimask_output=False,
    )
    dt_ms = (time.monotonic() - t0) * 1e3
    centroid = _largest_component_centroid(np.asarray(masks[0], dtype=bool))

    if centroid is None:
      with self._uv_lock:
        self._lost_count += 1
        held = self._latest_uv.copy()
      logger.warning(
        "ObjectTracker: empty mask (%.0f ms); holding last uv=%s (lost_count=%d)",
        dt_ms, held.tolist(), self._lost_count,
      )
      return

    with self._uv_lock:
      self._latest_uv = np.asarray(centroid, dtype=np.float32)
      self._inference_count += 1
    if self._inference_count <= 3 or self._inference_count % 20 == 0:
      logger.info(
        "ObjectTracker step %d: uv=(%.1f, %.1f) in %.0f ms",
        self._inference_count, centroid[0], centroid[1], dt_ms,
      )


def click_prompt_uv(initial_bgr: np.ndarray, window_name: str) -> tuple[float, float]:
  """Block on a single mouse click in ``window_name`` and return ``(u, v)``.

  Used at controller startup to seed the SAM2 prompt with the operator's
  pick. Re-uses the runner's existing window so the operator sees the
  same overlays / HUD they've been looking at.
  """
  import cv2

  picked: dict[str, tuple[int, int]] = {}

  def _on_mouse(event, x, y, flags, _userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
      picked["uv"] = (int(x), int(y))

  cv2.setMouseCallback(window_name, _on_mouse)
  hint = initial_bgr.copy()
  cv2.putText(hint, "click the object to track, then press any key",
              (10, hint.shape[0] - 36),
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
  cv2.imshow(window_name, hint)

  while "uv" not in picked:
    if cv2.waitKey(20) == 27:  # ESC aborts
      cv2.setMouseCallback(window_name, lambda *_: None)
      raise KeyboardInterrupt("object-tracker click prompt aborted")
  # Detach the callback so the main loop's key handling stays clean.
  cv2.setMouseCallback(window_name, lambda *_: None)
  return float(picked["uv"][0]), float(picked["uv"][1])
