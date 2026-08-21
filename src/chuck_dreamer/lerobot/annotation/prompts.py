"""Per-episode keyframe prompts for the object pose estimator.

``ObjectPoseEstimator.estimate_episode`` requires a ``first_frame_prompt``
(2-D pixel coordinate, or list of them) because the white object on a
black mat with white fiducials can't be reliably auto-detected.

Prompts are EPISODE-scoped ``object_prompts`` artifacts in the artifact
store: per episode a ``{frame_idx: prompt}`` map (JSON keys are strings).
Each episode's first-frame click lives under frame_idx 0; additional
keyframe clicks (e.g. the last frame for end-anchor) sit under their
episode-relative frame index. ``prompt-episodes`` is the writing tool.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

# Type alias for what SAM2 / the estimator accepts.
Prompt = tuple[int, int] | list[tuple[int, int]]


def _coerce_entry(raw: Any) -> dict[int, "Prompt"]:
  """Normalize a store payload to ``{frame_idx: prompt}``. The legacy
  single-prompt form (a bare ``[u, v]``) is treated as the frame-0 click."""
  if isinstance(raw, dict):
    out: dict[int, "Prompt"] = {}
    for k, v in raw.items():
      try:
        fi = int(k)
      except (TypeError, ValueError):
        continue
      out[fi] = _coerce_prompt(v)
    return out
  return {0: _coerce_prompt(raw)}


def load_keyframe_prompts(store: Any, dataset_id: str,
                          episode_index: int) -> dict[int, Prompt]:
  """The full per-frame prompt map for an episode (empty if unannotated)."""
  from chuck_dreamer.store import for_episode

  scope = for_episode(dataset_id, episode_index)
  if not store.has("object_prompts", scope):
    return {}
  payload, _ = store.get("object_prompts", scope)
  return _coerce_entry(payload)


def save_keyframe_prompt(store: Any, dataset_id: str,
                         episode_index: int, frame_index: int, prompt: Prompt,
                         *, advisory: dict[str, str] | None = None) -> None:
  """Merge a ``(episode, frame_index) -> prompt`` keyframe into the
  episode's ``object_prompts`` artifact.

  ``advisory`` records pre-computed inputs an assisted proposal leaned on
  (calibration hashes) — re-annotation hints, never staleness (spec §9.3).
  """
  from chuck_dreamer.store import annotation_record, for_episode

  scope = for_episode(dataset_id, episode_index)
  existing = load_keyframe_prompts(store, dataset_id, episode_index)
  existing[int(frame_index)] = _coerce_prompt(prompt)
  payload = {str(fi): _prompt_to_json(pr) for fi, pr in sorted(existing.items())}
  store.put("object_prompts", scope,
            payload, annotation_record("object_prompts", scope,
                                       "prompt-episodes", advisory=advisory))


def click_object(
  rgb: np.ndarray, *, banner: str | None = None,
) -> tuple[int, int] | None:
  """Pop an OpenCV window; return the first left-click as ``(u, v)`` in image pixels.

  Hitting ``q`` / ``Q`` / Esc cancels. The window scales the image down
  if it exceeds the 1280x720 budget so the user doesn't get a giant
  fullscreen window for high-res cameras.
  """
  import cv2

  H, W = rgb.shape[:2]
  max_w, max_h = 1280, 720
  banner_h, footer_h = 64, 24
  scale = min(max_w / W, max_h / H, 1.0)
  win_w     = int(W * scale)
  canvas_h  = int(H * scale)
  win_h     = banner_h + canvas_h + footer_h
  win_name  = "object-prompt"

  state: dict[str, tuple[int, int] | bool | None] = {"click": None, "done": False}

  def on_mouse(event, x, y, _flags, _ud):
    if event != cv2.EVENT_LBUTTONDOWN:
      return
    if y < banner_h or y >= banner_h + canvas_h:
      return
    u = x / scale
    v = (y - banner_h) / scale
    if not (0 <= u < W and 0 <= v < H):
      return
    state["click"] = (int(round(u)), int(round(v)))
    state["done"]  = True

  cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
  cv2.setMouseCallback(win_name, on_mouse)
  bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
  if scale != 1.0:
    bgr = cv2.resize(bgr, (win_w, canvas_h), interpolation=cv2.INTER_AREA)

  try:
    while not state["done"]:
      canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
      canvas[:banner_h, :] = (32, 32, 32)
      canvas[banner_h + canvas_h:, :] = (48, 48, 48)
      canvas[banner_h:banner_h + canvas_h, :, :] = bgr
      if banner:
        cv2.putText(canvas, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (230, 230, 230), 1, cv2.LINE_AA)
      cv2.putText(canvas, "click anywhere on the object  |  q skip",
                  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                  (180, 220, 255), 1, cv2.LINE_AA)
      cv2.imshow(win_name, canvas)
      key = cv2.waitKey(16) & 0xFFFF
      if key in (ord('q'), ord('Q'), 27):
        state["done"] = True
        break
  finally:
    cv2.destroyWindow(win_name)
    cv2.waitKey(1)
  click = state["click"]
  return click if isinstance(click, tuple) else None


def _coerce_prompt(raw: Any) -> Prompt:
  """Accept ``[u, v]`` (single click) or ``[[u, v], ...]`` (multi-click)."""
  if (
    isinstance(raw, (list, tuple))
    and len(raw) > 0
    and isinstance(raw[0], (list, tuple))
  ):
    return [(int(p[0]), int(p[1])) for p in raw]
  if isinstance(raw, (list, tuple)) and len(raw) == 2:
    return (int(raw[0]), int(raw[1]))
  raise ValueError(f"object_prompts.json entry is malformed: {raw!r}")


def _prompt_to_json(prompt: Prompt) -> Any:
  if isinstance(prompt, list):
    return [[int(p[0]), int(p[1])] for p in prompt]
  return [int(prompt[0]), int(prompt[1])]


def _has_display() -> bool:
  """Best-effort check that an interactive display is available."""
  if os.environ.get("CHUCK_OL_HEADLESS"):
    return False
  # macOS GUI sessions don't set DISPLAY; trust the platform there.
  import sys
  if sys.platform == "darwin":
    return True
  return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
