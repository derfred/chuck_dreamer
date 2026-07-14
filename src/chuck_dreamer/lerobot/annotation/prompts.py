"""Per-episode first-frame prompts for the object pose estimator.

``ObjectPoseEstimator.estimate_episode`` requires a ``first_frame_prompt``
(2-D pixel coordinate, or list of them) because the white object on a
black mat with white fiducials can't be reliably auto-detected. The
LeRobot importer would otherwise have to either pop a click window for
every episode (annoying) or hard-code prompts (wrong).

This module owns a sidecar JSON cache under the existing calibration
cache directory keyed by ``(dataset_id, episode_index)``. The first
import for a dataset pops an OpenCV click window per episode and writes
the result back to the sidecar; subsequent imports read the sidecar and
run non-interactively.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from chuck_dreamer.store import dataset_cache_dir

_SIDECAR_NAME = "object_prompts.json"

# Type alias for what SAM2 / the estimator accepts.
Prompt = tuple[int, int] | list[tuple[int, int]]


def sidecar_path(cache_dir: Path | str, dataset_id: str) -> Path:
  return dataset_cache_dir(cache_dir, dataset_id) / _SIDECAR_NAME


# Sidecar schema (current): per episode, a map of {frame_idx -> prompt}.
# Each episode's first-frame click lives under frame_idx 0; additional
# keyframe clicks (e.g. the last frame for end-anchor) sit under their
# global frame index inside the episode. The legacy schema (per-episode
# single prompt without a frame_idx wrapper) is auto-migrated on read.

def _episode_entry(blob: dict[str, Any], episode_index: int
                   ) -> dict[int, "Prompt"] | None:
  """Normalize the per-episode entry to ``{frame_idx: prompt}`` or None."""
  raw = blob.get(str(episode_index))
  if raw is None:
    return None
  if isinstance(raw, dict):
    out: dict[int, "Prompt"] = {}
    for k, v in raw.items():
      try:
        fi = int(k)
      except (TypeError, ValueError):
        continue
      out[fi] = _coerce_prompt(v)
    return out
  # Legacy single-prompt form: treat as frame 0 click.
  return {0: _coerce_prompt(raw)}


def load_prompt(
  cache_dir: Path | str, dataset_id: str, episode_index: int,
) -> Prompt | None:
  """Return the first-frame (frame 0) cached prompt for an episode."""
  p = sidecar_path(cache_dir, dataset_id)
  if not p.exists():
    return None
  try:
    blob = json.loads(p.read_text())
  except (json.JSONDecodeError, OSError):
    return None
  entry = _episode_entry(blob, episode_index)
  if entry is None:
    return None
  return entry.get(0)


def load_keyframe_prompts(
  cache_dir: Path | str, dataset_id: str, episode_index: int,
) -> dict[int, Prompt]:
  """Return the full per-frame prompt map for an episode (may be empty)."""
  p = sidecar_path(cache_dir, dataset_id)
  if not p.exists():
    return {}
  try:
    blob = json.loads(p.read_text())
  except (json.JSONDecodeError, OSError):
    return {}
  entry = _episode_entry(blob, episode_index)
  return entry or {}


def save_prompt(
  cache_dir: Path | str, dataset_id: str, episode_index: int, prompt: Prompt,
) -> Path:
  """Save a first-frame (frame 0) prompt for an episode. Back-compat
  wrapper around ``save_keyframe_prompt(..., frame_index=0)``.
  """
  return save_keyframe_prompt(cache_dir, dataset_id, episode_index, 0, prompt)


def save_keyframe_prompt(
  cache_dir: Path | str, dataset_id: str,
  episode_index: int, frame_index: int, prompt: Prompt,
) -> Path:
  """Atomically merge a ``(episode, frame_index) -> prompt`` keyframe
  into the sidecar.

  Per-episode entries are normalized into the new ``{frame_idx: prompt}``
  shape on merge; legacy single-prompt entries get migrated transparently.
  """
  p = sidecar_path(cache_dir, dataset_id)
  p.parent.mkdir(parents=True, exist_ok=True)
  blob: dict[str, Any] = {}
  if p.exists():
    try:
      blob = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
      blob = {}

  ep_key = str(episode_index)
  existing = _episode_entry(blob, episode_index) or {}
  existing[int(frame_index)] = _coerce_prompt(prompt)
  blob[ep_key] = {str(fi): _prompt_to_json(pr) for fi, pr in sorted(existing.items())}

  fd, tmp = tempfile.mkstemp(prefix=".prompts-", suffix=".json", dir=str(p.parent))
  try:
    with os.fdopen(fd, "w") as f:
      json.dump(blob, f, indent=2, sort_keys=True)
    os.replace(tmp, p)
  except Exception:
    if os.path.exists(tmp):
      os.unlink(tmp)
    raise
  return p


def prompt_or_click(
  cache_dir: Path | str,
  dataset_id: str,
  episode_index: int,
  first_frame: np.ndarray,
  *,
  banner: str | None = None,
  interactive: bool | None = None,
) -> Prompt | None:
  """Resolve a first-frame prompt: prefer the sidecar, fall back to a click.

  When ``interactive`` is ``False`` (or auto-detected to be False because
  no display is available) and no sidecar entry exists, raise a
  ``RuntimeError`` naming the sidecar path so headless batch runs fail
  loudly instead of hanging on a `cv2` window that has nowhere to draw.

  Returns ``None`` only if the user cancels the click window.
  """
  cached = load_prompt(cache_dir, dataset_id, episode_index)
  if cached is not None:
    return cached

  if interactive is None:
    interactive = _has_display()
  if not interactive:
    raise RuntimeError(
      f"no prompt cached for {dataset_id!r} episode {episode_index} at "
      f"{sidecar_path(cache_dir, dataset_id)} and no interactive display is "
      f"available. Either run an interactive import once to populate the "
      f"sidecar, or write the file manually as "
      f"{{\"{episode_index}\": [u, v]}}.")

  msg = banner or f"{dataset_id}  episode {episode_index}"
  prompt = click_object(first_frame, banner=msg)
  if prompt is None:
    return None
  save_prompt(cache_dir, dataset_id, episode_index, prompt)
  return prompt


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
