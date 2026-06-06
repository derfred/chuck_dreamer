"""Vision analysis core (V5.2–V5.4): activity score, safe-box monitoring, and
rest-period detection.

This module is the PURE, GStreamer-free, OpenCV-free core. It operates on a
small `Mask` protocol — total pixel count plus "how many foreground pixels lie
inside / outside a rectangle" — so the algorithm and its debouncing are
unit-testable against a tiny fake mask without numpy or cv2 installed (the same
split as pipeline.py's pure builders vs. the gi-touching handles). The
OpenCV-backed background subtractor + mask production live in
`gst_vision_handle.py`, which calls `VisionAnalyzer.process_mask` per frame.

The three analyses share one foreground mask per frame (the architecture's
"cheap, already running" background-subtraction pipeline, §2.x), computed once:

  - **Activity score** (V5.2): fraction of foreground pixels in the whole frame,
    smoothed with an EMA. The broadest motion signal.
  - **Safe box** (V5.3): fraction of foreground pixels OUTSIDE a configured
    rectangle; a debounced threshold crossing is an anomaly.
  - **Rest** (V5.4): during a rest period, any activity above a (low) rest
    threshold for a debounce window is an anomaly.

Each detector is self-clearing: it emits a `*_violation` on entry and a
`*_cleared` on a sustained return below threshold, so the Slice-6 state machine
sees both edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fessel_schemas import AnomalyType, SafeBox


class Mask(Protocol):
  """A binary foreground mask. The OpenCV implementation wraps a numpy array;
  tests use a pure-Python fake. Only these three queries are needed."""

  @property
  def total_pixels(self) -> int: ...

  def foreground_pixels(self) -> int: ...

  def foreground_outside(self, box: SafeBox) -> int: ...


@dataclass
class FrameResult:
  """What one processed frame produced (returned to the vision thread).

  `events` is the list of edge events (violation/cleared) the frame triggered,
  each a (AnomalyType, details-dict) pair the thread turns into an MQTT publish
  and, for violations, an anomaly-recording trigger.
  """

  activity_score: float  # raw per-frame foreground fraction, [0,1]
  activity_score_ema: float
  events: list[tuple[AnomalyType, dict]]


class _Debouncer:
  """N-consecutive-frames debounce (plan §V5.3: "exceeds threshold for N
  consecutive frames"). A violation latches once `frames` consecutive frames
  were over threshold, and clears once `frames` consecutive frames were under.
  A single noisy frame resets the run, so a fly across the lens never latches
  and a momentary dip never clears a real, sustained violation."""

  def __init__(self, frames: int) -> None:
    self._frames = max(1, frames)
    self._over_run = 0  # consecutive over-threshold frames
    self._under_run = 0  # consecutive under-threshold frames
    self._active = False

  @property
  def active(self) -> bool:
    return self._active

  @property
  def streak(self) -> int:
    """The current consecutive over-threshold run length (for the event's
    `duration_frames`)."""
    return self._over_run

  def update(self, over: bool) -> str | None:
    """Feed one frame's over/under-threshold boolean. Returns "enter" when the
    violation latches, "exit" when it clears, or None on no edge."""
    if over:
      self._over_run += 1
      self._under_run = 0
    else:
      self._under_run += 1
      self._over_run = 0
    if not self._active and self._over_run >= self._frames:
      self._active = True
      return "enter"
    if self._active and self._under_run >= self._frames:
      self._active = False
      return "exit"
    return None


@dataclass
class VisionConfig:
  """Vision tunables (plan §V5.5). All have documented starting points in the
  config example; concrete values need real-system tuning."""

  ema_alpha: float = 0.1
  safe_box: SafeBox | None = None
  safe_box_threshold: float = 0.05
  safe_box_debounce_frames: int = 12
  rest_threshold: float = 0.03
  rest_debounce_frames: int = 30


class VisionAnalyzer:
  """Runs the three analyses over a stream of foreground masks. Pure logic:
  no GStreamer, no OpenCV — `process_mask` takes a Mask and returns a
  FrameResult. Stateful (EMA, debouncers, rest flag) across frames.

  `ring_segment_hint` is supplied per frame by the thread (the most-recent ring
  segment at trigger time); the analyzer stamps it onto violation events so
  V5.7 can lift the right window."""

  def __init__(self, cfg: VisionConfig) -> None:
    self._cfg = cfg
    self._ema = 0.0
    self._seen_frame = False
    self._safe_box = _Debouncer(cfg.safe_box_debounce_frames)
    self._rest = _Debouncer(cfg.rest_debounce_frames)
    self._in_rest = False
    self._rest_reason: str | None = None

  # --- rest-period state, driven by MQTT subscriptions (V5.4) ---------------

  def set_rest(self, in_rest: bool, reason: str | None) -> None:
    """Update whether we're in a rest period. When rest ends, the rest
    debouncer is reset so motion that resumes legitimately doesn't fire."""
    if not in_rest and self._in_rest:
      self._rest = _Debouncer(self._cfg.rest_debounce_frames)
    self._in_rest = in_rest
    self._rest_reason = reason if in_rest else None

  @property
  def in_rest(self) -> bool:
    return self._in_rest

  @property
  def activity_score_ema(self) -> float:
    return self._ema

  def update_config(self, cfg: VisionConfig) -> None:
    """Hot-apply tunables (SIGHUP, §2.13). Debounce-window changes take effect
    by rebuilding the debouncers; an in-flight latched state is reset (a
    conservative, next-window-clean choice)."""
    self._cfg = cfg
    self._safe_box = _Debouncer(cfg.safe_box_debounce_frames)
    self._rest = _Debouncer(cfg.rest_debounce_frames)

  def process_mask(self, mask: Mask, ring_segment_hint: str | None = None) -> FrameResult:
    total = mask.total_pixels
    if total <= 0:
      # Degenerate frame; treat as no activity, no events.
      return FrameResult(activity_score=0.0, activity_score_ema=self._ema, events=[])

    fg = mask.foreground_pixels()
    score = fg / total
    # EMA smoothing (raw per-frame is too noisy for downstream use).
    if not self._seen_frame:
      self._ema = score
      self._seen_frame = True
    else:
      a = self._cfg.ema_alpha
      self._ema = a * score + (1.0 - a) * self._ema

    events: list[tuple[AnomalyType, dict]] = []

    # --- safe box (V5.3): foreground OUTSIDE the box over threshold ----------
    if self._cfg.safe_box is not None:
      outside = mask.foreground_outside(self._cfg.safe_box)
      outside_fraction = outside / total
      edge = self._safe_box.update(outside_fraction > self._cfg.safe_box_threshold)
      if edge == "enter":
        events.append(
          (
            AnomalyType.safe_box_violation,
            {
              "outside_fraction": round(outside_fraction, 4),
              "threshold": self._cfg.safe_box_threshold,
              "duration_frames": self._safe_box.streak,
              "ring_segment_hint": ring_segment_hint,
            },
          )
        )
      elif edge == "exit":
        events.append(
          (
            AnomalyType.safe_box_cleared,
            {
              "outside_fraction": round(outside_fraction, 4),
              "threshold": self._cfg.safe_box_threshold,
            },
          )
        )

    # --- rest period (V5.4): activity during rest over the (low) rest thresh -
    if self._in_rest:
      edge = self._rest.update(score > self._cfg.rest_threshold)
      if edge == "enter":
        events.append(
          (
            AnomalyType.rest_violation,
            {
              "activity_score": round(score, 4),
              "threshold": self._cfg.rest_threshold,
              "rest_reason": self._rest_reason,
              "ring_segment_hint": ring_segment_hint,
            },
          )
        )
      elif edge == "exit":
        events.append(
          (
            AnomalyType.rest_violation_cleared,
            {
              "activity_score": round(score, 4),
              "threshold": self._cfg.rest_threshold,
              "rest_reason": self._rest_reason,
            },
          )
        )

    return FrameResult(activity_score=score, activity_score_ema=self._ema, events=events)


def vision_config_from_dict(d: dict) -> VisionConfig:
  """Build a VisionConfig from the `video.vision` config subtree (V5.5).
  Missing keys fall back to the documented defaults; a present `safe_box`
  rectangle is parsed into a SafeBox (validated by pydantic)."""
  sb = d.get("safe_box")
  safe_box = SafeBox(**sb) if isinstance(sb, dict) else None
  return VisionConfig(
    ema_alpha=float(d.get("ema_alpha", 0.1)),
    safe_box=safe_box,
    safe_box_threshold=float(d.get("safe_box_threshold", 0.05)),
    safe_box_debounce_frames=int(d.get("safe_box_debounce_frames", 12)),
    rest_threshold=float(d.get("rest_threshold", 0.03)),
    rest_debounce_frames=int(d.get("rest_debounce_frames", 30)),
  )
