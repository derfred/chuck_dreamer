"""Vision analysis core tests (V5.2–V5.4).

Pure logic: a tiny fake Mask feeds the analyzer foreground counts directly, so
the activity/safe-box/rest detectors and their debouncing are exercised without
numpy or OpenCV. Mirrors test_pipeline.py's "pure builders, no gi" approach.
"""

from fessel_schemas import AnomalyType, SafeBox

from video.vision import VisionAnalyzer, VisionConfig, vision_config_from_dict


class FakeMask:
  """A flat foreground mask described by a fraction inside vs. outside a box.

  `total` total pixels; `fg_inside` foreground pixels within `box`; `fg_outside`
  foreground pixels outside it. foreground_pixels = inside + outside.
  """

  def __init__(self, total: int, fg_inside: int, fg_outside: int) -> None:
    self._total = total
    self._inside = fg_inside
    self._outside = fg_outside

  @property
  def total_pixels(self) -> int:
    return self._total

  def foreground_pixels(self) -> int:
    return self._inside + self._outside

  def foreground_outside(self, box: SafeBox) -> int:  # noqa: ARG002 — fake ignores geometry
    return self._outside


BOX = SafeBox(x=10, y=10, w=80, h=80)


def _still() -> FakeMask:
  return FakeMask(total=1000, fg_inside=0, fg_outside=0)


def test_activity_score_is_foreground_fraction_with_ema():
  va = VisionAnalyzer(VisionConfig(ema_alpha=0.5))
  r1 = va.process_mask(FakeMask(1000, 200, 0))  # 20% foreground
  assert r1.activity_score == 0.2
  # First frame seeds the EMA to the raw score.
  assert r1.activity_score_ema == 0.2
  r2 = va.process_mask(FakeMask(1000, 0, 0))  # 0% foreground
  # EMA = 0.5*0 + 0.5*0.2 = 0.1
  assert abs(r2.activity_score_ema - 0.1) < 1e-9


def test_safe_box_violation_debounced_then_cleared():
  va = VisionAnalyzer(
    VisionConfig(safe_box=BOX, safe_box_threshold=0.05, safe_box_debounce_frames=4)
  )
  # 10% of pixels foreground OUTSIDE the box (> 5% threshold).
  over = FakeMask(total=1000, fg_inside=0, fg_outside=100)
  events = []
  for _ in range(3):  # 3 of 4 consecutive -> no latch yet
    events += va.process_mask(over, ring_segment_hint="seg-00042.ts").events
  assert events == []
  # The 4th consecutive over-frame latches the violation.
  res = va.process_mask(over, ring_segment_hint="seg-00042.ts")
  kinds = [t for t, _ in res.events]
  assert AnomalyType.safe_box_violation in kinds
  enter_payload = dict(res.events[0][1])
  assert enter_payload["ring_segment_hint"] == "seg-00042.ts"
  assert enter_payload["outside_fraction"] == 0.1
  assert enter_payload["duration_frames"] == 4
  # 4 sustained still frames clear it.
  cleared = []
  for _ in range(4):
    cleared += va.process_mask(_still()).events
  assert AnomalyType.safe_box_cleared in [t for t, _ in cleared]


def test_single_noisy_frame_does_not_trip_safe_box():
  va = VisionAnalyzer(
    VisionConfig(safe_box=BOX, safe_box_threshold=0.05, safe_box_debounce_frames=12)
  )
  over = FakeMask(total=1000, fg_inside=0, fg_outside=100)
  # One spurious frame in a sea of stillness must not latch (a fly across lens).
  res = va.process_mask(over)
  assert res.events == []
  for _ in range(20):
    assert va.process_mask(_still()).events == []


def test_rest_violation_only_when_in_rest():
  va = VisionAnalyzer(VisionConfig(rest_threshold=0.03, rest_debounce_frames=3))
  moving = FakeMask(total=1000, fg_inside=100, fg_outside=0)  # 10% > 3%
  # Not in a rest period: motion is fine, no rest event.
  for _ in range(5):
    assert all(t is not AnomalyType.rest_violation for t, _ in va.process_mask(moving).events)
  # Enter rest; sustained motion now violates.
  va.set_rest(True, "paused")
  events = []
  for _ in range(3):
    events += va.process_mask(moving).events
  kinds = [t for t, _ in events]
  assert AnomalyType.rest_violation in kinds
  payload = next(p for t, p in events if t is AnomalyType.rest_violation)
  assert payload["rest_reason"] == "paused"


def test_leaving_rest_resets_debounce():
  va = VisionAnalyzer(VisionConfig(rest_threshold=0.03, rest_debounce_frames=3))
  moving = FakeMask(total=1000, fg_inside=100, fg_outside=0)
  va.set_rest(True, "between_episodes")
  va.process_mask(moving)  # one hit toward the rest debounce
  va.set_rest(False, None)  # leaving rest resets the debouncer
  va.set_rest(True, "paused")
  # A single moving frame after the reset must not immediately latch.
  res = va.process_mask(moving)
  assert all(t is not AnomalyType.rest_violation for t, _ in res.events)


def test_config_from_dict_parses_safe_box():
  cfg = vision_config_from_dict(
    {
      "ema_alpha": 0.2,
      "safe_box": {"x": 100, "y": 60, "w": 440, "h": 240},
      "safe_box_threshold": 0.07,
      "rest_threshold": 0.02,
    }
  )
  assert cfg.ema_alpha == 0.2
  assert cfg.safe_box == SafeBox(x=100, y=60, w=440, h=240)
  assert cfg.safe_box_threshold == 0.07
  assert cfg.rest_threshold == 0.02
