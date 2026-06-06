"""Audio analysis core (V5.6): RMS level metering + spike detection.

Pure logic, GStreamer-free: `AudioAnalyzer.process_level` takes one `level`
message's RMS/peak dBFS pair plus the message's timestamp (monotonic seconds)
and returns any edge event it triggers. The GStreamer bus wiring that pulls
`level` messages and calls this lives in `gst_audio_handle.py`.

A **spike** (plan §V5.6) is when RMS exceeds `threshold_db` for at least
`min_duration_ms`, AND the rolling-window baseline (median of recent RMS) is at
least `baseline_gap_db` below the peak. The baseline gap is what distinguishes a
spike from a sustained loud environment: a constantly-loud room has a high
baseline, so its peaks don't clear the gap. The detector self-clears (emits a
`spike_cleared` when RMS returns near baseline), and does not re-fire while still
spiking (debounced by the latched `_spiking` flag).
"""

from __future__ import annotations

from dataclasses import dataclass

from fessel_schemas import AnomalyType


def _median(xs: list[float]) -> float:
  if not xs:
    return 0.0
  s = sorted(xs)
  n = len(s)
  mid = n // 2
  if n % 2:
    return s[mid]
  return (s[mid - 1] + s[mid]) / 2.0


@dataclass
class AudioConfig:
  """Audio tunables (plan §V5.6)."""

  rolling_window_seconds: float = 5.0
  spike_threshold_db: float = -15.0
  spike_min_duration_ms: int = 100
  spike_baseline_gap_db: float = 20.0


class AudioAnalyzer:
  """Tracks a rolling window of RMS dBFS samples and detects spikes.

  `process_level(rms_db, peak_db, now)` is called per `level` message (~the
  GStreamer level element's interval). `now` is monotonic seconds, injected so
  the detector is deterministic in tests (no wall clock). Returns a list of
  (AnomalyType, details) edge events — at most one per call."""

  def __init__(self, cfg: AudioConfig) -> None:
    self._cfg = cfg
    # Rolling (timestamp, rms_db) window for the baseline median.
    self._window: list[tuple[float, float]] = []
    self._spiking = False
    # When the current over-threshold run began (None when under threshold),
    # used to enforce min_duration_ms before latching a spike.
    self._over_since: float | None = None
    self._peak_db = -120.0  # peak seen during the current over-threshold run

  @property
  def latest_rms_db(self) -> float:
    return self._window[-1][1] if self._window else -120.0

  def update_config(self, cfg: AudioConfig) -> None:
    self._cfg = cfg

  def baseline_db(self) -> float:
    return _median([r for _, r in self._window])

  def process_level(
    self, rms_db: float, peak_db: float, now: float
  ) -> list[tuple[AnomalyType, dict]]:
    # Maintain the rolling window (evict samples older than the window).
    self._window.append((now, rms_db))
    cutoff = now - self._cfg.rolling_window_seconds
    while self._window and self._window[0][0] < cutoff:
      self._window.pop(0)

    events: list[tuple[AnomalyType, dict]] = []
    over = rms_db > self._cfg.spike_threshold_db

    if over:
      if self._over_since is None:
        self._over_since = now
        self._peak_db = peak_db
      else:
        self._peak_db = max(self._peak_db, peak_db)
      duration_ms = round((now - self._over_since) * 1000)
      baseline = self.baseline_db()
      gap_ok = (self._peak_db - baseline) >= self._cfg.spike_baseline_gap_db
      if not self._spiking and duration_ms >= self._cfg.spike_min_duration_ms and gap_ok:
        self._spiking = True
        events.append(
          (
            AnomalyType.audio_spike,
            {
              "peak_db": round(self._peak_db, 1),
              "baseline_db": round(baseline, 1),
              "duration_ms": duration_ms,
            },
          )
        )
    else:
      # Back under threshold. If we were spiking, clear; reset the run tracker.
      if self._spiking:
        self._spiking = False
        events.append(
          (
            AnomalyType.audio_spike_cleared,
            {
              "peak_db": round(self._peak_db, 1),
              "baseline_db": round(self.baseline_db(), 1),
            },
          )
        )
      self._over_since = None
      self._peak_db = -120.0

    return events


def audio_config_from_dict(d: dict) -> AudioConfig:
  """Build an AudioConfig from the `video.audio` config subtree (V5.6)."""
  return AudioConfig(
    rolling_window_seconds=float(d.get("rolling_window_seconds", 5.0)),
    spike_threshold_db=float(d.get("spike_threshold_db", -15.0)),
    spike_min_duration_ms=int(d.get("spike_min_duration_ms", 100)),
    spike_baseline_gap_db=float(d.get("spike_baseline_gap_db", 20.0)),
  )
