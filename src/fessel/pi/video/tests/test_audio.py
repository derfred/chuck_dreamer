"""Audio spike-detection core tests (V5.6).

Pure logic over (rms_db, peak_db, now) triples — no GStreamer. The injected
`now` makes the min-duration and rolling-window behaviour deterministic.
"""

from fessel_schemas import AnomalyType

from video.audio import AudioAnalyzer, AudioConfig, audio_config_from_dict


def _cfg(**kw) -> AudioConfig:
  base = dict(
    rolling_window_seconds=5.0,
    spike_threshold_db=-15.0,
    spike_min_duration_ms=100,
    spike_baseline_gap_db=20.0,
  )
  base.update(kw)
  return AudioConfig(**base)


def test_spike_requires_min_duration():
  aa = AudioAnalyzer(_cfg(spike_min_duration_ms=200))
  # Seed a quiet baseline so the gap is satisfied later.
  for i in range(5):
    aa.process_level(rms_db=-50.0, peak_db=-48.0, now=float(i) * 0.1)
  # A loud sample at t=1.0 — over threshold but under min-duration so far.
  ev = aa.process_level(rms_db=-5.0, peak_db=-3.0, now=1.0)
  assert ev == []
  # Still loud, but only 100ms in (< 200ms required).
  ev = aa.process_level(rms_db=-5.0, peak_db=-3.0, now=1.1)
  assert ev == []
  # Now 200ms of sustained loudness -> spike.
  ev = aa.process_level(rms_db=-5.0, peak_db=-3.0, now=1.2)
  assert [t for t, _ in ev] == [AnomalyType.audio_spike]
  payload = ev[0][1]
  assert payload["peak_db"] == -3.0
  assert payload["duration_ms"] >= 200


def test_sustained_loud_environment_does_not_spike():
  # Baseline median is loud, so peaks don't clear the baseline gap.
  aa = AudioAnalyzer(_cfg(spike_baseline_gap_db=20.0))
  for i in range(20):
    # RMS -10 (over the -15 threshold) but peak only -8 -> gap = -8 - (-10) = 2dB.
    ev = aa.process_level(rms_db=-10.0, peak_db=-8.0, now=float(i) * 0.1)
    assert ev == []


def test_spike_then_cleared():
  aa = AudioAnalyzer(_cfg(spike_min_duration_ms=0))
  for i in range(5):
    aa.process_level(rms_db=-50.0, peak_db=-48.0, now=float(i) * 0.1)
  # Loud -> immediate spike (min_duration_ms=0).
  ev = aa.process_level(rms_db=-4.0, peak_db=-2.0, now=1.0)
  assert [t for t, _ in ev] == [AnomalyType.audio_spike]
  # Still loud -> no re-fire (debounced by the latched flag).
  ev = aa.process_level(rms_db=-4.0, peak_db=-2.0, now=1.1)
  assert ev == []
  # Back to quiet -> cleared.
  ev = aa.process_level(rms_db=-50.0, peak_db=-48.0, now=1.2)
  assert [t for t, _ in ev] == [AnomalyType.audio_spike_cleared]


def test_latest_rms_and_baseline_track_window():
  aa = AudioAnalyzer(_cfg(rolling_window_seconds=1.0))
  aa.process_level(rms_db=-40.0, peak_db=-38.0, now=0.0)
  aa.process_level(rms_db=-20.0, peak_db=-18.0, now=0.5)
  assert aa.latest_rms_db == -20.0
  # At t=2.0, both prior samples are older than the 1s window -> evicted.
  aa.process_level(rms_db=-30.0, peak_db=-28.0, now=2.0)
  assert aa.baseline_db() == -30.0


def test_config_from_dict():
  cfg = audio_config_from_dict(
    {
      "rolling_window_seconds": 3,
      "spike_threshold_db": -12,
      "spike_min_duration_ms": 150,
      "spike_baseline_gap_db": 18,
    }
  )
  assert cfg.rolling_window_seconds == 3.0
  assert cfg.spike_threshold_db == -12.0
  assert cfg.spike_min_duration_ms == 150
  assert cfg.spike_baseline_gap_db == 18.0
