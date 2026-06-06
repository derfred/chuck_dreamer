"""video SIGHUP reload tests (§2.13): ring/recording values, next-spawn.

VideoApp.__init__ builds an MqttClient (no connect) and a RecordingStateMachine
(no thread start), so it constructs without GStreamer; we only exercise the
pure config apply/validate, not run()."""

import pytest

from video.main import VideoApp


def _app():
  # Minimal config; defaults fill the rest. No mqtt connect, no ring spawn.
  return VideoApp({"storage": {"ssd_path": "/tmp/ssd"}})


def test_apply_config_updates_ring_and_recording_values():
  app = _app()
  app.apply_config(
    {
      "ring": {"resolution": "1920x1080", "fps": 25, "bitrate_bps": 3_000_000},
      "recording": {"bitrate_bps": 9_000_000, "start_timeout_s": 20.0},
    }
  )
  # New ring values are stored (applied on next ring spawn).
  assert app._ring_cfg["resolution"] == "1920x1080"
  assert app._ring_cfg["fps"] == 25
  assert app._ring_cfg["bitrate_bps"] == 3_000_000
  # Defaults still fill keys the reload didn't set.
  assert app._ring_cfg["segment_seconds"] == 2
  # Recording values + the SM's start-timeout.
  assert app._rec_cfg["bitrate_bps"] == 9_000_000
  assert app._rec_sm._start_timeout_s == 20.0


def test_apply_config_recording_takes_effect_on_next_spawn():
  # The recording factory reads self._rec_cfg live, so a reloaded bitrate is
  # used by the NEXT recording without touching any in-flight one.
  app = _app()
  assert app._rec_cfg["bitrate_bps"] == 8_000_000  # default
  app.apply_config({"recording": {"bitrate_bps": 12_000_000}})
  assert app._rec_cfg["bitrate_bps"] == 12_000_000


def test_validate_config_rejects_bad_values():
  with pytest.raises(ValueError):
    VideoApp.validate_config({"ring": {"resolution": "not-a-res"}})
  with pytest.raises(ValueError):
    VideoApp.validate_config({"ring": {"fps": 0}})
  with pytest.raises(ValueError):
    VideoApp.validate_config({"recording": {"bitrate_bps": -1}})
  with pytest.raises(ValueError):
    VideoApp.validate_config({"recording": {"start_timeout_s": 0}})
  # A good config validates clean.
  VideoApp.validate_config(
    {"ring": {"resolution": "1280x720", "fps": 30, "bitrate_bps": 2_000_000}}
  )


# --- Slice 5: vision/audio/anomaly tunables reload ---------------------------


def test_apply_config_updates_vision_audio_tunables():
  app = _app()
  app.apply_config(
    {
      "vision": {
        "safe_box": {"x": 10, "y": 20, "w": 100, "h": 80},
        "safe_box_threshold": 0.09,
        "publish_summary_hz": 2,
      },
      "audio": {"spike_threshold_db": -12, "publish_level_hz": 4},
      "anomaly": {"before_seconds": 15, "after_seconds": 45, "auto_upload": True},
    }
  )
  # Vision detector picked up the new safe box + threshold.
  assert app._vision_analyzer._cfg.safe_box is not None
  assert app._vision_analyzer._cfg.safe_box_threshold == 0.09
  # Audio detector picked up the new spike threshold.
  assert app._audio_analyzer._cfg.spike_threshold_db == -12.0
  # Anomaly window + auto-upload applied to the recorder.
  assert app._recorder._window.before_seconds == 15.0
  assert app._recorder._window.after_seconds == 45.0
  assert app._recorder._auto_upload is True


def test_validate_config_rejects_malformed_safe_box():
  with pytest.raises(ValueError):
    # w must be > 0 (SafeBox pydantic constraint).
    VideoApp.validate_config({"vision": {"safe_box": {"x": 0, "y": 0, "w": 0, "h": 10}}})
  # A good safe box validates clean.
  VideoApp.validate_config({"vision": {"safe_box": {"x": 0, "y": 0, "w": 10, "h": 10}}})
