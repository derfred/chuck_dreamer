"""video.snapshot.ingest_url_base defaulting: falls back to the shared
webui_base (the config consolidation) unless snapshot: sets its own —
including an explicit "" to disable the freeze-frame push entirely.

VideoApp.__init__ builds an MqttClient (no connect), so it constructs without
GStreamer/a broker; see test_reload.py for the same pattern."""

from video.main import VideoApp


def test_snapshot_defaults_to_webui_base_when_unset():
  app = VideoApp({"storage": {"ssd_path": "/tmp/ssd"}, "webui_base": "http://webui.x:8001"})
  assert app._snapshot_cfg["ingest_url_base"] == "http://webui.x:8001"


def test_snapshot_explicit_url_overrides_webui_base():
  app = VideoApp(
    {
      "storage": {"ssd_path": "/tmp/ssd"},
      "webui_base": "http://webui.x:8001",
      "snapshot": {"ingest_url_base": "http://elsewhere:9000"},
    }
  )
  assert app._snapshot_cfg["ingest_url_base"] == "http://elsewhere:9000"


def test_snapshot_explicit_empty_string_disables_push():
  app = VideoApp(
    {
      "storage": {"ssd_path": "/tmp/ssd"},
      "webui_base": "http://webui.x:8001",
      "snapshot": {"ingest_url_base": ""},
    }
  )
  assert app._snapshot_cfg["ingest_url_base"] == ""


def test_snapshot_no_webui_base_and_no_override_is_empty():
  app = VideoApp({"storage": {"ssd_path": "/tmp/ssd"}})
  assert app._snapshot_cfg["ingest_url_base"] == ""
