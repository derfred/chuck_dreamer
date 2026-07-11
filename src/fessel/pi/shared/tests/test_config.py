"""load_component_config tests — the single-fessel.yaml -> per-process view
(Requirements §6). Each process gets the shared `mqtt`/`storage` sections plus
its own subtree flattened to the top level, with no cross-component leakage."""

import textwrap

from fessel_shared import load_component_config, load_yaml_config

FESSEL_YAML = textwrap.dedent(
  """
  mqtt:
    host: 127.0.0.1
    port: 1883
  storage:
    ssd_path: /mnt/ssd
  video:
    whip: {endpoint: http://webui:8001/whip/ingest}
    ring: {fps: 30}
  supervisor:
    http: {port: 8443}
    control: {wiz: {driver: fake}}
  uploader:
    driver: http
    ingest_url_base: "http://webui.tailnet.ts.net:8001"
    upload: {poll_interval_s: 10}
  """
)


def _write(tmp_path, text):
  p = tmp_path / "fessel.yaml"
  p.write_text(text)
  return p


def test_shared_sections_present_in_every_component(tmp_path):
  p = _write(tmp_path, FESSEL_YAML)
  for comp in ("video", "supervisor", "uploader"):
    cfg = load_component_config(p, comp)
    assert cfg["mqtt"] == {"host": "127.0.0.1", "port": 1883}
    assert cfg["storage"] == {"ssd_path": "/mnt/ssd"}


def test_component_subtree_flattened_to_top_level(tmp_path):
  p = _write(tmp_path, FESSEL_YAML)
  video = load_component_config(p, "video")
  assert video["whip"] == {"endpoint": "http://webui:8001/whip/ingest"}
  assert video["ring"] == {"fps": 30}
  sup = load_component_config(p, "supervisor")
  assert sup["http"] == {"port": 8443}
  assert sup["control"]["wiz"]["driver"] == "fake"
  up = load_component_config(p, "uploader")
  assert up["driver"] == "http"
  assert up["ingest_url_base"].endswith(":8001")
  assert up["upload"]["poll_interval_s"] == 10


def test_no_cross_component_leakage(tmp_path):
  p = _write(tmp_path, FESSEL_YAML)
  video = load_component_config(p, "video")
  # video must not see supervisor/uploader keys (or the raw subtree names).
  for absent in ("control", "http", "ingest_url_base", "upload", "video", "supervisor", "uploader"):
    assert absent not in video, absent


def test_missing_file_is_empty(tmp_path):
  cfg = load_component_config(tmp_path / "nope.yaml", "video")
  assert cfg == {}


def test_missing_component_subtree_yields_shared_only(tmp_path):
  # A fessel.yaml with shared sections but no `uploader:` subtree -> the
  # uploader still gets mqtt/storage (and falls back to runtime defaults for the
  # rest), rather than crashing.
  p = _write(
    tmp_path,
    textwrap.dedent(
      """
      mqtt: {host: 127.0.0.1, port: 1883}
      storage: {ssd_path: /mnt/ssd}
      video: {srt: {host: h}}
      """
    ),
  )
  up = load_component_config(p, "uploader")
  assert up["mqtt"]["host"] == "127.0.0.1"
  assert up["storage"]["ssd_path"] == "/mnt/ssd"
  assert "srt" not in up  # video's key didn't leak


def test_flat_legacy_file_falls_back_to_top_level_keys(tmp_path):
  # A flat, single-component file (no component subtrees) still loads: shared
  # sections + any other top-level keys are returned as this component's. Keeps
  # a hand-written dev/test config working.
  p = _write(
    tmp_path,
    textwrap.dedent(
      """
      mqtt: {host: 127.0.0.1}
      storage: {ssd_path: /tmp/ssd}
      driver: fake
      upload: {poll_interval_s: 2}
      """
    ),
  )
  up = load_component_config(p, "uploader")
  assert up["driver"] == "fake"
  assert up["upload"] == {"poll_interval_s": 2}
  assert up["storage"]["ssd_path"] == "/tmp/ssd"


def test_load_yaml_config_still_available(tmp_path):
  # The raw loader is retained (mosquitto.conf is not YAML, but other callers
  # and tests may want the unmerged document).
  p = _write(tmp_path, FESSEL_YAML)
  raw = load_yaml_config(p)
  assert set(raw) == {"mqtt", "storage", "video", "supervisor", "uploader"}


def test_base_url_placeholder_substituted_everywhere(tmp_path):
  # video (WHIP + snapshot) and uploader all reach the webui pod's tailscale
  # sidecar on one hostname/port (B5.5.7 / recording-ingest consolidation): a
  # single ${webui_base} placeholder, referenced from three subtrees.
  p = _write(
    tmp_path,
    textwrap.dedent(
      """
      webui_base: "http://webui.example.ts.net:8001"
      mqtt: {host: 127.0.0.1, port: 1883}
      storage: {ssd_path: /mnt/ssd}
      video:
        whip: {endpoint: "${webui_base}/whip/ingest"}
        snapshot: {ingest_url_base: "${webui_base}"}
      uploader:
        driver: http
        ingest_url_base: "${webui_base}"
      """
    ),
  )
  video = load_component_config(p, "video")
  assert video["whip"]["endpoint"] == "http://webui.example.ts.net:8001/whip/ingest"
  assert video["snapshot"]["ingest_url_base"] == "http://webui.example.ts.net:8001"
  up = load_component_config(p, "uploader")
  assert up["ingest_url_base"] == "http://webui.example.ts.net:8001"


def test_multiple_base_url_placeholders_substituted_independently(tmp_path):
  # The substitution mechanism itself is generic (any top-level scalar string
  # key), not hardcoded to one name — verify two distinct keys both resolve.
  p = _write(
    tmp_path,
    textwrap.dedent(
      """
      webui_base: "http://webui.example.ts.net:8001"
      jetson_base: "http://192.168.1.40:8080"
      mqtt: {host: 127.0.0.1, port: 1883}
      storage: {ssd_path: /mnt/ssd}
      supervisor:
        control: {jetson: {base_url: "${jetson_base}"}}
      video:
        whip: {endpoint: "${webui_base}/whip/ingest"}
      """
    ),
  )
  sup = load_component_config(p, "supervisor")
  assert sup["control"]["jetson"]["base_url"] == "http://192.168.1.40:8080"
  video = load_component_config(p, "video")
  assert video["whip"]["endpoint"] == "http://webui.example.ts.net:8001/whip/ingest"


def test_no_base_url_key_leaves_placeholder_literal(tmp_path):
  # Without a top-level `webui_base:`, ${webui_base} is left untouched rather
  # than crashing — matches the "tolerant of a legacy/flat file" contract.
  p = _write(
    tmp_path,
    textwrap.dedent(
      """
      mqtt: {host: 127.0.0.1}
      storage: {ssd_path: /mnt/ssd}
      uploader: {driver: http, ingest_url_base: "${webui_base}/x"}
      """
    ),
  )
  up = load_component_config(p, "uploader")
  assert up["ingest_url_base"] == "${webui_base}/x"


def test_shipped_example_loads_per_component():
  # Guard: the shipped fessel.yaml.example must produce a sane per-component
  # view for all three processes, so the example can't silently drift from the
  # loader contract (the dpkg seeds this file verbatim).
  from pathlib import Path

  example = (
    Path(__file__).resolve().parents[2] / "deploy" / "fessel.yaml.example"
  )
  assert example.is_file(), example
  video = load_component_config(example, "video")
  assert video["mqtt"]["port"] == 1883
  assert video["storage"]["ssd_path"] == "/mnt/ssd"
  # The ring IS the recording buffer: one `recording` section, no `ring` block.
  assert "ring" not in video
  assert video["recording"]["fps"] == 30 and video["recording"]["max_lookback_seconds"] == 120.0
  assert "control" not in video and "ingest_url_base" not in video
  # webui_base is the single shared source for video's WHIP endpoint, the
  # uploader's (defaulted-in-code) ingest_url_base, and the snapshot push —
  # no per-site ingest_url_base key is written in the shipped example.
  assert video["webui_base"] == "http://webui.tailnet.ts.net:8001"
  assert video["whip"]["endpoint"] == video["webui_base"] + "/whip/ingest"
  assert "ingest_url_base" not in video["snapshot"]
  sup = load_component_config(example, "supervisor")
  assert sup["http"]["port"] == 8443 and sup["control"]["plugs"]["arm"]["address"]
  up = load_component_config(example, "uploader")
  assert up["driver"] == "http" and up["webui_base"] == video["webui_base"]
  assert "ingest_url_base" not in up
  assert up["upload"]["poll_interval_s"] == 10.0
