"""uploader process entrypoint (V4.7 / V5.5.1).

A sibling of `video` (the spec leaves the process boundary open; a separate
process isolates upload failures from the capture pipeline). It watches the SSD
upload_queue, ships flagged recordings to webui-backend's tailnet-only
recording-ingest endpoint over HTTPS with bounded retries/backoff, updates each
recording's metadata.json + publishes per-recording progress and periodic
backlog gauges to the Pi-internal MQTT bus (which supervisor bridges to
Prometheus in Slice 7).

Slice 5.5: the Pi holds NO cluster-store credentials. Auth to the ingest
endpoint is by Tailscale identity at the network layer; the uploader is an
HTTPS client only (no MinIO SDK, no S3 keys).

Config — the `uploader:` subtree of the single fessel.yaml (Requirements §6),
plus the shared `mqtt`/`storage` sections, flattened by load_component_config:
  mqtt:     {host, port}             # shared
  storage:  {ssd_path}               # shared; default /mnt/ssd (same mount as video)
  uploader: {driver, ingest_url_base, timeout_s, verify_tls}
  upload:   {backoff_s: [..], poll_interval_s, backlog_interval_s}
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time

from fessel_schemas import UploadGate
from fessel_shared import ConfigReloader, MqttClient, Storage, load_component_config
from fessel_shared import topics

from .core import DEFAULT_BACKOFF_S, Uploader
from .ingest import build_ingest_client

log = logging.getLogger(__name__)

# Single Pi config (Requirements §6); this process reads its `uploader:` subtree
# plus the shared `mqtt`/`storage` sections. FESSEL_CONFIG overrides the path.
CONFIG_PATH = os.environ.get("FESSEL_CONFIG", "/etc/fessel/fessel.yaml")
DEFAULT_POLL_INTERVAL_S = 10.0
DEFAULT_BACKLOG_INTERVAL_S = 30.0


def _ingest_config_with_env(cfg: dict) -> dict:
  """Overlay env-var overrides on the `uploader` ingest config block. The
  architecture (§5.5) injects deploy-specific values via environment; this lets
  the integration env point the test-Pi's uploader at the in-cluster webui
  Service (plain HTTP) and the `http` driver, rather than the baked-in `fake`
  the fessel.test.yaml defaults to.

  Env keys (all optional): FESSEL_INGEST_{DRIVER,URL_BASE,VERIFY_TLS}.
  VERIFY_TLS is parsed as a bool ("true"/"1"/"yes")."""
  out = dict(cfg)
  env = os.environ
  for key, env_name in (
    ("driver", "FESSEL_INGEST_DRIVER"),
    ("ingest_url_base", "FESSEL_INGEST_URL_BASE"),
  ):
    val = env.get(env_name)
    if val is not None:
      out[key] = val
  verify = env.get("FESSEL_INGEST_VERIFY_TLS")
  if verify is not None:
    out["verify_tls"] = verify.lower() in ("true", "1", "yes")
  return out


class UploaderApp:
  def __init__(self, config: dict) -> None:
    mqtt_cfg = config.get("mqtt", {})
    self._mqtt = MqttClient(
      client_id="uploader",
      host=mqtt_cfg.get("host", "127.0.0.1"),
      port=mqtt_cfg.get("port", 1883),
    )
    storage_cfg = config.get("storage", {})
    self._storage = Storage(storage_cfg.get("ssd_path", "/mnt/ssd"))
    # Boot snapshots of the restart-only sections, so a SIGHUP reload can detect
    # (and warn about) a change to something it cannot hot-apply (§2.13).
    self._boot_mqtt = config.get("mqtt")
    self._boot_storage = storage_cfg or None
    # _boot_ingest is set below from the env-resolved uploader config.

    upload_cfg = config.get("upload", {})
    self._poll_interval = float(upload_cfg.get("poll_interval_s", DEFAULT_POLL_INTERVAL_S))
    self._backlog_interval = float(
      upload_cfg.get("backlog_interval_s", DEFAULT_BACKLOG_INTERVAL_S)
    )

    ingest_cfg = _ingest_config_with_env(config.get("uploader", {}))
    self._boot_ingest = ingest_cfg
    client = build_ingest_client(ingest_cfg)
    self._uploader = Uploader(
      storage=self._storage,
      client=client,
      publish=self._mqtt.publish,
      backoff_s=upload_cfg.get("backoff_s") or DEFAULT_BACKOFF_S,
    )
    self._stop = threading.Event()
    # §2.12 upload gate. Default allowed=True: until the coordinator's retained
    # gate arrives (it publishes immediately on start, so the window is tiny),
    # draining the queue is the data-safe default — uploads resuming a beat
    # early under contention is benign (lowest-priority stream; WebRTC self-
    # adapts; the coordinator gates it within one tick).
    self._uploads_allowed = True
    self._gate_lock = threading.Lock()

  def _on_gate(self, _topic: str, gate: UploadGate) -> None:
    with self._gate_lock:
      self._uploads_allowed = gate.uploads_allowed
    log.info(
      '{"event":"upload_gate_recv","uploads_allowed":%s,"reason":"%s"}',
      "true" if gate.uploads_allowed else "false",
      gate.reason,
    )

  def _allowed(self) -> bool:
    with self._gate_lock:
      return self._uploads_allowed

  # --- SIGHUP reload (§2.13): only the hot-reloadable `upload` tunables -------

  @staticmethod
  def validate_config(cfg: dict) -> None:
    """Reject a config that would break the loop's invariants (intervals must be
    positive numbers; backoff, if present, a non-empty list of numbers). Raises
    on a bad config so the reloader keeps the running one."""
    up = cfg.get("upload", {})
    for key in ("poll_interval_s", "backlog_interval_s"):
      if key in up and not (isinstance(up[key], (int, float)) and up[key] > 0):
        raise ValueError(f"upload.{key} must be a positive number")
    bo = up.get("backoff_s")
    if bo is not None and (not isinstance(bo, list) or not bo):
      raise ValueError("upload.backoff_s must be a non-empty list")

  def apply_config(self, cfg: dict) -> None:
    """Apply the hot-reloadable subset (§2.13 conservative scope): the `upload`
    poll/backlog intervals (read live at the loop top) and the retry backoff.
    Restart-only keys (mqtt, storage.ssd_path, uploader.*) are NOT applied here;
    a change to them is logged so the operator knows a restart is needed."""
    up = cfg.get("upload", {})
    self._poll_interval = float(up.get("poll_interval_s", DEFAULT_POLL_INTERVAL_S))
    self._backlog_interval = float(up.get("backlog_interval_s", DEFAULT_BACKLOG_INTERVAL_S))
    self._uploader.set_backoff(up.get("backoff_s") or DEFAULT_BACKOFF_S)
    # Surface restart-only drift (best-effort comparison against boot values).
    if cfg.get("mqtt") != self._boot_mqtt or cfg.get("storage") != self._boot_storage:
      log.warning(
        '{"event":"config_reload","note":"mqtt/storage changed; restart required to apply"}'
      )
    if _ingest_config_with_env(cfg.get("uploader", {})) != self._boot_ingest:
      log.warning(
        '{"event":"config_reload","note":"uploader ingest config changed; restart required to apply"}'
      )

  def run(self) -> None:
    self._storage.ensure_layout()
    self._mqtt.connect()
    # Honour the bandwidth gate (§2.12): suspend uploads while a viewer is
    # present. Retained, so we get the current value on connect.
    self._mqtt.subscribe_model(
      topics.CMD_UPLOAD_GATE, UploadGate.model_validate, self._on_gate, qos=topics.QOS_CMD
    )
    self._mqtt.loop_start()

    signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
    signal.signal(signal.SIGINT, lambda *_: self._stop.set())
    # SIGHUP -> reload the `upload` tunables (§2.13).
    reloader = ConfigReloader(
      path=CONFIG_PATH,
      component="uploader",
      validate=self.validate_config,
      apply=self.apply_config,
    )
    reloader.install()

    last_backlog = 0.0
    while not self._stop.is_set():
      # While the gate denies uploads, skip the pass entirely: markers stay on
      # disk (the durable-queue contract makes "pause" = "don't scan"), so
      # nothing is lost and the queue drains once a viewer leaves.
      outcomes = self._uploader.process_pending() if self._allowed() else []
      # Publish backlog gauges on their own slower cadence (V4.8) regardless of
      # the gate, so a viewer camping long enough to starve uploads is still
      # visible (the Slice-7 >3h alert lands here).
      now = time.monotonic()
      if now - last_backlog >= self._backlog_interval:
        self._uploader.publish_backlog()
        last_backlog = now
      # Wait the poll interval, OR a longer backoff if a retryable failure was
      # seen this pass (the queue can't drain faster than the network allows).
      wait = self._poll_interval
      retrying = [o for o in outcomes if o.state.value == "uploading"]
      if retrying:
        max_attempt = max(o.attempts for o in retrying)
        wait = max(wait, self._uploader.backoff_for(max_attempt))
      self._stop.wait(wait)

    log.info("uploader shutting down")
    reloader.close()
    self._mqtt.loop_stop()
    self._mqtt.disconnect()


def main() -> None:
  logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
  )
  config = load_component_config(CONFIG_PATH, "uploader")
  UploaderApp(config).run()


if __name__ == "__main__":
  main()
