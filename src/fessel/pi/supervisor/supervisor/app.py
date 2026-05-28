"""supervisor HTTP control plane (S1.1) + MQTT relay + heartbeat.

For Slice 1 supervisor's only control-plane job is to relay live
activate/deactivate from mediamtx (over Tailscale) onto the Pi-internal
MQTT bus. It is NOT an authorization gate — authorization was settled
upstream at mediamtx's token check.

Endpoints (called by mediamtx's runOnDemand/runOnUnDemand over Tailscale;
authenticated at the network layer by Tailscale identity):

  GET  /healthz
  POST /control/live/activate    body {path, mode}  -> arm/video/cmd/live/activate
  POST /control/live/deactivate  body {path}        -> arm/video/cmd/live/deactivate

No safety state machine yet (that is Slice 6).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fessel_schemas import LiveActivate, LiveDeactivate, mode_from_canonical
from fessel_shared import MqttClient, load_yaml_config
from fessel_shared import topics
from pydantic import BaseModel

log = logging.getLogger(__name__)

CONFIG_PATH = os.environ.get("SUPERVISOR_CONFIG", "/etc/fessel/supervisor.yaml")
HEARTBEAT_INTERVAL_S = 5.0


class ActivateRequest(BaseModel):
  """mediamtx runOnDemand sends mode as the canonical string, not an object."""

  path: str
  mode: str


class DeactivateRequest(BaseModel):
  path: str


class Relay:
  """Owns the MQTT client and the heartbeat thread."""

  def __init__(self, config: dict) -> None:
    mqtt_cfg = config.get("mqtt", {})
    self._mqtt = MqttClient(
      client_id="supervisor",
      host=mqtt_cfg.get("host", "127.0.0.1"),
      port=mqtt_cfg.get("port", 1883),
    )
    self._stop = threading.Event()
    self._hb_thread = threading.Thread(target=self._heartbeat_loop, name="hb", daemon=True)

  def start(self) -> None:
    self._mqtt.connect()
    self._mqtt.loop_start()
    self._hb_thread.start()

  def stop(self) -> None:
    self._stop.set()
    self._mqtt.loop_stop()
    self._mqtt.disconnect()

  def _heartbeat_loop(self) -> None:
    while not self._stop.is_set():
      self._mqtt.publish(
        topics.HEARTBEAT,
        {"from": "supervisor", "ts": time.time()},
        qos=topics.QOS_HEARTBEAT,
      )
      self._stop.wait(HEARTBEAT_INTERVAL_S)

  def publish_activate(self, cmd: LiveActivate) -> None:
    self._mqtt.publish(
      topics.CMD_LIVE_ACTIVATE, cmd.model_dump(), qos=topics.QOS_CMD, retain=False
    )

  def publish_deactivate(self, cmd: LiveDeactivate) -> None:
    self._mqtt.publish(
      topics.CMD_LIVE_DEACTIVATE, cmd.model_dump(), qos=topics.QOS_CMD, retain=False
    )


def create_app(config: dict | None = None) -> FastAPI:
  cfg = config if config is not None else load_yaml_config(CONFIG_PATH)
  relay = Relay(cfg)

  @asynccontextmanager
  async def lifespan(app: FastAPI):  # noqa: ANN001
    relay.start()
    yield
    relay.stop()

  app = FastAPI(title="fessel-supervisor", lifespan=lifespan)
  app.state.relay = relay

  @app.get("/healthz")
  def healthz() -> dict:
    return {"status": "ok"}

  @app.post("/control/live/activate")
  def live_activate(req: ActivateRequest) -> dict:
    try:
      mode = mode_from_canonical(req.mode)
    except ValueError as e:
      raise HTTPException(status_code=400, detail=str(e)) from e
    cmd = LiveActivate(path=req.path, mode=mode)
    log.info("relay activate path=%s mode=%s", cmd.path, cmd.mode)
    relay.publish_activate(cmd)
    return {"relayed": "activate", "path": cmd.path}

  @app.post("/control/live/deactivate")
  def live_deactivate(req: DeactivateRequest) -> dict:
    cmd = LiveDeactivate(path=req.path)
    log.info("relay deactivate path=%s", cmd.path)
    relay.publish_deactivate(cmd)
    return {"relayed": "deactivate", "path": cmd.path}

  return app


# Module-level app for `uvicorn supervisor.app:app`.
app = create_app() if os.environ.get("SUPERVISOR_EAGER_APP") else None
