"""supervisor entrypoint: runs the control-plane HTTP server.

In production the control plane is HTTPS (terminated here, reachable from
the cluster over Tailscale). For local dev it serves plain HTTP. TLS
material, when present, is configured via supervisor.yaml:
  http: {host, port, tls_certfile, tls_keyfile}
"""

from __future__ import annotations

import logging
import os

import uvicorn
from fessel_shared import load_yaml_config

from .app import create_app

CONFIG_PATH = os.environ.get("SUPERVISOR_CONFIG", "/etc/fessel/supervisor.yaml")


def main() -> None:
  logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
  )
  config = load_yaml_config(CONFIG_PATH)
  http_cfg = config.get("http", {})
  app = create_app(config)
  uvicorn.run(
    app,
    host=http_cfg.get("host", "0.0.0.0"),
    port=http_cfg.get("port", 8443),
    ssl_certfile=http_cfg.get("tls_certfile"),
    ssl_keyfile=http_cfg.get("tls_keyfile"),
    log_config=None,
  )


if __name__ == "__main__":
  main()
