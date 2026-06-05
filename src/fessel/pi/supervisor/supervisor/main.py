"""supervisor entrypoint: runs the control-plane HTTP server.

In production the control plane is HTTPS (terminated here, reachable from
the cluster over Tailscale). For local dev it serves plain HTTP. TLS
material, when present, is configured under the `supervisor.http` section of
the single fessel.yaml (Requirements §6):
  supervisor: { http: {host, port, tls_certfile, tls_keyfile} }
"""

from __future__ import annotations

import logging

import uvicorn
from fessel_shared import load_component_config

from .app import CONFIG_PATH, create_app


def main() -> None:
  logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
  )
  config = load_component_config(CONFIG_PATH, "supervisor")
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
