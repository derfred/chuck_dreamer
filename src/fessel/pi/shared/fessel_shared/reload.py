"""SIGHUP config reload, shared by the Pi processes (Requirements §6).

`The Pi reloads on SIGHUP after validation.` The hard parts are (a) doing it
safely from a signal handler and (b) "after validation" — a bad edit must leave
the running config intact, never crash the process or apply a half-parsed file.

Design:

  - SIGHUP only sets an `threading.Event`; no real work runs in the handler
    (signal handlers interrupt the main thread between bytecodes, so file I/O +
    object rebuilds there are asking for re-entrancy trouble). A small daemon
    thread waits on the event and does the reload. This works identically
    whether the process's main thread runs a loop (video, uploader) or blocks
    in uvicorn (supervisor) — the reload never touches the main thread.
  - On each fire: re-read the component config, run the process-supplied
    `validate` (raises on a bad config), and only then call `apply`. If read or
    validate raises, the old running config is kept and the error is logged.

Scope is deliberately conservative (the operator chose "tunables only"): each
process's `apply` updates only hot-reloadable tunables (timers, retry counts,
backoff, encoder *values* applied on next spawn). Re-binding open sockets
(MQTT/HTTP), plug→device mappings, or a running encoder is NOT done here —
those are restart-only, and `apply` should log any changed restart-only key so
the operator knows a restart is needed.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable
from typing import Any

from .config import load_component_config

log = logging.getLogger(__name__)

# validate(new_cfg) -> None; raise (any exception) to reject the new config.
ValidateFn = Callable[[dict[str, Any]], None]
# apply(new_cfg) -> None; mutate the live hot-reloadable tunables.
ApplyFn = Callable[[dict[str, Any]], None]


class ConfigReloader:
  """Owns the SIGHUP → validate → apply path for one process.

  `validate` and `apply` are process-specific. `apply` runs only after
  `validate` accepts the freshly-read config; a rejected/unreadable config
  leaves the running process untouched (the §6 "after validation" guarantee).
  """

  def __init__(
    self,
    *,
    path: str,
    component: str,
    validate: ValidateFn,
    apply: ApplyFn,
  ) -> None:
    self._path = path
    self._component = component
    self._validate = validate
    self._apply = apply
    self._event = threading.Event()
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._loop, name="config-reload", daemon=True)

  def install(self) -> None:
    """Register the SIGHUP handler and start the reload worker.

    `signal.signal` only works on the main thread. In production each process
    installs this from its main thread; under a test harness that runs the app
    in a worker thread (Starlette TestClient), signal registration raises
    ValueError — we swallow it and still start the worker, so `reload_once()`
    remains exercisable in tests. SIGHUP just won't be wired in that context,
    which is fine: tests drive the reload directly, not via the signal."""
    try:
      signal.signal(signal.SIGHUP, self._on_sighup)
    except ValueError:
      log.debug("SIGHUP not registered (not main thread); reload worker still active")
    self._thread.start()

  def close(self) -> None:
    self._stop.set()
    self._event.set()  # wake the worker so it can exit

  def _on_sighup(self, _signum, _frame) -> None:  # noqa: ANN001
    # Minimal: just wake the worker. No I/O, no locks, no rebuilds here.
    self._event.set()

  def _loop(self) -> None:
    while not self._stop.is_set():
      self._event.wait()
      if self._stop.is_set():
        return
      self._event.clear()
      self.reload_once()

  def reload_once(self) -> bool:
    """Re-read, validate, and apply once. Returns True if applied, False if the
    new config was rejected/unreadable (old config kept). Synchronous — used by
    the worker and directly by tests."""
    try:
      new_cfg = load_component_config(self._path, self._component)
    except Exception as e:  # noqa: BLE001 — a malformed file must not crash us
      log.error('{"event":"config_reload","outcome":"read_failed","error":"%s"}', e)
      return False
    try:
      self._validate(new_cfg)
    except Exception as e:  # noqa: BLE001 — validation rejects -> keep old
      log.error(
        '{"event":"config_reload","outcome":"invalid","error":"%s"}',
        f"{type(e).__name__}: {e}",
      )
      return False
    try:
      self._apply(new_cfg)
    except Exception:  # noqa: BLE001 — apply itself failing must not crash
      log.exception('{"event":"config_reload","outcome":"apply_failed"}')
      return False
    log.info('{"event":"config_reload","outcome":"applied","component":"%s"}', self._component)
    return True
