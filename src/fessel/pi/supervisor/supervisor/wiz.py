"""WiZ plug control over the local LAN, send-and-verify (S3.1).

This is the most safety-critical code in Slice 3. WiZ plugs are driven by
pywizlight over **UDP** (port 38899) — an unacknowledged, best-effort
transport. A lost power-off command leaves the actuator in the wrong state
with no indication. For a safety actuator, "sent" is not "done".

Every state change is therefore **send-and-verify**:

  1. Issue the command (turn_off / turn_on).
  2. Read the state back (getPilot).
  3. If the read-back matches the intent -> success.
  4. Otherwise retry, up to a small bounded number of attempts with a short
     delay between them.
  5. If all attempts fail -> raise PlugError. We do NOT pretend success;
     the operator and any future safety logic must see the failure.

pywizlight is async-only; the sync FastAPI handlers call `set_state` which
drives the async work to completion in its own event loop. The actual
pywizlight bulb is hidden behind the `Bulb` protocol so tests can substitute
a fake (including one that drops commands, the failure mode we must surface).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Protocol

log = logging.getLogger(__name__)


class Bulb(Protocol):
  """The slice of the pywizlight `wizlight` API we depend on.

  Kept minimal so a fake bulb (tests) and the real pywizlight bulb are
  interchangeable. All methods are async, matching pywizlight.
  """

  async def turn_on(self) -> None: ...

  async def turn_off(self) -> None: ...

  async def is_on(self) -> bool:
    """Read the plug's current power state back from the device (getPilot)."""
    ...


class PlugError(RuntimeError):
  """Raised when a state change could not be verified after all retries.

  `name` identifies the plug; `observed` is the last state read back (None if
  the read itself failed / the device was unreachable).
  """

  def __init__(self, name: str, intent: bool, observed: bool | None, detail: str) -> None:
    super().__init__(detail)
    self.name = name
    self.intent = intent
    self.observed = observed
    self.detail = detail


@dataclass(frozen=True)
class PlugConfig:
  name: str
  address: str
  retries: int = 3
  retry_delay_s: float = 0.2


class PlugController:
  """Send-and-verify controller for a single WiZ plug.

  The bulb factory is injected so production uses pywizlight and tests use a
  fake. `set_state(on)` is the only mutating entry point; it returns the
  verified boolean state on success and raises PlugError on failure.
  """

  def __init__(self, cfg: PlugConfig, bulb: Bulb) -> None:
    self._cfg = cfg
    self._bulb = bulb

  @property
  def name(self) -> str:
    return self._cfg.name

  def set_retry_policy(self, retries: int, retry_delay_s: float) -> None:
    """Update the send-and-verify retry counts (SIGHUP reload, §2.13). Only the
    retry policy is mutable — the plug's name and address (which physical device
    it actuates) are NOT reloadable; re-binding a safety actuator at runtime is
    restart-only."""
    self._cfg = replace(self._cfg, retries=int(retries), retry_delay_s=float(retry_delay_s))

  async def _apply(self, on: bool) -> None:
    if on:
      await self._bulb.turn_on()
    else:
      await self._bulb.turn_off()

  async def set_state_async(self, on: bool) -> bool:
    """Issue + verify the state change. Returns the verified state or raises.

    The command is re-issued on each attempt (not just re-read): a lost UDP
    command needs another send, not just another read.
    """
    intent = on
    observed: bool | None = None
    last_detail = ""
    attempts = max(1, self._cfg.retries)
    for attempt in range(1, attempts + 1):
      try:
        await self._apply(on)
        observed = await self._bulb.is_on()
      except Exception as e:  # noqa: BLE001 — pywizlight raises a broad set (network, timeout)
        observed = None
        last_detail = f"{type(e).__name__}: {e}"
        log.warning(
          '{"event":"plug_attempt","plug":"%s","intent":%s,"attempt":%d,'
          '"observed":null,"outcome":"error","detail":"%s"}',
          self._cfg.name,
          str(intent).lower(),
          attempt,
          last_detail,
        )
      else:
        ok = observed == intent
        log.info(
          '{"event":"plug_attempt","plug":"%s","intent":%s,"attempt":%d,'
          '"observed":%s,"outcome":"%s"}',
          self._cfg.name,
          str(intent).lower(),
          attempt,
          str(observed).lower(),
          "ok" if ok else "mismatch",
        )
        if ok:
          return observed
        last_detail = (
          f"plug reported {'on' if observed else 'off'} after issuing {'on' if on else 'off'}"
        )
      if attempt < attempts:
        await asyncio.sleep(self._cfg.retry_delay_s)
    raise PlugError(self._cfg.name, intent, observed, last_detail or "verify failed")

  async def read_state_async(self) -> bool:
    """Read the current verified state without issuing a command (refresh)."""
    return await self._bulb.is_on()

  def set_state(self, on: bool) -> bool:
    """Sync wrapper for the FastAPI handlers (drives the async work to done)."""
    return asyncio.run(self.set_state_async(on))

  def read_state(self) -> bool:
    return asyncio.run(self.read_state_async())


# --- pywizlight-backed bulb --------------------------------------------------


class WizBulb:
  """Real pywizlight bulb. Imported lazily so the dependency is only needed at
  runtime on the Pi (tests inject a fake and never import pywizlight)."""

  def __init__(self, address: str) -> None:
    from pywizlight import wizlight

    self._light = wizlight(address)

  async def turn_on(self) -> None:
    from pywizlight import PilotBuilder

    await self._light.turn_on(PilotBuilder())

  async def turn_off(self) -> None:
    await self._light.turn_off()

  async def is_on(self) -> bool:
    # getPilot refreshes and returns a PilotParser; get_state() is the power
    # boolean. Force a fresh read (don't trust a cached pilot).
    state = await self._light.updateState()
    if state is None:
      raise RuntimeError("getPilot returned no state")
    return bool(state.get_state())


# --- in-memory fake bulb (integration test only) -----------------------------


class FakeBulb:
  """In-memory bulb for the in-cluster integration harness — NOT for production.

  It exists so the per-PR test can exercise the full send-and-verify loop, the
  cached-state update, the retained MQTT publish, and the /state reporting
  without a real WiZ plug or any UDP. Only the ~10-line pywizlight call
  (WizBulb) is bypassed.

  `mode` is the load-bearing knob:
    - "ok"   : commands take effect (the happy path).
    - "drop" : every turn_* command is silently ignored, exactly like a lost
               UDP packet — so send-and-verify can never confirm the change and
               must surface a PlugError -> 503. This is how the harness proves
               the safety property deterministically (a dedicated always-drop
               plug), with no runtime back channel and no race.
  """

  def __init__(self, *, initial_on: bool = True, mode: str = "ok") -> None:
    self._on = initial_on
    self._mode = mode

  async def turn_on(self) -> None:
    if self._mode != "drop":
      self._on = True

  async def turn_off(self) -> None:
    if self._mode != "drop":
      self._on = False

  async def is_on(self) -> bool:
    return self._on


def build_controller(
  cfg: PlugConfig, *, driver: str = "pywizlight", mode: str = "ok"
) -> PlugController:
  """Factory for a PlugController.

  `driver` selects the backing bulb:
    - "pywizlight" (default, production): a real WizBulb dialing cfg.address.
    - "fake" (integration test only): an in-memory FakeBulb; `mode` ("ok" |
      "drop") sets whether it honours or silently drops commands.

  The production path is unchanged — `fake` is opt-in via config and never
  reached unless `supervisor.control.wiz.driver` explicitly asks for it.
  """
  if driver == "fake":
    return PlugController(cfg, FakeBulb(mode=mode))
  return PlugController(cfg, WizBulb(cfg.address))
