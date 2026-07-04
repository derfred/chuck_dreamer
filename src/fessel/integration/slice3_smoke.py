#!/usr/bin/env python3
"""Slice 3 control-plane end-to-end smoke check (X3.1).

  ⚠️  THIS TOUCHES REAL HARDWARE AND CUTS REAL POWER.  ⚠️

This is NOT part of the Slice 1.5 integration harness (driver/run_tests.py),
which deliberately avoids real WiZ plugs and a real Jetson. Run this on demand,
against a deployment wired to the actual actuators, when you want to prove the
Slice 3 surface end-to-end. It will pause/resume the Jetson and power-cycle the
Jetson plug, so only run it when that is safe.

The scenario (X3.1):

  1. POST /api/control/pause   -> Jetson /state reflects paused.
  2. POST /api/control/resume  -> Jetson reflects resumed.
  3. POST /api/control/shutdown/jetson -> /api/state shows the plug off
     (verified), optionally cross-checked via direct getPilot() on the plug.
  4. POST /api/control/poweron/jetson  -> recovery; plug on again.

It reaches webui-backend the same way the operator's browser does, but supplies
the oauth2-proxy identity header itself (oauth2-proxy is out of this loop, same
as the Slice 1.5/2 driver). Configure via env:

  WEBUI                 backend base URL (default http://webui:8000)
  FESSEL_AUTH_USER_HEADER / FESSEL_TEST_OPERATOR   identity header + value
  PLUG_NAME             which plug to power-cycle (default 'jetson' — safer
                        than the arm; the arm cut is left to manual testing)
  WIZ_PLUG_ADDRESS      if set, cross-check the plug directly via pywizlight
  ACTION_TIMEOUT_S      per-action wait for the expected state (default 15)

Exit code 0 on success, 1 on the first failed assertion.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

WEBUI = os.environ.get("WEBUI", "http://webui:8000").rstrip("/")
AUTH_USER_HEADER = os.environ.get("FESSEL_AUTH_USER_HEADER", "X-Auth-Request-User")
AUTH_USER = os.environ.get("FESSEL_TEST_OPERATOR", "slice3-smoke")
PLUG_NAME = os.environ.get("PLUG_NAME", "jetson")
WIZ_PLUG_ADDRESS = os.environ.get("WIZ_PLUG_ADDRESS")
ACTION_TIMEOUT_S = float(os.environ.get("ACTION_TIMEOUT_S", "15"))

HEADERS = {AUTH_USER_HEADER: AUTH_USER}


def _req(method: str, path: str) -> tuple[int, dict]:
  req = urllib.request.Request(f"{WEBUI}{path}", method=method, headers=HEADERS)  # noqa: S310
  try:
    with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310
      body = r.read()
      return r.status, (json.loads(body) if body else {})
  except urllib.error.HTTPError as e:
    body = e.read()
    try:
      return e.code, json.loads(body)
    except ValueError:
      return e.code, {"detail": body.decode(errors="replace")}


def get_state() -> dict:
  status, body = _req("GET", "/api/state")
  if status != 200:
    raise AssertionError(f"GET /api/state -> {status}: {body}")
  return body


def control(action: str) -> dict:
  status, body = _req("POST", f"/api/control/{action}")
  if status != 200:
    raise AssertionError(f"POST /api/control/{action} -> {status}: {body}")
  return body


def wait_until(predicate, what: str) -> dict:
  deadline = time.time() + ACTION_TIMEOUT_S
  last: dict = {}
  while time.time() < deadline:
    last = get_state()
    if predicate(last):
      return last
    time.sleep(0.5)
  raise AssertionError(f"timed out waiting for {what}; last state={last}")


def jetson_state(state: dict) -> str | None:
  j = state.get("jetson") or {}
  s = j.get("state")
  return s if isinstance(s, str) else None


def plug_on(state: dict, name: str) -> bool | None:
  p = (state.get("plugs") or {}).get(name) or {}
  return p.get("on") if p.get("verified") else None


def verify_plug_directly(address: str, expect_on: bool) -> None:
  """Cross-check the WiZ plug independently of supervisor's cached view."""
  import asyncio

  from pywizlight import wizlight

  async def read() -> bool:
    light = wizlight(address)
    st = await light.updateState()
    return bool(st.get_state())

  observed = asyncio.run(read())
  if observed != expect_on:
    raise AssertionError(
      f"direct getPilot disagrees: plug reports on={observed}, expected {expect_on}"
    )


def main() -> int:
  print(f"[smoke] target {WEBUI}, plug={PLUG_NAME}", flush=True)

  # 1. pause -> Jetson paused.
  control("pause")
  wait_until(lambda s: jetson_state(s) == "paused", "jetson paused")
  print("[PASS] pause", flush=True)

  # 2. resume -> Jetson active.
  control("resume")
  wait_until(lambda s: jetson_state(s) in ("active", "resumed", "running"), "jetson resumed")
  print("[PASS] resume", flush=True)

  # 3. shutdown plug -> verified off (and optionally cross-checked).
  control(f"shutdown/{PLUG_NAME}")
  wait_until(lambda s: plug_on(s, PLUG_NAME) is False, f"{PLUG_NAME} plug verified off")
  if WIZ_PLUG_ADDRESS:
    verify_plug_directly(WIZ_PLUG_ADDRESS, expect_on=False)
  print(f"[PASS] shutdown/{PLUG_NAME} (verified off)", flush=True)

  # 4. poweron plug -> verified on (recovery).
  control(f"poweron/{PLUG_NAME}")
  wait_until(lambda s: plug_on(s, PLUG_NAME) is True, f"{PLUG_NAME} plug verified on")
  if WIZ_PLUG_ADDRESS:
    verify_plug_directly(WIZ_PLUG_ADDRESS, expect_on=True)
  print(f"[PASS] poweron/{PLUG_NAME} (recovered)", flush=True)

  print("[smoke] all Slice 3 control-plane assertions passed", flush=True)
  return 0


if __name__ == "__main__":
  try:
    sys.exit(main())
  except AssertionError as e:
    print(f"[FAIL] {e}", flush=True)
    sys.exit(1)
