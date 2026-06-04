"""WiZ send-and-verify tests (S3.1) — the most safety-critical code in Slice 3.

The load-bearing behaviours:
  - a command that takes effect verifies on the first attempt;
  - a *dropped* command (UDP loss) is re-issued and verifies on a later
    attempt — "sent" alone is never "done";
  - a command that never takes effect exhausts the retries and raises
    PlugError carrying the observed state (we do NOT pretend success);
  - an exception from the bulb (network unreachable) also raises PlugError
    with observed=None after exhausting retries.
"""

import pytest

from supervisor.wiz import Bulb, PlugConfig, PlugController, PlugError


class FakeBulb(Bulb):
  """In-memory bulb. `drop_first` simulates the first N turn_* commands being
  lost on the wire (state does not change); `raise_on` makes is_on raise."""

  def __init__(self, on: bool = True, drop_first: int = 0, raise_reads: int = 0) -> None:
    self._on = on
    self._drop = drop_first
    self._raise_reads = raise_reads
    self.commands: list[bool] = []

  async def turn_on(self) -> None:
    self.commands.append(True)
    if self._drop > 0:
      self._drop -= 1
      return  # command lost on the wire
    self._on = True

  async def turn_off(self) -> None:
    self.commands.append(False)
    if self._drop > 0:
      self._drop -= 1
      return
    self._on = False

  async def is_on(self) -> bool:
    if self._raise_reads > 0:
      self._raise_reads -= 1
      raise OSError("network unreachable")
    return self._on


def controller(bulb: Bulb, retries: int = 3) -> PlugController:
  return PlugController(
    PlugConfig(name="arm", address="x", retries=retries, retry_delay_s=0.0), bulb
  )


def test_first_attempt_success():
  bulb = FakeBulb(on=True)
  c = controller(bulb)
  assert c.set_state(False) is False
  assert bulb.commands == [False]  # one issue, verified


def test_dropped_command_is_reissued_and_verifies():
  # The first turn_off is lost on the wire; send-and-verify must re-issue.
  bulb = FakeBulb(on=True, drop_first=1)
  c = controller(bulb)
  assert c.set_state(False) is False
  # Two issues: the lost one and the one that took effect.
  assert bulb.commands == [False, False]


def test_never_takes_effect_raises_with_observed_state():
  # Every command is dropped: retries exhaust, PlugError reports observed=on.
  bulb = FakeBulb(on=True, drop_first=99)
  c = controller(bulb, retries=3)
  with pytest.raises(PlugError) as ei:
    c.set_state(False)
  err = ei.value
  assert err.name == "arm"
  assert err.intent is False
  assert err.observed is True  # the plug never went off
  assert len(bulb.commands) == 3  # all three attempts re-issued the command


def test_read_failure_raises_with_observed_none():
  # is_on raises on every attempt -> observed is None, still PlugError.
  bulb = FakeBulb(on=True, raise_reads=99)
  c = controller(bulb, retries=2)
  with pytest.raises(PlugError) as ei:
    c.set_state(False)
  assert ei.value.observed is None


def test_read_state_does_not_issue_command():
  bulb = FakeBulb(on=True)
  c = controller(bulb)
  assert c.read_state() is True
  assert bulb.commands == []  # pure read, no command issued


# --- the integration-test fake bulb + driver selection (sketch §2.1) ---------
# The in-memory FakeBulb shipped in the module (distinct from this file's test
# helper) is what the in-cluster harness uses. Its `drop` mode is how the
# per-PR test forces a verify failure deterministically.


def test_module_fake_bulb_ok_mode_honours_commands():
  from supervisor.wiz import FakeBulb as ModuleFakeBulb

  c = controller(ModuleFakeBulb(initial_on=True, mode="ok"))
  assert c.set_state(False) is False
  assert c.set_state(True) is True


def test_module_fake_bulb_drop_mode_fails_verify():
  # `drop` ignores every command (like lost UDP) -> verify never confirms ->
  # PlugError. This is the safety-path the harness asserts via an always-drop
  # plug, with no runtime back channel.
  from supervisor.wiz import FakeBulb as ModuleFakeBulb

  c = controller(ModuleFakeBulb(initial_on=True, mode="drop"), retries=3)
  with pytest.raises(PlugError) as ei:
    c.set_state(False)
  assert ei.value.observed is True  # stayed on; never went off


def test_build_controller_selects_fake_driver():
  from supervisor.wiz import FakeBulb as ModuleFakeBulb
  from supervisor.wiz import build_controller

  cfg = PlugConfig(name="arm", address="fake", retries=1, retry_delay_s=0.0)
  fake = build_controller(cfg, driver="fake", mode="drop")
  assert isinstance(fake._bulb, ModuleFakeBulb)
  # (The pywizlight branch constructs a real WizBulb -> wizlight(), which grabs
  # an event loop and is exercised only on the Pi; not unit-tested here.)


def test_build_control_plane_fake_driver_no_address_needed():
  # With driver=fake a plug needs no address; an always-drop plug is declarable.
  from supervisor.control import build_control_plane

  published: list = []
  ctl = build_control_plane(
    {
      "control": {
        "wiz": {"driver": "fake", "retries": 1, "retry_delay_s": 0.0},
        "plugs": {"jetson": {}, "armfail": {"mode": "drop"}},
      }
    },
    lambda *a: published.append(a),
  )
  # Happy plug verifies; the always-drop plug raises PlugError.
  assert ctl.poweron("jetson").on is True
  with pytest.raises(PlugError):
    ctl.shutdown("armfail")
