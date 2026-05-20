"""Real-arm policies for ``main.py run``.

All policies share the same surface:

  reset(initial_obs)  -> None
  act(obs) -> action  # numpy array, shape policy-specific

``obs`` is whatever :class:`~chuck_dreamer.real.run.runner.Runner` builds
per step — see that module for the schema. Policies that don't need an
image just ignore it.

Three implementations:

  * :class:`ManualPolicy` — mirrors joint positions from an SO-101 leader
                            arm onto the follower (teleoperation).
  * :class:`DreamerPolicyAdapter` — wraps a trained Dreamer model's actor
                                    (``model.actor_action`` on the
                                    posterior state).
  * :class:`CEMPolicyAdapter` — wraps :class:`~chuck_dreamer.dreamer.cem_policy.CEMPolicy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .arm import LeaderArm, LeaderArmConfig


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class RealPolicy(Protocol):
  def reset(self, initial_obs: dict) -> None: ...
  def act(self, obs: dict) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Manual teleop policy: SO-101 leader -> follower joint mirror
# ---------------------------------------------------------------------------


@dataclass
class ManualPolicyConfig:
  """Configuration for the leader-arm-driven manual policy.

  ``port`` is the serial device the leader bus connects to (e.g.
  ``"/dev/tty.usbmodemXXXX"``). The leader uses lerobot's ``SOLeader``
  teleoperator under the hood — same protocol as the follower's bus,
  but read-only.
  """
  port:           str
  calibration_id: str | None = None
  use_degrees:    bool = True


class ManualPolicy:
  """Joint-mirroring policy driven by an SO-101 leader arm.

  Returns a ``(6,)`` joint qpos array (radians, simulator order) on each
  ``act()`` call, sampled directly from the leader's current pose. The
  runner is expected to pass that array to ``Arm.send_joint_qpos`` so
  the follower tracks the leader's motion.
  """

  def __init__(self, config: ManualPolicyConfig) -> None:
    self.config = config
    self._leader: LeaderArm | None = None

  def _ensure_leader(self) -> LeaderArm:
    if self._leader is None:
      self._leader = LeaderArm(LeaderArmConfig(
        port=self.config.port,
        calibration_id=self.config.calibration_id,
        use_degrees=self.config.use_degrees,
      ))
      self._leader.connect()
    return self._leader

  def reset(self, initial_obs: dict) -> None:
    self._ensure_leader()

  def act(self, obs: dict) -> np.ndarray:
    leader = self._ensure_leader()
    return leader.read_joint_qpos()

  # Symmetric with the rest of the run module; the runner calls this on
  # shutdown to release the serial port.
  def stop_listener(self) -> None:
    if self._leader is not None:
      self._leader.disconnect()
      self._leader = None


# ---------------------------------------------------------------------------
# Dreamer actor adapter
# ---------------------------------------------------------------------------


class DreamerPolicyAdapter:
  """Run a trained Dreamer model's actor on the live observation stream.

  Keeps a posterior-state tracker, advances it each step with the
  previous action + the new embedding, and queries the model's actor.
  Mirrors what :class:`~chuck_dreamer.dreamer.cem_policy.CEMPolicy` does
  internally, minus the CEM planning loop.
  """

  def __init__(self, model, *, sample: bool = False) -> None:
    self.model  = model
    self.sample = sample
    self._tracker = None
    self._prev_action: np.ndarray | None = None

  def reset(self, initial_obs: dict) -> None:
    self._tracker = self.model.initial_state(1)
    self._prev_action = np.zeros((self.model.action_dim,), dtype=np.float32)

  def act(self, obs: dict) -> np.ndarray:
    from ...training.observation import Observation

    assert self._tracker is not None and self._prev_action is not None
    obs_w = Observation.from_buffer_value(
      _project_obs_for_model(obs, self.model.obs_mode),
      self.model.obs_mode,
    ).add_batch_axis().to_buffer_value()

    embed  = self.model.encode(obs_w)
    act_in = self.model.coerce(self._prev_action[None])
    self._tracker = self.model.posterior_step(
      self._tracker, act_in, embed, sample=False,
    )
    feat = self.model.feat(self._tracker)
    action = self.model.actor_action(feat, sample=self.sample)
    # actor lives in the backend's tensor type; convert back to numpy.
    from ...dreamer.world_model import as_numpy
    action_np = np.asarray(as_numpy(action)).reshape(-1).astype(np.float32)
    self._prev_action = action_np
    return action_np


# ---------------------------------------------------------------------------
# CEM adapter — just plumbs the real observations into CEMPolicy.act
# ---------------------------------------------------------------------------


@dataclass
class CEMRunConfig:
  checkpoint:     str
  horizon:        int = 12
  num_samples:    int = 256
  num_elites:     int = 32
  num_iterations: int = 4
  discount:       float = 1.0
  action_low:     np.ndarray | None = None
  action_high:    np.ndarray | None = None


class CEMPolicyAdapter:
  def __init__(self, model, run_cfg: CEMRunConfig) -> None:
    from ...dreamer.cem_policy import CEMPolicy

    if run_cfg.action_low is None or run_cfg.action_high is None:
      raise ValueError(
        "CEMRunConfig requires action_low/action_high — pass env.action_space bounds."
      )
    self.model = model
    self._cem  = CEMPolicy(
      model,
      horizon=run_cfg.horizon,
      num_samples=run_cfg.num_samples,
      num_elites=run_cfg.num_elites,
      num_iterations=run_cfg.num_iterations,
      discount=run_cfg.discount,
      action_low=run_cfg.action_low,
      action_high=run_cfg.action_high,
    )

  def reset(self, initial_obs: dict) -> None:
    self._cem.reset(None)

  def act(self, obs: dict) -> np.ndarray:
    projected = _project_obs_for_model(obs, self.model.obs_mode)
    return self._cem.act(projected)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_obs_for_model(obs: dict, obs_mode: Any) -> dict:
  """Turn the runner's flat obs dict into the per-component dict the model wants.

  The runner provides ``image`` and ``arm_qpos`` only — no EE pose
  (no IK is run here), no object pose (no detector wired up), so
  ``ee``, ``object_xy``, and ``object_uv`` checkpoints cannot run on
  the real arm yet. ``joints`` and ``proprio`` both work because
  ``joint_qpos`` = ``arm_qpos``.
  """
  from ...training.observation import (
    EE, IMAGE, JOINTS, OBJECT_UV, OBJECT_XY, PROPRIO, normalize_obs_mode,
  )

  mode = normalize_obs_mode(obs_mode)
  out: dict[str, np.ndarray] = {}
  for c in mode:
    if c == IMAGE:
      out[c] = np.asarray(obs["image"], dtype=np.uint8)
    elif c in (JOINTS, PROPRIO):
      out[c] = np.asarray(obs["arm_qpos"], dtype=np.float32)
    elif c in (EE, OBJECT_XY, OBJECT_UV):
      raise ValueError(
        f"obs_mode component {c!r} is not supported on the real arm: "
        f"the runner does not compute ee_pos / ee_quat / object_xy / object_uv."
      )
    else:
      raise ValueError(f"unknown obs_mode component {c!r}")
  return out
