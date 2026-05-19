"""Real-arm policies for ``main.py run``.

All policies share the same surface:

  reset(initial_obs)            -> None
  act(obs) -> ee_action (7,)    # [x, y, z, qw, qx, qy, qz]

``obs`` is whatever :class:`~chuck_dreamer.real.run.runner.Runner` builds
per step — see that module for the schema. Policies that don't need an
image just ignore it.

Three implementations:

  * :class:`ManualPolicy` — captures arrow / w/a/s/d / q/e keystrokes and
                            commands an EE delta in world space.
  * :class:`DreamerPolicyAdapter` — wraps a trained Dreamer model's actor
                                    (``model.actor_action`` on the
                                    posterior state).
  * :class:`CEMPolicyAdapter` — wraps :class:`~chuck_dreamer.dreamer.cem_policy.CEMPolicy`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class RealPolicy(Protocol):
  def reset(self, initial_obs: dict) -> None: ...
  def act(self, obs: dict) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Manual keyboard policy
# ---------------------------------------------------------------------------


@dataclass
class ManualPolicyConfig:
  step:     float = 0.01
  home_xyz: tuple[float, float, float] = (0.0, -0.2, 0.15)


@dataclass
class _KeyState:
  """Pressed-key set, mutated by the listener thread, read by ``act``.

  ``lock`` protects ``pressed`` so the main loop sees a coherent snapshot
  even if a key is released between the read and the act.
  """
  pressed: set[str] = field(default_factory=set)
  lock:    threading.Lock = field(default_factory=threading.Lock)


_MOVE_BINDINGS: dict[str, tuple[float, float, float]] = {
  # +x / -x: forward/back (away from arm base)
  "w": (0.0, +1.0, 0.0),
  "s": (0.0, -1.0, 0.0),
  # left/right
  "a": (-1.0, 0.0, 0.0),
  "d": (+1.0, 0.0, 0.0),
  # up/down
  "q": (0.0, 0.0, +1.0),
  "e": (0.0, 0.0, -1.0),
}


class ManualPolicy:
  """Keyboard-driven EE-delta policy.

  Holds the most recent commanded EE pose; each :meth:`act` adds the
  pressed-key direction × ``step`` to that pose. The hold-orientation
  quaternion is captured from the first observation so the gripper keeps
  whatever pose it started in.
  """

  def __init__(self, config: ManualPolicyConfig) -> None:
    self.config = config
    self._keys  = _KeyState()
    self._listener = None
    self._cmd_xyz:  np.ndarray | None = None
    self._hold_quat: np.ndarray | None = None

  # ------------------------------------------------------------------
  # Key listener (pynput is a project dep)
  # ------------------------------------------------------------------

  def start_listener(self) -> None:
    if self._listener is not None:
      return
    from pynput import keyboard  # lazy import — pynput pulls in X / Quartz

    def _name(key) -> str | None:
      try:
        return key.char.lower() if key.char else None
      except AttributeError:
        return None

    def on_press(key):
      n = _name(key)
      if n is None:
        return
      with self._keys.lock:
        self._keys.pressed.add(n)

    def on_release(key):
      n = _name(key)
      if n is None:
        return
      with self._keys.lock:
        self._keys.pressed.discard(n)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    self._listener = listener

  def stop_listener(self) -> None:
    if self._listener is not None:
      self._listener.stop()
      self._listener = None

  # ------------------------------------------------------------------
  # Policy API
  # ------------------------------------------------------------------

  def reset(self, initial_obs: dict) -> None:
    # No EE-pose feed from the runner anymore — seed from the configured
    # home pose and an identity orientation. Callers that wire IK back in
    # later can replace this with a real reading.
    self._cmd_xyz   = np.asarray(self.config.home_xyz, dtype=np.float64)
    self._hold_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    self.start_listener()

  def act(self, obs: dict) -> np.ndarray:
    if self._cmd_xyz is None or self._hold_quat is None:
      raise RuntimeError("ManualPolicy.act() called before reset()")
    with self._keys.lock:
      pressed = set(self._keys.pressed)

    delta = np.zeros(3, dtype=np.float64)
    for k, dir_ in _MOVE_BINDINGS.items():
      if k in pressed:
        delta += np.asarray(dir_, dtype=np.float64)
    self._cmd_xyz = self._cmd_xyz + delta * self.config.step

    return np.concatenate([
      self._cmd_xyz.astype(np.float32),
      self._hold_quat.astype(np.float32),
    ])


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


class CEMPolicyAdapter:
  def __init__(self, model, run_cfg: CEMRunConfig) -> None:
    from ...dreamer.cem_policy import CEMPolicy

    self.model = model
    self._cem  = CEMPolicy(
      model,
      horizon=run_cfg.horizon,
      num_samples=run_cfg.num_samples,
      num_elites=run_cfg.num_elites,
      num_iterations=run_cfg.num_iterations,
    )

  def reset(self, initial_obs: dict) -> None:
    self._cem.reset(None)

  def act(self, obs: dict) -> np.ndarray:
    projected = _project_obs_for_model(obs, self.model.obs_mode)
    return self._cem.act(projected)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_obs_for_model(obs: dict, obs_mode: str) -> Any:
  """Turn the runner's flat obs dict into what the trained model wants.

  Only ``image`` mode is supported on the real arm: ``state`` and
  ``image_proprio`` both need EE pose / object pose fields that this
  runner doesn't compute (no IK, no object detector wired up).
  """
  if obs_mode == "image":
    return np.asarray(obs["image"], dtype=np.uint8)
  raise ValueError(
    f"obs_mode={obs_mode!r} is not supported on the real arm: "
    "only image-mode checkpoints can run here for now."
  )
