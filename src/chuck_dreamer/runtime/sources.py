"""Scripted policies for M1 — go-to-pose and sine sweep.

Until the learned policy is integrated (M6), the runtime is driven by
scripted joint-space setpoints (project plan M1: "go-to-pose, sine sweep —
no policy"). These are real :class:`~chuck_dreamer.policy.Policy`
implementations so the runtime has a single producer abstraction: the policy
loop calls ``reset`` once then ``act(obs)`` each step, exactly as it will for
Dreamer at M6. Swapping in a different policy is a config change, no new loop
code.

Each scripted policy is a pure function of its own elapsed clock: ``act``
reads ``obs.t`` (seconds since :meth:`reset`) and ignores everything else, so
the trajectory is asserted in tests without threads. The underlying time →
target math lives in :meth:`target_at` for that reason.

``reset`` accepts either a joint vector (the runtime hands over the current
measured pose) or a :class:`SceneConfig` whose ``joint_initial_qpos`` is used
— mirroring how :class:`~chuck_dreamer.policy.GatedPolicy` takes "whatever
the inner policy expects". Scripted policies may deliberately command targets
outside the safety envelope; the kernel's clamp is what holds the boundary,
and exercising it is the point.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from .modalities import RuntimeObservation


def _start_qpos(scene_or_q0: Any) -> np.ndarray:
  """Extract the starting joint vector from a joint array or a SceneConfig."""
  if hasattr(scene_or_q0, "joint_initial_qpos"):
    q0 = scene_or_q0.joint_initial_qpos
    if q0 is None:
      raise ValueError("scene has no joint_initial_qpos to anchor the policy")
    return np.asarray(q0, dtype=np.float64)
  return np.asarray(scene_or_q0, dtype=np.float64)


class GoToPose:
  """Linearly interpolate from the reset pose to ``q_goal`` over ``duration_s``.

  Holds ``q_goal`` once ``t >= duration_s``. ``q_goal`` may be a per-joint
  list/array; pass it resolved (the harness turns the config ``"home"``
  sentinel into the home qpos before constructing the policy).
  """

  def __init__(self, q_goal: np.ndarray, duration_s: float) -> None:
    self._q_goal = np.asarray(q_goal, dtype=np.float64)
    if duration_s <= 0:
      raise ValueError("duration_s must be positive")
    self._duration = float(duration_s)
    self._q0 = np.zeros_like(self._q_goal)

  def reset(self, scene_or_q0: Any) -> None:
    q0 = _start_qpos(scene_or_q0)
    if q0.shape != self._q_goal.shape:
      raise ValueError(
        f"start pose shape {q0.shape} != q_goal shape {self._q_goal.shape}")
    self._q0 = q0.copy()

  def target_at(self, t: float) -> np.ndarray:
    """Joint target at ``t`` seconds since :meth:`reset` (pure)."""
    frac = 1.0 if t >= self._duration else max(0.0, t) / self._duration
    return self._q0 + frac * (self._q_goal - self._q0)

  def act(self, obs: RuntimeObservation) -> np.ndarray:
    return self.target_at(obs.t)


class ManualPolicy:
  """Pass-through teleop: command the leader joints carried on the observation.

  The leader-reader (:mod:`chuck_dreamer.runtime.teleop`) feeds
  ``obs.leader_qpos`` via the policy loop — :class:`ManualPolicy` does *not* own
  the reader (M2 modality decision, spec §3.8). :meth:`act` returns that reading
  as the joint-space action. When the leader reading is absent (no leader
  configured, or before its first poll) it returns a safe no-op — the last pose
  it commanded, falling back to the measured pose at boot — so the kernel simply
  holds. The kernel's slew + clamp shapes and bounds the motion exactly as for a
  scripted setpoint, so teleop respects the M1 envelope unchanged.
  """

  def __init__(self) -> None:
    self._last_cmd: np.ndarray | None = None

  def reset(self, scene_or_q0: Any) -> None:
    # Anchor the no-op fallback at the start pose (shared with the scripted sources).
    self._last_cmd = _start_qpos(scene_or_q0).copy()

  def act(self, obs: RuntimeObservation) -> np.ndarray:
    leader = obs.leader_qpos
    if leader is not None:
      cmd            = np.asarray(leader, dtype=np.float64)
      self._last_cmd = cmd
      return cmd
    # No-op: hold the last commanded pose; fall back to the measured joints
    # before the first leader reading. The kernel slews to it with zero delta.
    if self._last_cmd is not None:
      return self._last_cmd
    return np.asarray(obs.q_meas, dtype=np.float64)


class SineSweep:
  """Per-joint sinusoid ``center + amplitude * sin(2*pi*freq*t + phase)``.

  ``amplitude`` / ``freq_hz`` / ``phase`` accept either a scalar (broadcast
  to all joints) or a per-joint sequence. ``center`` defaults to the reset
  pose so the sweep oscillates around wherever the arm started.
  """

  def __init__(
    self,
    amplitude: float | np.ndarray,
    freq_hz: float | np.ndarray,
    phase: float | np.ndarray = 0.0,
    center: np.ndarray | None = None,
  ) -> None:
    self._amp = np.asarray(amplitude, dtype=np.float64)
    self._freq = np.asarray(freq_hz, dtype=np.float64)
    self._phase = np.asarray(phase, dtype=np.float64)
    self._center_cfg = None if center is None else np.asarray(center, dtype=np.float64)
    self._center: np.ndarray | None = self._center_cfg

  def reset(self, scene_or_q0: Any) -> None:
    q0 = _start_qpos(scene_or_q0)
    n = q0.shape[0]
    self._center = q0.copy() if self._center_cfg is None else self._center_cfg
    # Broadcast all params to the joint count now that we know it.
    self._amp = np.broadcast_to(self._amp, (n,)).astype(np.float64)
    self._freq = np.broadcast_to(self._freq, (n,)).astype(np.float64)
    self._phase = np.broadcast_to(self._phase, (n,)).astype(np.float64)
    self._center = np.broadcast_to(self._center, (n,)).astype(np.float64)

  def target_at(self, t: float) -> np.ndarray:
    """Joint target at ``t`` seconds since :meth:`reset` (pure)."""
    if self._center is None:
      raise RuntimeError("SineSweep.target_at called before reset")
    return cast(np.ndarray, self._center + self._amp * np.sin(
      2.0 * np.pi * self._freq * t + self._phase))

  def act(self, obs: RuntimeObservation) -> np.ndarray:
    return self.target_at(obs.t)
