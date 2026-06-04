"""The minimal observation a runtime policy receives each step.

Until the sensing/perception half lands (M3, spec §3.3/§5), the runtime has
no real :class:`~chuck_dreamer.training.observation.Observation` to hand a
policy. :class:`RuntimeObservation` is the small forward-compatible stand-in
the policy loop passes to ``policy.act``: the elapsed time since
``policy.reset`` (so a scripted policy is a pure function of its own clock)
and the latest measured joint positions. When M3 introduces the full
modality-bearing observation, it subsumes this — a policy that only reads
``t`` / ``q_meas`` keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RuntimeObservation:
  """What a policy sees at one step: elapsed time + measured joints.

  M2 (teleop, spec §3.8) adds the optional ``leader_qpos`` modality: when a
  leader arm is configured the policy loop attaches the freshest leader reading
  here. It is optional and defaulted to ``None`` so a policy reading only
  ``t`` / ``q_meas`` (``GoToPose``, ``SineSweep``) is unaffected; M3's modality
  registry subsumes this field.
  """

  t: float
  """Seconds since the producing :meth:`Policy.reset` call."""

  q_meas: np.ndarray
  """Latest measured joint positions, shape ``(n_joints,)``."""

  leader_qpos: np.ndarray | None = None
  """Latest leader joint positions in follower order + radians, shape
  ``(n_joints,)``; ``None`` when no leader is configured or before its first read."""

  @property
  def has_leader(self) -> bool:
    """Whether a leader-modality reading is present this step."""
    return self.leader_qpos is not None
