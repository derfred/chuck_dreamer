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
  """What a policy sees at one step: elapsed time + measured joints."""

  t: float
  """Seconds since the producing :meth:`Policy.reset` call."""

  q_meas: np.ndarray
  """Latest measured joint positions, shape ``(n_joints,)``."""
