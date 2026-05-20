"""CEM with a probe-based custom reward.

Instead of scoring imagined trajectories with the world model's trained
``reward_head`` (which is frozen at training time against whatever
``cfg.env.reward`` was active then), this policy:

  1. Runs the same prior rollout the standard ``CEMPolicy`` does.
  2. Concatenates ``[h, s]`` per step into a "feat" tensor.
  3. Passes feat through a small MLP **probe** that was trained to
     predict task-space state (``object_xy``, ``ee_xy``).
  4. Plugs the predicted state into a hand-coded Python ``RewardFn``
     (any of the ``reward.py`` dataclasses) to get a scalar per step.
  5. Sums across the horizon for CEM's argmax.

The benefit: you can change the reward shape without retraining the
WM. The probe is fast to fit (see ``scripts/train_probe.py``), and the
reward function lives in plain Python so iteration is fast.

The cost: prediction accuracy is bounded by the probe's R², which the
``latent_probe`` notebook shows peaks around 0.5 for ``object_xy`` and
0.85 for ``ee_xy`` ~10-25 RSSM steps post-burn-in. Beyond that window
the latent diverges and the probe outputs are noise — so keep the CEM
horizon short (≤ 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..reward import RewardFn
from .cem_policy import CEMPolicy
from .world_model import State, as_numpy, rollout


# ---------------------------------------------------------------------------
# Probe loader
# ---------------------------------------------------------------------------


@dataclass
class ProbeBundle:
  """Loaded MLP probe + the normalization stats it was trained with.

  Apply with ``predict_batch(feat) -> (object_xy, ee_xy)``. Stays on
  CPU + numpy throughout — the probe is small enough that GPU dispatch
  overhead would dominate per-batch cost.
  """

  weights:    list[tuple[np.ndarray, np.ndarray]]   # (W, b) per Linear layer
  x_mean:     np.ndarray
  x_std:      np.ndarray
  y_mean:     np.ndarray
  y_std:      np.ndarray
  target_names: tuple[str, ...]
  in_dim:     int
  out_dim:    int

  @classmethod
  def from_file(cls, path: str | Path) -> "ProbeBundle":
    """Load weights + stats from a torch.save'd file written by
    :mod:`scripts.train_probe`. Converts each Linear's params to numpy
    so inference works without torch on the hot path.
    """
    import torch

    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    state = blob["state_dict"]
    # state_dict keys look like `net.0.weight`, `net.0.bias`,
    # `net.2.weight`, `net.2.bias`, ... — every other module is a Linear.
    weights: list[tuple[np.ndarray, np.ndarray]] = []
    layer_idx = 0
    while True:
      wkey = f"net.{layer_idx}.weight"
      bkey = f"net.{layer_idx}.bias"
      if wkey not in state:
        break
      weights.append((
        state[wkey].cpu().numpy().astype(np.float32),
        state[bkey].cpu().numpy().astype(np.float32),
      ))
      layer_idx += 2  # skip the ReLU module index

    return cls(
      weights      = weights,
      x_mean       = np.asarray(blob["x_mean"], dtype=np.float32).reshape(-1),
      x_std        = np.asarray(blob["x_std"],  dtype=np.float32).reshape(-1),
      y_mean       = np.asarray(blob["y_mean"], dtype=np.float32).reshape(-1),
      y_std        = np.asarray(blob["y_std"],  dtype=np.float32).reshape(-1),
      target_names = tuple(blob.get("target_names", [])),
      in_dim       = int(blob["in_dim"]),
      out_dim      = int(blob["out_dim"]),
    )

  def predict_batch(self, feat: np.ndarray) -> np.ndarray:
    """``(..., in_dim)`` latent → ``(..., out_dim)`` predicted target.

    Standardizes inputs with the training-time stats, runs the MLP
    (Linear→ReLU→Linear→ReLU→…→Linear), denormalizes outputs.
    """
    x = (feat.astype(np.float32) - self.x_mean) / self.x_std
    last = len(self.weights) - 1
    for i, (W, b) in enumerate(self.weights):
      x = x @ W.T + b
      if i < last:
        x = np.maximum(x, 0.0)
    return x * self.y_std + self.y_mean


# ---------------------------------------------------------------------------
# Reward adapter
# ---------------------------------------------------------------------------


@dataclass
class _PlanInfo:
  """Light-weight stand-in for ``sim.step_info.StepInfo`` that the
  reward dataclasses consume. Only carries the fields the existing
  reward classes actually read: ``object_xy``, ``ee_pos`` (3,),
  ``goal_xy``, ``step``, ``obstacle_xy`` (NaN by default).

  Built per (sample, horizon-step) inside :meth:`CEMProbePolicy._score`
  from the probe's predictions + a fixed ``goal_xy``.
  """

  object_xy:   np.ndarray
  ee_pos:      np.ndarray
  ee_quat:     np.ndarray
  goal_xy:     np.ndarray
  step:        int
  time:        float = 0.0


def _default_ee_quat() -> np.ndarray:
  return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class CEMProbePolicy(CEMPolicy):
  """CEM whose per-step reward comes from ``probe(latent)`` + Python reward.

  Drop-in replacement for :class:`CEMPolicy` from the runner's
  perspective — same ``reset`` / ``act`` interface. Only ``_score``
  differs.
  """

  def __init__(
    self,
    model,
    *,
    probe:    ProbeBundle,
    reward_fn: RewardFn,
    goal_xy:  tuple[float, float],
    horizon:        int = 8,
    num_samples:    int = 256,
    num_elites:     int = 32,
    num_iterations: int = 4,
    init_std:       float = 1.0,
    min_std:        float = 0.1,
    discount:       float = 1.0,
    action_low:     float | np.ndarray = -1.0,
    action_high:    float | np.ndarray = 1.0,
    warm_start:     bool = True,
  ) -> None:
    super().__init__(
      model,
      horizon=horizon, num_samples=num_samples, num_elites=num_elites,
      num_iterations=num_iterations, init_std=init_std, min_std=min_std,
      discount=discount, action_low=action_low, action_high=action_high,
      warm_start=warm_start,
    )
    self.probe     = probe
    self.reward_fn = reward_fn
    self.goal_xy   = np.asarray(goal_xy, dtype=np.float32)
    # Per-step discount geometric weights for the horizon. Same shape
    # as CEMPolicy's internal weighting.
    self._step_discount = np.power(
      float(discount), np.arange(self.horizon, dtype=np.float32),
    )

  # ------------------------------------------------------------------
  # Overridden scoring
  # ------------------------------------------------------------------

  def _score(self, plan_state: State, samples: np.ndarray) -> np.ndarray:
    """Per-rollout return = Σ_t discount^t · reward_fn(probe(latent_t)).

    Differences from :meth:`CEMPolicy._score`:
      * ``predict_reward=False`` — we don't use the WM's reward head.
      * Reads ``traj.stack_feat()`` instead of ``traj.rewards``.
      * Vectorized probe inference across ``(N_samples, H)``.
      * Loops the reward function across (sample, step) since the
        existing reward dataclasses accept one ``StepInfo`` at a time.
        For N=256, H=8 that's ~2k calls per CEM iteration — each one
        is just a few np ops, so the loop is well under one ms.
    """
    with self._inference_context():
      traj = rollout(
        self.model,
        init_state=plan_state,
        horizon=self.horizon,
        mode="prior",
        actions=samples,
        sample=False,
        predict_reward=False,
      )
    feat = as_numpy(traj.stack_feat())               # (N, H, feat_dim)
    N, H, _ = feat.shape

    # Probe each (sample, step) latent → predicted (object_xy, ee_xy).
    pred = self.probe.predict_batch(feat.reshape(N * H, -1)).reshape(N, H, -1)
    # We expect 4-D output [object_x, object_y, ee_x, ee_y]; if the
    # probe was trained with a different layout, slot it in here.
    pred_object_xy = pred[..., :2]
    pred_ee_xy     = pred[..., 2:4] if pred.shape[-1] >= 4 else None

    ee_quat0 = _default_ee_quat()
    rewards = np.empty((N, H), dtype=np.float32)
    for n in range(N):
      for h in range(H):
        obj = pred_object_xy[n, h].astype(np.float32)
        if pred_ee_xy is not None:
          ee = np.asarray([pred_ee_xy[n, h, 0], pred_ee_xy[n, h, 1], 0.0], dtype=np.float32)
        else:
          ee = np.zeros(3, dtype=np.float32)
        info = _PlanInfo(
          object_xy=obj, ee_pos=ee, ee_quat=ee_quat0,
          goal_xy=self.goal_xy, step=h,
        )
        rewards[n, h] = float(self.reward_fn(info))

    return (rewards * self._step_discount[None, :]).sum(axis=-1)
