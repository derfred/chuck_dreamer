"""Train an MLP probe: WM latent (h, s) -> (object_xy, ee_xy).

Why this exists
---------------
CEM at inference scores imagined rollouts using the world model's
reward head, which was trained against whatever ``cfg.env.reward`` was
active during the WM's own training. To use a different reward shape
(e.g. ``PushReward``) without retraining the WM, we need to extract
task-space state (where is the object, where is the gripper) from the
latent at planning time. A small MLP fit on (latent, ground-truth)
pairs does this — see Frederik's R² plot for evidence that the latent
carries this information for ~10-25 RSSM steps after burn-in.

This script:
  1. Loads a chuck_dreamer checkpoint.
  2. Runs prior-only rollouts on a batch of sim episodes
     (`main.py generate-scenes` produces these).
  3. Pairs prior latents (h, s) with ground-truth object_xy / ee_xy
     read from the episode's raw arrays.
  4. Trains an MLP probe + saves weights + per-axis normalization
     stats to a single .pt file.

The saved file is loaded by ``CEMProbePolicy`` at inference time —
its scoring path replaces ``reward_head(latent)`` with
``handcoded_reward(probe(latent))``.

Usage
-----
    uv run --active python scripts/train_probe.py \\
        --checkpoint checkpoints/frederik/final.safetensors \\
        --data       data/probe_data/ \\
        --output     checkpoints/probe_object_xy.pt \\
        --num-episodes 200 \\
        --epochs       200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Probe MLP
# ---------------------------------------------------------------------------


class ProbeMLP(nn.Module):
  """Plain MLP: latent -> task-space state.

  Two hidden layers of 256 ReLU is plenty for this regression — there
  are only ~100k training examples (200 episodes × ~500 steps). Bigger
  nets overfit fast.
  """

  def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int):
    super().__init__()
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
      layers.append(nn.Linear(prev, h))
      layers.append(nn.ReLU())
      prev = h
    layers.append(nn.Linear(prev, out_dim))
    self.net = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_pairs(ckpt, data_dir: str, num_episodes: int, burn_in: int):
  """For each eval episode in ``data_dir``, run a stitched prior rollout
  (posterior during burn-in, prior tail) and pair each latent with the
  ground-truth ``object_xy`` and ``ee_xy`` from the episode's raw arrays.

  Returns ``(X, Y)`` numpy arrays of shape ``(N_total_steps, h+s_dim)``
  and ``(N_total_steps, 4)``. Trains on **prior** latents so the probe
  generalizes to what CEM imagines (CEM rollouts the world model's
  prior — the no-observation forward dynamics).
  """
  from chuck_dreamer.eval import load_eval_episodes, run_split_rollout

  obs_mode = ckpt.env.obs_mode
  sources = [{"path": data_dir, "format": "rerun", "num_episodes": num_episodes}]

  Xs: list[np.ndarray] = []
  Ys: list[np.ndarray] = []
  kept = 0
  for episode in load_eval_episodes(
    ckpt.config, sources=sources,
    min_len=burn_in + 1, max_episodes=num_episodes,
  ):
    T = episode.num_actions
    if T <= burn_in + 1:
      continue
    # Posterior across the whole episode (we need it to anchor the burn-in).
    post = run_split_rollout(
      ckpt.model, episode,
      burn_in=0, horizon=T, obs_mode=obs_mode,
      decode=False, predict_reward=False,
    )
    # Prior tail starting from the burn-in anchor.
    prior_tail = run_split_rollout(
      ckpt.model, episode,
      burn_in=burn_in, horizon=T - burn_in, obs_mode=obs_mode,
      decode=False, predict_reward=False,
    )
    if post is None or prior_tail is None:
      continue

    # Stitch a length-T prior trajectory.
    h_post = post.h_posterior
    s_post = post.s_posterior
    h_prior = np.concatenate([h_post[:burn_in], prior_tail.h_prior], axis=0)[:T]
    s_prior = np.concatenate([s_post[:burn_in], prior_tail.s_prior], axis=0)[:T]

    raw = episode.raw
    object_xy = np.asarray(raw["object_xy"], dtype=np.float32)[:T]
    ee_pos    = np.asarray(raw["ee_pos"],    dtype=np.float32)[:T]
    ee_xy     = ee_pos[:, :2]

    X = np.concatenate([h_prior, s_prior], axis=-1).astype(np.float32)
    Y = np.concatenate([object_xy, ee_xy], axis=-1).astype(np.float32)
    Xs.append(X)
    Ys.append(Y)
    kept += 1

  if not Xs:
    raise RuntimeError(f"No usable episodes in {data_dir!r}")
  return np.concatenate(Xs, 0), np.concatenate(Ys, 0), kept


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  ss_res = ((y_true - y_pred) ** 2).sum()
  ss_tot = ((y_true - y_true.mean(0, keepdims=True)) ** 2).sum()
  return 1.0 - ss_res / max(ss_tot, 1e-12)


def train(
  X: np.ndarray, Y: np.ndarray,
  *, hidden: tuple[int, ...] = (256, 256),
  epochs: int = 200, lr: float = 1e-3, batch: int = 512,
  val_frac: float = 0.2, device: str = "cpu",
):
  """Z-score normalize, split 80/20, fit MLP. Returns ``(net, stats)``.

  Stats include the per-axis means/stds used to (de)normalize at
  inference; the probe wrapper applies them on every call.
  """
  Xm = X.mean(0, keepdims=True);             Xs = X.std(0, keepdims=True) + 1e-6
  Ym = Y.mean(0, keepdims=True);             Ys = Y.std(0, keepdims=True) + 1e-6
  Xn = (X - Xm) / Xs
  Yn = (Y - Ym) / Ys

  rng = np.random.default_rng(seed=0)
  perm = rng.permutation(len(Xn))
  n_te = int(len(Xn) * val_frac)
  te_idx, tr_idx = perm[:n_te], perm[n_te:]
  Xtr, Ytr = Xn[tr_idx], Yn[tr_idx]
  Xte, Yte = Xn[te_idx], Yn[te_idx]

  net = ProbeMLP(Xn.shape[1], hidden, Yn.shape[1]).to(device)
  opt = torch.optim.Adam(net.parameters(), lr=lr)
  crit = nn.MSELoss()

  Xtr_t = torch.from_numpy(Xtr).float().to(device)
  Ytr_t = torch.from_numpy(Ytr).float().to(device)
  Xte_t = torch.from_numpy(Xte).float().to(device)
  Yte_t = torch.from_numpy(Yte).float().to(device)

  print(f"\nProbe training: in_dim={Xn.shape[1]} out_dim={Yn.shape[1]} "
        f"hidden={hidden} epochs={epochs} batch={batch} lr={lr}")
  for epoch in range(epochs):
    perm_t = torch.randperm(len(Xtr_t))
    for start in range(0, len(Xtr_t), batch):
      idx = perm_t[start:start + batch]
      yhat = net(Xtr_t[idx])
      loss = crit(yhat, Ytr_t[idx])
      opt.zero_grad(); loss.backward(); opt.step()
    if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
      with torch.no_grad():
        te_loss = crit(net(Xte_t), Yte_t).item()
      print(f"  epoch {epoch:4d}  train_loss={loss.item():.5f}  test_loss={te_loss:.5f}")

  with torch.no_grad():
    yhat = (net(Xte_t).cpu().numpy() * Ys + Ym)
    ytrue = (Yte_t.cpu().numpy() * Ys + Ym)
  r2_overall = _r2(ytrue, yhat)
  r2_obj     = _r2(ytrue[:, :2], yhat[:, :2])
  r2_ee      = _r2(ytrue[:, 2:], yhat[:, 2:])
  print(f"\nfinal test R² — overall={r2_overall:+.3f}  "
        f"object_xy={r2_obj:+.3f}  ee_xy={r2_ee:+.3f}")

  stats = dict(
    x_mean=Xm.astype(np.float32), x_std=Xs.astype(np.float32),
    y_mean=Ym.astype(np.float32), y_std=Ys.astype(np.float32),
    in_dim=int(Xn.shape[1]), out_dim=int(Yn.shape[1]),
    hidden=list(hidden),
    target_names=["object_x", "object_y", "ee_x", "ee_y"],
    r2_overall=float(r2_overall),
    r2_object_xy=float(r2_obj),
    r2_ee_xy=float(r2_ee),
  )
  return net, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint",   required=True)
  ap.add_argument("--data",         required=True,
                  help="Probe training data directory (from main.py generate-scenes)")
  ap.add_argument("--output",       default="checkpoints/probe_object_xy.pt")
  ap.add_argument("--burn-in",      default=5, type=int,
                  help="Posterior anchor steps before switching to prior")
  ap.add_argument("--num-episodes", default=200, type=int)
  ap.add_argument("--epochs",       default=200, type=int)
  ap.add_argument("--device",       default="mps",
                  help="torch device for loading the checkpoint + training probe")
  args = ap.parse_args()

  from chuck_dreamer.eval import load_checkpoint

  t0 = time.time()
  print(f"Loading checkpoint {args.checkpoint} (device={args.device}) ...")
  ckpt = load_checkpoint(args.checkpoint, device=args.device)

  print(f"Collecting (latent, target) pairs from {args.data} "
        f"(num_episodes={args.num_episodes}, burn_in={args.burn_in}) ...")
  X, Y, n_eps = collect_pairs(ckpt, args.data, args.num_episodes, args.burn_in)
  print(f"  episodes used: {n_eps}")
  print(f"  X shape: {X.shape}   Y shape: {Y.shape}")

  net, stats = train(X, Y, epochs=args.epochs, device=args.device)

  out = Path(args.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  payload = {"state_dict": net.state_dict(), **stats}
  torch.save(payload, out)
  print(f"\nWrote probe → {out}  ({out.stat().st_size / 1024:.1f} KB)")
  print(f"Total time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
  main()
