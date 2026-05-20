"""Cross-backend numerical-parity verifier for the Dreamer world model.

Three subcommands let you partition the work across machines (build
weights on one host, run them on another, compare on a third) while
guaranteeing both backends see bit-identical inputs and bit-identical
on-disk weights.

  weights   Build a model with a chosen backend, randomly initialize its
            parameters, and write a single ``.safetensors`` checkpoint
            plus an ``inputs.npz`` fixture (sampled with a fixed numpy
            seed so different runs of `outputs` consume the same data).

  outputs   Load the checkpoint with the chosen backend, run every
            forward head we want to compare (decoder, reward, actor
            mean, critic, RSSM ``_compute_h``), and write a flat
            ``.npz`` of numpy arrays.

  compare   Read two output ``.npz`` files (e.g. one mlx, one cuda) and
            print per-tensor max/mean absolute differences. Returns a
            non-zero exit code if any tensor exceeds ``--atol``.

Typical roundtrip (CPU torch ↔ MLX, on this Apple-silicon laptop)::

  python scripts/verify_mlx_torch_parity.py weights -d cpu  -o env.obs_mode='[image,state]' -p /tmp/parity
  python scripts/verify_mlx_torch_parity.py outputs -d cpu  -p /tmp/parity --tag torch
  python scripts/verify_mlx_torch_parity.py outputs -d mlx  -p /tmp/parity --tag mlx
  python scripts/verify_mlx_torch_parity.py compare         -p /tmp/parity --tags torch mlx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from chuck_dreamer.config import load_config
from chuck_dreamer.dreamer import build_model

# Keys in the safetensors metadata block where we stash the fixture
# shapes so `outputs` can rebuild the model without the user repeating
# the construction-time flags.
META_OBS_SHAPE  = "verify.obs_shape"
META_ACTION_DIM = "verify.action_dim"
META_FIXTURE_B  = "verify.fixture_batch"
META_FIXTURE_S  = "verify.fixture_seed"

WEIGHTS_FILENAME = "weights.safetensors"
INPUTS_FILENAME  = "inputs.npz"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_obs_shape(cfg, image_extra: dict[str, int] | None = None) -> dict[str, tuple[int, ...]]:
  """Build a dict of per-component shapes matching ``cfg.env.obs_mode``.

  We don't want this script to spin up the simulator just to read
  ``observation_spec``, so we synthesize shapes from the config:

    * ``image`` uses ``cfg.model.encoder.image_size`` for H and W with
      3 channels.
    * ``state`` defaults to 5 dims (a reasonable proprio length); the
      ``--state-dim`` flag overrides.
  """
  image_extra = image_extra or {}
  obs_shape: dict[str, tuple[int, ...]] = {}
  for c in cfg.env.obs_mode:
    if c == "image":
      img = int(cfg.model.encoder.image_size)
      obs_shape[c] = (img, img, 3)
    elif c == "state":
      obs_shape[c] = (int(image_extra.get("state_dim", 5)),)
    else:
      raise ValueError(f"unknown obs component {c!r}")
  return obs_shape


def _make_inputs(
  obs_shape: dict[str, tuple[int, ...]],
  action_dim: int,
  feat_dim: int,
  embed_dim: int,
  stoch_dim: int,
  deter_dim: int,
  batch: int,
  seed: int,
) -> dict[str, np.ndarray]:
  """Seed-deterministic fixture batch consumed by every forward head."""
  rng = np.random.default_rng(seed)
  inputs: dict[str, np.ndarray] = {
    "feat":   rng.standard_normal((batch, feat_dim)).astype(np.float32),
    "embed":  rng.standard_normal((batch, embed_dim)).astype(np.float32),
    "prev_s": rng.standard_normal((batch, stoch_dim)).astype(np.float32),
    "prev_a": rng.standard_normal((batch, action_dim)).astype(np.float32),
    "prev_h": rng.standard_normal((batch, deter_dim)).astype(np.float32),
  }
  # Per-component obs samples used to exercise the encoder / encode→
  # posterior_step roundtrip.
  for name, shape in obs_shape.items():
    if name == "image":
      inputs[f"obs_{name}"] = rng.integers(0, 256, size=(batch, *shape), dtype=np.uint8)
    else:
      inputs[f"obs_{name}"] = rng.standard_normal((batch, *shape)).astype(np.float32)
  return inputs


def _to_backend(model, arr: np.ndarray):
  """Lift a numpy array to whatever tensor type ``model.encode`` expects."""
  return model.coerce(arr)


def _to_numpy(x) -> np.ndarray:
  """Best-effort lift back to numpy. Handles torch.Tensor, mlx.array, dict."""
  if hasattr(x, "detach"):  # torch
    return x.detach().cpu().numpy()
  # mlx: numpy bridge via np.asarray after eval-on-touch; just use np.asarray.
  return np.asarray(x)


def _run_forwards(model, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """Run every forward head we compare across backends and return numpy."""
  feat   = _to_backend(model, inputs["feat"])
  prev_s = _to_backend(model, inputs["prev_s"])
  prev_a = _to_backend(model, inputs["prev_a"])
  prev_h = _to_backend(model, inputs["prev_h"])

  out: dict[str, np.ndarray] = {}

  # Decoder: one entry per component, prefixed so they don't collide
  # with the rest of the outputs.
  decoded = model.decode(feat)
  for k, v in decoded.items():
    out[f"decoder.{k}"] = _to_numpy(v)

  out["reward"] = _to_numpy(model.predict_reward(feat))
  out["critic"] = _to_numpy(model.critic(feat))

  # Actor: compare the deterministic mode (sampling would differ across
  # backends' RNGs even with matching weights).
  out["actor_mode"] = _to_numpy(model.actor_action(feat, sample=False))

  # RSSM deterministic step: most sensitive to the GRU bias repack.
  h = model.rssm._compute_h(prev_s, prev_a, prev_h)
  out["rssm_h"] = _to_numpy(h)

  # Encoder + posterior_step: exercises the CNN encoder reshape and the
  # posterior head. Build the obs dict the encoder expects.
  obs = {}
  for k, v in inputs.items():
    if k.startswith("obs_"):
      name = k[len("obs_"):]
      obs[name] = _to_backend(model, v)
  embed = model.encode(obs)
  out["embed"] = _to_numpy(embed)
  init = model.initial_state(int(inputs["prev_s"].shape[0]))
  post = model.posterior_step(init, prev_a, embed, sample=False)
  out["post_h"] = _to_numpy(post.h)
  out["post_s"] = _to_numpy(post.s)

  return out


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_weights(args: argparse.Namespace) -> int:
  out_dir = Path(args.path)
  out_dir.mkdir(parents=True, exist_ok=True)

  cfg = load_config(args.config, overrides=tuple(args.override))
  cfg.hardware.device = args.device

  obs_shape  = _resolve_obs_shape(cfg, {"state_dim": args.state_dim})
  action_dim = args.action_dim

  # Build with the chosen backend; the backend's own random init seeds
  # the weights. Two runs with the same backend + seed produce identical
  # checkpoints only if the backend's RNG is deterministic w.r.t. that
  # seed — we don't try to enforce that here (the contract is that
  # `outputs` reads whatever file `weights` wrote).
  model = build_model(cfg, obs_shape=obs_shape, action_dim=action_dim)

  extra = {
    META_OBS_SHAPE:  json.dumps({k: list(v) for k, v in obs_shape.items()}),
    META_ACTION_DIM: str(action_dim),
    META_FIXTURE_B:  str(args.batch),
    META_FIXTURE_S:  str(args.input_seed),
  }
  weights_path = out_dir / WEIGHTS_FILENAME
  model.save(str(weights_path), extra_metadata=extra)

  feat_dim  = int(cfg.model.rssm.stoch_size + cfg.model.rssm.deter_size)
  embed_dim = int(cfg.model.encoder.embed_size)
  inputs    = _make_inputs(
    obs_shape, action_dim, feat_dim, embed_dim,
    int(cfg.model.rssm.stoch_size), int(cfg.model.rssm.deter_size),
    args.batch, args.input_seed,
  )
  np.savez(out_dir / INPUTS_FILENAME, **inputs)

  print(f"wrote {weights_path}")
  print(f"wrote {out_dir / INPUTS_FILENAME}")
  print(f"obs_shape   = {obs_shape}")
  print(f"action_dim  = {action_dim}")
  print(f"batch       = {args.batch}, input_seed = {args.input_seed}")
  return 0


def cmd_outputs(args: argparse.Namespace) -> int:
  out_dir = Path(args.path)
  weights_path = out_dir / WEIGHTS_FILENAME
  inputs_path  = out_dir / INPUTS_FILENAME
  if not weights_path.exists():
    print(f"weights not found at {weights_path}", file=sys.stderr)
    return 2

  # Pull the construction-time shapes out of the checkpoint metadata so
  # the user only specifies them once (at `weights` time).
  from safetensors import safe_open
  with safe_open(str(weights_path), framework="np") as f:
    meta = f.metadata() or {}
  obs_shape  = {k: tuple(v) for k, v in json.loads(meta[META_OBS_SHAPE]).items()}
  action_dim = int(meta[META_ACTION_DIM])

  cfg = load_config(args.config, overrides=tuple(args.override))
  cfg.hardware.device = args.device

  model = build_model(cfg, obs_shape=obs_shape, action_dim=action_dim)
  model.load(str(weights_path))

  with np.load(inputs_path) as data:
    inputs = {k: data[k] for k in data.files}

  out = _run_forwards(model, inputs)

  tag_path = out_dir / f"outputs_{args.tag}.npz"
  np.savez(tag_path, **out)
  print(f"wrote {tag_path}")
  for k, v in out.items():
    print(f"  {k:24s} shape={tuple(v.shape)} dtype={v.dtype}")
  return 0


def cmd_compare(args: argparse.Namespace) -> int:
  in_dir = Path(args.path)
  tag_a, tag_b = args.tags
  path_a = in_dir / f"outputs_{tag_a}.npz"
  path_b = in_dir / f"outputs_{tag_b}.npz"
  for p in (path_a, path_b):
    if not p.exists():
      print(f"missing {p}", file=sys.stderr)
      return 2

  a = np.load(path_a)
  b = np.load(path_b)

  shared    = sorted(set(a.files) & set(b.files))
  only_a    = sorted(set(a.files) - set(b.files))
  only_b    = sorted(set(b.files) - set(a.files))
  worst_key = None
  worst_val = -np.inf
  failures: list[str] = []

  print(f"{'tensor':<24s} {'max_abs':>14s} {'mean_abs':>14s} {'shape':>20s} status")
  for k in shared:
    arr_a = a[k]
    arr_b = b[k]
    if arr_a.shape != arr_b.shape:
      print(f"{k:<24s} shape mismatch: {arr_a.shape} vs {arr_b.shape}")
      failures.append(k)
      continue
    diff = np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))
    mx_  = float(diff.max())
    mn_  = float(diff.mean())
    if mx_ > worst_val:
      worst_val, worst_key = mx_, k
    flag = "ok" if mx_ <= args.atol else "FAIL"
    if mx_ > args.atol:
      failures.append(k)
    print(f"{k:<24s} {mx_:14.6e} {mn_:14.6e} {str(arr_a.shape):>20s} {flag}")

  for k in only_a:
    print(f"{k:<24s} only in {tag_a}")
  for k in only_b:
    print(f"{k:<24s} only in {tag_b}")

  print()
  print(f"worst tensor: {worst_key} with max_abs_diff={worst_val:.6e} (atol={args.atol:.1e})")

  if failures or only_a or only_b:
    print(f"FAIL — {len(failures)} tensor(s) over atol, "
          f"{len(only_a) + len(only_b)} key(s) missing on one side")
    return 1
  print("PASS")
  return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser, *, needs_device: bool) -> None:
  p.add_argument("-p", "--path", required=True, help="Directory for parity artifacts.")
  p.add_argument("-c", "--config", default=None, help="Path to a YAML config (defaults to configs/default.yaml).")
  p.add_argument("-o", "--override", action="append", default=[],
                 help="Dotted-path config overrides, e.g. -o env.obs_mode='[image]'. Repeatable.")
  if needs_device:
    p.add_argument("-d", "--device", required=True,
                   choices=("mlx", "cpu", "cuda", "mps", "auto"),
                   help="Backend device to use.")


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  subs = parser.add_subparsers(dest="cmd", required=True)

  pw = subs.add_parser("weights", help="Build & save sample weights + fixture inputs.")
  _add_common(pw, needs_device=True)
  pw.add_argument("--action-dim", type=int, default=6)
  pw.add_argument("--state-dim",  type=int, default=5,
                  help="Dimensionality of the 'state' obs component (when present).")
  pw.add_argument("--batch",      type=int, default=4)
  pw.add_argument("--input-seed", type=int, default=0,
                  help="numpy seed used to draw the fixture batch.")
  pw.set_defaults(func=cmd_weights)

  po = subs.add_parser("outputs", help="Run forwards with one backend, save outputs.")
  _add_common(po, needs_device=True)
  po.add_argument("--tag", required=True,
                  help="Label embedded in the output filename (e.g. 'mlx', 'cuda').")
  po.set_defaults(func=cmd_outputs)

  pc = subs.add_parser("compare", help="Compare two output sets.")
  _add_common(pc, needs_device=False)
  pc.add_argument("--tags", nargs=2, required=True, metavar=("TAG_A", "TAG_B"),
                  help="Two tags from prior `outputs` runs to compare.")
  pc.add_argument("--atol", type=float, default=1e-4,
                  help="Per-tensor max-abs-diff tolerance for the PASS/FAIL verdict.")
  pc.set_defaults(func=cmd_compare)

  args = parser.parse_args(argv)
  return int(args.func(args))


if __name__ == "__main__":
  sys.exit(main())
