"""PyTorch implementation of the Dreamer v1 world model.

Mirrors :mod:`.mlx_model`: same module structure (encoder / decoder /
RSSM / reward / actor / critic / optional aux) and the same on-disk
checkpoint format. The public surface implements the
:class:`~chuck_dreamer.dreamer.world_model.WorldModel` protocol so
:func:`~chuck_dreamer.dreamer.world_model.rollout` composes against it
unchanged.

On-disk format vs. torch's default ``state_dict``
-------------------------------------------------

:meth:`DreamerTorchModel.save` does NOT write a plain torch state_dict.
It writes the MLX-format flat key dict (see :mod:`.mlx_model`) so the
two backends share one ``.safetensors`` schema. A torch-saved
checkpoint loads under MLX and vice versa. The deviations from what
``self._wm_bundle.state_dict()`` would produce:

  * **Sequential children get an extra ``layers`` segment.** MLX's
    ``nn.Sequential`` exposes children under ``layers.N``; torch
    exposes them as ``N``. We insert ``layers`` on save and strip it
    on load. e.g. torch ``convs.0.weight`` ↔ disk
    ``convs.layers.0.weight``.

  * **``MultiModalEncoder.branches`` / ``MultiModalDecoder.heads`` are
    flattened.** Torch wraps the per-component sub-modules in an
    ``nn.ModuleDict`` (`encoder.branches.<name>`); MLX has no
    ModuleDict and stores them as plain attributes named ``b_<name>``
    / ``h_<name>``. The ModuleDict level is collapsed on disk.

  * **``_ChannelLayerNorm`` is unwrapped.** This module is a torch-only
    NCHW→NHWC→LN→NHWC→NCHW shim around ``nn.LayerNorm`` that the MLX
    backend does not need (its convs are NHWC native). We hoist the
    inner ``nn.LayerNorm`` weights onto the wrapper's path so the leaf
    matches MLX's bare ``nn.LayerNorm`` on the channel axis.

  * **Conv weights are permuted.** torch ``Conv2d`` weights are
    ``(C_out, C_in, kH, kW)`` and ``ConvTranspose2d`` weights are
    ``(C_in, C_out, kH, kW)``; both backends store NHWC-style
    ``(C_out, kH, kW, C_in)`` on disk. Bias tensors are unchanged.

  * **GRU is repacked.** torch ``nn.GRUCell`` carries four parameter
    tensors ``{weight_ih, weight_hh, bias_ih, bias_hh}``. MLX
    ``nn.GRU`` carries ``{Wx, Wh, b, bhn}``: ``Wx=weight_ih``,
    ``Wh=weight_hh``, and the (r, z) biases are folded into a single
    ``b`` because the MLX forward has no hidden-side bias on those
    gates. We sum on save (``b[r] = bias_ih[r] + bias_hh[r]``, same
    for z) and on load put the entire sum into ``bias_ih`` with
    ``bias_hh`` zeroed for those rows. The forward pass is invariant
    to this split.

  * **Image tensors are NHWC at every public/disk boundary.** The
    torch convs run NCHW internally, but the encoder transposes back
    to NHWC before the projection (and the decoder reshapes from NHWC
    after the projection) so ``proj.weight`` is bit-for-bit
    interchangeable across backends. Without that extra transpose the
    flat projection slot would mean (c, h, w) on torch but (h, w, c)
    on MLX, and shared weights would diverge.

  * **Optimizer state is not written.** The MLX backend persists Adam
    state under ``opt_*`` prefixes; torch's optimizer state has no
    matching tree layout, so we omit it. Resuming a torch training run
    from a checkpoint reinitializes the optimizer.

Translation lives in :meth:`DreamerTorchModel._iter_mlx_state` (save)
and :meth:`DreamerTorchModel._apply_mlx_state` (load), with the
module-walk rules and the GRU/conv tensor reshapes factored into the
``_iter_mlx_children``, ``_torch_grucell_to_mlx``, and
``_mlx_to_torch_grucell`` helpers below.
"""

from __future__ import annotations

import os
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file as _st_load_file
from safetensors.torch import save_file as _st_save_file

from ..training.observation import IMAGE, Observation, normalize_obs_mode
from .world_model import Distribution, State

# Same key as the MLX backend so the YAML round-trip stays consistent
# *within* a backend; we don't attempt to load an MLX checkpoint from
# here, but the metadata convention is at least uniform.
CONFIG_METADATA_KEY = "config_yaml"


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(name: str) -> torch.device:
  """Map a ``config.hardware.device`` string to a :class:`torch.device`.

  ``"auto"`` picks cuda → mps → cpu in that order. Anything else passes
  through. ``"mlx"`` is rejected here — that backend lives in
  :mod:`.mlx_model`.
  """
  if name == "mlx":
    raise ValueError("resolve_device called with 'mlx'; use the MLX backend")
  if name == "auto":
    if torch.cuda.is_available():
      return torch.device("cuda")
    if torch.backends.mps.is_available():
      return torch.device("mps")
    return torch.device("cpu")
  return torch.device(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.ELU) -> nn.Sequential:
  """Standard MLP: Linear -> LayerNorm -> act -> ... -> Linear (no LN/act on output).

  Mirrors :func:`mlx_model._mlp`. LayerNorm after each hidden Linear keeps
  intermediate magnitudes bounded so the encoder/heads can't run away
  during early training.
  """
  layers: list[nn.Module] = []
  prev = in_dim
  for h in hidden:
    layers.append(nn.Linear(prev, h))
    layers.append(nn.LayerNorm(h))
    layers.append(act())
    prev = h
  layers.append(nn.Linear(prev, out_dim))
  return nn.Sequential(*layers)


class _ChannelLayerNorm(nn.Module):
  """LayerNorm over the channel axis of an NCHW tensor.

  Matches the MLX backend's ``nn.LayerNorm(C)`` applied on NHWC: per-pixel
  per-sample normalization across the channel dimension. Torch's
  ``nn.LayerNorm`` normalizes the trailing dim, which in NCHW is the
  spatial-W axis — wrong. We permute to NHWC, normalize, permute back.
  """

  def __init__(self, num_channels: int):
    super().__init__()
    self.norm = nn.LayerNorm(num_channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # (N, C, H, W) -> (N, H, W, C) -> LN over last dim -> back
    return cast(torch.Tensor, self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())


def feat(state: State) -> torch.Tensor:
  """Concatenate deterministic and stochastic state into the feature fed
  to decoder, reward head, actor, and critic."""
  return torch.cat([state.h, state.s], dim=-1)


def kl_gaussian(
  mean_q: torch.Tensor,
  std_q:  torch.Tensor,
  mean_p: torch.Tensor,
  std_p:  torch.Tensor,
) -> torch.Tensor:
  """KL(q || p) for diagonal Gaussians, per-dimension. Matches the MLX
  backend's formulation."""
  var_q = std_q ** 2
  var_p = std_p ** 2
  return torch.log(std_p / std_q) + (var_q + (mean_q - mean_p) ** 2) / (2.0 * var_p) - 0.5


def _same_pad(kernel: int, stride: int) -> int:
  """Same convention as the MLX encoder. For ``kernel=4 stride=2`` this
  gives ``padding=1``, halving spatial dims cleanly."""
  return max(0, (kernel - stride) // 2)


# ---------------------------------------------------------------------------
# Encoder / Decoder — state mode
# ---------------------------------------------------------------------------


class MLPEncoder(nn.Module):
  """For state-based observations (joint qpos + ee pose + object pose, etc.)."""

  def __init__(self, obs_dim: int, hidden: tuple[int, ...], embed_dim: int):
    super().__init__()
    self.net = _mlp(obs_dim, hidden, embed_dim)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    """obs: (..., obs_dim) -> embedding: (..., embed_dim)"""
    return cast(torch.Tensor, self.net(obs))


class MLPDecoder(nn.Module):
  """Reconstructs state-based observations from latent features."""

  def __init__(self, feat_dim: int, hidden: tuple[int, ...], obs_dim: int):
    super().__init__()
    self.net = _mlp(feat_dim, hidden, obs_dim)

  def forward(self, feat_in: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, self.net(feat_in))


# ---------------------------------------------------------------------------
# CNN encoder / decoder for image obs (NHWC at boundary, NCHW internally)
# ---------------------------------------------------------------------------


class CNNEncoder(nn.Module):
  """Dreamer-style CNN over ``(H, W, 3)`` uint8 images.

  Public boundary is NHWC to match MLX and the replay buffer. Internally
  the convs run NCHW (PyTorch's native layout); the encoder transposes
  the input once on the way in.

  Inputs may have arbitrary leading batch dims; they get flattened, run
  through the conv stack, then reshaped to ``(..., embed_dim)``. Inputs
  are normalized to ``[-0.5, 0.5]``.
  """

  def __init__(
    self,
    in_channels: int,
    channels: tuple[int, ...],
    kernels:  tuple[int, ...],
    strides:  tuple[int, ...],
    embed_dim: int,
    image_size: int,
  ):
    super().__init__()
    if not (len(channels) == len(kernels) == len(strides)):
      raise ValueError(
        f"CNNEncoder: channels/kernels/strides must have equal length; "
        f"got {len(channels)}/{len(kernels)}/{len(strides)}"
      )
    self.image_size = image_size

    # LayerNorm over the channel axis after each conv. _ChannelLayerNorm
    # permutes NCHW -> NHWC to apply LN(C) and back; this matches the MLX
    # backend's ``nn.LayerNorm(C)`` on NHWC exactly.
    convs: list[nn.Module] = []
    prev = in_channels
    for c, k, s in zip(channels, kernels, strides):
      convs.append(nn.Conv2d(prev, c, kernel_size=k, stride=s, padding=_same_pad(k, s)))
      convs.append(_ChannelLayerNorm(c))
      convs.append(nn.ELU())
      prev = c
    self.convs = nn.Sequential(*convs)

    # End spatial side after strided convs (each layer divides cleanly).
    spatial = image_size
    for s in strides:
      spatial //= int(s)
    self._flat_dim = prev * spatial * spatial
    self.proj      = nn.Linear(self._flat_dim, embed_dim)
    self.proj_norm = nn.LayerNorm(embed_dim)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    """obs: (..., H, W, 3). Returns (..., embed_dim)."""
    # Normalize uint8 → float in [-0.5, 0.5] to match the encoder's range
    # contract. Float inputs are assumed already-normalized.
    if obs.dtype == torch.uint8:
      x = obs.float() / 255.0 - 0.5
    else:
      x = obs
    lead = x.shape[:-3]
    H, W, C = x.shape[-3], x.shape[-2], x.shape[-1]
    x = x.reshape(-1, H, W, C)
    x = x.permute(0, 3, 1, 2).contiguous()           # NHWC -> NCHW
    x = self.convs(x)
    # Flatten in NHWC order before the projection so the linear sees the
    # same channel layout as the MLX backend (which never leaves NHWC).
    # Without this, ``proj.weight`` columns would map to a different
    # (c, h, w) ordering across backends and the same on-disk weights
    # would produce different embeddings.
    x = x.permute(0, 2, 3, 1).contiguous()           # NCHW -> NHWC
    x = x.reshape(x.shape[0], -1)
    x = self.proj_norm(self.proj(x))
    return cast(torch.Tensor, x.reshape(*lead, -1))


class CNNDecoder(nn.Module):
  """Mirror of :class:`CNNEncoder`. Maps ``(..., feat_dim) -> (..., H, W, 3)``.

  Output is in ``[-0.5, 0.5]``, matching the encoder's normalization.
  """

  def __init__(
    self,
    feat_dim: int,
    channels: tuple[int, ...],   # encoder order, e.g. (32, 64, 128, 256)
    kernels:  tuple[int, ...],
    strides:  tuple[int, ...],
    out_channels: int,
    image_size: int,
  ):
    super().__init__()
    if not (len(channels) == len(kernels) == len(strides)):
      raise ValueError(
        f"CNNDecoder: channels/kernels/strides must have equal length; "
        f"got {len(channels)}/{len(kernels)}/{len(strides)}"
      )
    self.image_size = image_size

    rev_channels = tuple(reversed(channels))
    rev_kernels  = tuple(reversed(kernels))
    rev_strides  = tuple(reversed(strides))

    spatial = image_size
    for s in strides:
      spatial //= int(s)
    self._init_spatial = spatial
    self._init_channels = rev_channels[0]
    self.proj = nn.Linear(feat_dim, self._init_channels * spatial * spatial)

    deconvs: list[nn.Module] = []
    prev = rev_channels[0]
    next_channels = list(rev_channels[1:]) + [out_channels]
    # LayerNorm + ELU follow every intermediate deconv; the final deconv
    # produces the image and stays bare so the output range isn't squashed.
    for i, (c_out, k, s) in enumerate(zip(next_channels, rev_kernels, rev_strides)):
      deconvs.append(
        nn.ConvTranspose2d(prev, c_out, kernel_size=k, stride=s, padding=_same_pad(k, s))
      )
      if i < len(next_channels) - 1:
        deconvs.append(_ChannelLayerNorm(c_out))
        deconvs.append(nn.ELU())
      prev = c_out
    self.deconvs = nn.Sequential(*deconvs)

  def forward(self, feat_in: torch.Tensor) -> torch.Tensor:
    lead = feat_in.shape[:-1]
    x = self.proj(feat_in)
    # Match the MLX backend's reshape ordering: interpret the projection
    # output as NHWC and permute into NCHW for the deconv stack. This
    # keeps ``proj.weight`` rows tied to the same (h, w, c) ordering on
    # both backends so the same on-disk weights produce the same image.
    x = x.reshape(-1, self._init_spatial, self._init_spatial, self._init_channels)  # NHWC
    x = x.permute(0, 3, 1, 2).contiguous()                                            # NHWC -> NCHW
    x = self.deconvs(x)
    # NCHW -> NHWC for the public boundary.
    x = x.permute(0, 2, 3, 1).contiguous()
    H, W, C = x.shape[-3], x.shape[-2], x.shape[-1]
    return cast(torch.Tensor, x.reshape(*lead, H, W, C))


# ---------------------------------------------------------------------------
# Multi-modal encoder / decoder
# ---------------------------------------------------------------------------


class MultiModalEncoder(nn.Module):
  """Per-component branches; concatenate the embeddings and fuse to ``embed_dim``.

  Each component name maps to a sub-module that takes that component's
  array and returns an ``(..., embed_dim)`` tensor. With a single
  component the fuse layer is the identity (no extra Linear); with N>1
  components a ``Linear(N*embed_dim → embed_dim) + LayerNorm`` mixes
  them — same recipe the old ``ImageProprioEncoder`` used for N=2.
  """

  def __init__(self, branches: dict[str, nn.Module], embed_dim: int):
    super().__init__()
    if not branches:
      raise ValueError("MultiModalEncoder needs at least one branch")
    # Sorted by config order, but Python 3.7+ dicts preserve insertion
    # order — preserve the order the caller passed so the fuse layer's
    # input slots are deterministic across runs.
    self._names: tuple[str, ...] = tuple(branches.keys())
    self.branches = nn.ModuleDict(branches)
    n = len(self._names)
    if n == 1:
      self.fuse: nn.Module     = nn.Identity()
      self.fuse_norm: nn.Module = nn.Identity()
    else:
      self.fuse      = nn.Linear(n * embed_dim, embed_dim)
      self.fuse_norm = nn.LayerNorm(embed_dim)

  def forward(self, obs: dict) -> torch.Tensor:
    feats = [self.branches[name](obs[name]) for name in self._names]
    if len(feats) == 1:
      return cast(torch.Tensor, feats[0])
    return cast(torch.Tensor, self.fuse_norm(self.fuse(torch.cat(feats, dim=-1))))


class MultiModalDecoder(nn.Module):
  """Per-component reconstruction head; returns a dict of reconstructions."""

  def __init__(self, heads: dict[str, nn.Module]):
    super().__init__()
    if not heads:
      raise ValueError("MultiModalDecoder needs at least one head")
    self._names: tuple[str, ...] = tuple(heads.keys())
    self.heads = nn.ModuleDict(heads)

  def forward(self, feat_in: torch.Tensor) -> dict:
    return {name: self.heads[name](feat_in) for name in self._names}


# ---------------------------------------------------------------------------
# RSSM
# ---------------------------------------------------------------------------


class RSSM(nn.Module):
  """Dreamer v1 RSSM. Same dynamics as :class:`mlx_model.RSSM`.

  Uses :class:`nn.GRUCell` for the single-step interface — avoids the
  ``(B, 1, D)`` reshape dance the MLX version needs.
  """

  def __init__(
    self,
    action_dim: int,
    embed_dim:  int,
    stoch_dim:  int = 30,
    deter_dim:  int = 200,
    hidden:     int = 200,
    min_std:    float = 0.1,
  ):
    super().__init__()
    self.action_dim = action_dim
    self.stoch_dim  = stoch_dim
    self.deter_dim  = deter_dim
    self.min_std    = min_std

    # See note on _mlp: LayerNorm after each hidden Linear bounds the
    # recurrent path and the posterior/prior heads so the GRU input and
    # the (mean, raw_std) outputs stay in a healthy range across BPTT.
    self.pre_gru = nn.Sequential(
      nn.Linear(stoch_dim + action_dim, hidden),
      nn.LayerNorm(hidden),
      nn.ELU(),
    )
    self.gru = nn.GRUCell(input_size=hidden, hidden_size=deter_dim)

    self.prior_net = nn.Sequential(
      nn.Linear(deter_dim, hidden), nn.LayerNorm(hidden), nn.ELU(),
      nn.Linear(hidden, 2 * stoch_dim),
    )
    self.post_net = nn.Sequential(
      nn.Linear(deter_dim + embed_dim, hidden), nn.LayerNorm(hidden), nn.ELU(),
      nn.Linear(hidden, 2 * stoch_dim),
    )

  def initial_state(self, batch_size: int, *, device: torch.device | None = None) -> State:
    dev = device if device is not None else next(self.parameters()).device
    h = torch.zeros(batch_size, self.deter_dim, device=dev)
    s = torch.zeros(batch_size, self.stoch_dim, device=dev)
    placeholder = Distribution(mean=s, std=torch.ones(batch_size, self.stoch_dim, device=dev))
    return State(h=h, s=s, prior=placeholder, posterior=None)

  def _split_dist(self, out: torch.Tensor) -> Distribution:
    mean, raw_std = out.chunk(2, dim=-1)
    std = F.softplus(raw_std) + self.min_std
    return Distribution(mean=mean, std=std)

  def _sample(self, dist: Distribution) -> torch.Tensor:
    eps = torch.randn_like(dist.mean)
    return cast(torch.Tensor, dist.mean + dist.std * eps)

  def _compute_h(self, prev_s: torch.Tensor, prev_a: torch.Tensor, prev_h: torch.Tensor) -> torch.Tensor:
    x = self.pre_gru(torch.cat([prev_s, prev_a], dim=-1))   # (B, hidden)
    return cast(torch.Tensor, self.gru(x, prev_h))           # (B, deter_dim)

  def img_step(self, prev_state: State, prev_action: torch.Tensor, *, sample: bool = True) -> State:
    h     = self._compute_h(prev_state.s, prev_action, prev_state.h)
    prior = self._split_dist(self.prior_net(h))
    s     = self._sample(prior) if sample else prior.mean
    return State(h=h, s=s, prior=prior, posterior=None)

  def obs_step(
    self,
    prev_state:  State,
    prev_action: torch.Tensor,
    embed:       torch.Tensor,
    *,
    sample: bool = True,
  ) -> State:
    h         = self._compute_h(prev_state.s, prev_action, prev_state.h)
    prior     = self._split_dist(self.prior_net(h))
    post_in   = torch.cat([h, embed], dim=-1)
    posterior = self._split_dist(self.post_net(post_in))
    s         = self._sample(posterior) if sample else posterior.mean
    return State(h=h, s=s, prior=prior, posterior=posterior)

  def observe(
    self,
    embeds:  torch.Tensor,   # (B, T, embed_dim)
    actions: torch.Tensor,   # (B, T, action_dim)
    init:    State | None = None,
  ) -> list[State]:
    B, T   = embeds.shape[0], embeds.shape[1]
    state  = init if init is not None else self.initial_state(B, device=embeds.device)
    states = []
    for t in range(T):
      state = self.obs_step(state, actions[:, t], embeds[:, t])
      states.append(state)
    return states

  def imagine(self, init_state: State, policy_fn, horizon: int) -> list[State]:
    state = init_state
    traj = [state]
    for _ in range(horizon):
      action = policy_fn(feat(state))
      state = self.img_step(state, action)
      traj.append(state)
    return traj


# ---------------------------------------------------------------------------
# Reward head, aux head, actor, critic
# ---------------------------------------------------------------------------


class RewardHead(nn.Module):
  def __init__(self, feat_dim: int, hidden: tuple[int, ...] = (200, 200)):
    super().__init__()
    self.net = _mlp(feat_dim, hidden, out_dim=1)

  def forward(self, feat_in: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, self.net(feat_in).squeeze(-1))


class AuxHead(nn.Module):
  """Privileged regression target (object_xy ‖ ee_pos) decoded from feat."""

  def __init__(self, feat_dim: int, target_dim: int, hidden: tuple[int, ...] = (200, 200)):
    super().__init__()
    self.target_dim = target_dim
    self.net = _mlp(feat_dim, hidden, out_dim=target_dim)

  def forward(self, feat_in: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, self.net(feat_in))


class Actor(nn.Module):
  """Tanh-squashed Gaussian. Matches the MLX backend's conventions."""

  def __init__(
    self,
    feat_dim:    int,
    action_dim:  int,
    hidden:      tuple[int, ...] = (300, 300, 300),
    init_std:    float = 5.0,
    min_std:     float = 1e-4,
    mean_scale:  float = 5.0,
  ):
    super().__init__()
    self.action_dim = action_dim
    self.init_std   = init_std
    self.min_std    = min_std
    self.mean_scale = mean_scale
    self.net = _mlp(feat_dim, hidden, out_dim=2 * action_dim)
    self._raw_std_offset = init_std

  def _dist(self, feat_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = self.net(feat_in)
    mean, raw_std = out.chunk(2, dim=-1)
    mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
    std = F.softplus(raw_std + self._raw_std_offset) + self.min_std
    return mean, std

  def forward(self, feat_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, std = self._dist(feat_in)
    eps = torch.randn_like(mean)
    raw = mean + std * eps
    action = torch.tanh(raw)
    return action, mean, std

  def mode(self, feat_in: torch.Tensor) -> torch.Tensor:
    mean, _ = self._dist(feat_in)
    return torch.tanh(mean)


class Critic(nn.Module):
  def __init__(self, feat_dim: int, hidden: tuple[int, ...] = (300, 300, 300)):
    super().__init__()
    self.net = _mlp(feat_dim, hidden, out_dim=1)

  def forward(self, feat_in: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, self.net(feat_in).squeeze(-1))


class _WMBundle(nn.Module):
  """Groups the modules that share the world-model optimizer."""

  def __init__(
    self,
    encoder: nn.Module,
    rssm: RSSM,
    decoder: nn.Module,
    reward: RewardHead,
    aux: AuxHead | None = None,
  ):
    super().__init__()
    self.encoder = encoder
    self.rssm    = rssm
    self.decoder = decoder
    self.reward  = reward
    if aux is not None:
      # Only register when present so the param tree (and checkpoints)
      # match between configs that disable the aux head and ones that
      # enable it.
      self.aux = aux


# ---------------------------------------------------------------------------
# DreamerTorchModel
# ---------------------------------------------------------------------------


def load_config_from_checkpoint(path: str):
  """Read the embedded training config from a torch safetensors checkpoint.

  Returns the DictConfig that was saved alongside the weights, or
  ``None`` if the checkpoint has no metadata. Mirrors the MLX backend's
  helper of the same name; kept separate because the load codepath differs.
  """
  # safetensors stores metadata in the file header; load_file does NOT
  # return it. We re-open with the low-level interface.
  from safetensors import safe_open
  with safe_open(path, framework="pt") as f:
    meta = f.metadata() or {}
  yaml_str = meta.get(CONFIG_METADATA_KEY)
  if not yaml_str:
    return None
  return OmegaConf.create(yaml_str)


class DreamerTorchModel:
  """PyTorch Dreamer model. Implements :class:`WorldModel` and offers the
  same trainer-facing surface (:meth:`wm_update`, :meth:`save`,
  :meth:`load`) as :class:`mlx_model.DreamerMLXModel`."""

  encoder: nn.Module
  decoder: nn.Module

  def __init__(self, config, obs_shape, action_dim: int, training: bool = True):
    self.config     = config
    self.training   = training
    self.feat_dim   = config.model.rssm.stoch_size + config.model.rssm.deter_size
    self.action_dim = action_dim
    self.stoch_dim  = int(config.model.rssm.stoch_size)
    self.deter_dim  = int(config.model.rssm.deter_size)

    self.device = resolve_device(config.hardware.device)

    self.obs_mode = normalize_obs_mode(config.env.obs_mode)

    if not isinstance(obs_shape, dict):
      raise ValueError(
        f"obs_shape must be a dict keyed by component name; got {type(obs_shape).__name__}"
      )
    missing = [c for c in self.obs_mode if c not in obs_shape]
    if missing:
      raise ValueError(f"obs_shape missing components {missing!r} (mode={self.obs_mode!r})")

    enc_hidden = tuple(config.model.encoder.mlp_hidden)
    dec_hidden = tuple(config.model.decoder.mlp_hidden)
    embed_size = int(config.model.encoder.embed_size)

    # CNN params only consulted when an image component is present.
    if IMAGE in self.obs_mode:
      enc_channels    = tuple(config.model.encoder.cnn_channels)
      enc_kernels     = tuple(config.model.encoder.cnn_kernels)
      enc_strides     = tuple(config.model.encoder.cnn_strides)
      dec_channels    = tuple(config.model.decoder.cnn_channels)
      dec_kernels     = tuple(config.model.decoder.cnn_kernels)
      dec_strides     = tuple(config.model.decoder.cnn_strides)
      self.image_size = int(config.model.encoder.image_size)
    else:
      self.image_size = None

    enc_branches: dict[str, nn.Module] = {}
    dec_heads:    dict[str, nn.Module] = {}
    for c in self.obs_mode:
      shape = tuple(obs_shape[c])
      if c == IMAGE:
        if len(shape) != 3:
          raise ValueError(f"image component expects (H, W, C) shape; got {shape}")
        in_channels = int(shape[2])
        enc_branches[c] = CNNEncoder(in_channels, enc_channels, enc_kernels, enc_strides,
                                     embed_dim=embed_size, image_size=self.image_size)
        dec_heads[c] = CNNDecoder(self.feat_dim, dec_channels, dec_kernels, dec_strides,
                                  out_channels=in_channels, image_size=self.image_size)
      else:
        if len(shape) != 1:
          raise ValueError(f"vector component {c!r} expects 1-D shape; got {shape}")
        vec_dim = int(shape[0])
        enc_branches[c] = MLPEncoder(vec_dim, enc_hidden, embed_size)
        dec_heads[c]    = MLPDecoder(self.feat_dim, dec_hidden, vec_dim)

    self.encoder = MultiModalEncoder(enc_branches, embed_dim=embed_size)
    self.decoder = MultiModalDecoder(dec_heads)

    self.rssm = RSSM(
      action_dim=action_dim,
      embed_dim=embed_size,
      stoch_dim=config.model.rssm.stoch_size,
      deter_dim=config.model.rssm.deter_size,
      hidden=config.model.rssm.hidden_size,
      min_std=config.model.rssm.min_stddev,
    )
    self.reward_head = RewardHead(self.feat_dim, tuple(config.model.reward.hidden))

    self.aux_head: AuxHead | None = None
    if config.training.losses.aux_scale > 0.0:
      aux_cfg = config.model.aux
      self.aux_head = AuxHead(self.feat_dim, target_dim=int(aux_cfg.target_dim), hidden=tuple(aux_cfg.hidden))

    self.actor = Actor(
      feat_dim=self.feat_dim,
      action_dim=action_dim,
      hidden=tuple(config.model.actor.hidden),
      init_std=config.model.actor.init_stddev,
      min_std=config.model.actor.min_stddev,
      mean_scale=config.model.actor.mean_scale,
    )
    self.critic = Critic(self.feat_dim, tuple(config.model.critic.hidden))

    # The WM optimizer covers encoder + rssm + decoder + reward + aux.
    # We keep the same grouping as the MLX backend so trainer-side logic
    # (and the eventual actor/critic loop) reads identically.
    self._wm_bundle = _WMBundle(
      self.encoder, self.rssm, self.decoder, self.reward_head, self.aux_head
    )
    self._wm_bundle.to(self.device)
    self.actor.to(self.device)
    self.critic.to(self.device)

    if training:
      adam_eps = config.training.optimizer.adam_eps
      self._opt_wm     = torch.optim.Adam(self._wm_bundle.parameters(),
                                          lr=config.training.optimizer.wm_lr,
                                          eps=adam_eps)
      self._opt_actor  = torch.optim.Adam(self.actor.parameters(),
                                          lr=config.training.optimizer.actor_lr,
                                          eps=adam_eps)
      self._opt_critic = torch.optim.Adam(self.critic.parameters(),
                                          lr=config.training.optimizer.critic_lr,
                                          eps=adam_eps)

  # ---------------------------------------------------------------------
  # WorldModel protocol
  # ---------------------------------------------------------------------

  def _to_device(self, x):
    """Move a tensor / dict-of-tensors to the model's device. Numpy arrays
    are converted via :func:`torch.as_tensor`."""
    if isinstance(x, dict):
      return {k: self._to_device(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
      return x.to(self.device)
    return torch.as_tensor(np.asarray(x)).to(self.device)

  def coerce(self, x):
    """Lift numpy → ``torch.Tensor`` on the model's device.

    Delegates to :meth:`_to_device`, which already handles ndarray and
    dict-of-ndarray inputs (and is a no-op for tensors already on-device).
    """
    return self._to_device(x)

  def initial_state(self, batch_size: int) -> State:
    return self.rssm.initial_state(batch_size, device=self.device)

  def encode(self, obs) -> torch.Tensor:
    return cast(torch.Tensor, self.encoder(self._to_device(obs)))

  def posterior_step(
    self, state: State, action: torch.Tensor, embed: torch.Tensor, *, sample: bool = True,
  ) -> State:
    return self.rssm.obs_step(state, action, embed, sample=sample)

  def prior_step(self, state: State, action: torch.Tensor, *, sample: bool = True) -> State:
    return self.rssm.img_step(state, action, sample=sample)

  def feat(self, state: State) -> torch.Tensor:
    return feat(state)

  def decode(self, feats: torch.Tensor):
    return self.decoder(feats)

  def predict_reward(self, feats: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, self.reward_head(feats))

  def actor_action(self, feats: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
    if sample:
      action, _, _ = self.actor(feats)
      return cast(torch.Tensor, action)
    return self.actor.mode(feats)

  def stack_along_time(self, xs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(xs, dim=1)

  def broadcast_to(self, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Backend-native broadcast. Used by CEM to expand a B=1 state across samples."""
    return torch.broadcast_to(x, shape)

  # ---------------------------------------------------------------------
  # Training-time surface (mirrors mlx_model.DreamerMLXModel)
  # ---------------------------------------------------------------------

  def _recon_loss(self, recon: Observation, target: Observation) -> torch.Tensor:
    """Sum of per-component reconstruction losses.

    Image components use pixelwise MSE with optional focus-mask
    weighting (``1 + focus_scale * mask`` over the spatial axes).
    Vector components use plain MSE summed over the feature axis.
    Both backends produce float reconstructions matching the encoder's
    pre-normalized image range.
    """
    total: torch.Tensor | None = None
    for name in target.components:
      r = recon.components[name]
      t = target.components[name]
      if name == IMAGE:
        diff = (r - t) ** 2
        if target.focus_mask is not None:
          diff = diff * (1 + self.config.training.losses.focus_scale * target.focus_mask[..., None])
        loss = diff.sum(dim=(-3, -2, -1)).mean()
      else:
        loss = ((r - t) ** 2).sum(-1).mean()
      total = loss if total is None else total + loss
    assert total is not None  # from_components rejects empty dicts
    return cast(torch.Tensor, total)

  def _move_batch(self, batch: dict) -> dict:
    """Move every leaf in ``batch`` to ``self.device``. Image leaves stay
    uint8 — the encoder/decoder normalize on the fly."""
    out: dict = {}
    for k, v in batch.items():
      if isinstance(v, dict):
        out[k] = {kk: self._to_device(vv) for kk, vv in v.items()}
      else:
        out[k] = self._to_device(v)
    return out

  def _wm_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
    target_obs = Observation.from_buffer_value(
      batch["obs"], self.obs_mode, focus_mask=batch.get("focus_mask"),
    )
    action = batch["action"]
    reward = batch["reward"]
    B, T   = target_obs.leading_shape[0], target_obs.leading_shape[1]

    embeds = self._wm_bundle.encoder(target_obs.to_buffer_value())   # (B, T, embed_dim)
    state  = self._wm_bundle.rssm.initial_state(B, device=self.device)
    states: list[State] = []
    kls_per_dim = []
    for t in range(T):
      state = self._wm_bundle.rssm.obs_step(state, action[:, t], embeds[:, t])
      states.append(state)
      assert state.posterior is not None
      kl_t = kl_gaussian(
        state.posterior.mean, state.posterior.std,
        state.prior.mean,     state.prior.std,
      )
      kls_per_dim.append(kl_t)

    feats    = torch.stack([feat(s) for s in states], dim=1)   # (B, T, feat_dim)
    recon    = self._wm_bundle.decoder(feats)
    rew_pred = self._wm_bundle.reward(feats)

    recon_obs  = Observation.from_buffer_value(recon, self.obs_mode)
    recon_loss = self._recon_loss(recon_obs, target_obs)
    rew_loss   = ((rew_pred - reward) ** 2).mean()

    kl_per_dim_btmean = torch.stack(kls_per_dim, dim=1).mean(dim=(0, 1))  # (stoch_dim,)
    kl                = kl_per_dim_btmean.sum()
    kl                = torch.maximum(kl, torch.tensor(self.config.training.losses.free_nats,
                                                       device=self.device))

    loss = (self.config.training.losses.recon_scale * recon_loss
         + self.config.training.losses.reward_scale * rew_loss
         + self.config.training.losses.kl_scale     * kl)

    aux_out: dict = {"recon": recon_loss, "rew": rew_loss, "kl": kl,
                     "kl_per_dim": kl_per_dim_btmean,
                     "post_states": states}
    if self.aux_head is not None:
      aux_pred       = self.aux_head(feats)
      aux_loss       = ((aux_pred - batch["aux_target"]) ** 2).sum(-1).mean()
      loss           = loss + self.config.training.losses.aux_scale * aux_loss
      aux_out["aux"] = aux_loss

    return loss, aux_out

  def wm_update(self, batch, tracker=None):
    """Compute world-model loss and step the WM optimizer."""
    batch = self._move_batch(batch)
    self._opt_wm.zero_grad(set_to_none=True)
    loss, aux = self._wm_loss(batch)
    loss.backward()

    grad_norm = None
    if self.config.training.optimizer.gradient_clipping > 0:
      max_norm  = self.config.training.optimizer.gradient_clipping
      grad_norm = torch.nn.utils.clip_grad_norm_(self._wm_bundle.parameters(), max_norm)
    self._opt_wm.step()

    if tracker is not None:
      kl_per_dim = aux["kl_per_dim"].detach().cpu().numpy()
      logs = {
        "wm/loss":  float(loss.item()),
        "wm/recon": float(aux["recon"].item()),
        "wm/rew":   float(aux["rew"].item()),
        "wm/kl":    float(aux["kl"].item()),
      }
      logs["wm/kl_active_dims"]    = float((kl_per_dim > 0.1).sum())
      logs["wm/kl_mean_per_dim"]   = float(kl_per_dim.mean())
      logs["wm/kl_per_dim_min"]    = float(kl_per_dim.min())
      logs["wm/kl_per_dim_p10"]    = float(np.percentile(kl_per_dim, 10))
      logs["wm/kl_per_dim_median"] = float(np.percentile(kl_per_dim, 50))
      logs["wm/kl_per_dim_p90"]    = float(np.percentile(kl_per_dim, 90))
      logs["wm/kl_per_dim_max"]    = float(kl_per_dim.max())
      if "aux" in aux:
        logs["wm/aux"] = float(aux["aux"].item())
      if grad_norm is not None:
        gn = float(grad_norm.item())
        max_norm = self.config.training.optimizer.gradient_clipping
        logs["wm/grad_norm"]    = gn
        logs["wm/grad_clipped"] = float(gn > max_norm)
      tracker.log(logs)

    return aux["post_states"]

  # ---------------------------------------------------------------------
  # Persistence — MLX-compatible on-disk format
  # ---------------------------------------------------------------------
  #
  # See the module docstring for the full translation table. The two
  # helpers below are the single source of truth for the mapping; both
  # save/load and any future format-debugging tool go through them.

  def _iter_mlx_state(self) -> dict[str, torch.Tensor]:
    """Snapshot all trainable params as MLX-format flat keys.

    Walks the torch module tree top-down, translating each segment to
    its MLX equivalent and permuting/splitting per-leaf-module as
    needed. Returns a flat ``{mlx_key: cpu_tensor}`` dict ready for
    safetensors.
    """
    out: dict[str, torch.Tensor] = {}
    for prefix, module in (("wm", self._wm_bundle),
                           ("actor",  self.actor),
                           ("critic", self.critic)):
      self._collect_mlx_state(prefix, module, out)
    return out

  def _collect_mlx_state(
    self, prefix: str, module: nn.Module, out: dict[str, torch.Tensor],
  ) -> None:
    # Direct parameters on this module.
    for name, p in module.named_parameters(recurse=False):
      key = f"{prefix}.{name}"
      out[key] = p.detach().cpu().contiguous()
    # Then descend into children. For each child, compute its MLX path
    # segment(s) and dispatch by leaf type.
    for child_name, child in _iter_mlx_children(module):
      qualified = f"{prefix}.{child_name}" if child_name else prefix
      if isinstance(child, nn.Conv2d):
        out[f"{qualified}.weight"] = child.weight.detach().cpu().permute(0, 2, 3, 1).contiguous()
        if child.bias is not None:
          out[f"{qualified}.bias"] = child.bias.detach().cpu().contiguous()
      elif isinstance(child, nn.ConvTranspose2d):
        out[f"{qualified}.weight"] = child.weight.detach().cpu().permute(1, 2, 3, 0).contiguous()
        if child.bias is not None:
          out[f"{qualified}.bias"] = child.bias.detach().cpu().contiguous()
      elif isinstance(child, nn.GRUCell):
        Wx, Wh, b, bhn = _torch_grucell_to_mlx(child)
        out[f"{qualified}.Wx"]  = Wx
        out[f"{qualified}.Wh"]  = Wh
        out[f"{qualified}.b"]   = b
        out[f"{qualified}.bhn"] = bhn
      elif isinstance(child, _ChannelLayerNorm):
        # The wrapper has a single sub-LayerNorm; MLX stores it bare so
        # we hoist the inner weights to the wrapper's path.
        ln = child.norm
        out[f"{qualified}.weight"] = ln.weight.detach().cpu().contiguous()
        out[f"{qualified}.bias"]   = ln.bias.detach().cpu().contiguous()
      else:
        self._collect_mlx_state(qualified, child, out)

  def _apply_mlx_state(self, weights: dict[str, torch.Tensor]) -> None:
    """Inverse of :meth:`_iter_mlx_state` — populate torch modules from
    an MLX-format flat dict.
    """
    for prefix, module in (("wm", self._wm_bundle),
                           ("actor",  self.actor),
                           ("critic", self.critic)):
      self._scatter_mlx_state(prefix, module, weights)
    self._wm_bundle.to(self.device)
    self.actor.to(self.device)
    self.critic.to(self.device)

  def _scatter_mlx_state(
    self, prefix: str, module: nn.Module, weights: dict[str, torch.Tensor],
  ) -> None:
    with torch.no_grad():
      for name, p in module.named_parameters(recurse=False):
        src = weights[f"{prefix}.{name}"]
        p.copy_(src.to(p.device, dtype=p.dtype))
      for child_name, child in _iter_mlx_children(module):
        qualified = f"{prefix}.{child_name}" if child_name else prefix
        if isinstance(child, nn.Conv2d):
          w = weights[f"{qualified}.weight"].permute(0, 3, 1, 2).contiguous()
          child.weight.copy_(w.to(child.weight.device, dtype=child.weight.dtype))
          if child.bias is not None:
            child.bias.copy_(
              weights[f"{qualified}.bias"].to(child.bias.device, dtype=child.bias.dtype)
            )
        elif isinstance(child, nn.ConvTranspose2d):
          w = weights[f"{qualified}.weight"].permute(3, 0, 1, 2).contiguous()
          child.weight.copy_(w.to(child.weight.device, dtype=child.weight.dtype))
          if child.bias is not None:
            child.bias.copy_(
              weights[f"{qualified}.bias"].to(child.bias.device, dtype=child.bias.dtype)
            )
        elif isinstance(child, nn.GRUCell):
          _mlx_to_torch_grucell(
            child,
            Wx  = weights[f"{qualified}.Wx"],
            Wh  = weights[f"{qualified}.Wh"],
            b   = weights[f"{qualified}.b"],
            bhn = weights[f"{qualified}.bhn"],
          )
        elif isinstance(child, _ChannelLayerNorm):
          ln = child.norm
          ln.weight.copy_(
            weights[f"{qualified}.weight"].to(ln.weight.device, dtype=ln.weight.dtype)
          )
          ln.bias.copy_(
            weights[f"{qualified}.bias"].to(ln.bias.device, dtype=ln.bias.dtype)
          )
        else:
          self._scatter_mlx_state(qualified, child, weights)

  def save(self, path: str, extra_metadata: dict[str, str] | None = None) -> None:
    """Persist weights to a safetensors file with the training config in
    metadata.

    The on-disk key layout matches the MLX backend (see module docstring);
    optimizer state is NOT included.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    weights = self._iter_mlx_state()
    metadata = {CONFIG_METADATA_KEY: OmegaConf.to_yaml(self.config, resolve=True)}
    if extra_metadata:
      if CONFIG_METADATA_KEY in extra_metadata:
        raise ValueError(f"extra_metadata may not set reserved key {CONFIG_METADATA_KEY!r}")
      metadata.update(extra_metadata)
    _st_save_file(weights, path, metadata=metadata)

  def load(self, path: str) -> dict[str, str]:
    """Load weights written by :meth:`save` (or by the MLX backend).
    Returns the caller-supplied ``extra_metadata`` dict that was passed
    at save time.
    """
    from safetensors import safe_open
    weights: dict[str, torch.Tensor] = _st_load_file(path)
    with safe_open(path, framework="pt") as f:
      meta = f.metadata() or {}
    self._apply_mlx_state(weights)
    return {k: v for k, v in meta.items() if k != CONFIG_METADATA_KEY}


# ---------------------------------------------------------------------------
# MLX-format translation helpers (module-level so they can be tested
# independently of model state).
# ---------------------------------------------------------------------------


def _iter_mlx_children(module: nn.Module):
  """Yield ``(mlx_segment, child_module)`` pairs for the MLX-layout walk.

  Three cases diverge from ``module.named_children()``:

    * ``nn.Sequential`` wraps its numeric children under a ``layers``
      attribute in MLX, so we emit ``layers.<i>`` instead of ``<i>``.
    * ``MultiModalEncoder.branches`` (``nn.ModuleDict``) entries are
      individual ``b_<name>`` attributes on the MLX twin — there's no
      enclosing dict, so we skip the dict level and rename the entries.
      Ditto ``MultiModalDecoder.heads`` → ``h_<name>``.
  """
  if isinstance(module, nn.Sequential):
    for name, child in module.named_children():
      yield f"layers.{name}", child
    return
  if isinstance(module, MultiModalEncoder):
    for name, child in module.named_children():
      if name == "branches":
        for branch_name, branch in child.named_children():
          yield f"b_{branch_name}", branch
      else:
        yield name, child
    return
  if isinstance(module, MultiModalDecoder):
    for name, child in module.named_children():
      if name == "heads":
        for head_name, head in child.named_children():
          yield f"h_{head_name}", head
      else:
        yield name, child
    return
  for name, child in module.named_children():
    yield name, child


def _torch_grucell_to_mlx(
  gru: nn.GRUCell,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Convert a torch ``GRUCell`` state_dict to MLX ``GRU`` tensors.

  Both backends share the (r, z, n) gate ordering. The MLX layer carries
  a single combined bias ``b`` for the (r, z) gates because its forward
  adds the hidden projection without a hidden-side bias on those gates;
  for the n gate it carries a separate ``bhn`` applied inside the
  ``r * (W_hn h + bhn)`` term. The conversions:

    Wx  = weight_ih                       # (3H, in)
    Wh  = weight_hh                       # (3H, H)
    b[r] = bias_ih[r] + bias_hh[r]
    b[z] = bias_ih[z] + bias_hh[z]
    b[n] = bias_ih[n]
    bhn  = bias_hh[n]                     # (H,)
  """
  H        = gru.hidden_size
  b_ih     = gru.bias_ih.detach().cpu()
  b_hh     = gru.bias_hh.detach().cpu()
  b_rz     = b_ih[: 2 * H] + b_hh[: 2 * H]
  b_n_ih   = b_ih[2 * H :]
  b_n_hh   = b_hh[2 * H :]
  b        = torch.cat([b_rz, b_n_ih], dim=0).contiguous()
  return (
    gru.weight_ih.detach().cpu().contiguous(),
    gru.weight_hh.detach().cpu().contiguous(),
    b,
    b_n_hh.contiguous(),
  )


def _mlx_to_torch_grucell(
  gru: nn.GRUCell,
  *, Wx: torch.Tensor, Wh: torch.Tensor, b: torch.Tensor, bhn: torch.Tensor,
) -> None:
  """Inverse of :func:`_torch_grucell_to_mlx`.

  The split of the (r, z) bias between ``bias_ih`` and ``bias_hh`` is
  ambiguous — MLX only stores the sum. We park the entire sum in
  ``bias_ih`` and zero out the matching ``bias_hh`` rows; the GRU
  forward pass is invariant to this choice.
  """
  H   = gru.hidden_size
  dev = gru.weight_ih.device
  dt  = gru.weight_ih.dtype
  with torch.no_grad():
    gru.weight_ih.copy_(Wx.to(dev, dtype=dt))
    gru.weight_hh.copy_(Wh.to(gru.weight_hh.device, dtype=gru.weight_hh.dtype))
    b_ih = torch.zeros_like(gru.bias_ih)
    b_hh = torch.zeros_like(gru.bias_hh)
    b_ih[: 2 * H] = b[: 2 * H].to(b_ih.device, dtype=b_ih.dtype)
    b_ih[2 * H :] = b[2 * H :].to(b_ih.device, dtype=b_ih.dtype)
    b_hh[2 * H :] = bhn.to(b_hh.device, dtype=b_hh.dtype)
    gru.bias_ih.copy_(b_ih)
    gru.bias_hh.copy_(b_hh)
