"""PyTorch implementation of the Dreamer v1 world model.

Mirrors :mod:`.mlx_model`: same module structure (encoder / decoder / RSSM
/ reward / actor / critic / optional aux) and the same checkpoint schema
embedded in safetensors metadata. The public surface implements the
:class:`~chuck_dreamer.dreamer.world_model.WorldModel` protocol so
:func:`~chuck_dreamer.dreamer.world_model.rollout` composes against it
unchanged.

Cross-backend incompatibilities — these are known and intentional. The
MLX and PyTorch checkpoints are NOT interchangeable. Items to resolve if
cross-backend portability becomes required:

  * Conv2d weight layout. PyTorch's nn.Conv2d weight is
    ``(C_out, C_in, kH, kW)``; MLX's is ``(C_out, kH, kW, C_in)``. Same for
    ConvTranspose2d. A loader would need to permute conv weights on
    cross-backend reads.
  * Image tensor layout. PyTorch convs are NCHW natively; MLX is NHWC.
    This backend transposes at the encoder input and decoder output so
    the public surface stays NHWC, matching the MLX backend and the
    replay buffer. Trajectories produced by either backend have
    identical shapes.
  * GRU implementation. MLX uses ``nn.GRU`` driven with ``(B, 1, D)``
    inputs; this backend uses ``nn.GRUCell`` which takes ``(B, D)``
    directly. Weights map across (W_ih, W_hh, b_ih, b_hh) but the
    parameter names differ.
  * Parameter initialization. Default inits differ between mlx.nn and
    torch.nn — randomness in weight scale propagates to the first few
    training steps. Long-run training is unaffected.
  * Save format. Both use ``.safetensors``, but the key prefixes
    (``wm.rssm.gru.weight_ih`` vs ``wm.rssm.gru.Wx``) come from the
    framework's own state_dict / tree_flatten output. Cross-backend
    loaders would need a key translation table.
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

from ..training.observation import Observation
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
  """Standard MLP: Linear -> act -> ... -> Linear (no act on output)."""
  layers: list[nn.Module] = []
  prev = in_dim
  for h in hidden:
    layers.append(nn.Linear(prev, h))
    layers.append(act())
    prev = h
  layers.append(nn.Linear(prev, out_dim))
  return nn.Sequential(*layers)


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

    convs: list[nn.Module] = []
    prev = in_channels
    for c, k, s in zip(channels, kernels, strides):
      convs.append(nn.Conv2d(prev, c, kernel_size=k, stride=s, padding=_same_pad(k, s)))
      convs.append(nn.ELU())
      prev = c
    self.convs = nn.Sequential(*convs)

    # End spatial side after strided convs (each layer divides cleanly).
    spatial = image_size
    for s in strides:
      spatial //= int(s)
    self._flat_dim = prev * spatial * spatial
    self.proj = nn.Linear(self._flat_dim, embed_dim)

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
    x = x.reshape(x.shape[0], -1)
    x = self.proj(x)
    return x.reshape(*lead, -1)


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
    for i, (c_out, k, s) in enumerate(zip(next_channels, rev_kernels, rev_strides)):
      deconvs.append(
        nn.ConvTranspose2d(prev, c_out, kernel_size=k, stride=s, padding=_same_pad(k, s))
      )
      if i < len(next_channels) - 1:
        deconvs.append(nn.ELU())
      prev = c_out
    self.deconvs = nn.Sequential(*deconvs)

  def forward(self, feat_in: torch.Tensor) -> torch.Tensor:
    lead = feat_in.shape[:-1]
    x = self.proj(feat_in)
    x = x.reshape(-1, self._init_channels, self._init_spatial, self._init_spatial)  # NCHW
    x = self.deconvs(x)
    # NCHW -> NHWC for the public boundary.
    x = x.permute(0, 2, 3, 1).contiguous()
    H, W, C = x.shape[-3], x.shape[-2], x.shape[-1]
    return x.reshape(*lead, H, W, C)


# ---------------------------------------------------------------------------
# image_proprio: combine a CNN image branch with an MLP proprio branch
# ---------------------------------------------------------------------------


class ImageProprioEncoder(nn.Module):
  """Concat CNN(image) and MLP(proprio) features, project to ``embed_dim``."""

  def __init__(
    self,
    cnn: CNNEncoder,
    proprio_dim: int,
    proprio_hidden: tuple[int, ...],
    embed_dim: int,
  ):
    super().__init__()
    self.cnn = cnn
    self.proprio_mlp = _mlp(proprio_dim, proprio_hidden, embed_dim)
    self.fuse = nn.Linear(2 * embed_dim, embed_dim)

  def forward(self, obs: dict) -> torch.Tensor:
    img_feat = self.cnn(obs["image"])
    pro_feat = self.proprio_mlp(obs["proprio"])
    return cast(torch.Tensor, self.fuse(torch.cat([img_feat, pro_feat], dim=-1)))


class ImageProprioDecoder(nn.Module):
  """Two-headed decoder: CNN reconstructs the image, MLP the proprio vector."""

  def __init__(
    self,
    feat_dim: int,
    cnn: CNNDecoder,
    proprio_hidden: tuple[int, ...],
    proprio_dim: int,
  ):
    super().__init__()
    self.cnn = cnn
    self.proprio_mlp = _mlp(feat_dim, proprio_hidden, proprio_dim)

  def forward(self, feat_in: torch.Tensor) -> dict:
    return {
      "image":   self.cnn(feat_in),
      "proprio": self.proprio_mlp(feat_in),
    }


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

    self.pre_gru = nn.Sequential(
      nn.Linear(stoch_dim + action_dim, hidden),
      nn.ELU(),
    )
    self.gru = nn.GRUCell(input_size=hidden, hidden_size=deter_dim)

    self.prior_net = nn.Sequential(
      nn.Linear(deter_dim, hidden), nn.ELU(),
      nn.Linear(hidden, 2 * stoch_dim),
    )
    self.post_net = nn.Sequential(
      nn.Linear(deter_dim + embed_dim, hidden), nn.ELU(),
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
    return dist.mean + dist.std * eps

  def _compute_h(self, prev_s: torch.Tensor, prev_a: torch.Tensor, prev_h: torch.Tensor) -> torch.Tensor:
    x = self.pre_gru(torch.cat([prev_s, prev_a], dim=-1))   # (B, hidden)
    return self.gru(x, prev_h)                              # (B, deter_dim)

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
    return self.net(feat_in).squeeze(-1)


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
    return self.net(feat_in).squeeze(-1)


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

    obs_mode = config.env.obs_mode
    self.obs_mode = obs_mode

    enc_hidden = tuple(config.model.encoder.mlp_hidden)
    dec_hidden = tuple(config.model.decoder.mlp_hidden)
    embed_size = int(config.model.encoder.embed_size)

    if obs_mode == "state":
      if not (isinstance(obs_shape, tuple) and len(obs_shape) == 1):
        raise ValueError(f"state obs_mode expects 1-D obs_shape; got {obs_shape}")
      obs_dim          = int(obs_shape[0])
      self.encoder     = MLPEncoder(obs_dim, enc_hidden, embed_size)
      self.decoder     = MLPDecoder(self.feat_dim, dec_hidden, obs_dim)
      self.image_size  = None
      self.proprio_dim = None

    elif obs_mode in ("image", "image_proprio"):
      enc_channels    = tuple(config.model.encoder.cnn_channels)
      enc_kernels     = tuple(config.model.encoder.cnn_kernels)
      enc_strides     = tuple(config.model.encoder.cnn_strides)
      dec_channels    = tuple(config.model.decoder.cnn_channels)
      dec_kernels     = tuple(config.model.decoder.cnn_kernels)
      dec_strides     = tuple(config.model.decoder.cnn_strides)
      image_size      = int(config.model.encoder.image_size)
      self.image_size = image_size

      if obs_mode == "image":
        if not (isinstance(obs_shape, tuple) and len(obs_shape) == 3):
          raise ValueError(f"image obs_mode expects 3-D obs_shape; got {obs_shape}")
        in_channels = int(obs_shape[2])
        self.encoder = CNNEncoder(in_channels, enc_channels, enc_kernels, enc_strides,
                                  embed_dim=embed_size, image_size=image_size)
        self.decoder = CNNDecoder(self.feat_dim, dec_channels, dec_kernels, dec_strides,
                                  out_channels=in_channels, image_size=image_size)
        self.proprio_dim = None

      else:  # image_proprio
        if not (hasattr(obs_shape, "__getitem__") and "image" in obs_shape and "proprio" in obs_shape):
          raise ValueError(
            f"image_proprio obs_mode expects shape dict with 'image' and 'proprio' keys; got {obs_shape}"
          )
        img_shape        = tuple(obs_shape["image"])
        pro_shape        = tuple(obs_shape["proprio"])
        in_channels      = int(img_shape[2])
        proprio_dim      = int(pro_shape[0])
        self.proprio_dim = proprio_dim
        cnn_enc          = CNNEncoder(in_channels, enc_channels, enc_kernels, enc_strides,
                                      embed_dim=embed_size, image_size=image_size)
        cnn_dec          = CNNDecoder(self.feat_dim, dec_channels, dec_kernels, dec_strides,
                                      out_channels=in_channels, image_size=image_size)
        self.encoder     = ImageProprioEncoder(cnn_enc, proprio_dim, enc_hidden, embed_size)
        self.decoder     = ImageProprioDecoder(self.feat_dim, cnn_dec, dec_hidden, proprio_dim)

    else:
      raise ValueError(f"unknown obs_mode={obs_mode!r}")

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
    # ``recon`` and ``target`` are both Observations in the same mode.
    # Image fields arrive pre-normalized (``[-0.5, 0.5]``) so the decoder
    # output and target share a scale. ``target.focus_mask`` (B, T, H, W)
    # multiplies image-pixel errors by ``1 + focus_scale * mask``.
    if target.mode == "state":
      return ((recon.state - target.state) ** 2).sum(-1).mean()
    diff = (recon.image - target.image) ** 2
    if target.focus_mask is not None:
      diff = diff * (1 + self.config.training.losses.focus_scale * target.focus_mask[..., None])
    img_loss = diff.sum(dim=(-3, -2, -1)).mean()
    if target.mode == "image":
      return img_loss
    if target.mode == "image_proprio":
      pro_loss = ((recon.proprio - target.proprio) ** 2).sum(-1).mean()
      return img_loss + pro_loss
    raise ValueError(f"unknown obs mode: {target.mode!r}")

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
  # Persistence
  # ---------------------------------------------------------------------

  def _named_param_dict(self) -> dict[str, torch.Tensor]:
    """Flat ``{prefix.name: tensor}`` snapshot of all trainable params.

    Keys mirror the MLX backend's prefixes (``wm.*``, ``actor.*``,
    ``critic.*``) but the trailing portion is the torch state_dict path,
    e.g. ``wm.rssm.gru.weight_ih`` rather than ``wm.rssm.gru.Wx``.
    """
    out: dict[str, torch.Tensor] = {}
    for prefix, module in (("wm", self._wm_bundle),
                           ("actor",  self.actor),
                           ("critic", self.critic)):
      for k, v in module.state_dict().items():
        out[f"{prefix}.{k}"] = v.detach().cpu().contiguous()
    return out

  def _load_named_params(self, weights: dict[str, torch.Tensor]) -> None:
    def take(prefix: str) -> dict[str, torch.Tensor]:
      plen = len(prefix) + 1
      return {k[plen:]: v for k, v in weights.items() if k.startswith(prefix + ".")}

    self._wm_bundle.load_state_dict(take("wm"))
    self.actor.load_state_dict(take("actor"))
    self.critic.load_state_dict(take("critic"))
    self._wm_bundle.to(self.device)
    self.actor.to(self.device)
    self.critic.to(self.device)

  def save(self, path: str, extra_metadata: dict[str, str] | None = None) -> None:
    """Persist weights to a safetensors file with the training config in
    metadata. Optimizer state is NOT included.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    weights = self._named_param_dict()
    metadata = {CONFIG_METADATA_KEY: OmegaConf.to_yaml(self.config, resolve=True)}
    if extra_metadata:
      if CONFIG_METADATA_KEY in extra_metadata:
        raise ValueError(f"extra_metadata may not set reserved key {CONFIG_METADATA_KEY!r}")
      metadata.update(extra_metadata)
    _st_save_file(weights, path, metadata=metadata)

  def load(self, path: str) -> dict[str, str]:
    """Load weights written by :meth:`save`. Returns the caller-supplied
    ``extra_metadata`` dict that was passed at save time.
    """
    from safetensors import safe_open
    weights: dict[str, torch.Tensor] = _st_load_file(path)
    with safe_open(path, framework="pt") as f:
      meta = f.metadata() or {}
    self._load_named_params(weights)
    return {k: v for k, v in meta.items() if k != CONFIG_METADATA_KEY}
