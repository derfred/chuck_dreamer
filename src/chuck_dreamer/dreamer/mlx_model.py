import os
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from omegaconf import OmegaConf

from .world_model import Distribution, State

# Key under which the OmegaConf-serialized training config is stored in the
# safetensors metadata block.
CONFIG_METADATA_KEY = "config_yaml"


def load_config_from_checkpoint(path: str):
  """Read the embedded training config from a .safetensors checkpoint.

  Returns the DictConfig that was saved alongside the weights, or ``None``
  if the checkpoint has no metadata. Callers that need shape-affecting
  fields (env.obs_mode, env.act_mode, model.*) should use this rather than
  re-reading the on-disk default config, which may have drifted.

  The saved YAML is fully resolved (see :meth:`DreamerMLXModel.save`), so
  this does not depend on any custom OmegaConf resolvers being registered.
  """
  _, metadata = mx.load(path, return_metadata=True)
  yaml_str = metadata.get(CONFIG_METADATA_KEY)
  if not yaml_str:
    return None
  return OmegaConf.create(yaml_str)


# Helpers
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


def feat(state: State) -> mx.array:
  """Concatenate deterministic and stochastic state into the feature fed
  to decoder, reward head, actor, and critic."""
  return mx.concatenate([state.h, state.s], axis=-1)


def kl_gaussian(
  mean_q: mx.array,
  std_q:  mx.array,
  mean_p: mx.array,
  std_p:  mx.array,
) -> mx.array:
  """KL(q || p) for diagonal Gaussians, per-dimension.

  Returns the per-dim KL with the same shape as the inputs. Caller decides
  whether to sum over the latent dim (typical) or reduce differently.

  Args:
    mean_q, std_q: parameters of q, the posterior in Dreamer's ELBO.
    mean_p, std_p: parameters of p, the prior.

  All four arrays broadcast together; typical shape is (B, stoch_dim) or
  (B, T, stoch_dim).
  """
  var_q = std_q ** 2
  var_p = std_p ** 2
  return (
    mx.log(std_p / std_q)
    + (var_q + (mean_q - mean_p) ** 2) / (2.0 * var_p)
    - 0.5
  )


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class MLPEncoder(nn.Module):
  """For state-based observations (joint qpos + ee pose + object pose, etc.).
  Produces an embedding the RSSM consumes."""

  def __init__(self, obs_dim: int, hidden: tuple[int, ...], embed_dim: int):
    super().__init__()
    self.net = _mlp(obs_dim, hidden, embed_dim)

  def __call__(self, obs: mx.array) -> mx.array:
    """obs: (..., obs_dim) -> embedding: (..., embed_dim)"""
    return cast(mx.array, self.net(obs))


class MLPDecoder(nn.Module):
  """Reconstructs state-based observations from latent features."""

  def __init__(self, feat_dim: int, hidden: tuple[int, ...], obs_dim: int):
    super().__init__()
    self.net = _mlp(feat_dim, hidden, obs_dim)

  def __call__(self, feat_in: mx.array) -> mx.array:
    return cast(mx.array, self.net(feat_in))


# ---------------------------------------------------------------------------
# CNN encoder / decoder for image obs
# ---------------------------------------------------------------------------


def _same_pad(kernel: int, stride: int) -> int:
  """Padding that makes a stride-2-style conv halve spatial dims cleanly.

  For kernel=4 stride=2 this gives padding=1, for kernel=3 stride=1
  padding=1, etc.: enough on each side that ``out = in / stride``.
  """
  return max(0, (kernel - stride) // 2)


class CNNEncoder(nn.Module):
  """Dreamer-style CNN over (H, W, 3) uint8 images.

  Inputs may have arbitrary batch dims; they are flattened, run through
  the conv stack in NHWC, then flattened to (..., embed_dim).
  Inputs are normalized to ``[-0.5, 0.5]``.
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

    # Spatial dim after the strided convs (with `_same_pad` choice above):
    # each layer divides cleanly by its stride, so end side = image_size / prod(strides).
    spatial = image_size
    for s in strides:
      spatial //= int(s)
    self._flat_dim = prev * spatial * spatial
    self.proj = nn.Linear(self._flat_dim, embed_dim)

  def __call__(self, obs: mx.array) -> mx.array:
    """obs: (..., H, W, 3) — uint8 or float. Returns (..., embed_dim)."""
    x = obs.astype(mx.float32) / 255.0 - 0.5 if obs.dtype == mx.uint8 else obs
    lead = x.shape[:-3]
    H, W, C = x.shape[-3], x.shape[-2], x.shape[-1]
    x = x.reshape((-1, H, W, C))
    x = self.convs(x)
    x = x.reshape((x.shape[0], -1))
    x = self.proj(x)
    return cast(mx.array, x.reshape((*lead, -1)))


class CNNDecoder(nn.Module):
  """Mirror of :class:`CNNEncoder`. Maps (..., feat_dim) -> (..., H, W, 3).

  Output is in ``[-0.5, 0.5]`` (matching the encoder's normalization);
  callers comparing against uint8 images should normalize the target the
  same way.
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
    # Build N-1 transposed-conv stages that step back through the encoder
    # widths; the last stage outputs `out_channels` directly.
    next_channels = list(rev_channels[1:]) + [out_channels]
    for i, (c_out, k, s) in enumerate(zip(next_channels, rev_kernels, rev_strides)):
      deconvs.append(
        nn.ConvTranspose2d(prev, c_out, kernel_size=k, stride=s, padding=_same_pad(k, s))
      )
      if i < len(next_channels) - 1:
        deconvs.append(nn.ELU())
      prev = c_out
    self.deconvs = nn.Sequential(*deconvs)

  def __call__(self, feat_in: mx.array) -> mx.array:
    lead = feat_in.shape[:-1]
    x = self.proj(feat_in)
    x = x.reshape((-1, self._init_spatial, self._init_spatial, self._init_channels))
    x = self.deconvs(x)
    H, W, C = x.shape[-3], x.shape[-2], x.shape[-1]
    return cast(mx.array, x.reshape((*lead, H, W, C)))


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

  def __call__(self, obs: dict) -> mx.array:
    img_feat = self.cnn(obs["image"])
    pro_feat = self.proprio_mlp(obs["proprio"])
    return cast(mx.array, self.fuse(mx.concatenate([img_feat, pro_feat], axis=-1)))


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

  def __call__(self, feat_in: mx.array) -> dict:
    return {
      "image":   self.cnn(feat_in),
      "proprio": self.proprio_mlp(feat_in),
    }


# ---------------------------------------------------------------------------
# RSSM: Recurrent State Space Model
# ---------------------------------------------------------------------------


class RSSM(nn.Module):
  """Dreamer v1 RSSM.

  State = (h, s) where:
    h_t = GRU(h_{t-1}, fc([s_{t-1}, a_{t-1}]))         # deterministic
    s_t ~ N(prior(h_t))                                 # stochastic prior
    s_t ~ N(posterior(h_t, embed_t)) when observation present

  Methods:
    initial_state(B)              -> dict with zero h, s
    img_step(state, action)       -> next state using prior only (imagination)
    obs_step(state, action, embed)-> next state using posterior (training)
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

    # pre-GRU MLP combining (s_{t-1}, a_{t-1})
    self.pre_gru = nn.Sequential(
      nn.Linear(stoch_dim + action_dim, hidden),
      nn.ELU(),
    )
    # GRU cell. Single-step interface; we drive it manually.
    self.gru = nn.GRU(input_size=hidden, hidden_size=deter_dim)

    # Prior: h -> (mean, std) of s
    self.prior_net = nn.Sequential(
      nn.Linear(deter_dim, hidden), nn.ELU(),
      nn.Linear(hidden, 2 * stoch_dim),
    )

    # Posterior: (h, embed) -> (mean, std) of s
    self.post_net = nn.Sequential(
      nn.Linear(deter_dim + embed_dim, hidden), nn.ELU(),
      nn.Linear(hidden, 2 * stoch_dim),
    )

  # ------------------- helpers -------------------

  def initial_state(self, batch_size: int) -> State:
    """Zero-initialized state.

    The ``prior`` field carries a placeholder distribution (mean=0,
    std=1); the first dynamics step overwrites it before anything reads
    it. ``posterior`` is ``None`` until the first :meth:`obs_step`.
    """
    h = mx.zeros((batch_size, self.deter_dim))
    s = mx.zeros((batch_size, self.stoch_dim))
    placeholder = Distribution(mean=s, std=mx.ones((batch_size, self.stoch_dim)))
    return State(h=h, s=s, prior=placeholder, posterior=None)

  def _split_dist(self, out: mx.array) -> Distribution:
    mean, std = mx.split(out, 2, axis=-1)
    std = nn.softplus(std) + self.min_std
    return Distribution(mean=cast(mx.array, mean), std=cast(mx.array, std))

  def _sample(self, dist: Distribution) -> mx.array:
    return cast(mx.array, dist.mean + dist.std * mx.random.normal(dist.mean.shape))

  # ------------------- core ops -------------------

  def _compute_h(self, prev_s: mx.array, prev_a: mx.array, prev_h: mx.array) -> mx.array:
    """One GRU step: h_t = GRU(h_{t-1}, mlp([s_{t-1}, a_{t-1}]))."""
    x = self.pre_gru(mx.concatenate([prev_s, prev_a], axis=-1))   # (B, hidden)
    # MLX GRU expects (B, T, D); we pass T=1 and squeeze.
    h_seq = self.gru(x[:, None, :], prev_h)                       # (B, 1, deter_dim)
    return cast(mx.array, h_seq[:, 0])                            # (B, deter_dim)

  def img_step(self, prev_state: State, prev_action: mx.array, *, sample: bool = True) -> State:
    """Imagination step: predict s_t from prior only, no observation.

    ``sample=False`` returns ``s = prior.mean`` (no noise) — used for
    deterministic open-loop rollouts during eval.
    """
    h     = self._compute_h(prev_state.s, prev_action, prev_state.h)
    prior = self._split_dist(self.prior_net(h))
    s     = self._sample(prior) if sample else prior.mean
    return State(h=h, s=s, prior=prior, posterior=None)

  def obs_step(self, prev_state: State, prev_action: mx.array, embed: mx.array, *, sample: bool = True) -> State:
    """Observation step: run prior, then refine with posterior using embed.

    ``sample=False`` returns ``s = posterior.mean`` (no noise) — used for
    deterministic posterior rollouts during eval / probing.
    """
    h         = self._compute_h(prev_state.s, prev_action, prev_state.h)
    prior     = self._split_dist(self.prior_net(h))
    post_in   = mx.concatenate([h, embed], axis=-1)
    posterior = self._split_dist(self.post_net(post_in))
    s         = self._sample(posterior) if sample else posterior.mean
    return State(h=h, s=s, prior=prior, posterior=posterior)

  def observe(
    self,
    embeds:  mx.array,   # (B, T, embed_dim)
    actions: mx.array,   # (B, T, action_dim)
    init:    State | None = None,
  ) -> list[State]:
    """Roll the posterior forward for T steps. Returns a list of T States."""
    B, T   = embeds.shape[0], embeds.shape[1]
    state  = init if init is not None else self.initial_state(B)
    states = []
    for t in range(T):
      state = self.obs_step(state, actions[:, t], embeds[:, t])
      states.append(state)
    return states

  def imagine(
    self,
    init_state: State,                          # (N, ...) starting point
    policy_fn,                                   # feat -> action
    horizon: int,
  ) -> list[State]:
    """Roll the prior forward for `horizon` steps using a policy.
    policy_fn maps a feature tensor (N, feat_dim) to an action (N, action_dim).
    Returns a list of length horizon+1 (including the start state)."""
    state = init_state
    traj = [state]
    for _ in range(horizon):
      action = policy_fn(feat(state))
      state = self.img_step(state, action)
      traj.append(state)
    return traj


# ---------------------------------------------------------------------------
# Reward head
# ---------------------------------------------------------------------------


class RewardHead(nn.Module):
  """Predicts scalar reward from latent features."""

  def __init__(self, feat_dim: int, hidden: tuple[int, ...] = (200, 200)):
    super().__init__()
    self.net = _mlp(feat_dim, hidden, out_dim=1)

  def __call__(self, feat_in: mx.array) -> mx.array:
    return cast(mx.array, self.net(feat_in).squeeze(-1))


class AuxHead(nn.Module):
  """Auxiliary regression head from latent features to a privileged target.

  Used at training time only: a small MLP that reads ``feat(state)`` and
  predicts a low-dim ground-truth quantity (e.g. object_xy + ee_pos) that
  the model would otherwise fail to encode under pixel-MSE — small but
  task-relevant features get drowned out by the bulk of the image. The
  aux MSE creates direct gradient pressure on the encoder/RSSM to keep
  that information in the latent. Discarded at rollout/eval time.
  """

  def __init__(self, feat_dim: int, target_dim: int, hidden: tuple[int, ...] = (200, 200)):
    super().__init__()
    self.target_dim = target_dim
    self.net = _mlp(feat_dim, hidden, out_dim=target_dim)

  def __call__(self, feat_in: mx.array) -> mx.array:
    return cast(mx.array, self.net(feat_in))


# ---------------------------------------------------------------------------
# Actor: tanh-squashed Gaussian
# ---------------------------------------------------------------------------

class Actor(nn.Module):
  """Stochastic actor for continuous control. Outputs a tanh-squashed Gaussian.

  Implementation details matching Dreamer v1:
    - mean is passed through `mean_scale * tanh(mean / mean_scale)` to keep it
      in (-mean_scale, +mean_scale) before the final tanh squash on the sample.
    - std uses softplus(raw + init_std) + min_std so initial std ~= 1.0.

  Returns (action, mean_pre_squash, std).
  """

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
    # init_std offset: softplus(0 + init_std_raw) + min_std ~= 1.0
    # We add init_std as a constant inside softplus; pre-compute the offset.
    self._raw_std_offset = init_std

  def _dist(self, feat_in: mx.array) -> tuple[mx.array, mx.array]:
    out = self.net(feat_in)
    mean, raw_std = mx.split(out, 2, axis=-1)
    mean = self.mean_scale * mx.tanh(mean / self.mean_scale)
    std = nn.softplus(raw_std + self._raw_std_offset) + self.min_std
    return mean, std

  def __call__(self, feat_in: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Sample an action with reparameterization. Returns (action, mean, std).
    The action is tanh-squashed; mean/std describe the pre-squash Gaussian."""
    mean, std = self._dist(feat_in)
    eps = mx.random.normal(mean.shape)
    raw = mean + std * eps
    action = mx.tanh(raw)
    return action, mean, std

  def mode(self, feat_in: mx.array) -> mx.array:
    """Deterministic action for evaluation: tanh of the mean."""
    mean, _ = self._dist(feat_in)
    return mx.tanh(mean)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class Critic(nn.Module):
  """Predicts scalar state value V(s) from latent features."""

  def __init__(self, feat_dim: int, hidden: tuple[int, ...] = (300, 300, 300)):
    super().__init__()
    self.net = _mlp(feat_dim, hidden, out_dim=1)

  def __call__(self, feat_in: mx.array) -> mx.array:
    return cast(mx.array, self.net(feat_in).squeeze(-1))


class _WMBundle(nn.Module):
  """Helper class to bundle RSSM, decoder, reward head, and (optional)
  aux head for joint WM updates.

  ``aux`` is None when disabled; the loss skips it and no params land
  in the saved parameter tree.
  """
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
    self.aux     = aux


class DreamerMLXModel:
  encoder: nn.Module
  decoder: nn.Module

  def __init__(self, config, obs_shape, action_dim: int, training: bool = True):
    self.config     = config
    self.training   = training
    self.feat_dim   = config.model.rssm.stoch_size + config.model.rssm.deter_size
    self.action_dim = action_dim
    self.stoch_dim  = int(config.model.rssm.stoch_size)
    self.deter_dim  = int(config.model.rssm.deter_size)

    self.obs_mode = config.env.obs_mode

    enc_hidden = tuple(config.model.encoder.mlp_hidden)
    dec_hidden = tuple(config.model.decoder.mlp_hidden)
    embed_size = int(config.model.encoder.embed_size)

    if self.obs_mode == "state":
      if not (isinstance(obs_shape, tuple) and len(obs_shape) == 1):
        raise ValueError(f"state obs_mode expects 1-D obs_shape; got {obs_shape}")
      obs_dim          = int(obs_shape[0])
      self.encoder     = MLPEncoder(obs_dim, enc_hidden, embed_size)
      self.decoder     = MLPDecoder(self.feat_dim, dec_hidden, obs_dim)
      self.image_size  = None
      self.proprio_dim = None

    elif self.obs_mode in ("image", "image_proprio"):
      enc_channels    = tuple(config.model.encoder.cnn_channels)
      enc_kernels     = tuple(config.model.encoder.cnn_kernels)
      enc_strides     = tuple(config.model.encoder.cnn_strides)
      dec_channels    = tuple(config.model.decoder.cnn_channels)
      dec_kernels     = tuple(config.model.decoder.cnn_kernels)
      dec_strides     = tuple(config.model.decoder.cnn_strides)
      image_size      = int(config.model.encoder.image_size)
      self.image_size = image_size

      if self.obs_mode == "image":
        # obs_shape is the image shape (H, W, 3); 3-D ndarray.
        if not (isinstance(obs_shape, tuple) and len(obs_shape) == 3):
          raise ValueError(f"image obs_mode expects 3-D obs_shape; got {obs_shape}")
        in_channels = int(obs_shape[2])
        self.encoder = CNNEncoder(in_channels, enc_channels, enc_kernels, enc_strides,
                                  embed_dim=embed_size, image_size=image_size)
        self.decoder = CNNDecoder(self.feat_dim, dec_channels, dec_kernels, dec_strides,
                                  out_channels=in_channels, image_size=image_size)
        self.proprio_dim = None

      else:  # image_proprio
        # obs_shape is a dict-like {"image": (H,W,3), "proprio": (P,)}.
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
      raise ValueError(f"unknown obs_mode={self.obs_mode!r}")

    self.rssm = RSSM(
      action_dim=action_dim,
      embed_dim=config.model.encoder.embed_size,
      stoch_dim=config.model.rssm.stoch_size,
      deter_dim=config.model.rssm.deter_size,
      hidden=config.model.rssm.hidden_size,
      min_std=config.model.rssm.min_stddev,
    )
    self.reward_head = RewardHead(self.feat_dim, tuple(config.model.reward.hidden))

    # Aux head: gated by aux_scale > 0. When off, no params on disk and
    # no forward pass. Target dims must match the buffer's aux_target
    # slice (object_xy + ee_pos = 5).
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

    if training:
      self._opt_wm     = optim.Adam(learning_rate=config.training.optimizer.wm_lr,     eps=config.training.optimizer.adam_eps)
      self._opt_actor  = optim.Adam(learning_rate=config.training.optimizer.actor_lr,  eps=config.training.optimizer.adam_eps)
      self._opt_critic = optim.Adam(learning_rate=config.training.optimizer.critic_lr, eps=config.training.optimizer.adam_eps)

      self._wm_bundle = _WMBundle(
        self.encoder, self.rssm, self.decoder, self.reward_head, self.aux_head
      )
      self._wm_grad   = nn.value_and_grad(self._wm_bundle, self._wm_loss_fn)

  # ---------------------------------------------------------------------
  # WorldModel protocol (see :mod:`.world_model`). The training-time
  # methods above (``_wm_loss_fn``, ``wm_update``, ``save``, ``load``) stay
  # backend-specific; everything below is the surface :func:`rollout`
  # composes against.
  # ---------------------------------------------------------------------

  def initial_state(self, batch_size: int) -> State:
    return self.rssm.initial_state(batch_size)

  def coerce(self, x):
    """Lift numpy → ``mx.array`` so loader-side inputs feed backend ops.

    Recursive on dicts. ``mx.array`` inputs pass through unchanged so
    repeat calls are cheap.
    """
    if isinstance(x, dict):
      return {k: self.coerce(v) for k, v in x.items()}
    if isinstance(x, mx.array):
      return x
    return mx.array(np.asarray(x))

  def encode(self, obs) -> mx.array:
    return cast(mx.array, self.encoder(self.coerce(obs)))

  def posterior_step(self, state: State, action: mx.array, embed: mx.array, *, sample: bool = True) -> State:
    return self.rssm.obs_step(state, action, embed, sample=sample)

  def prior_step(self, state: State, action: mx.array, *, sample: bool = True) -> State:
    return self.rssm.img_step(state, action, sample=sample)

  def feat(self, state: State) -> mx.array:
    return feat(state)

  def decode(self, feats: mx.array):
    return self.decoder(feats)

  def predict_reward(self, feats: mx.array) -> mx.array:
    return cast(mx.array, self.reward_head(feats))

  def actor_action(self, feats: mx.array, *, sample: bool = True) -> mx.array:
    if sample:
      action, _, _ = self.actor(feats)
      return cast(mx.array, action)
    return self.actor.mode(feats)

  def stack_along_time(self, xs: list[mx.array]) -> mx.array:
    """Stack a list of per-step backend tensors along a new time axis at index 1."""
    return cast(mx.array, mx.stack(xs, axis=1))

  def broadcast_to(self, x: mx.array, shape: tuple[int, ...]) -> mx.array:
    """Backend-native broadcast. Used by CEM to expand a B=1 state across samples."""
    return cast(mx.array, mx.broadcast_to(x, shape))

  def _recon_loss(self, recon, obs) -> mx.array:
    """Reconstruction loss appropriate for the configured ``obs_mode``.

    ``obs`` arrives pre-normalized from :class:`ImageProcessor` /
    :class:`ImageProprioProcessor` (float in ``[-0.5, 0.5]``), so the
    decoder output and target already share a scale.
    """
    if self.obs_mode == "state":
      return cast(mx.array, ((recon - obs) ** 2).sum(-1).mean())
    if self.obs_mode == "image":
      return cast(mx.array, ((recon - obs) ** 2).sum(axis=(-3, -2, -1)).mean())
    if self.obs_mode == "image_proprio":
      img_loss = ((recon["image"]   - obs["image"])   ** 2).sum(axis=(-3, -2, -1)).mean()
      pro_loss = ((recon["proprio"] - obs["proprio"]) ** 2).sum(-1).mean()
      return cast(mx.array, img_loss + pro_loss)
    raise ValueError(f"unknown obs_mode={self.obs_mode!r}")

  def _wm_loss_fn(self, wm_modules, batch):
    obs    = batch["obs"]      # (B, T, ...) ndarray or dict of ndarrays
    action = batch["action"]   # (B, T, action_dim)
    reward = batch["reward"]   # (B, T)
    if isinstance(obs, dict):
      any_leaf = next(iter(obs.values()))
      B, T     = any_leaf.shape[0], any_leaf.shape[1]
    else:
      B, T = obs.shape[0], obs.shape[1]

    embeds = wm_modules.encoder(obs)           # (B, T, embed_dim)
    state  = wm_modules.rssm.initial_state(B)
    states: list[State] = []
    kls_per_dim = []
    for t in range(T):
      action_t = action[:, t]
      embed_t  = embeds[:, t]
      state    = wm_modules.rssm.obs_step(state, action_t, embed_t)
      states.append(state)
      assert state.posterior is not None  # obs_step always populates it
      kl_t_per_dim = kl_gaussian(
        state.posterior.mean, state.posterior.std,
        state.prior.mean,     state.prior.std,
      )
      kls_per_dim.append(kl_t_per_dim)

    feats    = mx.stack([feat(s) for s in states], axis=1)   # (B, T, feat_dim)
    recon    = wm_modules.decoder(feats)
    rew_pred = wm_modules.reward(feats)

    recon_loss = self._recon_loss(recon, obs)
    rew_loss   = ((rew_pred - reward) ** 2).mean()

    # Per-dim KL averaged over batch and time: (stoch_dim,)
    kl_per_dim_btmean = mx.stack(kls_per_dim, axis=1).mean(axis=(0, 1))
    kl                = kl_per_dim_btmean.sum()
    kl                = mx.maximum(kl, mx.array(self.config.training.losses.free_nats))

    loss = (self.config.training.losses.recon_scale * recon_loss
         + self.config.training.losses.reward_scale * rew_loss
         + self.config.training.losses.kl_scale     * kl)

    aux_out: dict = {"recon": recon_loss, "rew": rew_loss, "kl": kl,
                     "kl_per_dim": kl_per_dim_btmean,
                     "post_states": states}
    if wm_modules.aux is not None:
      aux_pred       = wm_modules.aux(feats)
      aux_loss       = ((aux_pred - batch["aux_target"]) ** 2).sum(-1).mean()
      loss           = loss + self.config.training.losses.aux_scale * aux_loss
      aux_out["aux"] = aux_loss

    return loss, aux_out

  def wm_update(self, batch, tracker=None):
    """Compute model loss and gradients for a batch of sequences.

    The replay buffer hands us numpy arrays so it stays backend-agnostic;
    we lift to ``mx.array`` once here before entering the autodiff loop.
    """
    (loss, aux), grads = self._wm_grad(self._wm_bundle, self.coerce(batch))

    grad_norm = None
    if self.config.training.optimizer.gradient_clipping > 0:
      max_norm         = self.config.training.optimizer.gradient_clipping
      grads, grad_norm = optim.clip_grad_norm(grads, max_norm)
    self._opt_wm.update(self._wm_bundle, grads)

    mx.eval(self._wm_bundle.parameters(), loss, aux["kl_per_dim"])

    if tracker is not None:
      logs = {
        "wm/loss":  loss.item(),
        "wm/recon": aux["recon"].item(),
        "wm/rew":   aux["rew"].item(),
        "wm/kl":    aux["kl"].item(),
      }
      kl_per_dim                   = np.asarray(aux["kl_per_dim"])
      active_threshold             = 0.1
      logs["wm/kl_active_dims"]    = float((kl_per_dim > active_threshold).sum())
      logs["wm/kl_mean_per_dim"]   = float(kl_per_dim.mean())
      logs["wm/kl_per_dim_min"]    = float(kl_per_dim.min())
      logs["wm/kl_per_dim_p10"]    = float(np.percentile(kl_per_dim, 10))
      logs["wm/kl_per_dim_median"] = float(np.percentile(kl_per_dim, 50))
      logs["wm/kl_per_dim_p90"]    = float(np.percentile(kl_per_dim, 90))
      logs["wm/kl_per_dim_max"]    = float(kl_per_dim.max())
      if "aux" in aux:
        logs["wm/aux"] = aux["aux"].item()
      if grad_norm is not None:
        gn = grad_norm.item()
        logs["wm/grad_norm"]    = gn
        logs["wm/grad_clipped"] = float(gn > max_norm)
      tracker.log(logs)

    return aux["post_states"]

  def save(self, path: str, extra_metadata: dict[str, str] | None = None) -> None:
    """Save model weights (and optimizer state during training) to ``path``.

    Writes a single ``.safetensors`` file containing flat keys for the
    world-model bundle, actor, and critic. When ``training`` is True the
    Adam optimizer states are written alongside.

    ``extra_metadata`` is merged into the safetensors metadata block. The
    ``CONFIG_METADATA_KEY`` slot is reserved; passing it raises ValueError.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def flat(prefix, tree):
      return {f"{prefix}.{k}": v for k, v in tree_flatten(tree)}

    weights: dict = {}
    wm_bundle = self._wm_bundle if self.training else _WMBundle(
      self.encoder, self.rssm, self.decoder, self.reward_head, self.aux_head
    )
    weights.update(flat("wm", wm_bundle.parameters()))
    weights.update(flat("actor",  self.actor.parameters()))
    weights.update(flat("critic", self.critic.parameters()))

    if self.training:
      weights.update(flat("opt_wm",     self._opt_wm.state))
      weights.update(flat("opt_actor",  self._opt_actor.state))
      weights.update(flat("opt_critic", self._opt_critic.state))

    # ``resolve=True`` bakes interpolated fields (e.g. the
    # ``derive_image_size`` resolver on ``model.encoder.image_size``) into
    # plain values, so the saved YAML is self-contained: loading it does not
    # require the resolver to be registered in the reader process.
    metadata = {CONFIG_METADATA_KEY: OmegaConf.to_yaml(self.config, resolve=True)}
    if extra_metadata:
      if CONFIG_METADATA_KEY in extra_metadata:
        raise ValueError(f"extra_metadata may not set reserved key {CONFIG_METADATA_KEY!r}")
      metadata.update(extra_metadata)
    mx.save_safetensors(path, weights, metadata=metadata)

  def load(self, path: str) -> dict[str, str]:
    """Load weights written by :meth:`save`.

    Returns the caller-supplied ``extra_metadata`` dict that was passed to
    :meth:`save` (the reserved ``CONFIG_METADATA_KEY`` is filtered out).
    Empty dict if the checkpoint has no extra metadata.
    """
    flat_weights, metadata = mx.load(path, return_metadata=True)

    def take(prefix):
      plen = len(prefix) + 1
      return [(k[plen:], v) for k, v in flat_weights.items() if k.startswith(prefix + ".")]

    wm_bundle = self._wm_bundle if self.training else _WMBundle(
      self.encoder, self.rssm, self.decoder, self.reward_head, self.aux_head
    )
    wm_bundle.update(tree_unflatten(take("wm")))
    self.actor.update(tree_unflatten(take("actor")))
    self.critic.update(tree_unflatten(take("critic")))

    if self.training:
      opt_wm     = take("opt_wm")
      opt_actor  = take("opt_actor")
      opt_critic = take("opt_critic")
      if opt_wm:
        self._opt_wm.state = tree_unflatten(opt_wm)
      if opt_actor:
        self._opt_actor.state = tree_unflatten(opt_actor)
      if opt_critic:
        self._opt_critic.state = tree_unflatten(opt_critic)

    mx.eval(self.actor.parameters(), self.critic.parameters(), wm_bundle.parameters())

    return {k: v for k, v in metadata.items() if k != CONFIG_METADATA_KEY}
