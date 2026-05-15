"""Shape and contract tests for the Dreamer MLX model.

These exercise the documented interfaces of every component (encoder,
decoder, RSSM, reward head, actor, critic) plus an end-to-end build
from the project's default config. They don't train anything, so they
run in well under a second on CPU/MPS.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from chuck_dreamer.config import load_config  # noqa: E402
from chuck_dreamer.dreamer import build_model  # noqa: E402
from chuck_dreamer.dreamer.mlx_model import (  # noqa: E402
  Actor,
  AuxHead,
  CNNDecoder,
  CNNEncoder,
  Critic,
  DreamerMLXModel,
  ImageProprioDecoder,
  ImageProprioEncoder,
  MLPDecoder,
  MLPEncoder,
  RewardHead,
  RSSM,
  feat,
)
from chuck_dreamer.config import derive_image_size  # noqa: E402


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


def test_mlp_encoder_output_shape():
  enc = MLPEncoder(obs_dim=15, hidden=(64, 64), embed_dim=32)
  out = enc(mx.zeros((4, 15)))
  assert out.shape == (4, 32)


def test_mlp_decoder_output_shape():
  dec = MLPDecoder(feat_dim=230, hidden=(64, 64), obs_dim=15)
  out = dec(mx.zeros((4, 230)))
  assert out.shape == (4, 15)


# ---------------------------------------------------------------------------
# feat()
# ---------------------------------------------------------------------------


def test_feat_concatenates_h_and_s():
  from chuck_dreamer.dreamer import Distribution, State
  zeros = mx.zeros((4, 30))
  state = State(h=mx.zeros((4, 200)), s=mx.ones((4, 30)),
                prior=Distribution(mean=zeros, std=mx.ones((4, 30))))
  f = feat(state)
  assert f.shape == (4, 230)
  # First deter_dim entries come from h (zeros), rest from s (ones).
  assert float(mx.sum(f[:, :200])) == 0.0
  assert float(mx.sum(f[:, 200:])) == pytest.approx(4 * 30)


# ---------------------------------------------------------------------------
# RSSM
# ---------------------------------------------------------------------------


def _make_rssm(action_dim=6, embed_dim=32, stoch=8, deter=16, hidden=16, min_std=0.1):
  return RSSM(
    action_dim=action_dim,
    embed_dim=embed_dim,
    stoch_dim=stoch,
    deter_dim=deter,
    hidden=hidden,
    min_std=min_std,
  )


def test_rssm_initial_state_shapes():
  rssm = _make_rssm()
  st = rssm.initial_state(batch_size=5)
  assert st.h.shape == (5, 16)
  assert st.s.shape == (5, 8)
  # Initial posterior is None — no observation has been seen yet.
  assert st.posterior is None


def test_rssm_img_step_returns_prior_only():
  rssm = _make_rssm()
  st = rssm.initial_state(3)
  out = rssm.img_step(st, mx.zeros((3, 6)))
  assert out.h.shape == (3, 16)
  assert out.s.shape == (3, 8)
  assert out.prior.mean.shape == (3, 8)
  assert out.prior.std.shape == (3, 8)
  # img_step doesn't see an observation; no posterior.
  assert out.posterior is None


def test_rssm_obs_step_returns_prior_and_posterior():
  rssm = _make_rssm()
  st = rssm.initial_state(3)
  out = rssm.obs_step(st, mx.zeros((3, 6)), mx.zeros((3, 32)))
  assert out.h.shape == (3, 16)
  assert out.s.shape == (3, 8)
  assert out.prior.mean.shape == (3, 8)
  assert out.prior.std.shape  == (3, 8)
  assert out.posterior is not None
  assert out.posterior.mean.shape == (3, 8)
  assert out.posterior.std.shape  == (3, 8)


def test_rssm_std_respects_min_std():
  min_std = 0.25
  rssm = _make_rssm(min_std=min_std)
  st = rssm.initial_state(2)
  out = rssm.obs_step(st, mx.zeros((2, 6)), mx.zeros((2, 32)))
  # softplus(.) + min_std >= min_std, always.
  assert float(mx.min(out.prior.std)) >= min_std
  assert out.posterior is not None
  assert float(mx.min(out.posterior.std)) >= min_std


def test_rssm_observe_returns_one_state_per_step():
  rssm = _make_rssm()
  B, T = 2, 5
  embeds = mx.zeros((B, T, 32))
  actions = mx.zeros((B, T, 6))
  states = rssm.observe(embeds, actions)
  assert len(states) == T
  for s in states:
    assert s.h.shape == (B, 16)
    assert s.s.shape == (B, 8)


def test_rssm_imagine_returns_horizon_plus_one_states():
  rssm = _make_rssm()
  init = rssm.initial_state(2)

  feat_dim = rssm.deter_dim + rssm.stoch_dim
  calls: list[tuple[int, ...]] = []

  def policy_fn(f):
    calls.append(tuple(f.shape))
    return mx.zeros((f.shape[0], rssm.action_dim))

  traj = rssm.imagine(init, policy_fn, horizon=4)
  assert len(traj) == 5  # init + 4 imagined states
  assert traj[0] is init
  # Policy is called once per imagined step on the concatenated feature.
  assert calls == [(2, feat_dim)] * 4


# ---------------------------------------------------------------------------
# Reward head / Critic
# ---------------------------------------------------------------------------


def test_reward_head_squeezes_trailing_dim():
  head = RewardHead(feat_dim=230, hidden=(32, 32))
  out = head(mx.zeros((6, 230)))
  assert out.shape == (6,)


def test_critic_squeezes_trailing_dim():
  critic = Critic(feat_dim=230, hidden=(32, 32))
  out = critic(mx.zeros((6, 230)))
  assert out.shape == (6,)


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------


def _make_actor(feat_dim=230, action_dim=6, **kw):
  return Actor(feat_dim=feat_dim, action_dim=action_dim, hidden=(32, 32), **kw)


def test_actor_call_shapes_and_action_in_tanh_range():
  actor = _make_actor()
  action, mean, std = actor(mx.zeros((4, 230)))
  assert action.shape == (4, 6)
  assert mean.shape == (4, 6)
  assert std.shape == (4, 6)
  # tanh squash bounds the sampled action.
  assert float(mx.max(action)) <= 1.0
  assert float(mx.min(action)) >= -1.0


def test_actor_mode_is_in_tanh_range_and_deterministic():
  actor = _make_actor()
  f = mx.zeros((4, 230))
  a1 = actor.mode(f)
  a2 = actor.mode(f)
  assert a1.shape == (4, 6)
  assert float(mx.max(a1)) <= 1.0
  assert float(mx.min(a1)) >= -1.0
  # mode() has no sampling, so two calls must agree exactly.
  assert mx.array_equal(a1, a2)


def test_actor_std_respects_min_std():
  min_std = 0.05
  actor = _make_actor(min_std=min_std, init_std=5.0)
  _, _, std = actor(mx.zeros((4, 230)))
  assert float(mx.min(std)) >= min_std


def test_actor_initial_std_uses_init_std_offset():
  """Dropping the init_std offset would collapse std toward min_std at init.
  We don't pin a single value (random init weights make raw_std nonzero),
  but assert std lives in the band the offset is meant to produce."""
  init_std = 5.0
  actor = _make_actor(init_std=init_std, min_std=1e-4)
  _, _, std = actor(mx.zeros((4, 230)))
  # softplus is monotonic; with init_std=5 the smallest possible std is
  # well above 1.0. If the offset was dropped the std would sit near 0.7
  # (softplus(small) ≈ ln 2). 2.0 is a safe lower bound that catches that.
  assert float(mx.min(std)) >= 2.0


# ---------------------------------------------------------------------------
# DreamerMLXModel — config wiring + end-to-end forward pass
# ---------------------------------------------------------------------------


def test_dreamer_model_builds_from_default_config():
  cfg = load_config()
  cfg.env.obs_mode = "state"
  model = build_model(cfg, obs_shape=(15,), action_dim=6)
  assert isinstance(model, DreamerMLXModel)
  assert isinstance(model.encoder, MLPEncoder)
  assert isinstance(model.decoder, MLPDecoder)
  assert isinstance(model.rssm, RSSM)
  assert isinstance(model.reward_head, RewardHead)
  assert isinstance(model.actor, Actor)
  assert isinstance(model.critic, Critic)


def test_dreamer_model_end_to_end_forward_shapes():
  cfg = load_config()
  cfg.env.obs_mode = "state"
  B, T, obs_dim, action_dim = 2, 4, 15, 6
  model = build_model(cfg, obs_shape=(obs_dim,), action_dim=action_dim)

  obs = mx.zeros((B, T, obs_dim))
  actions = mx.zeros((B, T, action_dim))

  embeds = model.encoder(obs)
  embed_size = cfg.model.encoder.embed_size
  assert embeds.shape == (B, T, embed_size)

  states = model.rssm.observe(embeds, actions)
  assert len(states) == T

  feat_dim = cfg.model.rssm.stoch_size + cfg.model.rssm.deter_size
  f = feat(states[-1])
  assert f.shape == (B, feat_dim)

  assert model.decoder(f).shape == (B, obs_dim)
  assert model.reward_head(f).shape == (B,)

  action, mean, std = model.actor(f)
  assert action.shape == (B, action_dim)
  assert mean.shape == (B, action_dim)
  assert std.shape == (B, action_dim)

  assert model.critic(f).shape == (B,)


def test_dreamer_model_imagine_uses_actor_and_yields_horizon_plus_one():
  cfg = load_config()
  cfg.env.obs_mode = "state"
  model = build_model(cfg, obs_shape=(15,), action_dim=6)

  init = model.rssm.initial_state(batch_size=3)

  def policy_fn(f):
    a, _, _ = model.actor(f)
    return a

  traj = model.rssm.imagine(init, policy_fn, horizon=5)
  assert len(traj) == 6
  for st in traj:
    assert st.h.shape == (3, cfg.model.rssm.deter_size)
    assert st.s.shape == (3, cfg.model.rssm.stoch_size)


def test_dreamer_model_rejects_unsupported_obs_mode():
  cfg = load_config()
  cfg.env.obs_mode = "depth"  # not in {state, image, image_proprio}
  with pytest.raises(ValueError):
    build_model(cfg, obs_shape=(64, 64, 3), action_dim=7)


def test_build_model_rejects_unknown_device():
  cfg = load_config()
  cfg.env.obs_mode = "state"
  cfg.hardware.device = "not-a-real-backend"
  with pytest.raises(ValueError):
    build_model(cfg, obs_shape=(15,), action_dim=6)


# ---------------------------------------------------------------------------
# CNN encoder / decoder shapes
# ---------------------------------------------------------------------------


def test_cnn_encoder_output_shape_uint8_input():
  strides = (2, 2, 2, 2)
  img = derive_image_size(strides)  # 64
  enc = CNNEncoder(in_channels=3, channels=(32, 64, 128, 256),
                   kernels=(4, 4, 4, 4), strides=strides,
                   embed_dim=200, image_size=img)
  x = mx.zeros((2, 5, img, img, 3), dtype=mx.uint8)
  assert enc(x).shape == (2, 5, 200)


def test_cnn_decoder_recovers_image_shape():
  strides = (2, 2, 2, 2)
  img = derive_image_size(strides)
  dec = CNNDecoder(feat_dim=230, channels=(32, 64, 128, 256),
                   kernels=(4, 4, 4, 4), strides=strides,
                   out_channels=3, image_size=img)
  f = mx.zeros((2, 5, 230))
  out = dec(f)
  assert out.shape == (2, 5, img, img, 3)


def test_cnn_encoder_validates_unequal_lengths():
  with pytest.raises(ValueError):
    CNNEncoder(in_channels=3,
               channels=(32, 64),
               kernels=(4, 4, 4),  # mismatched length
               strides=(2, 2),
               embed_dim=10, image_size=16)


def test_image_proprio_encoder_decoder_roundtrip_shapes():
  strides = (2, 2)
  img = derive_image_size(strides)
  cnn_enc = CNNEncoder(in_channels=3, channels=(32, 64), kernels=(4, 4),
                       strides=strides, embed_dim=64, image_size=img)
  cnn_dec = CNNDecoder(feat_dim=80, channels=(32, 64), kernels=(4, 4),
                       strides=strides, out_channels=3, image_size=img)
  enc = ImageProprioEncoder(cnn_enc, proprio_dim=13, proprio_hidden=(64,), embed_dim=64)
  dec = ImageProprioDecoder(feat_dim=80, cnn=cnn_dec, proprio_hidden=(64,), proprio_dim=13)

  obs = {"image": mx.zeros((2, 3, img, img, 3), dtype=mx.uint8),
         "proprio": mx.zeros((2, 3, 13))}
  e = enc(obs)
  assert e.shape == (2, 3, 64)

  out = dec(mx.zeros((2, 3, 80)))
  assert out["image"].shape == (2, 3, img, img, 3)
  assert out["proprio"].shape == (2, 3, 13)


# ---------------------------------------------------------------------------
# DreamerMLXModel: image / image_proprio modes
# ---------------------------------------------------------------------------


def test_dreamer_model_builds_image_mode():
  cfg = load_config()
  cfg.env.obs_mode = "image"
  img = derive_image_size(tuple(cfg.model.encoder.cnn_strides))
  model = build_model(cfg, obs_shape=(img, img, 3), action_dim=6)
  assert isinstance(model, DreamerMLXModel)
  assert model.obs_mode == "image"
  assert model.image_size == img
  assert isinstance(model.encoder, CNNEncoder)
  assert isinstance(model.decoder, CNNDecoder)


def test_dreamer_model_builds_image_proprio_mode():
  cfg = load_config()
  cfg.env.obs_mode = "image_proprio"
  img = derive_image_size(tuple(cfg.model.encoder.cnn_strides))
  model = build_model(cfg, obs_shape={"image": (img, img, 3), "proprio": (13,)}, action_dim=6)
  assert model.obs_mode == "image_proprio"
  assert model.image_size == img
  assert model.proprio_dim == 13
  assert isinstance(model.encoder, ImageProprioEncoder)
  assert isinstance(model.decoder, ImageProprioDecoder)


def test_dreamer_image_recon_loss_on_normalized_obs():
  cfg = load_config()
  cfg.env.obs_mode = "image"
  img = derive_image_size(tuple(cfg.model.encoder.cnn_strides))
  model = build_model(cfg, obs_shape=(img, img, 3), action_dim=6)
  # Obs arrive pre-normalized from :class:`ImageProcessor` (float in
  # ``[-0.5, 0.5]``). Use uint8-zero's normalized value (-0.5) as a
  # fixed target so the per-pixel error is 0.5.
  recon = mx.zeros((2, 3, img, img, 3))
  obs   = mx.full((2, 3, img, img, 3), -0.5)
  loss = model._recon_loss(recon, obs)
  expected = 0.25 * img * img * 3
  assert abs(float(loss.item()) - expected) < 1e-3


def test_dreamer_image_proprio_recon_loss_sums_image_and_proprio():
  cfg = load_config()
  cfg.env.obs_mode = "image_proprio"
  img = derive_image_size(tuple(cfg.model.encoder.cnn_strides))
  model = build_model(cfg, obs_shape={"image": (img, img, 3), "proprio": (4,)}, action_dim=6)
  recon = {"image":   mx.zeros((2, 3, img, img, 3)),
           "proprio": mx.zeros((2, 3, 4))}
  # Image obs pre-normalized to -0.5 (uint8-zero post-normalize).
  obs   = {"image":   mx.full((2, 3, img, img, 3), -0.5),
           "proprio": mx.ones((2, 3, 4))}
  loss = model._recon_loss(recon, obs)
  expected_img = 0.25 * img * img * 3
  expected_pro = 4.0  # ((0 - 1)**2).sum(-1) == 4 per (b,t), mean over (B,T) = 4.
  assert abs(float(loss.item()) - (expected_img + expected_pro)) < 1e-3


def test_dreamer_state_recon_loss_unchanged():
  cfg = load_config()
  cfg.env.obs_mode = "state"
  model = build_model(cfg, obs_shape=(5,), action_dim=6)
  recon = mx.zeros((2, 3, 5))
  obs   = mx.ones((2, 3, 5))
  loss = model._recon_loss(recon, obs)
  assert abs(float(loss.item()) - 5.0) < 1e-6  # 5 dims, all (0-1)^2 = 1, summed.


# ---------------------------------------------------------------------------
# Aux head
# ---------------------------------------------------------------------------


def test_aux_head_output_shape():
  head = AuxHead(feat_dim=230, target_dim=5, hidden=(32, 32))
  out = head(mx.zeros((4, 230)))
  assert out.shape == (4, 5)


def test_aux_head_disabled_by_default():
  """aux_scale=0 in the default config means no AuxHead instance is created.
  Skipping construction keeps the parameter tree (and checkpoint key set)
  identical to pre-aux runs."""
  cfg = load_config()
  cfg.env.obs_mode = "state"
  model = build_model(cfg, obs_shape=(5,), action_dim=6)
  assert model.aux_head is None


def test_aux_head_built_when_aux_scale_positive():
  cfg = load_config()
  cfg.env.obs_mode = "state"
  cfg.training.losses.aux_scale = 1.0
  model = build_model(cfg, obs_shape=(5,), action_dim=6)
  assert isinstance(model.aux_head, AuxHead)
  assert model.aux_head.target_dim == int(cfg.model.aux.target_dim)


def test_wm_loss_uses_aux_when_head_present():
  cfg = load_config()
  cfg.env.obs_mode = "state"
  cfg.training.losses.aux_scale = 1.0
  model = build_model(cfg, obs_shape=(5,), action_dim=6)
  batch = {
    "obs":        mx.zeros((2, 4, 5)),
    "action":     mx.zeros((2, 4, 6)),
    "reward":     mx.zeros((2, 4)),
    "aux_target": mx.ones((2, 4, int(cfg.model.aux.target_dim))),
  }
  _, aux = model._wm_loss_fn(model._wm_bundle, batch)
  # aux_target=ones; an untrained head with random init produces nonzero
  # MSE — only assert the loss landed in the aux dict.
  assert "aux" in aux
  assert float(aux["aux"].item()) > 0.0


def test_aux_head_round_trips_through_save_load(tmp_path):
  cfg = load_config()
  cfg.env.obs_mode = "state"
  cfg.training.losses.aux_scale = 1.0
  model = build_model(cfg, obs_shape=(5,), action_dim=6)

  path = str(tmp_path / "ckpt.safetensors")
  model.save(path)

  # Build a fresh model and reload — aux head must be present and weights
  # must match (we test via output equality on a fixed input).
  model2 = build_model(cfg, obs_shape=(5,), action_dim=6)
  model2.load(path)

  f = mx.zeros((3, cfg.model.rssm.stoch_size + cfg.model.rssm.deter_size))
  assert mx.array_equal(model.aux_head(f), model2.aux_head(f))
