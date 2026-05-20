"""Shape, contract, and persistence tests for the PyTorch backend.

These exercise the torch-specific surface: NHWC↔NCHW transposes on the
CNN encoder/decoder, multi-modal dict-shaped obs, device placement,
and the save/load round-trip including config metadata.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from chuck_dreamer.config import derive_image_size, load_config  # noqa: E402
from chuck_dreamer.dreamer import build_model  # noqa: E402
from chuck_dreamer.dreamer.torch_model import (  # noqa: E402
  Actor,
  AuxHead,
  CNNDecoder,
  CNNEncoder,
  Critic,
  DreamerTorchModel,
  MLPDecoder,
  MLPEncoder,
  MultiModalDecoder,
  MultiModalEncoder,
  RewardHead,
  RSSM,
  feat,
  load_config_from_checkpoint,
  resolve_device,
)


# ---------------------------------------------------------------------------
# Encoder / Decoder — state mode
# ---------------------------------------------------------------------------


def test_mlp_encoder_output_shape():
  enc = MLPEncoder(obs_dim=15, hidden=(64, 64), embed_dim=32)
  out = enc(torch.zeros(4, 15))
  assert out.shape == (4, 32)


def test_mlp_decoder_output_shape():
  dec = MLPDecoder(feat_dim=230, hidden=(64, 64), obs_dim=15)
  out = dec(torch.zeros(4, 230))
  assert out.shape == (4, 15)


# ---------------------------------------------------------------------------
# CNN encoder / decoder — NHWC at boundary
# ---------------------------------------------------------------------------


def test_cnn_encoder_output_shape_uint8_input():
  strides = (2, 2, 2, 2)
  img = derive_image_size(strides)
  enc = CNNEncoder(in_channels=3, channels=(32, 64, 128, 256),
                   kernels=(4, 4, 4, 4), strides=strides,
                   embed_dim=200, image_size=img)
  x = torch.zeros(2, 5, img, img, 3, dtype=torch.uint8)
  assert enc(x).shape == (2, 5, 200)


def test_cnn_decoder_recovers_image_shape():
  strides = (2, 2, 2, 2)
  img = derive_image_size(strides)
  dec = CNNDecoder(feat_dim=230, channels=(32, 64, 128, 256),
                   kernels=(4, 4, 4, 4), strides=strides,
                   out_channels=3, image_size=img)
  f = torch.zeros(2, 5, 230)
  out = dec(f)
  assert out.shape == (2, 5, img, img, 3)


def test_cnn_encoder_validates_unequal_lengths():
  with pytest.raises(ValueError):
    CNNEncoder(in_channels=3,
               channels=(32, 64),
               kernels=(4, 4, 4),  # mismatch
               strides=(2, 2),
               embed_dim=10, image_size=16)


def test_multi_modal_encoder_decoder_roundtrip_shapes():
  strides = (2, 2)
  img = derive_image_size(strides)
  cnn_enc = CNNEncoder(in_channels=3, channels=(32, 64), kernels=(4, 4),
                       strides=strides, embed_dim=64, image_size=img)
  cnn_dec = CNNDecoder(feat_dim=80, channels=(32, 64), kernels=(4, 4),
                       strides=strides, out_channels=3, image_size=img)
  mlp_enc = MLPEncoder(obs_dim=13, hidden=(64,), embed_dim=64)
  mlp_dec = MLPDecoder(feat_dim=80, hidden=(64,), obs_dim=13)
  enc = MultiModalEncoder({"image": cnn_enc, "proprio": mlp_enc}, embed_dim=64)
  dec = MultiModalDecoder({"image": cnn_dec, "proprio": mlp_dec})

  obs = {"image": torch.zeros(2, 3, img, img, 3, dtype=torch.uint8),
         "proprio": torch.zeros(2, 3, 13)}
  e = enc(obs)
  assert e.shape == (2, 3, 64)

  out = dec(torch.zeros(2, 3, 80))
  assert out["image"].shape == (2, 3, img, img, 3)
  assert out["proprio"].shape == (2, 3, 13)


# ---------------------------------------------------------------------------
# RSSM
# ---------------------------------------------------------------------------


def _make_rssm(action_dim=6, embed_dim=32, stoch=8, deter=16, hidden=16, min_std=0.1):
  return RSSM(action_dim=action_dim, embed_dim=embed_dim,
              stoch_dim=stoch, deter_dim=deter, hidden=hidden, min_std=min_std)


def test_rssm_initial_state_shapes():
  rssm = _make_rssm()
  st = rssm.initial_state(batch_size=5)
  assert st.h.shape == (5, 16)
  assert st.s.shape == (5, 8)
  assert st.posterior is None


def test_rssm_img_step_returns_prior_only():
  rssm = _make_rssm()
  st = rssm.initial_state(3)
  out = rssm.img_step(st, torch.zeros(3, 6))
  assert out.h.shape == (3, 16)
  assert out.s.shape == (3, 8)
  assert out.prior.mean.shape == (3, 8)
  assert out.posterior is None


def test_rssm_obs_step_returns_prior_and_posterior():
  rssm = _make_rssm()
  st = rssm.initial_state(3)
  out = rssm.obs_step(st, torch.zeros(3, 6), torch.zeros(3, 32))
  assert out.h.shape == (3, 16)
  assert out.s.shape == (3, 8)
  assert out.posterior is not None
  assert out.posterior.mean.shape == (3, 8)


def test_rssm_std_respects_min_std():
  min_std = 0.25
  rssm = _make_rssm(min_std=min_std)
  st = rssm.initial_state(2)
  with torch.no_grad():
    out = rssm.obs_step(st, torch.zeros(2, 6), torch.zeros(2, 32))
  assert float(out.prior.std.min()) >= min_std
  assert out.posterior is not None
  assert float(out.posterior.std.min()) >= min_std


def test_rssm_observe_returns_one_state_per_step():
  rssm = _make_rssm()
  B, T = 2, 5
  embeds  = torch.zeros(B, T, 32)
  actions = torch.zeros(B, T, 6)
  states = rssm.observe(embeds, actions)
  assert len(states) == T
  for s in states:
    assert s.h.shape == (B, 16)
    assert s.s.shape == (B, 8)


def test_rssm_imagine_returns_horizon_plus_one_states():
  rssm = _make_rssm()
  init = rssm.initial_state(2)
  calls: list[tuple[int, ...]] = []

  def policy_fn(f):
    calls.append(tuple(f.shape))
    return torch.zeros(f.shape[0], rssm.action_dim)
  traj = rssm.imagine(init, policy_fn, horizon=4)
  assert len(traj) == 5
  assert traj[0] is init
  feat_dim = rssm.deter_dim + rssm.stoch_dim
  assert calls == [(2, feat_dim)] * 4


# ---------------------------------------------------------------------------
# Reward head / Critic / Actor / Aux
# ---------------------------------------------------------------------------


def test_reward_head_squeezes_trailing_dim():
  head = RewardHead(feat_dim=230, hidden=(32, 32))
  out = head(torch.zeros(6, 230))
  assert out.shape == (6,)


def test_critic_squeezes_trailing_dim():
  critic = Critic(feat_dim=230, hidden=(32, 32))
  out = critic(torch.zeros(6, 230))
  assert out.shape == (6,)


def test_actor_call_shapes_and_action_in_tanh_range():
  actor = Actor(feat_dim=230, action_dim=6, hidden=(32, 32))
  with torch.no_grad():
    action, mean, std = actor(torch.zeros(4, 230))
  assert action.shape == (4, 6)
  assert mean.shape == (4, 6)
  assert std.shape == (4, 6)
  assert float(action.max()) <= 1.0
  assert float(action.min()) >= -1.0


def test_actor_mode_is_deterministic_and_in_range():
  actor = Actor(feat_dim=230, action_dim=6, hidden=(32, 32))
  f = torch.zeros(4, 230)
  with torch.no_grad():
    a1 = actor.mode(f)
    a2 = actor.mode(f)
  assert torch.equal(a1, a2)
  assert float(a1.max()) <= 1.0


def test_aux_head_output_shape():
  head = AuxHead(feat_dim=230, target_dim=5, hidden=(32, 32))
  out = head(torch.zeros(4, 230))
  assert out.shape == (4, 5)


# ---------------------------------------------------------------------------
# DreamerTorchModel — config wiring + end-to-end
# ---------------------------------------------------------------------------


def _state_cfg():
  cfg = load_config()
  cfg.env.obs_mode = ["state"]
  cfg.hardware.device = "cpu"
  return cfg


def test_dreamer_model_builds_from_default_config():
  cfg = _state_cfg()
  model = build_model(cfg, obs_shape={"state": (15,)}, action_dim=6)
  assert isinstance(model, DreamerTorchModel)
  assert isinstance(model.encoder, MultiModalEncoder)
  assert isinstance(model.decoder, MultiModalDecoder)
  assert isinstance(model.encoder.branches["state"], MLPEncoder)
  assert isinstance(model.decoder.heads["state"], MLPDecoder)


def test_dreamer_model_builds_image_mode():
  cfg = load_config()
  cfg.env.obs_mode = ["image"]
  cfg.hardware.device = "cpu"
  img = derive_image_size(tuple(cfg.model.encoder.cnn_strides))
  model = build_model(cfg, obs_shape={"image": (img, img, 3)}, action_dim=6)
  assert model.obs_mode == ("image",)
  assert isinstance(model.encoder.branches["image"], CNNEncoder)
  assert isinstance(model.decoder.heads["image"], CNNDecoder)


def test_dreamer_model_builds_image_proprio_mode():
  cfg = load_config()
  cfg.env.obs_mode = ["image", "proprio"]
  cfg.hardware.device = "cpu"
  img = derive_image_size(tuple(cfg.model.encoder.cnn_strides))
  model = build_model(cfg, obs_shape={"image": (img, img, 3), "proprio": (13,)}, action_dim=6)
  assert model.obs_mode == ("image", "proprio")
  assert model.image_size == img
  assert isinstance(model.encoder.branches["image"], CNNEncoder)
  assert isinstance(model.encoder.branches["proprio"], MLPEncoder)
  assert isinstance(model.decoder.heads["image"], CNNDecoder)
  assert isinstance(model.decoder.heads["proprio"], MLPDecoder)


def test_dreamer_model_end_to_end_forward_shapes():
  cfg = _state_cfg()
  B, T, obs_dim, action_dim = 2, 4, 15, 6
  model = build_model(cfg, obs_shape={"state": (obs_dim,)}, action_dim=action_dim)

  obs = {"state": torch.zeros(B, T, obs_dim)}
  actions = torch.zeros(B, T, action_dim)

  embeds = model.encoder(obs)
  embed_size = cfg.model.encoder.embed_size
  assert embeds.shape == (B, T, embed_size)

  states = model.rssm.observe(embeds, actions)
  assert len(states) == T

  feat_dim = cfg.model.rssm.stoch_size + cfg.model.rssm.deter_size
  f = feat(states[-1])
  assert f.shape == (B, feat_dim)
  out = model.decoder(f)
  assert out["state"].shape == (B, obs_dim)
  assert model.reward_head(f).shape == (B,)
  action, mean, std = model.actor(f)
  assert action.shape == (B, action_dim)
  assert model.critic(f).shape == (B,)


def test_dreamer_model_rejects_unsupported_obs_mode():
  cfg = load_config()
  cfg.env.obs_mode = ["depth"]
  cfg.hardware.device = "cpu"
  with pytest.raises(ValueError):
    build_model(cfg, obs_shape={"image": (64, 64, 3)}, action_dim=7)


# ---------------------------------------------------------------------------
# Aux head wiring
# ---------------------------------------------------------------------------


def test_aux_head_disabled_by_default():
  cfg = _state_cfg()
  model = build_model(cfg, obs_shape={"state": (5,)}, action_dim=6)
  assert model.aux_head is None


def test_aux_head_built_when_aux_scale_positive():
  cfg = _state_cfg()
  cfg.training.losses.aux_scale = 1.0
  model = build_model(cfg, obs_shape={"state": (5,)}, action_dim=6)
  assert isinstance(model.aux_head, AuxHead)
  assert model.aux_head.target_dim == int(cfg.model.aux.target_dim)


def test_wm_update_runs_and_advances_params():
  cfg = _state_cfg()
  model = build_model(cfg, obs_shape={"state": (5,)}, action_dim=6)
  # Snapshot the encoder's first linear layer's weight.
  first_linear = next(p for p in model._wm_bundle.encoder.parameters())
  before = first_linear.detach().clone()

  B, T = 2, 4
  batch = {
    "obs":    {"state": torch.randn(B, T, 5)},
    "action": torch.randn(B, T, 6),
    "reward": torch.randn(B, T),
  }
  post_states, loss_val = model.wm_update(batch)
  assert len(post_states) == T
  assert isinstance(loss_val, float)
  # WM optimizer ran; some param should have changed (non-zero gradient).
  assert not torch.equal(before, first_linear.detach())


def test_wm_update_with_aux_head_runs():
  cfg = _state_cfg()
  cfg.training.losses.aux_scale = 1.0
  model = build_model(cfg, obs_shape={"state": (5,)}, action_dim=6)
  B, T = 2, 3
  batch = {
    "obs":        {"state": torch.randn(B, T, 5)},
    "action":     torch.randn(B, T, 6),
    "reward":     torch.randn(B, T),
    "aux_target": torch.randn(B, T, int(cfg.model.aux.target_dim)),
  }
  model.wm_update(batch)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_load_round_trip_matches_forward_output(tmp_path):
  cfg = _state_cfg()
  model = build_model(cfg, obs_shape={"state": (5,)}, action_dim=6)
  f = torch.randn(3, cfg.model.rssm.stoch_size + cfg.model.rssm.deter_size)
  out_before = model.decoder(f)["state"].detach().clone()

  path = str(tmp_path / "ckpt.safetensors")
  model.save(path)

  model2 = build_model(cfg, obs_shape={"state": (5,)}, action_dim=6)
  # Different random init: outputs should differ before loading.
  assert not torch.allclose(model.decoder(f)["state"], model2.decoder(f)["state"])
  model2.load(path)
  assert torch.allclose(out_before, model2.decoder(f)["state"])


def test_save_embeds_config_yaml(tmp_path):
  cfg = _state_cfg()
  model = build_model(cfg, obs_shape={"state": (5,)}, action_dim=6)
  path = str(tmp_path / "ckpt.safetensors")
  model.save(path)
  loaded_cfg = load_config_from_checkpoint(path)
  assert loaded_cfg is not None
  assert list(loaded_cfg.env.obs_mode) == ["state"]


def test_save_extra_metadata_round_trips(tmp_path):
  cfg = _state_cfg()
  model = build_model(cfg, obs_shape={"state": (5,)}, action_dim=6)
  path = str(tmp_path / "ckpt.safetensors")
  model.save(path, extra_metadata={"iteration": "42"})
  model2 = build_model(cfg, obs_shape={"state": (5,)}, action_dim=6)
  extra = model2.load(path)
  assert extra == {"iteration": "42"}


def test_save_extra_metadata_rejects_reserved_key(tmp_path):
  cfg = _state_cfg()
  model = build_model(cfg, obs_shape={"state": (5,)}, action_dim=6)
  path = str(tmp_path / "ckpt.safetensors")
  with pytest.raises(ValueError):
    model.save(path, extra_metadata={"config_yaml": "x"})


def test_aux_head_round_trips_through_save_load(tmp_path):
  cfg = _state_cfg()
  cfg.training.losses.aux_scale = 1.0
  model = build_model(cfg, obs_shape={"state": (5,)}, action_dim=6)
  path = str(tmp_path / "ckpt.safetensors")
  model.save(path)

  model2 = build_model(cfg, obs_shape={"state": (5,)}, action_dim=6)
  model2.load(path)
  f = torch.zeros(3, cfg.model.rssm.stoch_size + cfg.model.rssm.deter_size)
  assert torch.allclose(model.aux_head(f), model2.aux_head(f))


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------


def test_resolve_device_auto_picks_an_available_backend():
  d = resolve_device("auto")
  assert d.type in ("cuda", "mps", "cpu")


def test_resolve_device_rejects_mlx():
  with pytest.raises(ValueError):
    resolve_device("mlx")


def test_model_params_live_on_resolved_device():
  cfg = _state_cfg()
  model = build_model(cfg, obs_shape={"state": (5,)}, action_dim=6)
  any_param = next(model._wm_bundle.parameters())
  assert any_param.device.type == model.device.type


def test_encode_accepts_numpy_input():
  """The encoder helper should handle numpy obs at the boundary — useful
  for callers like CEMPolicy that move between numpy and the model."""
  cfg = _state_cfg()
  model = build_model(cfg, obs_shape={"state": (5,)}, action_dim=6)
  emb = model.encode({"state": np.zeros((2, 3, 5), dtype=np.float32)})
  assert emb.shape == (2, 3, cfg.model.encoder.embed_size)
  assert emb.device.type == model.device.type
