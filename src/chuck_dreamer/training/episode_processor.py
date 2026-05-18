"""Convert raw sim-recorded episodes into the replay buffer's schema.

:class:`~chuck_dreamer.training.episode_dataset.Episode` carries raw
dicts of arrays straight from the writer format. The processor here
projects them onto the modal observation chosen by ``config.env.obs_mode``
and aligns with the buffer's ``(T+1 obs, T action)`` invariant.

A single :class:`EpisodeProcessor` parameterized by ``obs_mode`` replaces
the three per-mode processor classes that used to live here. Per-mode
transformations (resize, normalize, proprio concat) are owned by
:class:`~chuck_dreamer.training.observation.Observation`.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .observation import ObsMode, Observation


RawEpisode = dict[str, Any]
Episode = dict[str, Any]


_STEP_INFO_KEYS = ("object_xy", "ee_pos", "ee_quat")


class EpisodeProcessorProtocol(Protocol):
  """Converts a raw sim episode into replay-buffer schema."""

  def __call__(self, raw: RawEpisode) -> Episode: ...


def _slice_step_info(raw: RawEpisode, n: int) -> dict[str, np.ndarray]:
  """Extract per-step info columns sliced to length ``n``.

  In the recorded format, ``step_info[t]`` describes the state AFTER
  action_t — i.e. it aligns with the buffer's ``reward[t]`` (reward
  earned by taking ``action_t``). Callers pass ``n = T = N - 1`` so the
  result aligns with the action/reward/done axis.
  """
  out: dict[str, np.ndarray] = {}
  for key in _STEP_INFO_KEYS:
    out[key] = np.asarray(raw[key], dtype=np.float32)[:n]
  if "timestamp" in raw:
    out["time"] = np.asarray(raw["timestamp"], dtype=np.float32)[:n]
  return out


def _resolve_action(raw: RawEpisode, act_mode: str) -> np.ndarray:
  """Pick the action stream matching ``act_mode`` (``"joint"`` or ``"ee"``).

  Recordings can carry both ``joint_action`` and ``ee_action``; the choice
  is made by the training-time config, not the recording. Hard-errors
  when the requested stream is absent so a dim mismatch never reaches
  the world model.
  """
  key = "joint_action" if act_mode == "joint" else "ee_action" if act_mode == "ee" else None
  if key is None:
    raise ValueError(f"unknown act_mode={act_mode!r}; expected 'joint' or 'ee'")
  if key not in raw:
    raise KeyError(
      f"episode missing {key!r} (required by act_mode={act_mode!r}); "
      f"present action keys: "
      f"{sorted(k for k in raw if k in ('joint_action', 'ee_action')) or 'none'}"
    )
  return np.asarray(raw[key], dtype=np.float32)


def _pack(obs: Observation, raw: RawEpisode, act_mode: str) -> Episode:
  """Align a per-step :class:`Observation` with the buffer's layout.

  Sim episodes record N aligned steps of (obs_t, action_t, reward_t).
  We treat the N recorded obs as ``obs[0..N-1]`` and drop the last
  action/reward, yielding N obs and N-1 actions — i.e. T = N-1 with
  T+1 = N obs. The final ``done`` is set to True. ``step_info`` is
  sliced to length T.

  ``focus_mask`` (when present on ``obs``) rides in the episode dict as
  a sibling key — kept separate from ``obs`` so legacy consumers that
  read ``ep["obs"]`` as a plain ndarray/dict stay backwards-compatible.
  """
  N = obs.leading_shape[0]
  if N < 2:
    raise ValueError(f"episode too short: {N} steps (need >= 2)")

  action   = _resolve_action(raw, act_mode)[: N - 1]
  reward   = np.asarray(raw["reward"], dtype=np.float32)[: N - 1]
  done     = np.zeros((N - 1,), dtype=bool)
  done[-1] = True

  ep: Episode = {
    "obs": obs.to_buffer_value(),
    "action": action,
    "reward": reward,
    "done": done,
    "step_info": _slice_step_info(raw, N - 1),
  }
  if obs.focus_mask is not None:
    ep["focus_mask"] = obs.focus_mask
  if "goal_xy" in raw:
    ep["goal_xy"] = np.asarray(raw["goal_xy"], dtype=np.float32)
  if "tags" in raw:
    ep["tags"] = tuple(str(t) for t in raw["tags"])
  return ep


def _union_focus_mask(raw: RawEpisode, sources: tuple[str, ...]) -> np.ndarray | None:
  """OR-combine the named ``segmentation_*`` masks present in ``raw``.

  Returns ``None`` if none of the sources are present (silent fallback —
  loaders without segmentation data should still work). Missing
  individual sources are skipped (they may not be recorded for every
  episode), but at least one source must be present for the mask to be
  produced. The returned mask is float32 in ``{0, 1}``, shape ``(T, H, W)``.
  """
  layers: list[np.ndarray] = []
  for key in sources:
    if key in raw:
      layers.append(np.asarray(raw[key], dtype=bool))
  if not layers:
    return None
  union = layers[0]
  for layer in layers[1:]:
    union = union | layer
  return union.astype(np.float32)


def _build_observation(
  raw: RawEpisode,
  obs_mode: ObsMode,
  image_size: int | None,
  focus_mask_sources: tuple[str, ...] = (),
) -> Observation:
  """Project a raw recorded episode onto the configured observation mode.

  The transformations (resize, normalize, proprio concat) all live as
  methods on :class:`Observation` — this function just composes them.
  """
  if obs_mode == "state":
    ee_pos     = np.asarray(raw["ee_pos"],     dtype=np.float32)
    ee_quat    = np.asarray(raw["ee_quat"],    dtype=np.float32)
    object_xy  = np.asarray(raw["object_xy"],  dtype=np.float32)
    joint_qpos = np.asarray(raw["joint_qpos"], dtype=np.float32)
    return Observation.state_vector(
      np.concatenate([ee_pos, ee_quat, object_xy, joint_qpos], axis=1),
    )

  if image_size is None:
    raise ValueError(f"obs_mode={obs_mode!r} requires an image_size")
  images = np.asarray(raw["image"], dtype=np.uint8)
  fmask  = _union_focus_mask(raw, focus_mask_sources)

  if obs_mode == "image":
    obs = Observation.image_only(images, focus_mask=fmask)
    return obs.resize_image(image_size).normalize_image()

  if obs_mode == "image_proprio":
    ee_pos     = np.asarray(raw["ee_pos"],     dtype=np.float32)
    ee_quat    = np.asarray(raw["ee_quat"],    dtype=np.float32)
    joint_qpos = np.asarray(raw["joint_qpos"], dtype=np.float32)
    proprio    = np.concatenate([ee_pos, ee_quat, joint_qpos], axis=1)
    obs = Observation.image_proprio(images, proprio, focus_mask=fmask)
    return obs.resize_image(image_size).normalize_image()

  raise ValueError(f"unknown obs_mode={obs_mode!r}")


class EpisodeProcessor:
  """Project raw sim episodes onto a configured observation mode.

  One processor handles all three obs modes — ``state``, ``image``,
  ``image_proprio``. ``image_size`` is required for image modes and
  ignored for ``state``. ``focus_mask_sources`` names raw segmentation
  keys (e.g. ``"segmentation_target"``) whose union becomes the per-step
  focus mask attached to the observation; empty disables it.

  ``act_mode`` (``"joint"`` or ``"ee"``) picks which action stream to
  read off the recording. Recordings may carry both; the training config
  decides which view the world model sees.
  """

  def __init__(
    self,
    obs_mode: ObsMode,
    image_size: int | None = None,
    focus_mask_sources: tuple[str, ...] = (),
    act_mode: str = "ee",
  ):
    self.obs_mode = obs_mode
    self.image_size = None if image_size is None else int(image_size)
    self.focus_mask_sources = tuple(focus_mask_sources)
    self.act_mode = str(act_mode)

  def __call__(self, raw: RawEpisode) -> Episode:
    obs = _build_observation(raw, self.obs_mode, self.image_size, self.focus_mask_sources)
    return _pack(obs, raw, self.act_mode)


def processor_for(config) -> EpisodeProcessor:
  """Build the processor matching ``config.env.obs_mode``.

  For image modes ``config.model.encoder.image_size`` (computed by the
  ``derive_image_size`` OmegaConf resolver from the encoder strides) is
  the side length captured frames are resized to. ``focus_mask_sources``
  is pulled from ``training.losses.focus_masks`` and only applied to
  image-having modes. ``act_mode`` is read from ``config.env.act_mode``
  and selects which action stream the processor reads from each recording.
  """
  obs_mode = str(config.env.obs_mode)
  act_mode = str(config.env.act_mode)
  if obs_mode == "state":
    return EpisodeProcessor("state", act_mode=act_mode)
  if obs_mode in ("image", "image_proprio"):
    focus_sources = tuple(config.training.losses.get("focus_masks", []) or [])
    return EpisodeProcessor(
      obs_mode,
      image_size=int(config.model.encoder.image_size),
      focus_mask_sources=focus_sources,
      act_mode=act_mode,
    )
  raise ValueError(f"Unknown obs_mode: {obs_mode!r}")
