"""Episode writers — persist sim and eval episodes to HDF5 or Rerun ``.rrd`` files.

Two kinds of episodes share these writers:

  * **Sim collect** — full per-step record from a real rollout (image,
    action, reward, joint state, …). Written via :meth:`write_episode`.
  * **Eval** — what the world model saw vs. what it produced under a
    burn-in / open-loop split (raw obs, processed obs, posterior + prior
    reconstructions, latent ``h`` / ``s``). Written via
    :meth:`write_eval_episode`.

Each writer class supports both: the format and on-disk layout differ,
but the entry-point + metadata handling are shared.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("hdf5", "rerun")

# Filename prefixes — different kinds of episodes live alongside one
# another in the same dump dir, so :class:`EpisodeDataset` can find sim
# episodes without picking up eval dumps and vice versa.
EPISODE_FILENAME_PREFIX      = "episode"
EVAL_EPISODE_FILENAME_PREFIX = "eval"


def EpisodeWriter(output_dir: str, format: str = "hdf5"):
    """Factory that returns the concrete writer for the requested ``format``.

    The returned writer supports both :meth:`write_episode` (sim) and
    :meth:`write_eval_episode` (eval), so a single instance can persist
    both kinds.
    """
    if format == "hdf5":
        return HDF5EpisodeWriter(output_dir)
    if format == "rerun":
        return RerunEpisodeWriter(output_dir)
    raise ValueError(
        f"Unsupported format '{format}'. Supported formats: {SUPPORTED_FORMATS}.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


# Per-step columns shared by all action modes for sim writes.
# ``joint_action`` and ``ee_action`` may both ride along — the file is
# action-space-agnostic and the training pipeline picks which to read.
_REQUIRED_KEYS = (
    "image", "reward", "timestamp",
    "joint_qpos", "ee_pos", "ee_quat", "object_xy",
)


def _serialize_metadata_config(metadata: dict[str, Any] | None) -> str | None:
    """Return the ``config`` field of metadata as a JSON string, or None."""
    if metadata is None:
        return None
    cfg = metadata.get("config")
    if cfg is None:
        return None
    if isinstance(cfg, (str, bytes)):
        return cfg if isinstance(cfg, str) else cfg.decode("utf-8")
    return json.dumps(cfg if isinstance(cfg, dict) else asdict(cfg))  # type: ignore[arg-type]


def _collect_actions(episode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return ``{kind: array}`` for whichever of joint_action / ee_action are present.

    Recordings may carry both: the file is action-space-agnostic and the
    training pipeline picks which view to use at load time based on
    ``config.env.act_mode``. At least one must be present.
    """
    out: dict[str, np.ndarray] = {}
    for key in ("joint_action", "ee_action"):
        if key in episode:
            out[key] = np.asarray(episode[key], dtype=np.float32)
    if not out:
        raise KeyError(
            "episode is missing an action field (expected joint_action and/or ee_action)")
    return out


def _denormalize_recon_image(recon: np.ndarray) -> np.ndarray:
    """Map a decoder image output (float in ``[-0.5, 0.5]``) to uint8 RGB.

    Out-of-range values are clipped so we get a viewable image even when
    the model has not converged yet.
    """
    arr = np.asarray(recon, dtype=np.float32)
    arr = np.clip(arr + 0.5, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def _coerce_image_for_log(img: np.ndarray) -> np.ndarray:
    """Accept uint8 [0,255] or float images in [-0.5, 0.5] / [0, 1] and return uint8."""
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.min() < 0.0:
        arr = arr + 0.5
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def _to_unit_float(img: np.ndarray) -> np.ndarray:
    """Map an obs/recon image to float32 in ``[0, 1]`` for diffing.

    Mirrors :func:`_coerce_image_for_log` / :func:`_denormalize_recon_image`
    but keeps the result as floats so per-pixel differences keep their sign
    range and don't underflow at uint8.
    """
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    if arr.min() < 0.0:
        arr = arr + 0.5
    return np.clip(arr, 0.0, 1.0)


def _pixel_diff(obs: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """Return signed per-pixel difference ``obs - recon`` in ``[-1, 1]``."""
    return _to_unit_float(obs) - _to_unit_float(recon)


def _diff_to_uint8(diff: np.ndarray) -> np.ndarray:
    """Map a signed diff in ``[-1, 1]`` to a viewable uint8 image centred at 128."""
    arr = np.clip(diff, -1.0, 1.0)
    arr = (arr * 0.5 + 0.5) * 255.0
    return arr.astype(np.uint8)


def _rerun_metadata_props(metadata: dict[str, Any] | None, extra_keys: tuple[str, ...] = ()) -> dict[str, str]:
    """Project an episode-metadata dict onto Rerun-loggable string props.

    Always picks up the JSON-serialized ``config``, ``seed``, ``source``,
    and ``outcome`` fields when present. ``extra_keys`` adds further
    scalar fields (looked up with ``metadata.get(k)``, str-cast,
    none-filtered) so eval can surface ``iteration`` / ``episode_index``
    / ``burn_in`` without forking a second helper.
    """
    props: dict[str, str] = {}
    if metadata is None:
        return props
    cfg_json = _serialize_metadata_config(metadata)
    if cfg_json is not None:
        props["config"] = cfg_json
    if "seed" in metadata:
        props["seed"] = str(int(metadata["seed"]))
    if metadata.get("source") is not None:
        props["source"] = str(metadata["source"])
    if metadata.get("outcome") is not None:
        props["outcome"] = str(metadata["outcome"])
    for k in extra_keys:
        v = metadata.get(k) if metadata else None
        if v is not None:
            props[k] = str(v)
    return props


class _BaseEpisodeWriter:
    """Shared base — owns the output directory and per-recording paths."""

    file_extension: str = ""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, prefix: str, name_suffix: str) -> Path:
        return self.output_dir / f"{prefix}-{name_suffix}.{self.file_extension}"


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------


class HDF5EpisodeWriter(_BaseEpisodeWriter):
    """Writes both sim and eval episodes to HDF5 files.

    Sim episodes go to ``output_dir/episode-{suffix}.hdf5`` with the
    structure::

        images          (T, H, W, 3)    uint8
        joint_action    (T, n_joints)   float32   — optional, present when the caller supplies it
        ee_action       (T, 7)          float32   — optional, present when the caller supplies it
        rewards         (T,)            float32
        timestamps      (T,)            float32
        joint_qpos      (T, n_joints)   float32
        ee_pos          (T, 3)          float32
        ee_quat         (T, 4)          float32
        object_xy       (T, 2)          float32
        segmentation/                              — optional, present iff env produced masks
            target      (T, H, W)       bool
            goal        (T, H, W)       bool
            arm         (T, H, W)       bool      — optional, union of robot-arm geoms
            background  (T, H, W)       bool      — optional
            clutter     (T, K, H, W)    bool      — optional, K = scene clutter count
        metadata/
            config      scalar string (JSON)
            seed        scalar int64
            source      scalar string
            outcome     scalar string
            goal_xy     (2,) float32

    Files may carry one or both action streams. The on-disk recording is
    action-space-agnostic; :class:`EpisodeProcessor` picks which view to
    use at load time based on ``config.env.act_mode``.

    Eval episodes go to ``output_dir/eval-{suffix}.hdf5`` with::

        obs                (T, ...)            float32 — encoder input
        recon_posterior    (T, ...)
        recon_prior        (T, ...)
        diff_posterior     (T, ...) float32 — obs - recon_posterior (image only)
        diff_prior         (T, ...) float32 — obs - recon_prior     (image only)
        h_posterior        (T, deter_dim)
        s_posterior        (T, stoch_dim)
        h_prior            (T, deter_dim)
        s_prior            (T, stoch_dim)
        metadata/
            burn_in        int
            iteration      int (if present)
            episode_index  int (if present)
            seed           int (if present)
    """

    file_extension = "hdf5"

    @staticmethod
    def _write_segmentation_hdf5(f, episode: dict[str, Any]) -> None:
        """Persist any segmentation_* arrays in ``episode`` under ``segmentation/``.

        No-op for episodes that don't carry masks (e.g. lerobot imports).
        Mask names are stored without the ``segmentation_`` prefix.
        """
        seg_items = {
            k[len("segmentation_"):]: v
            for k, v in episode.items()
            if k.startswith("segmentation_")
        }
        if not seg_items:
            return
        seg_grp = f.create_group("segmentation")
        for name, mask in seg_items.items():
            seg_grp.create_dataset(
                name,
                data=np.asarray(mask, dtype=bool),
                compression="gzip",
                compression_opts=4,
            )

    def write_episode(
        self,
        episode: dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None,
        *,
        name_suffix: str,
    ) -> Path:
        actions = _collect_actions(episode)
        T = next(iter(actions.values())).shape[0]
        if T == 0:
            raise ValueError("episode must not be empty")

        rewards    = np.asarray(episode["reward"],     dtype=np.float32)
        timestamps = np.asarray(episode["timestamp"],  dtype=np.float32)
        joint_qpos = np.asarray(episode["joint_qpos"], dtype=np.float32)
        ee_pos     = np.asarray(episode["ee_pos"],     dtype=np.float32)
        ee_quat    = np.asarray(episode["ee_quat"],    dtype=np.float32)
        object_xy  = np.asarray(episode["object_xy"],  dtype=np.float32)
        images     = np.asarray(episode["image"],      dtype=np.uint8)

        ep_path = self._path_for(EPISODE_FILENAME_PREFIX, name_suffix)
        with h5py.File(ep_path, "w") as f:
            f.create_dataset("images",     data=images,  compression="gzip", compression_opts=4)
            for kind, arr in actions.items():
                f.create_dataset(kind, data=arr)
            f.create_dataset("rewards",    data=rewards)
            f.create_dataset("timestamps", data=timestamps)
            f.create_dataset("joint_qpos", data=joint_qpos)
            f.create_dataset("ee_pos",     data=ee_pos)
            f.create_dataset("ee_quat",    data=ee_quat)
            f.create_dataset("object_xy",  data=object_xy)

            self._write_segmentation_hdf5(f, episode)

            meta_grp = f.create_group("metadata")
            if metadata is not None:
                cfg = _serialize_metadata_config(metadata)
                if cfg is not None:
                    meta_grp.create_dataset("config", data=cfg)
                seed = metadata.get("seed", -1)
                meta_grp.create_dataset("seed", data=int(seed))
                source = metadata.get("source", "sim")
                meta_grp.create_dataset("source", data=str(source))
                outcome = metadata.get("outcome")
                if outcome is not None:
                    meta_grp.create_dataset("outcome", data=str(outcome))
                goal_xy = metadata.get("goal_xy")
                if goal_xy is not None:
                    meta_grp.create_dataset(
                        "goal_xy",
                        data=np.asarray(goal_xy, dtype=np.float32))
                tags = metadata.get("tags")
                if tags:
                    meta_grp.create_dataset(
                        "tags",
                        data=np.asarray([str(t) for t in tags], dtype=h5py.string_dtype()))

        return ep_path

    def write_eval_episode(
        self,
        episode: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        *,
        name_suffix: str,
    ) -> Path:
        ep_path = self._path_for(EVAL_EPISODE_FILENAME_PREFIX, name_suffix)
        obs             = np.asarray(episode["obs"],             dtype=np.float32)
        recon_posterior = np.asarray(episode["recon_posterior"], dtype=np.float32)
        recon_prior     = np.asarray(episode["recon_prior"],     dtype=np.float32)
        with h5py.File(ep_path, "w") as f:
            f.create_dataset("obs",             data=obs)
            f.create_dataset("recon_posterior", data=recon_posterior,
                             compression="gzip", compression_opts=4)
            f.create_dataset("recon_prior",     data=recon_prior,
                             compression="gzip", compression_opts=4)
            if recon_posterior.ndim == 4:
                f.create_dataset("diff_posterior",
                                 data=_pixel_diff(obs, recon_posterior).astype(np.float32),
                                 compression="gzip", compression_opts=4)
                f.create_dataset("diff_prior",
                                 data=_pixel_diff(obs, recon_prior).astype(np.float32),
                                 compression="gzip", compression_opts=4)
            f.create_dataset("h_posterior", data=np.asarray(episode["h_posterior"], dtype=np.float32))
            f.create_dataset("s_posterior", data=np.asarray(episode["s_posterior"], dtype=np.float32))
            f.create_dataset("h_prior",     data=np.asarray(episode["h_prior"],     dtype=np.float32))
            f.create_dataset("s_prior",     data=np.asarray(episode["s_prior"],     dtype=np.float32))

            meta_grp = f.create_group("metadata")
            if metadata is not None:
                cfg = _serialize_metadata_config(metadata)
                if cfg is not None:
                    meta_grp.create_dataset("config", data=cfg)
                for k in ("burn_in", "iteration", "episode_index", "seed"):
                    if metadata.get(k) is not None:
                        meta_grp.create_dataset(k, data=int(metadata[k]))
                if metadata.get("source") is not None:
                    meta_grp.create_dataset("source", data=str(metadata["source"]))
        return ep_path


# ---------------------------------------------------------------------------
# Rerun writer
# ---------------------------------------------------------------------------


class RerunEpisodeWriter(_BaseEpisodeWriter):
    """Writes both sim and eval episodes to Rerun ``.rrd`` files.

    See :class:`HDF5EpisodeWriter` for a description of the two payload
    schemas. Sim recordings log ``camera/image``, ``reward``, the action
    column, and per-step joint/EE/object signals; eval recordings log
    raw + processed obs, posterior + prior reconstructions, and the
    latent trajectories.

    Metadata is logged as static text docs on the ``metadata/`` entity
    so it surfaces in the viewer.
    """

    file_extension = "rrd"

    def __init__(self, output_dir: str) -> None:
        import rerun as rr  # noqa: F401 — fail fast if rerun is missing
        super().__init__(output_dir)

    def _new_recording(self, recording_id: str):
        import rerun as rr
        return rr.RecordingStream(
            application_id="chuck_dreamer",
            recording_id=recording_id,
        )

    @staticmethod
    def _log_metadata(rec, props: dict[str, str]) -> None:
        import rerun as rr
        for key, value in props.items():
            rec.log(f"metadata/{key}", rr.TextDocument(value), static=True)

    def write_episode(
        self,
        episode: dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None,
        *,
        name_suffix: str,
    ) -> Path:
        actions = _collect_actions(episode)
        T = next(iter(actions.values())).shape[0]
        if T == 0:
            raise ValueError("episode must not be empty")

        import rerun as rr

        ep_path = self._path_for(EPISODE_FILENAME_PREFIX, name_suffix)
        rec = self._new_recording(f"{EPISODE_FILENAME_PREFIX}-{name_suffix}")

        props = _rerun_metadata_props(metadata)
        if metadata is not None and metadata.get("goal_xy") is not None:
            goal = np.asarray(metadata["goal_xy"], dtype=np.float32)
            props["goal_xy"] = f"[{float(goal[0])}, {float(goal[1])}]"
        if metadata is not None and metadata.get("tags"):
            # Tags ride as a comma-joined string since Rerun metadata props are
            # dict[str, str]. The reader splits on commas to recover the tuple.
            props["tags"] = ",".join(str(t) for t in metadata["tags"])
        self._log_metadata(rec, props)

        images     = np.asarray(episode["image"],      dtype=np.uint8)
        rewards    = np.asarray(episode["reward"],     dtype=np.float32)
        timestamps = np.asarray(episode["timestamp"],  dtype=np.float32)
        joint_qpos = np.asarray(episode["joint_qpos"], dtype=np.float32)
        ee_pos     = np.asarray(episode["ee_pos"],     dtype=np.float32)
        ee_quat    = np.asarray(episode["ee_quat"],    dtype=np.float32)
        object_xy  = np.asarray(episode["object_xy"],  dtype=np.float32)

        seg_target     = episode.get("segmentation_target")
        seg_goal       = episode.get("segmentation_goal")
        seg_arm        = episode.get("segmentation_arm")
        seg_background = episode.get("segmentation_background")
        seg_clutter    = episode.get("segmentation_clutter")  # (T, K, H, W) or None

        for i in range(T):
            rec.set_time("step", sequence=i)
            rec.set_time("time", duration=float(timestamps[i]))

            rec.log("camera/image", rr.Image(images[i]))
            for kind, arr in actions.items():
                rec.log(kind, rr.Scalars(arr[i].tolist()))
            rec.log("reward",       rr.Scalars(float(rewards[i])))
            rec.log("joint_qpos",   rr.Scalars(joint_qpos[i].tolist()))
            rec.log("ee_pos",       rr.Scalars(ee_pos[i].tolist()))
            rec.log("ee_quat",      rr.Scalars(ee_quat[i].tolist()))
            rec.log("object_xy",    rr.Scalars(object_xy[i].tolist()))

            if seg_target is not None:
                rec.log("camera/seg/target",
                        rr.SegmentationImage(np.asarray(seg_target[i], dtype=np.uint8)))
            if seg_goal is not None:
                rec.log("camera/seg/goal",
                        rr.SegmentationImage(np.asarray(seg_goal[i], dtype=np.uint8)))
            if seg_arm is not None:
                rec.log("camera/seg/arm",
                        rr.SegmentationImage(np.asarray(seg_arm[i], dtype=np.uint8)))
            if seg_background is not None:
                rec.log("camera/seg/background",
                        rr.SegmentationImage(np.asarray(seg_background[i], dtype=np.uint8)))
            if seg_clutter is not None:
                # Aggregate clutter mask (union over K pieces).
                clutter_union = np.any(seg_clutter[i], axis=0)
                rec.log("camera/seg/clutter",
                        rr.SegmentationImage(clutter_union.astype(np.uint8)))
                for k in range(seg_clutter.shape[1]):
                    rec.log(f"camera/seg/clutter_{k}",
                            rr.SegmentationImage(np.asarray(seg_clutter[i, k], dtype=np.uint8)))

        rec.save(str(ep_path))
        return ep_path

    def write_eval_episode(
        self,
        episode: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        *,
        name_suffix: str,
    ) -> Path:
        import rerun as rr

        recording_id = f"{EVAL_EPISODE_FILENAME_PREFIX}-{name_suffix}"
        ep_path      = self._path_for(EVAL_EPISODE_FILENAME_PREFIX, name_suffix)
        rec          = self._new_recording(recording_id)

        self._log_metadata(rec, _rerun_metadata_props(
            metadata, extra_keys=("iteration", "episode_index", "burn_in"),
        ))

        obs             = episode["obs"]
        recon_posterior = np.asarray(episode["recon_posterior"], dtype=np.float32)
        recon_prior     = np.asarray(episode["recon_prior"],     dtype=np.float32)
        h_posterior     = np.asarray(episode["h_posterior"],     dtype=np.float32)
        s_posterior     = np.asarray(episode["s_posterior"],     dtype=np.float32)
        h_prior         = np.asarray(episode["h_prior"],         dtype=np.float32)
        s_prior         = np.asarray(episode["s_prior"],         dtype=np.float32)

        is_image = recon_posterior.ndim == 4   # (T, H, W, C)
        T        = recon_posterior.shape[0]

        for i in range(T):
            rec.set_time("step", sequence=i)

            if is_image:
                rec.log("obs",             rr.Image(_coerce_image_for_log(obs[i])))
                rec.log("recon/posterior", rr.Image(_denormalize_recon_image(recon_posterior[i])))
                rec.log("recon/prior",     rr.Image(_denormalize_recon_image(recon_prior[i])))
                rec.log("recon/diff_posterior",
                        rr.Image(_diff_to_uint8(_pixel_diff(obs[i], recon_posterior[i]))))
                rec.log("recon/diff_prior",
                        rr.Image(_diff_to_uint8(_pixel_diff(obs[i], recon_prior[i]))))
            else:
                rec.log("obs",             rr.Scalars(np.asarray(obs[i], dtype=np.float32).tolist()))
                rec.log("recon/posterior", rr.Scalars(recon_posterior[i].tolist()))
                rec.log("recon/prior",     rr.Scalars(recon_prior[i].tolist()))

            rec.log("latent/h",       rr.Scalars(h_posterior[i].tolist()))
            rec.log("latent/s",       rr.Scalars(s_posterior[i].tolist()))
            rec.log("latent/h_prior", rr.Scalars(h_prior[i].tolist()))
            rec.log("latent/s_prior", rr.Scalars(s_prior[i].tolist()))

        rec.save(str(ep_path))
        return ep_path
