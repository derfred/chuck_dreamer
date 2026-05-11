"""Episode writers — persist sim episodes to HDF5 or Rerun ``.rrd`` files."""

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

# All writer filenames share this prefix so `iter_episodes` can glob them
# regardless of which call site produced them.
EPISODE_FILENAME_PREFIX = "episode"


def EpisodeWriter(output_dir: str, format: str = "hdf5"):
    """Factory that returns the concrete writer for the requested ``format``.

    Supported formats: ``hdf5`` (default), ``rerun``.
    """
    if format == "hdf5":
        return HDF5EpisodeWriter(output_dir)
    if format == "rerun":
        return RerunEpisodeWriter(output_dir)
    raise ValueError(
        f"Unsupported format '{format}'. Supported formats: {SUPPORTED_FORMATS}.")


def _serialize_metadata_config(metadata: dict[str, Any] | None) -> str | None:
    if metadata is None:
        return None
    cfg = metadata.get("config")
    if cfg is None:
        return None
    if isinstance(cfg, (str, bytes)):
        return cfg if isinstance(cfg, str) else cfg.decode("utf-8")
    return json.dumps(cfg if isinstance(cfg, dict) else asdict(cfg))  # type: ignore[arg-type]


# Per-step columns shared by all action modes. ``joint_action`` xor
# ``ee_action`` is selected at write time based on which key the episode
# dict contains.
_REQUIRED_KEYS = (
    "image", "reward", "timestamp",
    "joint_qpos", "ee_pos", "ee_quat", "object_xy",
)


def _resolve_action(episode: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    """Return (kind, array) where kind is 'joint_action' or 'ee_action'.

    Episodes recorded under ``act_mode='joint'`` carry ``joint_action``;
    EE-mode episodes carry ``ee_action``. Exactly one must be present.
    """
    if "joint_action" in episode:
        return "joint_action", np.asarray(episode["joint_action"], dtype=np.float32)
    if "ee_action" in episode:
        return "ee_action", np.asarray(episode["ee_action"], dtype=np.float32)
    raise KeyError("episode is missing an action field (expected joint_action or ee_action)")


class HDF5EpisodeWriter:
    """
    Writes episodes to HDF5 files.

    Each episode is stored in ``output_dir/episode_NNNNN.hdf5`` with the
    structure::

        images          (T, H, W, 3)    uint8
        joint_action    (T, n_joints)   float32   — present iff act_mode == "joint"
        ee_action       (T, 7)          float32   — present iff act_mode == "ee"
        rewards         (T,)            float32
        timestamps      (T,)            float32
        joint_qpos      (T, n_joints)   float32
        ee_pos          (T, 3)          float32
        ee_quat         (T, 4)          float32
        object_xy       (T, 2)          float32
        metadata/
            config      scalar string (JSON)
            seed        scalar int64
            source      scalar string
            outcome     scalar string
            goal_xy     (2,) float32
            act_mode    scalar string ("joint" | "ee")
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_episode(
        self,
        episode: dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None,
        *,
        name_suffix: str,
    ) -> Path:
        action_kind, action = _resolve_action(episode)
        if action.shape[0] == 0:
            raise ValueError("episode must not be empty")

        rewards    = np.asarray(episode["reward"],     dtype=np.float32)
        timestamps = np.asarray(episode["timestamp"],  dtype=np.float32)
        joint_qpos = np.asarray(episode["joint_qpos"], dtype=np.float32)
        ee_pos     = np.asarray(episode["ee_pos"],     dtype=np.float32)
        ee_quat    = np.asarray(episode["ee_quat"],    dtype=np.float32)
        object_xy  = np.asarray(episode["object_xy"],  dtype=np.float32)
        images     = np.asarray(episode["image"],      dtype=np.uint8)

        ep_path = self.output_dir / f"{EPISODE_FILENAME_PREFIX}-{name_suffix}.hdf5"
        with h5py.File(ep_path, "w") as f:
            f.create_dataset("images",     data=images,  compression="gzip", compression_opts=4)
            f.create_dataset(action_kind,  data=action)
            f.create_dataset("rewards",    data=rewards)
            f.create_dataset("timestamps", data=timestamps)
            f.create_dataset("joint_qpos", data=joint_qpos)
            f.create_dataset("ee_pos",     data=ee_pos)
            f.create_dataset("ee_quat",    data=ee_quat)
            f.create_dataset("object_xy",  data=object_xy)

            meta_grp = f.create_group("metadata")
            act_mode = "joint" if action_kind == "joint_action" else "ee"
            meta_grp.create_dataset("act_mode", data=act_mode)
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

        return ep_path


class RerunEpisodeWriter:
    """
    Writes episodes to Rerun ``.rrd`` files (one per episode).

    Each episode is stored in ``output_dir/episode_NNNNN.rrd``. Images are
    logged as ``camera/image``; scalar/vector signals use the per-component
    ``Scalars`` archetype. Whichever of ``joint_action`` / ``ee_action``
    the episode contains is logged. Metadata is attached on the recording
    properties entity so it is visible in the viewer.
    """

    def __init__(self, output_dir: str) -> None:
        import rerun as rr  # noqa: F401  — fail fast if rerun is missing
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_episode(
        self,
        episode: dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None,
        *,
        name_suffix: str,
    ) -> Path:
        action_kind, action = _resolve_action(episode)
        if action.shape[0] == 0:
            raise ValueError("episode must not be empty")

        import rerun as rr

        recording_id = f"{EPISODE_FILENAME_PREFIX}-{name_suffix}"
        ep_path = self.output_dir / f"{recording_id}.rrd"
        rec = rr.RecordingStream(
            application_id="chuck_dreamer",
            recording_id=recording_id,
        )

        act_mode = "joint" if action_kind == "joint_action" else "ee"
        props: dict[str, Any] = {"act_mode": act_mode}
        if metadata is not None:
            cfg_json = _serialize_metadata_config(metadata)
            if cfg_json is not None:
                props["config"] = cfg_json
            if "seed" in metadata:
                props["seed"] = str(int(metadata["seed"]))
            if "source" in metadata:
                props["source"] = str(metadata["source"])
            if metadata.get("outcome") is not None:
                props["outcome"] = str(metadata["outcome"])
            if metadata.get("goal_xy") is not None:
                goal = np.asarray(metadata["goal_xy"], dtype=np.float32)
                props["goal_xy"] = f"[{float(goal[0])}, {float(goal[1])}]"
        for key, value in props.items():
            rec.log(f"metadata/{key}", rr.TextDocument(value), static=True)

        images     = np.asarray(episode["image"],      dtype=np.uint8)
        rewards    = np.asarray(episode["reward"],     dtype=np.float32)
        timestamps = np.asarray(episode["timestamp"],  dtype=np.float32)
        joint_qpos = np.asarray(episode["joint_qpos"], dtype=np.float32)
        ee_pos     = np.asarray(episode["ee_pos"],     dtype=np.float32)
        ee_quat    = np.asarray(episode["ee_quat"],    dtype=np.float32)
        object_xy  = np.asarray(episode["object_xy"],  dtype=np.float32)

        T = action.shape[0]
        for i in range(T):
            rec.set_time("step", sequence=i)
            rec.set_time("time", duration=float(timestamps[i]))

            rec.log("camera/image",   rr.Image(images[i]))
            rec.log(action_kind,      rr.Scalars(action[i].tolist()))
            rec.log("reward",         rr.Scalars(float(rewards[i])))
            rec.log("joint_qpos",     rr.Scalars(joint_qpos[i].tolist()))
            rec.log("ee_pos",         rr.Scalars(ee_pos[i].tolist()))
            rec.log("ee_quat",        rr.Scalars(ee_quat[i].tolist()))
            rec.log("object_xy",      rr.Scalars(object_xy[i].tolist()))

        rec.save(str(ep_path))
        return ep_path
