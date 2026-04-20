"""Data collection utilities: EpisodeWriter and random_push_policy."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
import mujoco  # type: ignore[import-untyped]

from .scene_config import SceneConfig


# ---------------------------------------------------------------------------
# EpisodeWriter
# ---------------------------------------------------------------------------

class EpisodeWriter:
    """
    Writes episodes to HDF5 files.

    Each episode is stored in ``output_dir/episode_NNNNN.hdf5`` with the
    structure::

        images      (T, H, W, 3)  uint8
        actions     (T, 3)        float32
        rewards     (T,)          float32
        metadata/
            config  scalar string (JSON)
            seed    scalar int64
            source  scalar string
    """

    def __init__(self, output_dir: str, format: str = "hdf5") -> None:
        if format != "hdf5":
            raise ValueError(
                f"Unsupported format '{format}'. Only 'hdf5' is supported.")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ep_count = self._count_existing_episodes()

    def _count_existing_episodes(self) -> int:
        return len(list(self.output_dir.glob("episode_*.hdf5")))

    def write_episode(
        self,
        episode: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Persist one episode.

        Parameters
        ----------
        episode:
            List of dicts with keys ``pre_image`` (H,W,3 uint8),
            ``post_image`` (H,W,3 uint8), ``action`` (3,),
            ``reward`` (float).
        metadata:
            Optional dict. May include ``config`` (SceneConfig or dict),
            ``seed`` (int), ``source`` (str).

        Returns
        -------
        Path to the written file.
        """
        if not episode:
            raise ValueError("episode must not be empty")

        actions = np.stack(
            [np.asarray(s["action"], dtype=np.float32) for s in episode])
        rewards = np.array([float(s["reward"])
                           for s in episode], dtype=np.float32)

        ep_path = self.output_dir / f"episode_{self._ep_count:05d}.hdf5"
        with h5py.File(ep_path, "w") as f:
            pre_images = np.stack([s["pre_image"]
                                  for s in episode], axis=0).astype(np.uint8)
            post_images = np.stack([s["post_image"]
                                   for s in episode], axis=0).astype(np.uint8)
            f.create_dataset(
                "pre_images",
                data=pre_images,
                compression="gzip",
                compression_opts=4)
            f.create_dataset(
                "post_images",
                data=post_images,
                compression="gzip",
                compression_opts=4)
            f.create_dataset("actions", data=actions)
            f.create_dataset("rewards", data=rewards)

            meta_grp = f.create_group("metadata")
            if metadata is not None:
                cfg = metadata.get("config")
                if cfg is not None:
                    if not isinstance(cfg, (str, bytes)):
                        cfg = json.dumps(
                            cfg if isinstance(cfg, dict)
                            else asdict(cfg)  # type: ignore[arg-type]
                        )
                    meta_grp.create_dataset("config", data=cfg)
                seed = metadata.get("seed", -1)
                meta_grp.create_dataset("seed", data=int(seed))
                source = metadata.get("source", "sim")
                meta_grp.create_dataset("source", data=str(source))

        self._ep_count += 1
        return ep_path


# ---------------------------------------------------------------------------
# Random push policy
# ---------------------------------------------------------------------------

class RandomPushPolicy:
    """
        A simple heuristic push policy.

        State 1 - initial: arm in rest position pointing to the middle of the table
        State 2 — approach: move the end-effector toward the object, maintain a standoff distance.
        State 3 — push: once close, move toward the goal.
        State 4 — done: push complete, hold position.

        Returns a (3,) float32 action [dx, dy, dz].
    """

    _PUSH_Z = 0.075          # EE height — midpoint of typical object side
    _APPROACH_SPEED = 0.02
    _PUSH_SPEED = 0.015
    _STANDOFF = 0.06         # approach to this distance behind the object
    _PUSH_THRESH = 0.08      # switch from approach to push phase
    _PUSH_DISTANCE = 0.12   # distance at which we consider the push done

    state: str = "initial"  # "initial", "ready", "approach", "push", "done"

    def __init__(self, config: SceneConfig, controller, rng: np.random.Generator | None = None) -> None:
        self.config     = config
        self.controller = controller
        self.rng        = rng if rng is not None else np.random.default_rng()

    @property
    def ready_xz(self) -> np.ndarray:
        base  = np.array(self.config.robot_base_pos[:2], dtype=np.float64)
        zero  = np.array([0.0, 0.0], dtype=np.float64)
        return base + (zero - base) * 0.3

    @property
    def goal_xy(self) -> np.ndarray:
        return np.array(self.config.goal_pos, dtype=np.float64)

    def _update_state(self, obs: dict[str, np.ndarray]) -> None:
        pass

    def _act_initial(self, _obs: dict[str, np.ndarray]) -> np.ndarray:
        target = np.append(self.ready_xz, self._PUSH_Z)
        step   = (target - _obs["ee_pos"]) * 0.2 + _obs["ee_pos"]
        return self.controller.ik_for_ee_pos(step, _obs["qpos"])

    def _act_ready(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        pass

    def _act_approach(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        pass

    def _act_push(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        pass

    def _act_done(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        "Returns a (3,) float32 action [dx, dy, dz]."

        ee_pos     = obs["ee_pos"].astype(np.float64)
        obj_pos_xy = obs["object_pos"].astype(np.float64)

        self._update_state(obs)
        return getattr(self, f"_act_{self.state}")(obs)  # type: ignore[attr-defined]


def random_push_policy(
    obs: dict[str, np.ndarray],
    config: SceneConfig,
) -> np.ndarray:
    """
    A simple heuristic push policy.

    Phase 1 — approach: move the end-effector toward the object.
    Phase 2 — push: once close, move toward the goal.

    Returns a (3,) float32 action [dx, dy, dz].
    """
    ee_pos     = obs["ee_pos"].astype(np.float64)
    obj_pos_xy = obs["object_pos"].astype(np.float64)
    goal_xy    = np.array(config.goal_pos, dtype=np.float64)
    ee_xy = ee_pos[:2]
    ee_z = float(ee_pos[2])

    # Push direction: from object toward goal
    push_dir = goal_xy - obj_pos_xy
    push_norm = np.linalg.norm(push_dir)
    if push_norm > 1e-6:
        push_dir /= push_norm
    else:
        push_dir = np.array([1.0, 0.0])

    # Approach target: stand off behind the object on the push axis
    approach_target = obj_pos_xy - push_dir * _STANDOFF

    dist_to_approach = float(np.linalg.norm(ee_xy - approach_target))

    if dist_to_approach > _PUSH_THRESH:
        # Phase 1: move to approach position behind the object
        direction = approach_target - ee_xy
        direction /= (np.linalg.norm(direction) + 1e-8)
        dx, dy = direction * _APPROACH_SPEED
    else:
        # Phase 2: push straight through toward the goal
        dx, dy = push_dir * _PUSH_SPEED
        dx += float(rng.uniform(-0.002, 0.002))
        dy += float(rng.uniform(-0.002, 0.002))

    # Descend toward pushing height
    dz = float(np.clip(_PUSH_Z - ee_z, -0.01, 0.01))

    dx = float(np.clip(dx, -0.02, 0.02))
    dy = float(np.clip(dy, -0.02, 0.02))

    return np.array([dx, dy, dz], dtype=np.float32)
