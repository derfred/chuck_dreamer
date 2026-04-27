"""Data collection utilities: EpisodeWriter, RandomPushPolicy, ScenePlayer."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import mujoco  # type: ignore[import-untyped]
import h5py  # type: ignore[import-untyped]
import numpy as np

from .scene_config import SceneConfig
from ..policy import Policy, Action

logger = logging.getLogger(__name__)


SUPPORTED_FORMATS = ("hdf5", "rerun")


# ---------------------------------------------------------------------------
# EpisodeWriter
# ---------------------------------------------------------------------------

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


class HDF5EpisodeWriter:
    """
    Writes episodes to HDF5 files.

    Each episode is stored in ``output_dir/episode_NNNNN.hdf5`` with the
    structure::

        images       (T, H, W, 3)    uint8
        actions      (T, n_joints)   float32
        rewards      (T,)            float32
        timestamps   (T,)            float32   seconds since episode start
        joint_qpos   (T, n_joints)   float32   arm joint positions
        ee_pos       (T, 3)          float32
        ee_quat      (T, 4)          float32
        object_xy    (T, 2)          float32
        metadata/
            config   scalar string (JSON)
            seed     scalar int64
            source   scalar string
            outcome  scalar string   "done" | "terminated" | "timeout" | "crashed"
            goal_xy  (2,) float32
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ep_count = len(list(self.output_dir.glob("episode_*.hdf5")))

    def write_episode(
        self,
        episode: dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Persist one episode.

        Parameters
        ----------
        episode:
            Dict of T-stacked arrays with keys ``image`` (T,H,W,3 uint8),
            ``action`` (T,n_joints), ``reward`` (T,), ``timestamp`` (T,),
            ``joint_qpos`` (T,n_joints), ``ee_pos`` (T,3), ``ee_quat`` (T,4),
            ``object_xy`` (T,2).
        metadata:
            Optional dict. May include ``config`` (SceneConfig or dict),
            ``seed`` (int), ``source`` (str).

        Returns
        -------
        Path to the written file.
        """
        if not episode or episode["action"].shape[0] == 0:
            raise ValueError("episode must not be empty")

        actions    = np.asarray(episode["action"],     dtype=np.float32)
        rewards    = np.asarray(episode["reward"],     dtype=np.float32)
        timestamps = np.asarray(episode["timestamp"],  dtype=np.float32)
        joint_qpos = np.asarray(episode["joint_qpos"], dtype=np.float32)
        ee_pos     = np.asarray(episode["ee_pos"],     dtype=np.float32)
        ee_quat    = np.asarray(episode["ee_quat"],    dtype=np.float32)
        object_xy  = np.asarray(episode["object_xy"],  dtype=np.float32)
        images     = np.asarray(episode["image"],      dtype=np.uint8)

        ep_path = self.output_dir / f"episode_{self._ep_count:05d}.hdf5"
        with h5py.File(ep_path, "w") as f:
            f.create_dataset("images",     data=images,  compression="gzip", compression_opts=4)
            f.create_dataset("actions",    data=actions)
            f.create_dataset("rewards",    data=rewards)
            f.create_dataset("timestamps", data=timestamps)
            f.create_dataset("joint_qpos", data=joint_qpos)
            f.create_dataset("ee_pos",     data=ee_pos)
            f.create_dataset("ee_quat",    data=ee_quat)
            f.create_dataset("object_xy",  data=object_xy)

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

        self._ep_count += 1
        return ep_path


class RerunEpisodeWriter:
    """
    Writes episodes to Rerun ``.rrd`` files (one per episode).

    Each episode is stored in ``output_dir/episode_NNNNN.rrd``. Images are
    logged as ``camera/image``; scalar/vector signals use
    the per-component ``Scalars`` archetype. Metadata is attached on the
    recording properties entity so it is visible in the viewer.
    """

    def __init__(self, output_dir: str) -> None:
        import rerun as rr  # noqa: F401  — fail fast if rerun is missing
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ep_count = len(list(self.output_dir.glob("episode_*.rrd")))

    def write_episode(
        self,
        episode: dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        if not episode or episode["action"].shape[0] == 0:
            raise ValueError("episode must not be empty")

        import rerun as rr

        ep_path = self.output_dir / f"episode_{self._ep_count:05d}.rrd"
        rec = rr.RecordingStream(
            application_id="chuck_dreamer",
            recording_id=f"episode_{self._ep_count:05d}",
        )

        if metadata is not None:
            cfg_json = _serialize_metadata_config(metadata)
            props: dict[str, Any] = {}
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
        actions    = np.asarray(episode["action"],     dtype=np.float32)
        rewards    = np.asarray(episode["reward"],     dtype=np.float32)
        timestamps = np.asarray(episode["timestamp"],  dtype=np.float32)
        joint_qpos = np.asarray(episode["joint_qpos"], dtype=np.float32)
        ee_pos     = np.asarray(episode["ee_pos"],     dtype=np.float32)
        ee_quat    = np.asarray(episode["ee_quat"],    dtype=np.float32)
        object_xy  = np.asarray(episode["object_xy"],  dtype=np.float32)

        T = actions.shape[0]
        for i in range(T):
            rec.set_time("step", sequence=i)
            rec.set_time("time", duration=float(timestamps[i]))

            rec.log("camera/image",  rr.Image(images[i]))
            rec.log("action",        rr.Scalars(actions[i].tolist()))
            rec.log("reward",        rr.Scalars(float(rewards[i])))
            rec.log("joint_qpos",    rr.Scalars(joint_qpos[i].tolist()))
            rec.log("ee_pos",        rr.Scalars(ee_pos[i].tolist()))
            rec.log("ee_quat",       rr.Scalars(ee_quat[i].tolist()))
            rec.log("object_xy",     rr.Scalars(object_xy[i].tolist()))

        rec.save(str(ep_path))
        self._ep_count += 1
        return ep_path


# ---------------------------------------------------------------------------
# Random push policy
# ---------------------------------------------------------------------------

class RandomPushPolicy(Policy):
  """
      A simple heuristic push policy.

      State 1 - initial: arm in rest position pointing to the middle of the table
      State 2 — approach: move the end-effector toward the object, maintain a standoff distance.
      State 3 — push: once close, move toward the goal.
      State 4 — done: push complete, hold position.

      Returns a (3,) float32 action [dx, dy, dz].
  """

  _PUSH_Z = 0.075          # EE height — midpoint of typical object side
  _MOVE_SPEED = 0.015
  _LOOKAHEAD  = 0.02   # meters ahead of projected position to command
  _STANDOFF = 0.06         # approach to this distance behind the object
  _CLOSE_THRESH = 0.08      # switch from approach to push phase

  state: str = "initial"  # "initial", "ready", "approach", "push", "done"

  def __init__(self) -> None:
    self.controller: Any = None
    self.start_xyz: np.ndarray | None = None

  @property
  def ready_xy(self) -> np.ndarray:
    base  = np.array(self.scene.robot_base_pos[:2], dtype=np.float64)
    zero  = np.array([0.0, 0.0], dtype=np.float64)
    return cast(np.ndarray, base + (zero - base) * 0.3)

  @property
  def goal_xy(self) -> np.ndarray:
    return np.array(self.scene.goal_pos, dtype=np.float64)

  @property
  def goal_xyz(self) -> np.ndarray:
    return cast(np.ndarray, np.append(self.goal_xy, self._PUSH_Z))

  @property
  def object_xy(self) -> np.ndarray:
    return np.array(self.scene.target.pos[:2], dtype=np.float64)

  @property
  def approach_xy(self) -> np.ndarray:
    push_dir  = self.goal_xy - self.object_xy
    push_dist = np.linalg.norm(push_dir)
    if push_dist > 1e-6:
      push_dir /= push_dist
    else:
      push_dir = np.array([1.0, 0.0])
    approach_point = self.object_xy - push_dir * self._STANDOFF
    return cast(np.ndarray, approach_point)

  @property
  def approach_xyz(self) -> np.ndarray:
    return cast(np.ndarray, np.append(self.approach_xy, self._PUSH_Z))

  def _determine_state(self, obs: dict[str, np.ndarray]) -> str | None:
      if self.state == "initial":
          if np.linalg.norm(obs["ee_pos"][:2] - self.ready_xy) < self._CLOSE_THRESH:
              return "ready"
      elif self.state == "approach":
          if np.linalg.norm(obs["ee_pos"] - self.approach_xyz) < self._CLOSE_THRESH:
              return "push"
      elif self.state == "push":
          if np.linalg.norm(obs["ee_pos"] - self.goal_xyz) < self._CLOSE_THRESH:
              return "done"
      return None

  def _step_to(
      self,
      start_xyz: np.ndarray,
      target_xyz: np.ndarray,
      obs: dict[str, np.ndarray],
  ) -> Action:
    if self.start_xyz is None:
      self.start_xyz = np.asarray(start_xyz, dtype=np.float64).copy()

    start = self.start_xyz
    end   = np.asarray(target_xyz, dtype=np.float64)
    ee    = np.asarray(obs["ee_pos"], dtype=np.float64)

    # Direction and total length of the segment.
    segment = end - start
    seg_len = float(np.linalg.norm(segment))
    if seg_len < 1e-6:
        return Action.from_qpos(self.controller.ik_for_ee_pos(end, obs["qpos"]))
    direction = segment / seg_len

    # Project current EE position onto the segment to find actual progress.
    # This is what makes it robust: if the EE lagged behind or got pushed
    # sideways by contact, we re-reference from where we actually are.
    progress = float(np.dot(ee - start, direction))
    progress = float(np.clip(progress, 0.0, seg_len))

    # Command a point a small lookahead past the projection, capped at the end.
    # Lookahead > step size gives the controller something to chase and
    # smooths out jitter from the projection.
    commanded_progress = min(progress + self._LOOKAHEAD, seg_len)
    commanded = start + direction * commanded_progress

    return Action.from_qpos(self.controller.ik_for_ee_pos(commanded, obs["qpos"]))

  def _act_initial(self, obs: dict[str, np.ndarray]) -> Action:
    target = np.append(self.ready_xy, self._PUSH_Z)
    return self._step_to(obs["ee_pos"], target, obs)

  def _act_ready(self, obs: dict[str, np.ndarray]) -> Action:
    return Action.from_qpos(obs["arm_qpos"])

  def _act_approach(self, obs: dict[str, np.ndarray]) -> Action:
    return self._step_to(obs["ee_pos"], self.approach_xyz, obs)

  def _act_push(self, obs: dict[str, np.ndarray]) -> Action:
    return self._step_to(self.approach_xyz, self.goal_xyz, obs)

  def _act_done(self, obs: dict[str, np.ndarray]) -> Action:
    return Action.from_qpos(obs["arm_qpos"])

  def reset(self, controller, scene: SceneConfig) -> None:
    self.controller = controller
    self.scene      = scene
    self.state      = "initial"
    self.start_xyz  = None

  def act(self, obs: dict[str, np.ndarray]) -> tuple[Action, str | None]:
    cur_state  = self.state
    next_state = self._determine_state(obs)
    changed    = next_state is not None and next_state != cur_state
    if changed:
      assert next_state is not None
      self.start_xyz = None
      self.state     = next_state

    action = getattr(self, f"_act_{self.state}")(obs)
    return action, (cur_state if changed else None)

  def is_done(self) -> bool:
    return self.state == "done"

  def insert_hints(self, viewer: mujoco.Viewer, rgba: np.ndarray = np.array([0.0, 1.0, 0.0, 0.8], dtype=np.float32)) -> None:
    """Insert custom geoms into the scene for visualization hints."""
    if self.state != "ready":
      return

    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
      return

    assert self.scene is not None

    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
      g,
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=np.array([0.02, 0.0, 0.0], dtype=np.float64),
      pos=self.goal_xyz,
      mat=np.eye(3).flatten(),
      rgba=rgba,
    )
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    viewer.user_scn.ngeom += 1

    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
      return

    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
      g,
      type=mujoco.mjtGeom.mjGEOM_ARROW,
      size=np.zeros(3),
      pos=np.zeros(3),
      mat=np.eye(3).flatten(),
      rgba=rgba,
    )
    mujoco.mjv_connector(
      g,
      mujoco.mjtGeom.mjGEOM_ARROW,
      0.01,
      self.approach_xyz,
      self.goal_xyz,
    )
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    viewer.user_scn.ngeom += 1

  def advance_from_ready(self) -> None:
    if self.state == "ready":
      self.state = "approach"
