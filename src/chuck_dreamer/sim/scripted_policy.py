"""Heuristic scripted push policy."""

from __future__ import annotations

import logging
from typing import cast

import mujoco  # type: ignore[import-untyped]
import numpy as np

from .scene_config import SceneConfig

logger = logging.getLogger(__name__)


class ScriptedPolicy:
  """
      A simple heuristic push policy.

      State 1 - initial: arm in rest position pointing to the middle of the table
      State 2 — approach: move the end-effector toward the object, maintain a standoff distance.
      State 3 — push: once close, move toward the goal.
      State 4 — done: push complete, hold position.

      Returns a (7,) float32 EE pose action ``[x, y, z, qw, qx, qy, qz]``.
      The env's IK warm-starts from the current joint configuration, so the
      lookahead-based segment chasing produces smooth joint motion.
  """

  _PUSH_Z = 0.075          # EE height — midpoint of typical object side
  _MOVE_SPEED = 0.015
  _LOOKAHEAD  = 0.02   # meters ahead of projected position to command
  _STANDOFF = 0.06         # approach to this distance behind the object
  _CLOSE_THRESH = 0.08      # switch from approach to push phase

  state: str = "initial"  # "initial", "ready", "approach", "push", "done"

  def __init__(self, auto_advance_from_ready: bool = False) -> None:
    self.start_xyz: np.ndarray | None = None
    self.scene: SceneConfig | None = None
    # Hold-orientation quaternion captured on first observation. Keeping the
    # gripper aligned with the home pose during the push is what we want; we
    # don't have a strong reason to rotate.
    self.hold_quat: np.ndarray | None = None
    # Headless episode collection skips the ``ready`` pause; the interactive
    # viewer keeps it so the user can press space to start the push.
    self.auto_advance_from_ready = auto_advance_from_ready

  @property
  def ready_xy(self) -> np.ndarray:
    assert self.scene is not None
    base  = np.array(self.scene.robot_base_pos[:2], dtype=np.float64)
    zero  = np.array([0.0, 0.0], dtype=np.float64)
    return cast(np.ndarray, base + (zero - base) * 0.3)

  @property
  def goal_xy(self) -> np.ndarray:
    assert self.scene is not None
    return np.array(self.scene.goal_xy, dtype=np.float64)

  @property
  def goal_xyz(self) -> np.ndarray:
    return cast(np.ndarray, np.append(self.goal_xy, self._PUSH_Z))

  @property
  def object_xy(self) -> np.ndarray:
    assert self.scene is not None
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

  def _pose(self, xyz: np.ndarray) -> np.ndarray:
    """Pack [x, y, z] + hold quaternion into a (7,) action."""
    assert self.hold_quat is not None
    return np.concatenate([
      np.asarray(xyz, dtype=np.float32),
      self.hold_quat.astype(np.float32),
    ])

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
  ) -> np.ndarray:
    if self.start_xyz is None:
      self.start_xyz = np.asarray(start_xyz, dtype=np.float64).copy()

    start = self.start_xyz
    end   = np.asarray(target_xyz, dtype=np.float64)
    ee    = np.asarray(obs["ee_pos"], dtype=np.float64)

    # Direction and total length of the segment.
    segment = end - start
    seg_len = float(np.linalg.norm(segment))
    if seg_len < 1e-6:
        return self._pose(end)
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

    return self._pose(commanded)

  def _act_initial(self, obs: dict[str, np.ndarray]) -> np.ndarray:
    target = np.append(self.ready_xy, self._PUSH_Z)
    return self._step_to(obs["ee_pos"], target, obs)

  def _act_ready(self, obs: dict[str, np.ndarray]) -> np.ndarray:
    return self._pose(np.asarray(obs["ee_pos"], dtype=np.float64))

  def _act_approach(self, obs: dict[str, np.ndarray]) -> np.ndarray:
    return self._step_to(obs["ee_pos"], self.approach_xyz, obs)

  def _act_push(self, obs: dict[str, np.ndarray]) -> np.ndarray:
    return self._step_to(self.approach_xyz, self.goal_xyz, obs)

  def _act_done(self, obs: dict[str, np.ndarray]) -> np.ndarray:
    return self._pose(np.asarray(obs["ee_pos"], dtype=np.float64))

  def reset(self, scene: SceneConfig) -> None:
    self.scene      = scene
    self.state      = "initial"
    self.start_xyz  = None
    self.hold_quat  = None

  def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
    if self.hold_quat is None:
      self.hold_quat = np.asarray(obs["ee_quat"], dtype=np.float32).copy()

    if self.auto_advance_from_ready and self.state == "ready":
      self.start_xyz = None
      self.state     = "approach"

    next_state = self._determine_state(obs)
    if next_state is not None and next_state != self.state:
      self.start_xyz = None
      self.state     = next_state

    return cast(np.ndarray, getattr(self, f"_act_{self.state}")(obs))

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

  def is_done(self) -> bool:
    return self.state == "done"
