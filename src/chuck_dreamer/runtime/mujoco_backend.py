"""``MujocoBackend`` — the SO-101 MuJoCo model behind the backend protocol.

Wraps the existing simulation (``SceneBuilder`` / ``SceneGenerator``,
``sim/pushing_env.py``) so the runtime's control kernel sits on top of sim
physics identically to how it will sit on the real arm.

An internal physics thread steps the sim in real time; ``read``/``write``
are guarded by a lock shared with it. An optional ``launch_passive`` viewer
(main thread only) draws the commanded setpoint and current EE as marker
overlays
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import cast

import mujoco  # type: ignore[import-untyped]
import numpy as np

from ..sim.scene_builder import SceneBuilder
from ..sim.scene_config import SceneConfig

_EE_SITE = "ee_site"


class MujocoBackend:
  """SO-101 MuJoCo simulation as a :class:`RobotBackend`."""

  def __init__(
    self,
    scene: SceneConfig,
    *,
    viewer: bool = False,
    realtime: bool = True,
    render_size: tuple[int, int] | None = None,
  ) -> None:
    self._scene          = scene
    self._viewer_enabled = viewer
    self._realtime       = realtime
    self._lock           = threading.Lock()

    self.model    = SceneBuilder().build(scene, render_size=render_size)
    self.data     = mujoco.MjData(self.model)
    self._fk_data = mujoco.MjData(self.model) # needed to deconflict with physics thread's mjData stack

    joint_names = scene.joint_names
    self._n     = len(joint_names)
    joint_ids   = [
      mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names
    ]
    self._qpos_adr = np.array([self.model.jnt_qposadr[j] for j in joint_ids])
    self._dof_adr  = np.array([self.model.jnt_dofadr[j] for j in joint_ids])
    self._ctrl_idx = np.array([
      mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
      for n in scene.actuator_names
    ])
    self._jnt_range  = self.model.jnt_range[joint_ids].copy()  # (n, 2)
    self._ctrl_range = self.model.actuator_ctrlrange[self._ctrl_idx].copy()
    self._ee_sid     = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, _EE_SITE)

    # Initialise to the scene's home pose, holding it (ctrl == qpos) so the
    # position actuators don't drive back to zero (mirrors Controller.reset_initial_qpos).
    q0                             = np.asarray(scene.joint_initial_qpos, dtype=np.float64)
    self.data.qpos[self._qpos_adr] = q0
    self.data.qvel[self._dof_adr]  = 0.0
    self.data.ctrl[self._ctrl_idx] = q0
    mujoco.mj_forward(self.model, self.data)

    self._stop = threading.Event()
    self._thread: threading.Thread | None = None

  @classmethod
  def from_config(cls, cfg, *, viewer: bool = False, realtime: bool = True) -> "MujocoBackend":
    """Build from the project config, sampling a scene via ``SceneGenerator``."""
    from ..sim.scene_generator import SceneGenerator

    scene = SceneGenerator(cfg).sample()
    h, w  = (int(x) for x in str(cfg.env.render_size).lower().split("x"))
    return cls(scene, viewer=viewer, realtime=realtime, render_size=(h, w))

  # -- backend protocol -----------------------------------------------------

  @property
  def n_joints(self) -> int:
    return self._n

  def read_positions(self) -> np.ndarray:
    with self._lock:
      return cast(np.ndarray, self.data.qpos[self._qpos_adr].copy())

  def last_positions(self) -> np.ndarray:
    return self.read_positions()

  def write_positions(self, q: np.ndarray) -> None:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (self._n,):
      raise ValueError(f"write_positions expects ({self._n},); got {q.shape}")
    with self._lock:
      self.data.ctrl[self._ctrl_idx] = q

  def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
    # Prefer true joint ranges; fall back to actuator ctrl range where a
    # joint is marked unlimited (lower == upper == 0 in MuJoCo).
    lower            = self._jnt_range[:, 0].copy()
    upper            = self._jnt_range[:, 1].copy()
    unlimited        = lower == upper
    lower[unlimited] = self._ctrl_range[unlimited, 0]
    upper[unlimited] = self._ctrl_range[unlimited, 1]
    return lower, upper

  def start(self) -> None:
    self._stop.clear()
    self._thread = threading.Thread(target=self._physics_run, name="runtime-mujoco", daemon=True)
    self._thread.start()

  def stop(self, join_timeout: float = 5.0) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=join_timeout)
      self._thread = None

  # -- physics + viewer -----------------------------------------------------

  def step(self, n: int = 1) -> None:
    """Deterministically advance the sim ``n`` steps (test/headless hook)."""
    with self._lock:
      for _ in range(n):
        mujoco.mj_step(self.model, self.data)

  def _physics_run(self) -> None:
    dt            = self.model.opt.timestep
    next_deadline = time.monotonic()
    while not self._stop.is_set():
      self.step(1)
      next_deadline += dt
      if self._realtime:
        sleep_for = next_deadline - time.monotonic()
        if sleep_for > 0:
          self._stop.wait(sleep_for)
        else:
          next_deadline = time.monotonic()

  def ee_pos(self) -> np.ndarray:
    with self._lock:
      return cast(np.ndarray, self.data.site_xpos[self._ee_sid].copy())

  def ee_quat(self) -> np.ndarray:
    """EE orientation as a ``(4,)`` quaternion ``[w, x, y, z]`` (sim convention).

    Mirrors ``ee_pos``; together they form the 7-D ``ee`` modality the
    perception layer emits (``ee_pos ‖ ee_quat``), matching
    ``Controller.get_ee_quat`` in ``sim/pushing_env.py``.
    """
    quat = np.zeros(4, dtype=np.float64)
    with self._lock:
      mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._ee_sid])
    return quat

  def physics_lock(self):
    """The mutex serializing all access to ``self.data``.

    The viewer's ``sync()`` reads ``self.data`` on the main thread while the
    physics thread writes it via ``mj_step``; the caller holds this around
    ``sync()`` (and overlay drawing) so the two never overlap.
    """
    return self._lock

  @contextlib.contextmanager
  def open_viewer(self):
    """Context-managed ``launch_passive`` viewer (main thread only)."""
    import mujoco.viewer  # type: ignore[import-untyped]

    with mujoco.viewer.launch_passive(self.model, self.data) as v:
      cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "main_camera")
      if cam_id >= 0:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        v.cam.fixedcamid = cam_id
      yield v

  def draw_overlays(self, viewer, *, q_cmd: np.ndarray | None = None) -> None:
    """Draw the current EE (red) and the commanded-pose EE (green) markers.

    TODO(M5): also draw the task-space safety box once the EE workspace
    envelope is defined.
    """
    viewer.user_scn.ngeom = 0
    self._add_sphere(viewer, self.ee_pos(), rgba=(1.0, 0.0, 0.0, 0.9))
    if q_cmd is not None:
      self._add_sphere(viewer, self._fk_ee(q_cmd), rgba=(0.0, 1.0, 0.0, 0.9))

  def _add_sphere(self, viewer, pos, *, rgba, radius: float = 0.02) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
      return
    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
      g, type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=np.array([radius, 0.0, 0.0]), pos=np.asarray(pos, dtype=np.float64),
      mat=np.eye(3).flatten(), rgba=np.asarray(rgba, dtype=np.float32),
    )
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    viewer.user_scn.ngeom += 1

  def _fk_ee(self, q: np.ndarray) -> np.ndarray:
    """EE position for joint configuration ``q`` via FK on persistent scratch.

    The whole read-copy-forward runs under ``self._lock``: ``mj_forward``
    touches the model's shared stack, so it must not overlap the physics
    thread's ``mj_step`` (otherwise "mjData stack in use"). The scratch
    buffer is reused (allocated once in ``__init__``) for the same reason —
    never allocate an MjData here on the render thread.
    """
    with self._lock:
      self._fk_data.qpos[:]              = self.data.qpos
      self._fk_data.qpos[self._qpos_adr] = np.asarray(q, dtype=np.float64)
      mujoco.mj_forward(self.model, self._fk_data)
      return cast(np.ndarray, self._fk_data.site_xpos[self._ee_sid].copy())
