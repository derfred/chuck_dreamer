"""Top-level run loop for the real arm.

Wires together :class:`~chuck_dreamer.real.run.camera.Camera`,
:class:`~chuck_dreamer.real.run.arm.Arm`,
:class:`~chuck_dreamer.real.run.controller.Controller`, and a chosen
:class:`~chuck_dreamer.real.run.policies.RealPolicy`. The observation
the policy receives is:

  obs = {
    "image":    (H, W, 3) uint8,
    "arm_qpos": (6,)      float32,
  }

There is no IK in this loop — policies are expected to emit something
the arm wrapper can consume directly (currently just a placeholder; the
arm's ``send_action`` is a no-op so the manual policy can be wired up
without hardware risk). The cv2 window blocks the main thread; ``s``
starts the controller, ``q``/ESC quits. The manual policy mirrors
joint positions from a leader arm onto the follower — see
:class:`~chuck_dreamer.real.run.policies.ManualPolicy`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2  # type: ignore[import-not-found]
import numpy as np
from lerobot.utils.robot_utils import precise_sleep

from .arm import Arm, ArmConfig
from .camera import Camera, CameraConfig
from .controller import Controller
from .policies import (
  CEMPolicyAdapter,
  CEMRunConfig,
  DreamerPolicyAdapter,
  ManualPolicy,
  ManualPolicyConfig,
  RealPolicy,
)

logger = logging.getLogger(__name__)

WINDOW_NAME = "chuck-dreamer · real"


@dataclass
class RunSettings:
  control_dt: float
  start_key:  str


# ---------------------------------------------------------------------------
# Factory: cfg.real.policy -> RealPolicy
# ---------------------------------------------------------------------------


_SO101_JOINTS = (
  "shoulder_pan", "shoulder_lift", "elbow_flex",
  "wrist_flex",   "wrist_roll",    "gripper",
)
_URDF_PATH = "assets/urdf/SO101/so101_meshless.urdf"


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
  """Convert [qw, qx, qy, qz] to a 3x3 rotation matrix.

  chuck_dreamer's EE actions encode quaternions as wxyz; scipy expects
  xyzw, so we re-order and delegate.
  """
  from scipy.spatial.transform import Rotation as R
  qw, qx, qy, qz = (float(c) for c in quat_wxyz)
  return R.from_quat([qx, qy, qz, qw]).as_matrix()


class _EEToJointsIK:
  """Adapter: 7-D EE action (CEM output) -> 6-D joint qpos (rad).

  Uses lerobot's placo-backed ``RobotKinematics`` to solve IK against
  the SO-101 URDF. Built once per run; each call re-uses the current
  follower qpos as the IK initial guess so successive plans stay near
  the live arm pose (faster convergence + fewer "flip" solutions).
  """

  def __init__(self) -> None:
    from lerobot.model.kinematics import RobotKinematics

    self.solver = RobotKinematics(
      urdf_path=_URDF_PATH,
      target_frame_name="gripper_frame_link",
      joint_names=list(_SO101_JOINTS),
    )

  def solve(self, ee_action: np.ndarray, current_qpos_rad: np.ndarray) -> np.ndarray:
    if ee_action.shape != (7,):
      raise ValueError(f"expected (7,) EE action, got {ee_action.shape}")
    T = np.eye(4)
    T[:3, 3]  = ee_action[:3]
    T[:3, :3] = _quat_wxyz_to_rotmat(ee_action[3:])
    # placo's set_joint bindings only accept Python double; numpy.float32
    # raises a "did not match C++ signature" error inside its C++ layer.
    # Force float64 here so IK works regardless of the upstream dtype.
    current_deg = np.rad2deg(np.asarray(current_qpos_rad, dtype=np.float64))
    target_deg  = self.solver.inverse_kinematics(current_deg, T)
    return np.deg2rad(np.asarray(target_deg, dtype=np.float64)).astype(np.float32)

  def current_ee_action(self, current_qpos_rad: np.ndarray) -> np.ndarray:
    """FK: ``(6,)`` joint qpos (rad) -> ``(7,)`` EE action ``[x,y,z,qw,qx,qy,qz]``.

    Used to warm-start CEM's action distribution at the current arm
    pose. Without this, CEM cold-starts with ``mean = zeros``, which in
    EE mode commands the gripper to the world origin with a degenerate
    zero quaternion — every sampled rollout is then unreachable noise.
    """
    from scipy.spatial.transform import Rotation as R
    current_deg = np.rad2deg(np.asarray(current_qpos_rad, dtype=np.float64))
    T = self.solver.forward_kinematics(current_deg)
    pos = T[:3, 3].astype(np.float32)
    # scipy returns [x, y, z, w]; chuck_dreamer's EE action is [w, x, y, z].
    qx, qy, qz, qw = R.from_matrix(T[:3, :3]).as_quat()
    quat_wxyz = np.asarray([qw, qx, qy, qz], dtype=np.float32)
    return np.concatenate([pos, quat_wxyz])


def _seed_cem_from_current_pose(
  policy: RealPolicy, ik: "_EEToJointsIK", current_qpos_rad: np.ndarray,
  *, init_std: float = 0.08, verbose: bool = False,
) -> None:
  """Poke ``CEMPolicy._mean`` / ``_prev_action`` / ``init_std`` after release.

  ``GatedPolicy.release`` resets the inner ``CEMPolicy``, which sets
  ``_mean = None`` (i.e. zeros on first ``act()``). For ``act_mode=ee``
  that means CEM begins by sampling action sequences around
  ``[0, 0, 0, 0, 0, 0, 0]`` — EE pose at the world origin with a
  degenerate zero quaternion, far outside reachable space. The reward
  head has nothing useful to score so CEM ends up playing essentially
  random actions, and you see the arm thrashing.

  This helper computes the current EE pose via FK and tiles it across
  the planning horizon as the initial action mean, plus drops
  ``init_std`` to a tight ball (5 cm position, 0.05 quat unit) so the
  first iteration's samples stay inside the reachable workspace. The
  refit / warm-start logic then carries the plan forward as usual.

  No-op for non-CEM policies (e.g. ``ManualPolicy``) and for non-EE
  CEM (action_dim != 7); the cold-start failure only manifests in EE
  mode.
  """
  from chuck_dreamer.policy import GatedPolicy

  if not isinstance(policy, GatedPolicy):
    return
  inner = policy.inner
  if not isinstance(inner, CEMPolicyAdapter):
    return
  cem = inner._cem
  if cem.model.action_dim != 7:
    return

  seed = ik.current_ee_action(current_qpos_rad)
  cem._mean        = np.tile(seed, (cem.horizon, 1)).astype(np.float32)
  cem._prev_action = seed.copy()
  cem.init_std     = float(init_std)
  if verbose:
    logger.info(
      "CEM seeded from current EE pose pos=%s quat=%s, init_std=%.3f",
      seed[:3].round(3), seed[3:].round(3), cem.init_std,
    )


def _hold_qpos(obs: dict) -> np.ndarray:
  """Hold-action for the real arm: command the current joint qpos.

  Used by :class:`~chuck_dreamer.policy.GatedPolicy` when wrapping a
  model-based policy on the real arm — the gate isn't actually called
  in practice (``Controller`` keeps the runner from sending anything
  while idle), but the wrapper still needs a sane default in case a
  caller does invoke ``act`` before release.
  """
  return np.asarray(obs["arm_qpos"], dtype=np.float32)


def _build_policy(cfg) -> RealPolicy:
  """Pick and construct the policy advertised by ``cfg.real.policy``.

  The manual policy is self-contained and runs immediately on start.
  The checkpoint-based policies (Dreamer actor, CEM) load a trained
  world model via the eval ``load_checkpoint`` path and are wrapped in
  :class:`~chuck_dreamer.policy.GatedPolicy` so the inner planner /
  actor only sees observations from the moment the operator hits the
  start key.
  """
  from ...policy import GatedPolicy

  ptype = cfg.real.policy.type
  if ptype == "manual":
    m = cfg.real.policy.manual
    if not m.port:
      raise ValueError("real.policy.manual.port must be set for the leader arm.")
    return ManualPolicy(ManualPolicyConfig(
      port=str(m.port),
      calibration_id=(None if m.calibration_id is None else str(m.calibration_id)),
      use_degrees=bool(m.use_degrees),
    ))
  if ptype == "dreamer":
    from ...eval.checkpoint import load_checkpoint
    d = cfg.real.policy.dreamer
    if not d.checkpoint:
      raise ValueError("real.policy.dreamer.checkpoint must be set")
    loaded = load_checkpoint(str(d.checkpoint), device=str(cfg.hardware.device))
    return GatedPolicy(
      DreamerPolicyAdapter(loaded.model, sample=bool(d.sample)),
      hold_action=_hold_qpos,
    )
  if ptype == "cem":
    from ...eval.checkpoint import load_checkpoint
    c = cfg.real.policy.cem
    if not c.checkpoint:
      raise ValueError("real.policy.cem.checkpoint must be set")
    loaded = load_checkpoint(str(c.checkpoint), device=str(cfg.hardware.device))
    # Action bounds come from the env that was rebuilt alongside the
    # checkpoint — that's the env CEM is implicitly planning for.
    action_space = loaded.env.action_space
    return GatedPolicy(
      CEMPolicyAdapter(
        loaded.model,
        CEMRunConfig(
          checkpoint=str(c.checkpoint),
          horizon=int(c.horizon),
          num_samples=int(c.num_samples),
          num_elites=int(c.num_elites),
          num_iterations=int(c.num_iterations),
          discount=float(c.get("discount", 1.0)) if hasattr(c, "get") else 1.0,
          action_low=action_space.low,
          action_high=action_space.high,
        ),
      ),
      hold_action=_hold_qpos,
    )
  if ptype == "cem_probe":
    from ...eval.checkpoint import load_checkpoint
    from .policies import CEMProbePolicyAdapter, CEMProbeRunConfig

    # Read from a real.policy.cem_probe block if present, fall back
    # to the standard cem block for shared knobs (horizon, num_samples
    # etc.) so users can flip ptype without re-doing all defaults.
    cp = cfg.real.policy.get("cem_probe", {}) if hasattr(cfg.real.policy, "get") else {}
    c  = cfg.real.policy.cem
    chk = cp.get("checkpoint") if hasattr(cp, "get") else None
    chk = chk or c.checkpoint
    if not chk:
      raise ValueError(
        "real.policy.cem_probe.checkpoint (or real.policy.cem.checkpoint) must be set."
      )
    probe_path  = cp.get("probe_path",  "") if hasattr(cp, "get") else ""
    reward_kind = cp.get("reward_kind", "push_shaped") if hasattr(cp, "get") else "push_shaped"
    goal_xy_raw = cp.get("goal_xy", [0.15, 0.18]) if hasattr(cp, "get") else [0.15, 0.18]
    goal_xy = (float(goal_xy_raw[0]), float(goal_xy_raw[1]))
    if not probe_path:
      raise ValueError(
        "real.policy.cem_probe.probe_path must be set — point at a .pt "
        "file from scripts/train_probe.py."
      )
    loaded = load_checkpoint(str(chk), device=str(cfg.hardware.device))
    action_space = loaded.env.action_space
    return GatedPolicy(
      CEMProbePolicyAdapter(
        loaded.model,
        CEMProbeRunConfig(
          checkpoint=str(chk),
          horizon=int(c.horizon),
          num_samples=int(c.num_samples),
          num_elites=int(c.num_elites),
          num_iterations=int(c.num_iterations),
          discount=float(c.get("discount", 1.0)) if hasattr(c, "get") else 1.0,
          action_low=action_space.low,
          action_high=action_space.high,
          probe_path=str(probe_path),
          reward_kind=str(reward_kind),
          goal_xy=goal_xy,
        ),
      ),
      hold_action=_hold_qpos,
    )
  raise ValueError(f"unknown real.policy.type: {ptype!r}")


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


def _draw_hud(frame: np.ndarray, state: str, fps: float, *, dry_run: bool) -> np.ndarray:
  """Annotate the live frame with the controller state and FPS."""
  out = frame.copy()
  cv2.putText(out, f"state: {state}", (10, 24),
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
  cv2.putText(out, f"fps:   {fps:5.1f}", (10, 48),
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
  if dry_run:
    cv2.putText(out, "DRY RUN (no arm)", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
  cv2.putText(out, "s: start    q/ESC: quit",
              (10, out.shape[0] - 12),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
  return out


def run(cfg) -> None:
  """Entry point for ``python main.py run``."""
  settings = RunSettings(
    control_dt=float(cfg.real.control_dt),
    start_key=str(cfg.real.start_key).lower(),
  )

  cam_cfg = CameraConfig(
    source=cfg.real.camera.source,
    width=int(cfg.real.camera.width),
    height=int(cfg.real.camera.height),
    fps=int(cfg.real.camera.fps),
    flip=bool(cfg.real.camera.flip),
  )
  arm_cfg = ArmConfig(
    port=(None if cfg.real.arm.port is None else str(cfg.real.arm.port)),
    calibration_id=(None if cfg.real.arm.calibration_id is None
                    else str(cfg.real.arm.calibration_id)),
    disable_torque_on_disconnect=bool(cfg.real.arm.disable_torque_on_disconnect),
    max_relative_target=(None if cfg.real.arm.max_relative_target is None
                         else float(cfg.real.arm.max_relative_target)),
    use_degrees=bool(cfg.real.arm.use_degrees),
  )

  policy = _build_policy(cfg)
  controller = Controller(policy)

  # Build the IK adapter once if the configured policy can emit
  # EE-mode (7-D) actions. We instantiate eagerly here (not inside
  # the action loop) so the URDF parse + placo init don't add per-step
  # latency. ``ManualPolicy`` already emits joint qpos so we skip it.
  ik: _EEToJointsIK | None = None
  if str(cfg.real.policy.type) in ("cem", "dreamer"):
    ik = _EEToJointsIK()
    logger.info("IK adapter built (URDF=%s)", _URDF_PATH)

  start_keycode = ord(settings.start_key[0])

  with Camera(cam_cfg) as camera, Arm(arm_cfg) as arm:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    if arm.is_dry_run:
      logger.warning(
        "real.arm.port is null — running without an arm. "
        "Camera and policy will run; joint commands are swallowed."
      )

    last_step_t = time.monotonic()
    fps_t       = last_step_t
    fps         = 0.0
    last_action: np.ndarray | None = None

    try:
      while True:
        loop_start = time.monotonic()

        bgr, rgb = camera.read()
        # FPS is a smoothed window over the camera read cadence.
        now = time.monotonic()
        dt = now - fps_t
        if dt > 0:
          fps = 0.9 * fps + 0.1 * (1.0 / dt)
        fps_t = now

        # Build obs from the live state of the world. Without an IK
        # solver we don't carry an EE pose; policies that need one have
        # to track it internally.
        current_qpos = arm.read_joint_qpos()
        obs = {
          "image":    rgb,
          "arm_qpos": current_qpos.astype(np.float32),
        }

        # Hand off to the controller at control_dt cadence — camera
        # refreshes faster than control to keep the HUD lively.
        if now - last_step_t >= settings.control_dt:
          # Re-seed CEM each step so the planning distribution stays
          # anchored on the live arm pose. Without this, ``_mean``
          # warm-starts from the previous step and drifts under the
          # reward-head's gradient noise — over many ticks the mean
          # walks off into unreachable territory and the follower
          # tries to teleport to a workspace-edge pose, slamming the
          # table. No-op for non-CEM/non-EE policies.
          if ik is not None and controller.state == "running":
            _seed_cem_from_current_pose(policy, ik, current_qpos)
          action = controller.step(obs)
          last_step_t = now
          if action is not None:
            last_action = action
            # The manual policy emits (6,) joint qpos directly; CEM /
            # Dreamer policies trained with ``act_mode=ee`` emit a (7,)
            # ``[x, y, z, qw, qx, qy, qz]`` EE pose that has to go
            # through IK before reaching the follower. Anything else
            # is logged and dropped (placeholder for future shapes).
            joint_action: np.ndarray | None = None
            if action.shape == (6,):
              joint_action = action
            elif action.shape == (7,) and ik is not None:
              try:
                joint_action = ik.solve(action, current_qpos)
              except Exception as e:  # noqa: BLE001 — placo errors vary
                logger.warning("IK solve failed: %s — holding pose", e)
            else:
              logger.warning(
                "unsupported action shape %s; holding pose", action.shape,
              )

            if joint_action is not None:
              try:
                arm.send_joint_qpos(joint_action)
              except Exception as e:  # noqa: BLE001 — serial faults vary
                logger.warning("send_joint_qpos failed: %s — holding pose", e)

        # Draw + window event pump.
        hud = _draw_hud(bgr, controller.state, fps, dry_run=arm.is_dry_run)
        cv2.imshow(WINDOW_NAME, hud)
        key = cv2.waitKey(1) & 0xFF

        if key == start_keycode:
          controller.start(obs)
          logger.info("controller started")
          # Seed CEM's action distribution from the current EE pose so
          # the first planning iteration samples reachable trajectories
          # (see _seed_cem_from_current_pose for the why). No-op for
          # non-CEM / non-EE policies.
          if ik is not None:
            _seed_cem_from_current_pose(policy, ik, current_qpos, verbose=True)
        elif key in (27, ord("q")):  # ESC or q
          logger.info("quit requested")
          break

        # Give the OS a sliver of time even when the camera is fast.
        # precise_sleep matches what upstream lerobot-teleoperate uses —
        # plain time.sleep oversleeps on macOS by 5–15 ms, which jitters
        # the control cadence visibly at 60 Hz.
        budget = (1.0 / max(cam_cfg.fps, 1)) - (time.monotonic() - loop_start)
        precise_sleep(budget)

    finally:
      if isinstance(policy, ManualPolicy):
        policy.stop_listener()
      controller.stop()
      cv2.destroyWindow(WINDOW_NAME)
      logger.info("last commanded action: %s", last_action)
