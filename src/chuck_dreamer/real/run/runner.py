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
    loaded = load_checkpoint(str(d.checkpoint))
    return GatedPolicy(
      DreamerPolicyAdapter(loaded.model, sample=bool(d.sample)),
      hold_action=_hold_qpos,
    )
  if ptype == "cem":
    from ...eval.checkpoint import load_checkpoint
    c = cfg.real.policy.cem
    if not c.checkpoint:
      raise ValueError("real.policy.cem.checkpoint must be set")
    loaded = load_checkpoint(str(c.checkpoint))
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
          action = controller.step(obs)
          last_step_t = now
          if action is not None:
            last_action = action
            # The manual policy emits a (6,) joint qpos; route it to the
            # follower so the leader's pose is mirrored. Model-based
            # policies (image-mode actor / CEM) still emit something
            # else (action_dim depends on the trainer); for those we
            # just log the action for now since IK is unwired.
            if action.shape == (6,):
              try:
                arm.send_joint_qpos(action)
              except Exception as e:  # noqa: BLE001 — serial faults vary
                logger.warning("send_joint_qpos failed: %s — holding pose", e)

        # Draw + window event pump.
        hud = _draw_hud(bgr, controller.state, fps, dry_run=arm.is_dry_run)
        cv2.imshow(WINDOW_NAME, hud)
        key = cv2.waitKey(1) & 0xFF

        if key == start_keycode:
          controller.start(obs)
          logger.info("controller started")
        elif key in (27, ord("q")):  # ESC or q
          logger.info("quit requested")
          break

        # Give the OS a sliver of time even when the camera is fast.
        budget = max(0.0, (1.0 / max(cam_cfg.fps, 1)) - (time.monotonic() - loop_start))
        if budget > 0:
          time.sleep(budget)

    finally:
      if isinstance(policy, ManualPolicy):
        policy.stop_listener()
      controller.stop()
      cv2.destroyWindow(WINDOW_NAME)
      logger.info("last commanded action: %s", last_action)
