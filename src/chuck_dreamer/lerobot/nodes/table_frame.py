"""Table-frame coordinate nodes (spec §7.7 + the frame-suffix scheme).

Every coordinate track exists in exactly one frame; crossing frames is an
explicit node consuming the DATASET-scoped ``table_to_arm`` transform
(``p_table = R @ p_arm + t``, metres — produced by
``import-lerobot extract-table-to-arm``):

  ee_pos_arm    (mm, arm)   → ee_pos_table    (mm, table)
  ee_quat_arm   (arm)       → ee_quat_table   (table)
  ee_pos/quat_table          → ee_action_table (mm + unitless, table)
  obj_pos_table (mm, table) → obj_pos_arm     (mm, arm)

The nodes are pure per-episode transforms; validity propagates untouched.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from chuck_dreamer.common.tracks import Frame, Persist, Track, TrackSpec, Unit

from ..context import RunContext
from ..harness import InputDecl
from .ee import POSE_MM_UNITS, shift_action

if TYPE_CHECKING:
  from ..trackset import NodeView


def _transform_mm(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
  """``(R, t_mm)`` from the store's ``table_to_arm`` payload (R unitless,
  t in metres → millimetres to match the coordinate tracks)."""
  R = np.asarray(payload["R"], dtype=np.float64)
  t = np.asarray(payload["t"], dtype=np.float64).reshape(3) * 1000.0
  return R, t


def _quat_from_R(R: np.ndarray) -> np.ndarray:
  """Rotation matrix → quaternion ``[w, x, y, z]`` (Shepperd's method,
  branch on the largest diagonal term for numerical stability)."""
  m = np.asarray(R, dtype=np.float64)
  tr = m[0, 0] + m[1, 1] + m[2, 2]
  if tr > 0:
    s = np.sqrt(tr + 1.0) * 2
    q = np.array([0.25 * s,
                  (m[2, 1] - m[1, 2]) / s,
                  (m[0, 2] - m[2, 0]) / s,
                  (m[1, 0] - m[0, 1]) / s])
  elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
    s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
    q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                  (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
  elif m[1, 1] >= m[2, 2]:
    s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
    q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                  0.25 * s, (m[1, 2] + m[2, 1]) / s])
  else:
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                  (m[1, 2] + m[2, 1]) / s, 0.25 * s])
  out: np.ndarray = q / np.linalg.norm(q)
  return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Hamilton product ``a ⊗ b`` for ``[w, x, y, z]`` quaternions;
  ``b`` may be ``(T, 4)``."""
  aw, ax, ay, az = a
  bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  return np.stack([
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ], axis=-1)


class _TableNodeBase:
  scope_level = "episode"
  lane = "producer"

  def __init__(self, ctx: RunContext) -> None:
    self.ctx = ctx


class EePosTableNode(_TableNodeBase):
  """Spec §7.7: EE position, arm → table frame."""
  name = "ee_pos_table"
  inputs = (InputDecl("ee_pos_arm", frame=Frame.ARM, unit=Unit.MM),
            InputDecl("table_to_arm"))
  outputs = (TrackSpec("ee_pos_table", np.float32, (3,), Frame.TABLE,
                       Unit.MM, persist=Persist.TRAJECTORY),)

  def run(self, view: "NodeView") -> list[Track]:
    p_arm = view.track("ee_pos_arm").values
    R, t = _transform_mm(view.artifact("table_to_arm"))
    p_table = (p_arm @ R.T + t).astype(np.float32)
    return [Track(self.outputs[0], p_table)]


class EeQuatTableNode(_TableNodeBase):
  """EE orientation, arm → table frame (rotation part of ``table_to_arm``)."""
  name = "ee_quat_table"
  inputs = (InputDecl("ee_quat_arm", frame=Frame.ARM),
            InputDecl("table_to_arm"))
  outputs = (TrackSpec("ee_quat_table", np.float32, (4,), Frame.TABLE,
                       Unit.UNITLESS, persist=Persist.TRAJECTORY),)

  def run(self, view: "NodeView") -> list[Track]:
    q_arm = view.track("ee_quat_arm").values
    R, _ = _transform_mm(view.artifact("table_to_arm"))
    q_table = _quat_mul(_quat_from_R(R), q_arm).astype(np.float32)
    return [Track(self.outputs[0], q_table)]


class EeActionTableNode(_TableNodeBase):
  """Spec §7.9 in the table frame: shift-by-one of the table-frame pose."""
  name = "ee_action_table"
  inputs = (InputDecl("ee_pos_table", frame=Frame.TABLE, unit=Unit.MM),
            InputDecl("ee_quat_table", frame=Frame.TABLE))
  outputs = (TrackSpec("ee_action_table", np.float32, (7,), Frame.TABLE,
                       POSE_MM_UNITS, persist=Persist.TRAJECTORY),)

  def run(self, view: "NodeView") -> list[Track]:
    pos  = view.track("ee_pos_table").values
    quat = view.track("ee_quat_table").values
    action = shift_action(np.concatenate([pos, quat], axis=-1))
    return [Track(self.outputs[0], action)]


class ObjPosArmNode(_TableNodeBase):
  """Object position, table → arm frame (inverse of ``table_to_arm``).
  Runs after the pose fit, so it shares the worker lane."""
  name = "obj_pos_arm"
  lane = "worker"
  inputs = (InputDecl("obj_pos_table", frame=Frame.TABLE, unit=Unit.MM),
            InputDecl("table_to_arm"))
  outputs = (TrackSpec("obj_pos_arm", np.float32, (3,), Frame.ARM,
                       Unit.MM, validity=True, persist=Persist.TRAJECTORY),)

  def run(self, view: "NodeView") -> list[Track]:
    track = view.track("obj_pos_table")
    R, t = _transform_mm(view.artifact("table_to_arm"))
    p_arm = ((track.values - t) @ R).astype(np.float32)   # R.T applied rowwise
    assert track.valid is not None
    return [Track(self.outputs[0], p_arm, track.valid.copy())]


def build_ee_table_nodes(ctx: RunContext) -> list[_TableNodeBase]:
  return [EePosTableNode(ctx), EeQuatTableNode(ctx), EeActionTableNode(ctx)]


def build_object_table_nodes(ctx: RunContext) -> list[_TableNodeBase]:
  return [ObjPosArmNode(ctx)]
