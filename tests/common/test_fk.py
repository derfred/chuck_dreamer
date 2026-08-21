from __future__ import annotations

import numpy as np
import pytest

from chuck_dreamer.common import FK_MODEL_PATH
from chuck_dreamer.common.fk import CANONICAL_JOINT_NAMES, FK, Q_HOME


@pytest.fixture(scope="module")
def fk() -> FK:
  return FK(FK_MODEL_PATH)


def test_q_maps_to_the_canonical_joints_in_order(fk):
  """The public `q` is in CANONICAL_JOINT_NAMES order. Pinocchio assigns the
  configuration slots, so this pins the two to agree rather than assuming
  the URDF happens to declare its joints in the same sequence."""
  assert [fk.model.names[j] for j in fk.joint_ids] == CANONICAL_JOINT_NAMES
  np.testing.assert_array_equal(fk.qpos_idx, np.arange(5))


def test_forward_is_plus_x_at_home(fk):
  """At the URDF's zero the arm extends along +x, away from the table edge.
  This is the frame convention downstream analysis assumes."""
  tip = fk(Q_HOME)
  horizontal = tip.copy()
  horizontal[2] = 0.0
  unit = horizontal / np.linalg.norm(horizontal)
  np.testing.assert_allclose(unit, [1.0, 0.0, 0.0], atol=1e-3)


def test_link_lengths_match_the_urdf_geometry(fk):
  """Frame-independent link lengths. These come from the CAD and are what
  make the model the *same arm* as the simulator's MJCF; a change here means
  the URDF was swapped, not merely re-zeroed."""
  chain = fk.fk_chain(Q_HOME)
  seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)
  # shoulder, upper arm, forearm, wrist, tip
  np.testing.assert_allclose(
    seg, [0.06478, 0.11600, 0.13500, 0.06372, 0.09845], atol=1e-4)


def test_fk_chain_ends_at_the_ee(fk):
  q = np.array([-0.712, 0.176, 0.471, 0.833, 0.124])
  chain = fk.fk_chain(q)
  assert chain.shape == (len(CANONICAL_JOINT_NAMES) + 1, 3)
  np.testing.assert_allclose(chain[-1], fk(q), atol=1e-12)


def test_dq_offsets_the_joints(fk):
  """`dq` is added to `q` before kinematics, so an offset on one and the
  equivalent shift on the other must agree."""
  q = np.array([-0.712, 0.176, 0.471, 0.833, 0.124])
  dq = np.array([0.01, -0.02, 0.03, 0.0, 0.0])
  np.testing.assert_allclose(fk(q, dq=dq), fk(q + dq), atol=1e-12)


def test_jaw_does_not_move_the_tip(fk):
  """The URDF carries a 6th joint (the jaw). It is a leaf off gripper_link
  and must not displace the tip -- `q` is deliberately 5 long."""
  q = np.array([-0.712, 0.176, 0.471, 0.833, 0.124])
  full = fk._configuration(q)
  assert full.shape == (fk.model.nq,)
  assert fk.model.nq == 6
  assert full[5] == 0.0


@pytest.mark.parametrize("bad", [np.zeros(4), np.zeros(6)])
def test_rejects_wrong_shaped_q(fk, bad):
  with pytest.raises(ValueError, match="q must be shape"):
    fk(bad)


def test_rejects_wrong_shaped_dq(fk):
  with pytest.raises(ValueError, match="dq must be shape"):
    fk(Q_HOME, dq=np.zeros(3))


def test_missing_model_raises():
  with pytest.raises(FileNotFoundError):
    FK("/nonexistent/so101.urdf")


def test_unknown_ee_frame_raises():
  with pytest.raises(ValueError, match="frame .* not found"):
    FK(FK_MODEL_PATH, ee_frame_name="no_such_frame")
