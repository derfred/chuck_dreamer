"""Unit tests for the M3 observation + modality-composition machinery.

Pure (no threads, no backend): the composed-set algebra, the pipeline
order-consistency check, the startup fail-fast, and the RuntimeObservation
surface (attribute back-compat + the modality dict + the training adapter).
"""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from chuck_dreamer.runtime.modalities import (
  EE,
  IMAGE,
  OBJECT_XY,
  ModalityError,
  RuntimeObservation,
  check_required,
  compose_modalities,
  required_modalities,
)


class _Sensor:
  def __init__(self, name, produces):
    self.name = name
    self.produces = tuple(produces)


class _Module:
  def __init__(self, name, consumes, emits):
    self.name = name
    self.consumes = tuple(consumes)
    self.emits = tuple(emits)


# -- compose_modalities -------------------------------------------------------


def test_compose_unions_base_sensors_and_modules():
  sensors = [_Sensor("cam", (IMAGE,))]
  perception = [_Module("loc", (IMAGE,), (OBJECT_XY,))]
  active = compose_modalities(sensors, perception)
  assert {"t", "q_meas", IMAGE, OBJECT_XY} <= active
  assert "leader_qpos" not in active


def test_compose_includes_leader_when_present():
  active = compose_modalities([], [], leader_present=True)
  assert "leader_qpos" in active
  assert compose_modalities([], [], leader_present=False) == frozenset({"t", "q_meas"})


def test_compose_raises_on_unsatisfiable_consume():
  # Localizer consumes `image`, but no sensor produces it.
  perception = [_Module("loc", (IMAGE,), (OBJECT_XY,))]
  with pytest.raises(ModalityError, match="image"):
    compose_modalities([], perception)


def test_compose_respects_pipeline_order():
  # A module that produces `image` placed *after* the consumer does not satisfy
  # it — order matters (no reordering).
  producer = _Module("producer", (), (IMAGE,))
  consumer = _Module("consumer", (IMAGE,), (OBJECT_XY,))
  with pytest.raises(ModalityError):
    compose_modalities([], [consumer, producer])
  # Correct order is fine.
  assert OBJECT_XY in compose_modalities([], [producer, consumer])


# -- required_modalities / check_required ------------------------------------


def test_required_modalities_from_config():
  rt = OmegaConf.create({"required_modalities": ["object_xy", "image"]})
  assert required_modalities(rt) == frozenset({"object_xy", "image"})
  assert required_modalities(OmegaConf.create({})) == frozenset()


def test_check_required_passes_when_satisfied():
  check_required(frozenset({IMAGE}), frozenset({"t", "q_meas", IMAGE}))


def test_check_required_raises_on_missing():
  with pytest.raises(ModalityError, match="object_xy"):
    check_required(frozenset({OBJECT_XY}), frozenset({"t", "q_meas", IMAGE}))


# -- RuntimeObservation -------------------------------------------------------


def _obs(**modalities):
  q = np.zeros(6)
  data = {"t": 0.5, "q_meas": q, **modalities}
  return RuntimeObservation(t=0.5, q_meas=q, modalities=data)


def test_attribute_backcompat():
  # The scripted-policy surface still works.
  o = _obs()
  assert o.t == 0.5
  assert o.q_meas.shape == (6,)
  assert o.leader_qpos is None
  assert o.has_leader is False


def test_leader_attribute_and_flag():
  q = np.zeros(6)
  o = RuntimeObservation(t=0.0, q_meas=q, leader_qpos=np.ones(6),
                         modalities={"t": 0.0, "q_meas": q, "leader_qpos": np.ones(6)})
  assert o.has_leader is True
  np.testing.assert_array_equal(o.leader_qpos, np.ones(6))


def test_get_has_present_and_typed_accessors():
  img = np.zeros((4, 4, 3), np.uint8)
  o = _obs(image=img, object_xy=np.array([1.0, 2.0], np.float32))
  assert o.has(IMAGE) and o.has(OBJECT_XY)
  assert not o.has(EE)
  assert o.get(EE) is None
  assert o.image is not None
  np.testing.assert_array_equal(o.object_xy, [1.0, 2.0])
  assert o.object_uv is None
  assert {IMAGE, OBJECT_XY, "t", "q_meas"} <= o.present()


def test_to_observation_builds_training_components():
  o = _obs(ee=np.arange(7, dtype=np.float32),
           object_xy=np.array([1.0, 2.0], np.float32))
  training = o.to_observation((EE, OBJECT_XY))
  assert set(training.components) == {EE, OBJECT_XY}
  np.testing.assert_array_equal(training.components[OBJECT_XY], [1.0, 2.0])


def test_to_observation_raises_on_absent_component():
  o = _obs()  # no object_xy this step
  with pytest.raises(ModalityError, match="object_xy"):
    o.to_observation((OBJECT_XY,))
