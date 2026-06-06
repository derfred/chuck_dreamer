"""The runtime perception pipeline (spec §3.2, §3.3).

A perception module declares the modalities it ``consumes`` and ``emits`` and
transforms a per-step modality dict in place. The pipeline runs the configured
modules **inline at the start of each policy step**, in order — so slow
perception reduces the effective policy rate but never touches the control loop
(which keeps slewing the last published setpoint). This mirrors the offline
``Stage`` protocol's ``produces`` / ``requires`` idea
(``lerobot/stages/base.py``) but is online and stateful: a module may hold
per-episode warm-start state, cleared by :meth:`PerceptionModule.reset` at the
top of each episode.

``EePoseModule`` is the cheap, real-time, dependency-free module M3 ships to
exercise the full sensor → perception → observation → Rerun path in sim. The
heavy SAM2 object localizer lives in ``perception_object.py`` (config-gated,
off by default).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .modalities import EE

__all__ = ["PerceptionModule", "PerceptionPipeline", "EePoseModule"]


@runtime_checkable
class PerceptionModule(Protocol):
  name: str
  consumes: tuple[str, ...]
  emits: tuple[str, ...]

  def reset(self) -> None:
    """Clear per-episode state (e.g. warm-start pose). Called once per episode."""
    ...

  def process(self, data: dict[str, np.ndarray], *, t: float) -> dict[str, np.ndarray]:
    """Read ``consumes`` from ``data``; return only the new ``emits`` entries.

    The pipeline merges the returned dict back into ``data`` so a later module
    can consume it. Returning an empty dict is the "gap" signal (e.g. the
    localizer found no mask this step): the modality is simply absent from the
    observation, which the typed accessors report as ``None``.
    """
    ...


class PerceptionPipeline:
  """Runs an ordered list of perception modules inline each policy step."""

  def __init__(self, modules: list[PerceptionModule]) -> None:
    self._modules = list(modules)

  @property
  def modules(self) -> list[PerceptionModule]:
    return list(self._modules)

  def reset(self) -> None:
    """Forward ``reset`` to every module (per-episode warm-start clear)."""
    for m in self._modules:
      m.reset()

  def run(self, data: dict[str, np.ndarray], *, t: float) -> dict[str, np.ndarray]:
    """Fold each module's emits into ``data``, in configured order."""
    for m in self._modules:
      data.update(m.process(data, t=t))
    return data


class EePoseModule:
  """Emit the 7-D ``ee`` modality (``ee_pos ‖ ee_quat``) from the backend FK.

  Cheap and real-time — no calibration, no camera, no torch. It reads the
  backend's measured end-effector pose, so it only works on a backend that
  exposes ``ee_pos`` / ``ee_quat`` (``MujocoBackend``); a backend without them
  raises at construction, surfacing the misconfiguration at startup.
  """

  name = "ee_pose"
  consumes: tuple[str, ...] = ()
  emits: tuple[str, ...] = (EE,)

  def __init__(self, backend) -> None:
    if not (hasattr(backend, "ee_pos") and hasattr(backend, "ee_quat")):
      raise TypeError(
        f"EePoseModule needs a backend exposing ee_pos()/ee_quat(); "
        f"{type(backend).__name__} does not")
    self._backend = backend

  @classmethod
  def from_config(cls, cfg, *, backend, **_params) -> "EePoseModule":
    return cls(backend)

  def reset(self) -> None:
    pass

  def process(self, data: dict[str, np.ndarray], *, t: float) -> dict[str, np.ndarray]:
    pos = np.asarray(self._backend.ee_pos(), dtype=np.float32)
    quat = np.asarray(self._backend.ee_quat(), dtype=np.float32)
    return {EE: np.concatenate([pos, quat])}
