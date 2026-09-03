"""The modality-bearing runtime observation, plus startup composition checks.

This is the M3 (spec §3.3/§5) replacement for the bare M0–M2 observation: a
:class:`RuntimeObservation` that still exposes the ``t`` / ``q_meas`` /
``leader_qpos`` *attributes* the scripted policies read (``GoToPose`` /
``SineSweep`` read ``obs.t``; ``ManualPolicy`` reads ``obs.leader_qpos`` /
``obs.has_leader`` / ``obs.q_meas``), but now also carries the full
``modalities`` dict produced by the sensor + perception layers and an
adapter to the training-side component-dict
:class:`~chuck_dreamer.training.observation.Observation` the Dreamer encoder
consumes (M6).

The modality *names* are the training component vocabulary
(``image`` / ``joints`` / ``proprio`` / ``ee`` / ``object_xy`` / ``object_uv``)
re-exported here so the runtime and the model agree on a single set, plus the
runtime-only ``leader_qpos`` and the always-present ``t`` / ``q_meas``.

:func:`compose_modalities` builds the active set the session produces and
validates the configured perception pipeline is order-consistent;
:func:`check_required` is the spec §9.1-step-5 fail-fast: a session whose
declared required modality is produced by nobody never starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from chuck_dreamer.training.observation import (
  EE,
  IMAGE,
  JOINTS,
  OBJECT_UV,
  OBJECT_XY,
  PROPRIO,
  VALID_COMPONENTS,
)
from chuck_dreamer.training.observation import Observation as TrainingObservation

if TYPE_CHECKING:
  from .control_state import ControlState
  from .perception import PerceptionModule
  from .sensors import Sensor
  from .teleop import LeaderState

__all__ = [
  "RuntimeObservation",
  "ModalityError",
  "compose_modalities",
  "required_modalities",
  "check_required",
  # re-exported modality names
  "IMAGE",
  "JOINTS",
  "PROPRIO",
  "EE",
  "OBJECT_XY",
  "OBJECT_UV",
  "VALID_COMPONENTS",
  "LEADER_QPOS",
  "T",
  "Q_MEAS",
]

# Runtime-only / always-present modality names. ``t`` and ``q_meas`` are built
# by the policy loop every step; ``leader_qpos`` is the M2 teleop modality.
T = "t"
Q_MEAS = "q_meas"
LEADER_QPOS = "leader_qpos"

# What the policy loop always produces before any sensor / perception runs.
_BASE_MODALITIES = frozenset({T, Q_MEAS})


class ModalityError(ValueError):
  """A required modality is unproduced, or a pipeline consume is unsatisfiable."""


@dataclass(frozen=True)
class RuntimeObservation:
  """What a policy sees at one step: clock + joints + all produced modalities.

  ``t`` / ``q_meas`` / ``leader_qpos`` stay as named attributes so the M0–M2
  scripted policies keep working unchanged; they are now projections of the
  :attr:`control` / :attr:`leader` state objects that carry the age and health
  a bare vector cannot (see :meth:`build`). ``modalities`` is the full dict the
  sensor + perception layers composed for this step (it *includes* ``t`` /
  ``q_meas`` / ``leader_qpos`` so introspection over :meth:`present` is
  complete); the typed accessors and :meth:`get` read from it.

  TODO(M6): revisit the observation interface before plugging in Dreamer. This
  surface is a deliberate bridge — attribute accessors for the scripted
  policies plus a ``modalities`` dict and a :meth:`to_observation` adapter onto
  the training component-dict ``Observation``. M6 should spend dedicated cycles
  reconciling the runtime observation, the training ``Observation``, and
  Dreamer's declared obs-space: units (e.g. object_xy is mm from the localizer
  vs metres in training), dtypes, image-normalization timing, world-frame
  conventions, and whether ``to_observation`` is the right seam or the runtime
  should build the training ``Observation`` natively. Do not treat this as
  final.
  """

  t: float
  """Seconds since the producing :meth:`Policy.reset` call."""

  q_meas: np.ndarray
  """Latest measured joint positions, shape ``(n_joints,)``."""

  leader_qpos: np.ndarray | None = None
  """Latest leader joint positions in follower order + radians, shape
  ``(n_joints,)``; ``None`` when no leader is configured or before its first
  read."""

  modalities: dict[str, np.ndarray] = field(default_factory=dict)
  """Every modality produced this step, keyed by modality name (sensor outputs,
  perception emits, and the base ``t`` / ``q_meas`` / ``leader_qpos``)."""

  control: "ControlState | None" = None
  """The control loop's own measurement for this step, straight off the control
  channel: the same object its safety layer acted on, carrying ``q_age``,
  :class:`~chuck_dreamer.runtime.control_state.FaultFlags` and whichever
  diagnostics the tick's read budget covered. ``q_meas`` is its ``q``.

  ``None`` only where no channel state was available -- a directly-constructed
  observation in a test, or a policy loop running without a control loop."""

  leader: "LeaderState | None" = None
  """The leader reading behind :attr:`leader_qpos`, with its ``age`` and ``ok``.

  The leader sits on a separate uncontended bus and is pulled directly by the
  policy loop rather than routed through the control channel, so it is sampled
  on a different clock from :attr:`control` -- ``age`` is how a consumer that
  cares can tell how far apart they are."""

  # -- introspection (spec §5) ----------------------------------------------

  def get(self, name: str, default=None):
    """The value for modality ``name``, or ``default`` if absent this step."""
    return self.modalities.get(name, default)

  def has(self, name: str) -> bool:
    """Whether modality ``name`` is present this step."""
    return name in self.modalities

  def present(self) -> frozenset[str]:
    """The set of modality names present in this observation."""
    return frozenset(self.modalities)

  # -- typed accessors over well-known modalities ---------------------------

  @property
  def image(self) -> np.ndarray | None:
    """The camera image modality, shape ``(H, W, 3)`` uint8, or ``None``."""
    return self.modalities.get(IMAGE)

  @property
  def object_xy(self) -> np.ndarray | None:
    """World-frame object position (estimator-native units), or ``None``."""
    return self.modalities.get(OBJECT_XY)

  @property
  def object_uv(self) -> np.ndarray | None:
    """Object pixel coordinates ``(u, v)`` in the camera image, or ``None``."""
    return self.modalities.get(OBJECT_UV)

  @property
  def has_leader(self) -> bool:
    """Whether a leader-modality reading is present this step."""
    return self.leader_qpos is not None

  @classmethod
  def build(
    cls,
    *,
    t: float,
    control: "ControlState",
    leader: "LeaderState | None",
    modalities: dict[str, np.ndarray],
  ) -> "RuntimeObservation":
    return cls(
      t=t,
      q_meas=control.q,
      leader_qpos=None if leader is None else leader.q,
      modalities=modalities,
      control=control,
      leader=leader,
    )

  # -- M6 adapter -----------------------------------------------------------

  def to_observation(self, obs_mode: tuple[str, ...]) -> TrainingObservation:
    """Build the training component-dict ``Observation`` for ``obs_mode``.

    Raises if any requested component is absent this step — the caller is
    expected to have validated the policy's obs-space against the composed
    modality set at startup (see :func:`check_required`), so a miss here is a
    per-step gap (e.g. the localizer found no mask) worth surfacing loudly.
    """
    missing = [c for c in obs_mode if c not in self.modalities]
    if missing:
      raise ModalityError(
        f"to_observation(): components {missing!r} absent this step "
        f"(present={sorted(self.modalities)!r})")
    return TrainingObservation.from_components(
      {c: self.modalities[c] for c in obs_mode})


# -- composition + fail-fast ---------------------------------------------------


def compose_modalities(
  sensors: "list[Sensor]",
  perception: "list[PerceptionModule]",
  *,
  leader_present: bool = False,
) -> frozenset[str]:
  """Return the active modality set and validate the pipeline order.

  Active set = ``{t, q_meas}`` (+ ``leader_qpos`` when a leader is enabled)
  ∪ every sensor's ``produces`` ∪ every perception module's ``emits``.

  Also validates the *configured* pipeline order: each module's ``consumes``
  must be a subset of everything produced before it (base ∪ all sensors ∪
  earlier modules). Unlike the offline stage machinery this does **not**
  reorder — the runtime pipeline is a deliberate, stateful authoring choice —
  it only raises :class:`ModalityError` on an unsatisfiable consume.
  """
  available: set[str] = set(_BASE_MODALITIES)
  if leader_present:
    available.add(LEADER_QPOS)
  for s in sensors:
    available.update(s.produces)
  for m in perception:
    missing = [c for c in m.consumes if c not in available]
    if missing:
      raise ModalityError(
        f"perception module {m.name!r} consumes {missing!r} which no sensor or "
        f"earlier module produces; available before it = {sorted(available)!r}")
    available.update(m.emits)
  return frozenset(available)


def required_modalities(rt) -> frozenset[str]:
  """The modality set the session declares it needs (``runtime.required_modalities``).

  At M6 the learned policy's declared obs-space unions into this; for M3 the
  scripted producers need nothing beyond the base set, so this is purely
  config-driven and lets the fail-fast be exercised today.
  """
  declared = rt.get("required_modalities", None)
  if declared is None:
    return frozenset()
  return frozenset(str(x) for x in declared)


def check_required(required: frozenset[str], available: frozenset[str]) -> None:
  """Fail fast (spec §9.1 step 5) if any required modality is unproduced."""
  missing = required - available
  if missing:
    raise ModalityError(
      f"required modality(s) {sorted(missing)!r} are not produced by any sensor "
      f"or perception module; available = {sorted(available)!r}")
