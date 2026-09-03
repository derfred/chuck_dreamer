"""The runtime orchestrator: build from config, run both loops, shut down.

:class:`Runtime` is the M0 "process that boots from a YAML session config":
it constructs the backend, channel, policy, telemetry sink, and the two loops
from ``cfg.runtime``, then owns their lifecycle. A SIGINT handler
(installed on the main thread) sets a shutdown event; the main thread waits
on it (or drives the viewer when enabled) and runs the ordered shutdown.

The producer is a :class:`~chuck_dreamer.policy.Policy`, built from
``runtime.policy`` by the same ``{target, params}`` registry path as the
backend (M0/M1 ships the scripted
:class:`~chuck_dreamer.runtime.sources.GoToPose` /
:class:`~chuck_dreamer.runtime.sources.SineSweep`; M6 swaps in Dreamer by
pointing that target elsewhere). :class:`PolicyLoop` calls ``policy.act(obs)``
each step with a :class:`RuntimeObservation` carrying elapsed time and the control loop's own
measurement of the arm.

Shutdown ordering (project plan): stop the policy loop first (no new
setpoints), ask the control loop to brake and wait for the arm to stop, then
stop the control loop, backend, and finally the sink (which drains + flushes —
the "flushed logs" guarantee).
"""

from __future__ import annotations

import inspect
import logging
import signal
import threading
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..policy import Policy
from .backend import RobotBackend, ViewableBackend
from .control_channel import ControlChannel
from .control_loop import ControlLoop, build_control_config
from .modalities import (
  RuntimeObservation,
  check_required,
  compose_modalities,
  required_modalities,
)
from .perception import PerceptionPipeline
from .registry import construct, import_symbol
from .rerun_sink import RerunSink
from .telemetry import TelemetryQueue, TelemetryRecord
from .threads import PacedLoop

if TYPE_CHECKING:
  from .perception import PerceptionModule
  from .sensors import Sensor
  from .teleop import LeaderReader

logger = logging.getLogger(__name__)

_FAKE_BACKEND_TARGET = "chuck_dreamer.runtime.backend:FakeBackend"


class PolicyLoop(PacedLoop):
  def __init__(
    self,
    policy: Policy,
    channel: ControlChannel,
    backend: RobotBackend,
    *,
    rate_hz: float,
    leader: "LeaderReader | None" = None,
    sensors: "list[Sensor] | None" = None,
    perception: PerceptionPipeline | None = None,
    rerun_sink: RerunSink | None = None,
    telemetry: TelemetryQueue | None = None,
  ) -> None:
    super().__init__("runtime-policy", rate_hz)
    self._policy     = policy
    self._channel    = channel
    self._backend    = backend
    self._leader     = leader
    self._sensors    = sensors or []
    self._perception = perception or PerceptionPipeline([])
    self._rerun_sink = rerun_sink
    self._telemetry  = telemetry
    # Only for the overrun event, which fires outside the loop body.
    self._step       = 0

  def _on_start(self) -> None:
    self._policy.reset(self._backend.home_qpos)
    self._perception.reset()

  def _on_overrun(self, late_by_s: float) -> None:
    if self._telemetry is not None:
      self._telemetry.emit_event(
        "policy_overrun", tick=self._step, detail=f"late_by={late_by_s:.4f}s")

  def _run(self) -> None:
    t0        = time.monotonic()
    last_time = t0
    step      = 0

    for _ in self.paced():
      t       = time.monotonic() - t0
      control = self._channel.state()

      if control is None:
        continue
      metrics = TelemetryRecord(t_wall=time.time(), t_mono=time.monotonic(), tick=step)

      with metrics.timed("step"):
        leader = self._leader.latest() if self._leader is not None else None

        data: dict[str, Any] = {"t": t, "q_meas": control.q}
        if leader is not None:
          data["leader_qpos"] = leader.q
        with metrics.timed("sensors"):
          for s in self._sensors:
            snap = s.latest()
            if snap:
              data.update(snap)
        with metrics.timed("perception"):
          data = self._perception.run(data, t=t)

        obs = RuntimeObservation.build(t=t, control=control, leader=leader, modalities=data)

        with metrics.timed("act"):
          action = self._policy.act(obs)
        self._channel.publish(action)

      if self._rerun_sink is not None:
        self._rerun_sink.log_observation(obs, step=step, t=t)

      if self._telemetry is not None:
        now = time.monotonic()
        metrics.update_policy(action=action, dt=now - last_time)
        last_time = now
        self._telemetry.emit(metrics)
      step       = step + 1
      self._step = step


class Runtime:
  """Boots the SO-101 runtime from ``cfg`` and owns its thread lifecycle."""

  backend: RobotBackend

  def __init__(self, cfg: DictConfig) -> None:
    self.cfg = cfg
    rt_cfg   = cfg.runtime

    self.backend = self._build_backend(rt_cfg)
    n_joints     = self.backend.n_joints

    # 3. Channel, telemetry queue, Rerun sink, loops.
    self.channel   = ControlChannel()
    self.telemetry = TelemetryQueue(maxsize=int(rt_cfg.logging.queue_maxsize))
    self.sink      = RerunSink(
      rt_cfg.logging.rerun.rrd_dir, self.telemetry, n_joints=n_joints,
      obs_queue_maxsize=int(rt_cfg.logging.rerun.get("obs_queue_maxsize", 256)),
    )

    self.control_loop = ControlLoop(build_control_config(rt_cfg, n_joints=n_joints), self.backend, self.channel, self.telemetry)
    self.policy       = self._build_policy(rt_cfg)
    self.leader       = self._build_leader(rt_cfg)

    # 4. Observation half (M3): sensors + perception pipeline, then compose the
    #    active modality set and fail fast (spec §9.1 step 4-5) before any thread
    #    starts if a required modality is unproduced.
    self.sensors     = self._build_sensors(rt_cfg)
    self.perception  = self._build_perception(rt_cfg)
    self.available   = compose_modalities(self.sensors, self.perception.modules, leader_present=self.leader is not None)
    check_required(required_modalities(rt_cfg), self.available)

    self.policy_loop = PolicyLoop(
      self.policy, self.channel, self.backend,
      rate_hz=float(rt_cfg.policy_rate_hz), leader=self.leader,
      sensors=self.sensors, perception=self.perception, rerun_sink=self.sink,
      telemetry=self.telemetry,
    )

    self._duration_s     = None if rt_cfg.duration_s is None else float(rt_cfg.duration_s)
    self._viewer_enabled = bool(rt_cfg.viewer.enabled)
    self._shutdown       = threading.Event()
    # One .rrd per episode (spec §3.11). Episode lifecycle is M7; until then a
    # single recording spans the run, named by start time so successive runs
    # don't clobber each other.
    self._recording_id = f"episode-{time.strftime('%Y%m%d-%H%M%S')}"

  # -- construction helpers -------------------------------------------------

  def _build_backend(self, rt: DictConfig) -> RobotBackend:
    """Construct the configured backend; it is the source of truth for the
    joint count, the joint envelope and the home pose.

    ``FakeBackend`` is the one backend with no model of its own, so the
    runtime hands it the configured envelope (and home pose, when set) at
    construction; everything else either reads config itself via
    ``from_config`` or carries its own geometry.
    """
    # `runtime.backend` is always a mapping {target: str, params: {...}}.
    target               = str(rt.backend.target)
    params               = _params_dict(rt.backend.get("params"))
    spec: dict[str, Any] = {"target": target, "params": params}

    if target == _FAKE_BACKEND_TARGET:
      limits = rt.control_loop.joint_limits
      if limits.get("lower") is None or limits.get("upper") is None:
        raise ValueError(
          "runtime.control_loop.joint_limits.lower/upper must be set for "
          "FakeBackend (it has no model to read joint limits from)")
      lower = _to_seq(limits.lower)
      upper = _to_seq(limits.upper)
      n     = len(lower)
      if len(upper) != n:
        raise ValueError(
          f"joint_limits.lower/upper length mismatch: {n} vs {len(upper)}")
      home = limits.get("home_qpos")
      return cast(RobotBackend, construct(
        spec, n_joints=n, lower=np.asarray(lower, dtype=np.float64),
        upper=np.asarray(upper, dtype=np.float64),
        q_init=None if home is None else _to_array(home, n)))

    # A config-aware backend (e.g. MujocoBackend, which samples a scene from
    # cfg) exposes a `from_config` classmethod; prefer it and feed it `params`
    # as kwargs. Otherwise construct the class directly from the spec.
    factory     = import_symbol(target)
    from_config = getattr(factory, "from_config", None)
    if callable(from_config):
      return cast(RobotBackend, from_config(self.cfg, **params))
    return cast(RobotBackend, construct(spec))

  def _build_policy(self, rt: DictConfig) -> Policy:
    spec    = _registry_spec(rt.get("policy"))
    factory = import_symbol(spec["target"])
    params  = _accepted_params(factory, spec["params"], spec["target"])
    return cast(Policy, construct({"target": spec["target"], "params": params}))

  def _build_leader(self, rt: DictConfig) -> "LeaderReader | None":
    """Construct the leader-reader when a leader is enabled, else None.

    The leader is a *sensor*, decoupled from the policy: it is read and exposed
    as the ``leader_qpos`` modality whenever ``runtime.leader.enabled`` is true,
    independent of the configured policy (spec §3.8 — "the leader modality is also
    available to any other policy that declares it"). ``ManualPolicy`` consumes
    it but does not require it, and any other policy may read it too.

    Reuses the construct/from_config convention from :meth:`_build_backend`. The
    follower jaw bounds (the last joint) are injected from the backend's joint
    limits so the gripper percentage->radian mapping always agrees with the
    bounds the control loop limits against; they are not read from config.
    """
    leader_cfg = rt.get("leader")
    if leader_cfg is None or not bool(leader_cfg.get("enabled")):
      return None

    target               = str(leader_cfg.target)
    params               = _params_dict(leader_cfg.get("params"))
    lower, upper         = self.backend.joint_limits()
    jaw_lo               = float(lower[-1])
    jaw_hi               = float(upper[-1])
    spec: dict[str, Any] = {"target": target, "params": params}

    # One construction contract (LeaderReader docstring): the resolved jaw
    # bounds are injected on *every* path, so a config-aware reader can never
    # silently miss the safety envelope. from_config is preferred (it also
    # absorbs params leaked from a previous target after a config merge);
    # readers without one get a direct construct with the same injection.
    factory     = import_symbol(target)
    from_config = getattr(factory, "from_config", None)
    if callable(from_config):
      return cast("LeaderReader",
                  from_config(self.cfg, jaw_lower=jaw_lo, jaw_upper=jaw_hi, **params))
    return cast("LeaderReader", construct(spec, jaw_lower=jaw_lo, jaw_upper=jaw_hi))

  def _build_sensors(self, rt: DictConfig) -> "list[Sensor]":
    """Construct the configured camera sensors (spec §3.1).

    Reuses the construct/from_config convention from :meth:`_build_backend`.
    Sensors that need the backend (``SimCameraSensor`` renders its model/data)
    get it injected when their ``from_config`` accepts a ``backend`` keyword.
    """
    return [
      cast("Sensor", self._construct_component(entry))
      for entry in (rt.get("sensors") or [])
    ]

  def _build_perception(self, rt: DictConfig) -> PerceptionPipeline:
    """Construct the ordered perception pipeline (spec §3.2).

    Same construction convention as sensors; modules run inline at the start of
    each policy step. Empty by default (the M3 scaffold) — enable modules
    (``EePoseModule`` / ``ObjectLocalizerModule``) explicitly in config.
    """
    modules = [
      cast("PerceptionModule", self._construct_component(entry))
      for entry in (rt.get("perception") or [])
    ]
    return PerceptionPipeline(modules)

  def _construct_component(self, entry) -> Any:
    """Build one sensor / perception module from a ``{target, params}`` entry.

    Prefers a ``from_config`` classmethod (fed ``self.cfg`` + params), injecting
    ``backend=self.backend`` only when the factory accepts it; falls back to
    direct ``construct`` of the spec, again injecting ``backend`` only if the
    constructor takes it. This mirrors :meth:`_build_backend` / :meth:`_build_leader`.
    """
    target               = str(entry["target"])
    params               = _params_dict(entry.get("params"))
    spec: dict[str, Any] = {"target": target, "params": params}
    factory              = import_symbol(target)

    from_config = getattr(factory, "from_config", None)
    if callable(from_config):
      kwargs = dict(params)
      if "backend" in inspect.signature(from_config).parameters:
        kwargs["backend"] = self.backend
      return from_config(self.cfg, **kwargs)

    extra: dict[str, Any] = {}
    if "backend" in inspect.signature(factory).parameters:
      extra["backend"] = self.backend
    return construct(spec, **extra)

  # -- lifecycle ------------------------------------------------------------

  def start(self) -> None:
    self.backend.start()
    for s in self.sensors:   # after backend: SimCameraSensor renders its model/data
      s.start()
    self.sink.start(self._recording_id)
    self.control_loop.start()
    if self.leader is not None:
      self.leader.start()
    self.policy_loop.start()

  def request_shutdown(self) -> None:
    self._shutdown.set()

  def run(self) -> None:
    """Boot, run until SIGINT / duration / viewer close, then shut down."""
    installed = _install_sigint(self.request_shutdown)
    self.start()
    try:
      if self._viewer_enabled:
        self._run_viewer_loop()
      else:
        self._shutdown.wait(self._duration_s)  # None blocks until SIGINT
    finally:
      if installed is not None:
        signal.signal(signal.SIGINT, installed)
      self.shutdown()

  def _run_viewer_loop(self) -> None:
    """Drive the backend's viewer on the main thread (macOS requirement)."""
    backend = self.backend
    if not isinstance(backend, ViewableBackend):
      logger.warning("viewer enabled but backend is not viewable; waiting headless")
      self._shutdown.wait(self._duration_s)
      return
    # The viewer reads the backend's mjData on this (main) thread while the
    # physics thread writes it; serialize sync() against mj_step under the
    # backend's lock. draw_overlays acquires that lock internally (briefly),
    # so it must run OUTSIDE the `with` to avoid re-entrant deadlock.
    deadline = None if self._duration_s is None else time.monotonic() + self._duration_s
    with backend.open_viewer() as viewer:
      while viewer.is_running() and not self._shutdown.is_set():
        backend.draw_overlays(viewer)
        with backend.physics_lock():
          viewer.sync()
        if deadline is not None and time.monotonic() >= deadline:
          break
        self._shutdown.wait(1.0 / 60.0)

  def shutdown(self) -> None:
    # stop new setpoints, brake, flush, tear down.
    self.policy_loop.stop()
    for s in self.sensors:   # no more observations after the policy loop stops
      s.stop()
    if self.leader is not None:
      self.leader.stop()
    if not self.control_loop.stop_for_shutdown():
      logger.warning("control loop did not come to rest in time; tearing down anyway")
    self.control_loop.stop()
    self.backend.stop()
    self.sink.stop()  # drains both queues + flushes the .rrd (one per episode)
    if self.telemetry.dropped:
      logger.warning("dropped %d telemetry records (queue full)", self.telemetry.dropped)
    if self.sink.dropped:
      logger.warning("dropped %d observation records (rerun queue full)", self.sink.dropped)


# -- small config coercions ----------------------------------------------------


def _registry_spec(node) -> dict[str, Any]:
  """Coerce a ``{target, params}`` config node to a plain registry spec.

  Accepts the shorthand where the node is just the import-path string, so
  ``policy: "pkg.mod:Thing"`` and the mapping form both construct.
  """
  if isinstance(node, str):
    return {"target": node, "params": {}}
  if node is None or node.get("target") is None:
    raise ValueError(
      "runtime.policy needs a 'target' import path, e.g. "
      "'chuck_dreamer.runtime.sources:SineSweep'")
  return {"target": str(node.target), "params": _params_dict(node.get("params"))}


def _accepted_params(factory, params: dict[str, Any], target: str) -> dict[str, Any]:
  """``params`` filtered to the keyword arguments ``factory`` actually accepts.

  A factory taking ``**kwargs`` accepts everything, so nothing is dropped there.
  """
  try:
    sig = inspect.signature(factory)
  except (TypeError, ValueError):
    return params  # not introspectable; let the constructor speak for itself
  if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
    return params
  accepted = {
    name for name, p in sig.parameters.items()
    if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
  }
  ignored = sorted(set(params) - accepted)
  if ignored:
    logger.warning("%s ignoring policy params it does not use: %s", target, ignored)
  return {k: v for k, v in params.items() if k in accepted}


def _params_dict(params_cfg) -> dict[str, Any]:
  """Coerce a registry-spec ``params`` config node to a plain str-keyed dict."""
  raw = (
    OmegaConf.to_container(params_cfg, resolve=True)
    if OmegaConf.is_config(params_cfg) else (params_cfg or {})
  )
  return {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {}


def _to_seq(value) -> list:
  """Coerce a config sequence node to a plain list (length is the joint count)."""
  raw = (OmegaConf.to_container(value, resolve=True)
         if OmegaConf.is_config(value) else value)
  if not isinstance(raw, (list, tuple)):
    raise ValueError(f"expected a sequence, got {type(raw).__name__}")
  return list(raw)


def _to_array(value, n: int) -> np.ndarray:
  """Coerce a scalar or sequence config value to a length-``n`` float array."""
  arr = np.asarray(OmegaConf.to_container(value, resolve=True)
                   if OmegaConf.is_config(value) else value, dtype=np.float64)
  if arr.ndim == 0:
    arr = np.full(n, float(arr))
  if arr.shape != (n,):
    raise ValueError(f"expected length-{n} value, got shape {arr.shape}")
  return arr


def _install_sigint(handler):
  """Install a SIGINT handler that calls ``handler``; return the previous one.

  Returns ``None`` (and installs nothing) when not on the main thread —
  ``signal.signal`` only works there, so tests that build a Runtime off the
  main thread silently skip it and drive shutdown directly.
  """
  if threading.current_thread() is not threading.main_thread():
    return None
  previous = signal.getsignal(signal.SIGINT)
  signal.signal(signal.SIGINT, lambda *_: handler())
  return previous
