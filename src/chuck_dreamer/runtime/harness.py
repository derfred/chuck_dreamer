"""The runtime orchestrator: build from config, run both loops, shut down.

:class:`Runtime` is the M0 "process that boots from a YAML session config":
it constructs the backend, channel, policy, telemetry sink, and the two loops
from ``cfg.runtime``, then owns their lifecycle. A SIGINT handler
(installed on the main thread) sets a shutdown event; the main thread waits
on it (or drives the viewer when enabled) and runs the ordered shutdown.

The producer is a :class:`~chuck_dreamer.policy.Policy` (M0/M1 ships the
scripted :class:`~chuck_dreamer.runtime.sources.GoToPose` /
:class:`~chuck_dreamer.runtime.sources.SineSweep`; M6 swaps in Dreamer via
config). :class:`PolicyLoop` calls ``policy.act(obs)`` each step with a
:class:`RuntimeObservation` carrying elapsed time and the measured joints.

Shutdown ordering (project plan): stop the policy loop first (no new
setpoints), ask the control loop to brake and wait for the arm to stop, then
stop the control loop, backend, and finally the sink (which drains + flushes —
the "flushed logs" guarantee).
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..policy import Policy
from .backend import RobotBackend, ViewableBackend
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
from .setpoint_channel import SetpointChannel
from .sources import GoToPose, ManualPolicy, SineSweep
from .telemetry import TelemetryQueue

if TYPE_CHECKING:
  from .perception import PerceptionModule
  from .sensors import Sensor
  from .teleop import LeaderReader

logger = logging.getLogger(__name__)

_FAKE_BACKEND_TARGET = "chuck_dreamer.runtime.backend:FakeBackend"


class PolicyLoop:
  """Threaded policy producer: sense → perceive → ``policy.act(obs)`` → channel.

  Resets the policy and the perception pipeline once against the backend's home
  pose, then, at the policy rate: samples each sensor's latest value, runs the
  perception pipeline inline, composes a :class:`RuntimeObservation`, publishes
  ``policy.act(obs)``, and (if a Rerun sink is wired) logs the observation. This
  is the exact producer path Dreamer will use at M6.

  Perception runs inline here, on the policy thread: a slow ``process`` only
  delays the next ``channel.publish`` — the control loop keeps slewing the last
  published setpoint, so slow perception reduces the *policy* rate, not control
  correctness (spec §3.2/§4.1). The ``sleep_for <= 0`` branch lets the loop run
  as fast as it can when a tick overruns its period.
  """

  def __init__(
    self,
    policy: Policy,
    channel: SetpointChannel,
    backend: RobotBackend,
    *,
    rate_hz: float,
    leader: "LeaderReader | None" = None,
    sensors: "list[Sensor] | None" = None,
    perception: PerceptionPipeline | None = None,
    rerun_sink: RerunSink | None = None,
  ) -> None:
    if rate_hz <= 0:
      raise ValueError("rate_hz must be positive")
    self._policy     = policy
    self._channel    = channel
    self._backend    = backend
    self._leader     = leader
    self._sensors    = sensors or []
    self._perception = perception or PerceptionPipeline([])
    self._rerun_sink = rerun_sink
    self._period     = 1.0 / float(rate_hz)
    self._stop       = threading.Event()
    self._thread: threading.Thread | None = None

  def start(self) -> None:
    self._policy.reset(self._backend.home_qpos)
    self._perception.reset()
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, name="runtime-policy", daemon=True)
    self._thread.start()

  def stop(self, join_timeout: float = 5.0) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=join_timeout)
      self._thread = None

  def _run(self) -> None:
    t0 = time.monotonic()
    next_deadline = t0
    step = 0
    while not self._stop.is_set():
      t      = time.monotonic() - t0
      q_meas = self._backend.last_positions()
      # latest() is non-blocking by contract, so the policy loop never waits on
      # serial / the camera; when no leader is configured leader_qpos is None.
      leader_qpos = self._leader.latest() if self._leader is not None else None

      # Compose the modality dict: base + sensors + perception emits. The dict
      # is the RuntimeObservation's full `modalities`; `t`/`q_meas`/`leader_qpos`
      # also ride as named fields for the scripted policies.
      data: dict[str, Any] = {"t": t, "q_meas": q_meas}
      if leader_qpos is not None:
        data["leader_qpos"] = leader_qpos
      for s in self._sensors:
        snap = s.latest()
        if snap:
          data.update(snap)
      data = self._perception.run(data, t=t)

      obs = RuntimeObservation(
        t=t, q_meas=q_meas, leader_qpos=leader_qpos, modalities=data,
      )
      self._channel.publish(self._policy.act(obs))
      if self._rerun_sink is not None:
        self._rerun_sink.log_observation(obs, step=step, t=t)
      step += 1

      next_deadline += self._period
      sleep_for = next_deadline - time.monotonic()
      self._stop.wait(sleep_for if sleep_for > 0 else 0)


class Runtime:
  """Boots the SO-101 runtime from ``cfg`` and owns its thread lifecycle."""

  backend: RobotBackend

  def __init__(self, cfg: DictConfig) -> None:
    self.cfg = cfg
    rt_cfg   = cfg.runtime

    self.backend = self._build_backend(rt_cfg)
    n_joints     = self.backend.n_joints

    # 3. Channel, telemetry queue, Rerun sink, loops.
    self.channel   = SetpointChannel()
    self.telemetry = TelemetryQueue(maxsize=int(rt_cfg.logging.queue_maxsize))
    self.sink      = RerunSink(
      rt_cfg.logging.rerun.rrd_dir, self.telemetry, n_joints=n_joints,
      obs_queue_maxsize=int(rt_cfg.logging.rerun.get("obs_queue_maxsize", 256)),
    )

    self.control_loop = ControlLoop(build_control_config(rt_cfg, n_joints=n_joints), self.backend, self.channel, self.telemetry)
    self.policy       = self._build_policy(rt_cfg)
    self.leader       = self._build_leader(rt_cfg)  # None unless source.kind == "manual"

    # 4. Observation half (M3): sensors + perception pipeline, then compose the
    #    active modality set and fail fast (spec §9.1 step 4-5) before any thread
    #    starts if a required modality is unproduced.
    self.sensors     = self._build_sensors(rt_cfg)
    self.perception  = self._build_perception(rt_cfg)
    self.available   = compose_modalities(
      self.sensors, self.perception.modules, leader_present=self.leader is not None)
    check_required(required_modalities(rt_cfg), self.available)

    self.policy_loop = PolicyLoop(
      self.policy, self.channel, self.backend,
      rate_hz=float(rt_cfg.policy_rate_hz), leader=self.leader,
      sensors=self.sensors, perception=self.perception, rerun_sink=self.sink,
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
    target = str(rt.backend.target)
    params = _params_dict(rt.backend.get("params"))
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
    kind = str(rt.source.kind)
    if kind == "go_to_pose":
      g    = rt.source.go_to_pose
      home = self.backend.home_qpos
      goal = home if g.q_goal == "home" else _to_array(g.q_goal, len(home))
      return GoToPose(q_goal=goal, duration_s=float(g.duration_s))
    if kind == "sine_sweep":
      s = rt.source.sine_sweep
      return SineSweep(
        amplitude=_scalar_or_array(s.amplitude),
        freq_hz=_scalar_or_array(s.freq_hz),
        phase=_scalar_or_array(s.phase),
      )
    if kind == "manual":
      # Pass-through teleop. Reads the leader modality off the obs when a leader
      # is enabled (built independently in _build_leader); holds otherwise.
      return ManualPolicy()
    # Anything else is a registry import path to a custom Policy.
    return cast(Policy, construct(kind))

  def _build_leader(self, rt: DictConfig) -> "LeaderReader | None":
    """Construct the leader-reader when a leader is enabled, else None.

    The leader is a *sensor*, decoupled from the policy: it is read and exposed
    as the ``leader_qpos`` modality whenever ``runtime.leader.enabled`` is true,
    independent of ``source.kind`` (spec §3.8 — "the leader modality is also
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
    import inspect

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


def _scalar_or_array(value):
  """Pass a scalar through; turn a config list into an array (source params)."""
  if OmegaConf.is_config(value):
    return np.asarray(OmegaConf.to_container(value, resolve=True), dtype=np.float64)
  return value


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
