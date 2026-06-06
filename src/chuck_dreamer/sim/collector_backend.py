"""Collector backends — sync (in-process) and async (worker pool).

The trainer talks to a :class:`CollectorBackend` instead of an
:class:`EpisodeCollector` directly. Two implementations:

  * :class:`SyncCollectorBackend` runs episodes inline in the trainer
    process. Matches today's behavior — no extra processes, no IPC.
    Default. Good for local dev and CI.

  * :class:`AsyncCollectorBackend` spawns ``num_workers`` long-lived
    worker processes, each running its own ``PushingEnv`` + policy.
    Workers collect continuously and stream episodes to the trainer
    via an :class:`mp.Queue`. Training and collection overlap; the
    trainer's "collect phase" reduces to draining the queue. Workers
    hold stale copies of the model weights; the trainer broadcasts
    updates every ``weight_broadcast_every_n_iters`` iterations.

Pick the backend via :func:`build_collector_backend(config, ...)` which
reads ``config.training.collector.mode``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import platform
import queue
from dataclasses import dataclass
from typing import Any, Protocol

from omegaconf import OmegaConf

from ..common.episode import Episode
from .episode_collector import EpisodeCollector
from .scene_config import SceneConfig

logger = logging.getLogger(__name__)


@dataclass
class CollectResult:
  """One collected episode + metadata, returned by every backend.

  ``episode`` is None when the rollout produced zero steps (e.g. an env
  reset that crashed immediately). ``scene`` is included so the trainer
  can call ``maybe_log_collect_episode`` without re-deriving it.
  """
  episode:   Episode | None
  outcome:   str
  scene:     SceneConfig | None
  worker_id: int = 0


class CollectorBackend(Protocol):
  """Interface the trainer uses to drive collection.

  Sync implementations execute ``drain`` inline; async ones return
  whatever workers have produced since the last call (blocking up to
  ``target`` results). Both shapes share this surface so the trainer
  doesn't branch on mode.
  """

  def drain(self, target: int) -> list[CollectResult]: ...
  def broadcast_weights(self, blob: bytes) -> None: ...
  def shutdown(self) -> None: ...


# ---------------------------------------------------------------------------
# Sync backend — one env, in-process, same as the pre-async behavior.
# ---------------------------------------------------------------------------


class SyncCollectorBackend:
  """Run episodes inline in the trainer process.

  ``broadcast_weights`` is a no-op: the policy already references the
  trainer's live model, so nothing needs to be copied.
  """

  def __init__(self, collector: EpisodeCollector, config) -> None:
    self.collector = collector
    self.config    = config

  def drain(self, target: int) -> list[CollectResult]:
    results: list[CollectResult] = []
    for _ in range(target):
      scene = self.collector.reset()
      if self.config.env.max_steps is not None:
        scene.max_steps = int(self.config.env.max_steps)
      episode, outcome = self.collector.run()
      results.append(CollectResult(
        episode=episode, outcome=outcome, scene=scene, worker_id=0,
      ))
    return results

  def broadcast_weights(self, blob: bytes) -> None:
    # Policy holds a live reference — no copy needed.
    del blob

  def shutdown(self) -> None:
    pass


# ---------------------------------------------------------------------------
# Async backend — worker pool, mp.Queue IPC.
# ---------------------------------------------------------------------------


def _set_mujoco_gl() -> None:
  """Match main.py's MUJOCO_GL=egl rule for headless Linux workers."""
  if (
    platform.system() == "Linux"
    and "MUJOCO_GL" not in os.environ
    and not os.environ.get("DISPLAY")
  ):
    os.environ["MUJOCO_GL"] = "egl"


def _worker_loop(
  worker_id:  int,
  config_yaml: str,
  episode_q:  "mp.Queue[CollectResult]",
  weights_q:  "mp.Queue[bytes]",
  stop_event,
) -> None:
  """Entry point for each collector worker process.

  ``config_yaml`` is the resolved config serialized as YAML — passing
  the live ``DictConfig`` through ``spawn`` is unreliable when custom
  resolvers are registered (see :mod:`chuck_dreamer.config`), so we
  re-parse on the worker side. Resolvers used by the trainer
  (``derive_image_size``) are registered at module import time and run
  again here.
  """
  _set_mujoco_gl()

  # Re-register resolvers; ``replace=True`` makes this idempotent in
  # repeat spawns (e.g. tests).
  from .. import config as _cfg_module  # noqa: F401  (side-effect: registers resolvers)
  config = OmegaConf.create(config_yaml)

  # Seed each worker distinctly so they don't generate identical scenes.
  worker_seed = int(config.seed) + 1000 * worker_id + 1
  config.seed = worker_seed

  # Force the worker's model copy onto CPU. The GPU belongs to the
  # trainer; if N workers each grab a CUDA context they fight the
  # trainer for memory and exhaust the device. Workers do one forward
  # per env step at small batch — CPU is fine here. MLX stays on MLX
  # since there's no separate device concept on Apple Silicon.
  if config.hardware.device != "mlx":
    config.hardware.device = "cpu"

  # Import here so torch/mlx initialize once per worker, not once per
  # module load on the main process.
  from .pushing_env import PushingEnv
  from .episode_collector import EpisodeCollector
  from .scripted_policy import ScriptedPolicy

  env = PushingEnv(config)

  policy: Any
  if config.training.policy == "scripted":
    policy = ScriptedPolicy(auto_advance_from_ready=True)
    model  = None
  elif config.training.policy == "dreamer":
    from ..dreamer import build_model, DreamerPolicy
    obs_shape = env.model_obs_shape
    action_dim = int(env.action_space.shape[0])
    model = build_model(config, obs_shape=obs_shape, action_dim=action_dim)
    policy = DreamerPolicy(model, act_mode=env.act_mode)
  else:
    raise ValueError(f"Unknown policy {config.training.policy!r}")

  collector = EpisodeCollector(env, policy)
  logger.info("collector worker %d started (seed=%d)", worker_id, worker_seed)

  while not stop_event.is_set():
    # Drain weights_q so we always apply the freshest blob and don't
    # lag behind by N broadcasts if the trainer outpaces us.
    latest_blob: bytes | None = None
    try:
      while True:
        latest_blob = weights_q.get_nowait()
    except queue.Empty:
      pass
    if latest_blob is not None and model is not None:
      try:
        model.load_state_bytes(latest_blob)
      except Exception as exc:
        logger.warning("worker %d failed to load weights: %s", worker_id, exc)

    try:
      scene = collector.reset()
      if config.env.max_steps is not None:
        scene.max_steps = int(config.env.max_steps)
      episode, outcome = collector.run()
    except Exception as exc:
      logger.warning("worker %d episode crashed during setup: %s", worker_id, exc)
      episode, outcome, scene = None, "crashed", None

    try:
      episode_q.put(CollectResult(
        episode=episode, outcome=outcome, scene=scene, worker_id=worker_id,
      ), timeout=1.0)
    except queue.Full:
      # Trainer is slow / queue capped — drop, log, continue. Skipping
      # episodes under backpressure is preferable to deadlocking.
      logger.debug("worker %d dropped episode (queue full)", worker_id)

  logger.info("collector worker %d exiting", worker_id)


class AsyncCollectorBackend:
  """Worker-pool collector backend.

  Workers loop continuously, pushing :class:`CollectResult` onto
  ``episode_q``. ``drain(target)`` returns whatever is already there,
  then blocks until at least ``target`` results arrive. ``broadcast_weights``
  fan-outs the blob to all workers via ``weights_q``.
  """

  def __init__(
    self,
    config,
    num_workers: int,
    max_pending: int = 64,
  ) -> None:
    if num_workers < 1:
      raise ValueError(f"num_workers must be >= 1; got {num_workers}")
    self.num_workers = num_workers

    ctx = mp.get_context("spawn")
    self.episode_q  = ctx.Queue(maxsize=max_pending)
    self.weights_q  = ctx.Queue(maxsize=2 * num_workers)
    self.stop_event = ctx.Event()

    # Serialize the config once; workers re-parse via OmegaConf.create.
    config_yaml = OmegaConf.to_yaml(config, resolve=True)

    self.workers: list[mp.process.BaseProcess] = []
    for i in range(num_workers):
      p = ctx.Process(
        target=_worker_loop,
        args=(i, config_yaml, self.episode_q, self.weights_q, self.stop_event),
        daemon=True,
        name=f"collector-{i}",
      )
      p.start()
      self.workers.append(p)
    logger.info("AsyncCollectorBackend started %d workers", num_workers)

  def drain(self, target: int) -> list[CollectResult]:
    """Return everything currently queued, then block for the rest.

    Caller passes the desired count for one iteration. We first take
    whatever's pending without blocking (so steady-state never waits
    when workers are ahead), then block for the remainder.
    """
    results: list[CollectResult] = []
    try:
      while len(results) < target:
        results.append(self.episode_q.get_nowait())
    except queue.Empty:
      pass

    while len(results) < target:
      # Periodic timeout so a dead worker doesn't hang the trainer
      # indefinitely — surface as a warning and keep waiting.
      try:
        results.append(self.episode_q.get(timeout=30.0))
      except queue.Empty:
        alive = sum(1 for p in self.workers if p.is_alive())
        logger.warning(
          "drain() waited 30s for %d more results; %d/%d workers alive",
          target - len(results), alive, len(self.workers),
        )
    return results

  def broadcast_weights(self, blob: bytes) -> None:
    """Push the latest weight blob to every worker.

    Naive fan-out: one queue put per worker. Workers drain the queue
    and only apply the freshest blob, so missed updates are harmless.
    Sized to ``2*num_workers`` so a slow worker can lag by one
    broadcast without dropping the next.
    """
    for _ in range(self.num_workers):
      try:
        self.weights_q.put_nowait(blob)
      except queue.Full:
        # Worker hasn't drained the previous one — that's fine, it
        # will pick up this blob (or a later one) on its next cycle.
        pass

  def shutdown(self, timeout: float = 5.0) -> None:
    self.stop_event.set()
    # Workers may be blocked on episode_q.put(); they use a 1s timeout
    # on put so they'll check stop_event soon. Drain in a loop until
    # everyone has exited or the timeout elapses.
    import time
    deadline = time.time() + timeout
    while time.time() < deadline and any(p.is_alive() for p in self.workers):
      try:
        while True:
          self.episode_q.get_nowait()
      except queue.Empty:
        pass
      time.sleep(0.05)

    for p in self.workers:
      p.join(timeout=0.5)
      if p.is_alive():
        logger.warning("worker %s did not exit cleanly; terminating", p.name)
        p.terminate()
        p.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_collector_backend(config, collector: EpisodeCollector) -> CollectorBackend:
  """Pick a backend based on ``config.training.collector.mode``.

  The ``collector`` argument is only used by the sync backend; async
  ignores it and constructs its own per-worker. Passing it in keeps
  the call site uniform.
  """
  mode = str(config.training.collector.mode).lower()
  if mode == "sync":
    return SyncCollectorBackend(collector, config)
  if mode == "async":
    return AsyncCollectorBackend(
      config,
      num_workers=int(config.training.collector.num_workers),
      max_pending=int(config.training.collector.max_pending),
    )
  raise ValueError(f"Unknown collector mode {mode!r}; expected 'sync' or 'async'")
