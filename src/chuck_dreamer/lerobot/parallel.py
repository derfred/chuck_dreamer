"""Parallel import orchestrator: GPU producer(s) feeding a CPU worker pool.

The serial importer (``importer.import_dataset``) runs every episode's stages
back-to-back in one process: decode → ``ee_pos`` → SAM2 ``segment:object`` (GPU)
→ ``object_pose`` (a CPU-bound per-frame Nelder-Mead fit) → write. The GPU sits
idle during the long pose fit and the pose fit can't use the other cores
(SciPy / cv2 hold the GIL), so wall-clock is roughly the *sum* of the two.

This module pipelines those two halves across episodes instead:

  * **Producer lane** (one thread per GPU): decode the next episode and run its
    ``"producer"`` stages (``ee_pos`` + SAM2 segmentation) on a pinned device,
    then hand ``(episode, metadata, masks)`` to the worker pool. Producers pull
    from a shared decode queue, so a slow episode on one GPU never stalls
    another. SAM2 / torch / CUDA are imported *only* here, in the main process.

  * **Worker lane** (a pool of ``spawn``-ed processes): run each episode's
    ``"worker"`` stages (``object_pose``) from the handed-in masks and write the
    result. Workers are pure NumPy / SciPy / cv2 + the calibration/mesh loaders
    — they never import torch or touch CUDA, so ``spawn`` is safe and cheap.

Steady-state throughput ≈ ``max(decode, G·segment, pose / N)`` rather than their
sum: with the pose fit dominating and ``N`` CPU workers, the GPU and decode time
hide almost entirely. ``G`` producer threads pinned to ``cuda:0..G-1`` raise the
segmentation ceiling ~G×.

The masks cross the process boundary as an explicit argument (not via the shared
``RunContext``), which is also what dissolves the per-episode mask-cache race:
each producer owns its own ``ctx``, each worker owns its own ``ctx``, and nothing
mutable is shared across episodes in flight.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from chuck_dreamer.common.episode import Episode
from chuck_dreamer.common.episode_writer import EpisodeWriter

from .importer import drop_video_field
from .trackset import EpisodeScope, TrackSet, run_episode

if TYPE_CHECKING:
  from chuck_dreamer.lerobot.episode_spec import EpisodeSlice
  from .pipeline import Run

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Queue payloads
# ---------------------------------------------------------------------------
@dataclass
class _Task:
  """One episode handed from a producer to a worker: the assembled
  :class:`Episode` (producer stages already applied) + its metadata. The SAM2
  masks the worker's ``object_pose`` stage consumes ride *inside* the episode as
  a non-persisted ``object_masks`` field, so there is no separate mask channel —
  the whole episode (scratch fields included) crosses the boundary as one
  pickled payload."""
  episode_index: int
  episode: Episode
  metadata: dict[str, Any]


@dataclass
class _Result:
  """A worker's outcome for one episode. Exactly one of ``out_path`` /
  ``error`` is set; ``error`` carries the stringified exception so it can
  cross the process boundary and be re-raised in the main process."""
  episode_index: int
  out_path: Path | None = None
  error: str | None = None


# Sentinels. ``_STOP`` ends a queue consumer; identity comparison only.
_STOP = "__STOP__"


# ---------------------------------------------------------------------------
# CPU worker process: object_pose + write
# ---------------------------------------------------------------------------
def _worker_main(
  config: Any,
  source_repo: str,
  output_dir: str,
  fmt: str,
  worker_node_names: tuple[str, ...],
  task_q: "mp.Queue[Any]",
  result_q: "mp.Queue[_Result]",
  *,
  drop_video: bool = False,
) -> None:
  """Worker-process entrypoint: build a CPU-only context + the run's worker-lane
  nodes once, then drain ``task_q``, running those nodes from the handed-in
  masks and writing each episode, until a ``_STOP`` sentinel arrives.

  ``worker_node_names`` is the dependency-ordered list of ``"worker"``-lane
  node names the parent resolved from the enabled flags (today just
  ``object_pose``; resolving by name keeps the worker correct if more are
  added). The producer's masks ride *inside* ``task.episode`` as non-persisted
  ``object_masks`` / ``object_masks.valid`` fields, so the worker's TrackSet
  reconstructs the validity-carrying track exactly as in the serial path — no
  ctx re-seeding, no separate mask channel.

  Imports stay torch-free: a worker only needs the calibration/mesh loaders and
  the writer, so ``spawn`` doesn't pay torch / CUDA init and never inherits a
  forked CUDA context. Per-episode failures are reported as ``_Result(error=)``
  rather than crashing the pool, so one bad episode doesn't sink the run."""
  # Built lazily inside the worker so the parent never imports the node
  # module's transitive deps before ``spawn``.
  from .context import RunContext
  from .nodes import NODE_TYPES

  ctx = RunContext(config=config, source_repo=source_repo)
  nodes = [NODE_TYPES[name](ctx) for name in worker_node_names]
  writer = EpisodeWriter(output_dir, format=fmt)

  while True:
    item = task_q.get()
    if item is _STOP:
      return
    task: _Task = item
    try:
      ctx.episode_index = task.episode_index
      ts                = TrackSet(EpisodeScope(source_repo, task.episode_index), task.episode, task.metadata)
      run_episode(nodes, ts)
      if drop_video:
        drop_video_field(task.episode)
      name_suffix = task.metadata.pop("_name_suffix")
      out_path    = writer.write_episode(task.episode, metadata=task.metadata, name_suffix=name_suffix)
      result_q.put(_Result(task.episode_index, out_path=out_path))
    except Exception as exc:  # noqa: BLE001 — ship the failure back, don't crash the pool
      logger.exception("worker: episode %d failed", task.episode_index)
      result_q.put(_Result(task.episode_index, error=f"{type(exc).__name__}: {exc}"))


# ---------------------------------------------------------------------------
# GPU producer thread: decode + ee_pos + segment, then enqueue
# ---------------------------------------------------------------------------
def _producer_main(
  run: Run,
  device: str | None,
  slice_q: "queue.Queue[Any]",
  task_q: "mp.Queue[Any]",
  result_q: "mp.Queue[_Result]",
  output_dir: str,
  fmt: str,
  tags: tuple[str, ...],
  name_prefix: str | None,
  has_worker_stages: bool,
) -> None:
  """Producer-thread body: pull episode slices off ``slice_q``, decode + run
  ``"producer"`` stages on ``device``, and hand each episode to the worker pool.

  Runs in the *main* process (CUDA-safe: torch/SAM2 live here), one thread per
  GPU. ``device`` (e.g. ``"cuda:1"``) is pushed onto the run's config so this
  producer's SAM2 stage targets its own GPU; ``None`` leaves the configured
  device untouched (single-GPU / CPU fallback).

  When the run has no worker-lane stages (``--no-object-pose``), the producer
  writes the episode itself — there is nothing to fan out — so this collapses
  to a per-GPU serial importer with no queue handoff."""
  from .importer import assemble_episode  # local: keeps importer the single assembler

  # Pin this producer to its device by overriding the run's segmentation device.
  # Each producer thread has its own Run (and thus its own ctx + registry), so
  # this never races another producer's device setting.
  if device is not None:
    run.set_object_localization_device(device)

  writer = EpisodeWriter(output_dir, format=fmt) if not has_worker_stages else None

  while True:
    try:
      item = slice_q.get_nowait()
    except queue.Empty:
      return
    if item is _STOP:
      return
    sl: EpisodeSlice = item
    try:
      assembled = assemble_episode(
        run, sl, tags=tags, name_prefix=name_prefix)
      if assembled is None:
        continue
      episode, metadata, name_suffix = assembled

      # Run the producer-lane nodes (the EE chain + SAM2 segmentation) on
      # this producer's ctx/device, through the harness. The SAM2 masks land
      # on the episode itself as a non-persisted field, so they travel to the
      # worker with the episode — nothing to lift off the ctx.
      ts = TrackSet(EpisodeScope(run.dataset_id, sl.episode_index),
                    episode, metadata)
      run_episode(run.lane_pipeline(sl.episode_index, "producer"), ts)

      if has_worker_stages:
        # The worker writes; pass the name suffix through metadata (it pops it
        # before the writer sees it). Lives on metadata so the _Task payload
        # stays a flat episode/metadata pair.
        metadata["_name_suffix"] = name_suffix
        task_q.put(_Task(sl.episode_index, episode, metadata))
      else:
        # Nothing for the worker pool — write here.
        assert writer is not None
        if run.drop_video:
          drop_video_field(episode)
        out_path = writer.write_episode(
          episode, metadata=metadata, name_suffix=name_suffix)
        result_q.put(_Result(sl.episode_index, out_path=out_path))
    except Exception as exc:  # noqa: BLE001 — surface the failure, keep going
      logger.exception("producer[%s]: episode %d failed", device, sl.episode_index)
      result_q.put(_Result(sl.episode_index,
                           error=f"{type(exc).__name__}: {exc}"))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def import_dataset_parallel(
  run: Run,
  output_dir: str,
  *,
  format: str = "hdf5",
  tags: tuple[str, ...] = (),
  name_prefix: str | None = None,
  jobs: int,
  devices: list[str] | None = None,
) -> Iterator[tuple[int, Path]]:
  """Parallel counterpart of :func:`importer.import_dataset`, same generator
  signature: yields ``(episode_index, output_path)`` as episodes finish.

  ``jobs`` is the number of CPU worker processes running ``object_pose`` +
  write. ``devices`` is the list of producer devices (e.g. ``["cuda:0",
  "cuda:1"]``); one producer thread is started per device. ``None`` ⇒ a single
  producer on the configured device.

  Episodes complete out of order (pose-fit time varies); output files are named
  by ``episode_index``, so order doesn't matter. A failure in any producer or
  worker is re-raised here after the pool drains, so the caller's
  ``FileNotFoundError → ClickException`` handling still fires."""
  slices = run.slices
  if not slices:
    raise RuntimeError(f"{run.dataset_id}: no episodes to import")

  if devices is None or not devices:
    devices = [None]  # type: ignore[list-item]  # single producer, configured device

  # The run's worker-lane nodes, dependency-ordered (today just object_pose).
  # If empty (e.g. --no-object-pose), producers write episodes directly and we
  # start no worker pool. Resolved off the first episode's pipeline — the node
  # *set* is the same for every episode (it's flag-driven, not episode-driven).
  worker_node_names = tuple(
    n.name for n in run.pipeline(slices[0].episode_index)
    if getattr(n, "lane", "worker") == "worker")
  has_worker_stages = bool(worker_node_names)

  # As in the serial path: with --no-video and no frame-consuming node, the
  # decode is skipped outright instead of decoded-then-discarded.
  if not run.decodes_video:
    logger.info("--no-video and no frame-consuming stage enabled: "
                "skipping video decode entirely")

  # ``spawn`` (not fork): the workers must not inherit any CUDA context, and a
  # producer thread may have initialised CUDA in this process by the time a
  # worker starts.
  mp_ctx = mp.get_context("spawn")
  task_q: "mp.Queue[Any]" = mp_ctx.Queue(maxsize=max(2 * jobs, 4))
  result_q: "mp.Queue[_Result]" = mp_ctx.Queue()

  # Feed producers from an in-process slice queue (cheap; slices are small).
  slice_q: "queue.Queue[Any]" = queue.Queue()
  for sl in slices:
    slice_q.put(sl)
  for _ in devices:
    slice_q.put(_STOP)

  workers: list[mp.process.BaseProcess] = []
  if has_worker_stages:
    for _ in range(jobs):
      p: mp.process.BaseProcess = mp_ctx.Process(
        target=_worker_main,
        args=(run.ctx.config, run.dataset_id, output_dir, format,
              worker_node_names, task_q, result_q),
        kwargs={"drop_video": run.drop_video},
        daemon=True,
      )
      p.start()
      workers.append(p)

  # One producer thread per device. Each gets its *own* Run so its ctx/device
  # don't collide with another producer's.
  producers: list[threading.Thread] = []
  for device in devices:
    prod_run = _clone_run_for_producer(run)
    t = threading.Thread(
      target=_producer_main,
      args=(prod_run, device, slice_q, task_q, result_q, output_dir,
            format, tags, name_prefix, has_worker_stages),
      daemon=True,
    )
    t.start()
    producers.append(t)

  # Drain results until every episode is accounted for. Producers run in this
  # process (join them to know when production is done); workers are separate
  # processes (sentinel + join).
  n_expected        = len(slices)
  n_done            = 0
  errors: list[str] = []

  # A background thread joins producers then sends worker stop sentinels, so
  # the main thread can keep yielding results as they arrive. ``stopped`` flips
  # once producers are joined and every worker has been told to stop — after
  # that, no new results can be produced beyond what's already in flight.
  stopped = threading.Event()

  def _shutdown_when_drained() -> None:
    for t in producers:
      t.join()
    for _ in workers:
      task_q.put(_STOP)
    stopped.set()

  shutdown = threading.Thread(target=_shutdown_when_drained, daemon=True)
  shutdown.start()

  while n_done < n_expected:
    try:
      res = result_q.get(timeout=1.0)
    except queue.Empty:
      # Dead-pool guard: only after production has provably stopped (producers
      # joined + workers told to stop) and every worker process has exited and
      # the result queue is drained can we conclude no more results will arrive.
      # Without this we'd hang forever on a hard crash mid-episode.
      if stopped.is_set() and \
         not any(p.is_alive() for p in workers) and result_q.empty():
        break
      continue
    n_done += 1
    if res.error is not None:
      errors.append(f"episode {res.episode_index}: {res.error}")
      continue
    assert res.out_path is not None
    yield res.episode_index, res.out_path

  shutdown.join(timeout=30.0)
  for p in workers:
    p.join(timeout=30.0)
    if p.is_alive():
      p.terminate()

  # Both producer- and worker-side failures arrive as _Result(error=) and are
  # collected in `errors`. Re-raise after the pool drains so partial output is
  # still written and the caller's error handling fires once.
  if errors:
    raise RuntimeError(
      f"{len(errors)} episode(s) failed during parallel import; first: "
      f"{errors[0]}")
  if n_done < n_expected:
    raise RuntimeError(
      f"parallel import ended with {n_done}/{n_expected} episodes accounted "
      f"for — a producer or worker process exited unexpectedly")


def _clone_run_for_producer(run: Run) -> Run:
  """A fresh :class:`Run` for one producer thread: same spec / params, its own
  *copied* config, and its own :class:`RunContext` + stage registry.

  Producers run concurrently and each mutates ``ctx.episode_index`` per episode
  *and* its config's segmentation device, so they must not share a context or a
  config object. (Inter-stage mask state lives on the :class:`Episode` now, not
  the ctx, so that is no longer a reason — but the episode-index and device
  isolation still are.) The config is deep-copied per producer so a device
  override on one never races another. Re-resolving the slices is cheap (it
  reads the cached ``meta/`` sidecars)."""
  from .pipeline import Run

  config = run.ctx.config
  if config is not None:
    from omegaconf import OmegaConf
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))

  clone = Run(config, run.spec, dict(run.params))
  return clone
