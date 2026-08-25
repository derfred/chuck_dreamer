"""Telemetry → Rerun sink: the M3 replacement for ``CsvSink``.

Both runtime threads feed one ``.rrd`` per episode on a shared timeline:

* the **control thread** keeps emitting flat :class:`TelemetryRecord`s to the
  existing :class:`TelemetryQueue` (q_meas / q_cmd / target / mode / clamp /
  events) — the record shape is unchanged, only the sink that drains it is new
  (telemetry.py's own promise: "M3 upgrades the *sink* to Rerun without
  changing the record shape");
* the **policy thread** calls :meth:`RerunSink.log_observation`, a non-blocking
  enqueue (drop-on-full, like ``TelemetryQueue.emit``) of the per-step camera
  frame, perception outputs, and composed observation onto the sink's own
  internal queue — so the policy loop never blocks on Rerun encoding.

A single logger thread drains both queues and logs to the recording, stamping
every entry with ``step`` (sequence) and ``time`` (duration) so control scalars
and camera frames line up in the viewer (spec §4.3). Entity-path conventions
and the ``RecordingStream`` / ``flush`` recipe are reused from
``common/episode_writer.py``'s ``RerunEpisodeWriter``; frames ride as per-step
``rr.Image`` rather than an encoded video blob — simpler for a streaming sink,
at the cost of a larger ``.rrd`` (a fine M3 trade).
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .modalities import EE, OBJECT_UV, OBJECT_XY
from .telemetry import TelemetryQueue, TelemetryRecord

if TYPE_CHECKING:
  from .modalities import RuntimeObservation

__all__ = ["RerunSink"]


class _ObsSnapshot:
  """A per-step policy-thread record: the composed observation + its step/time."""

  __slots__ = ("step", "t", "image", "q_meas", "leader_qpos", "vectors")

  def __init__(self, step: int, t: float, *, image, q_meas, leader_qpos, vectors):
    self.step = step
    self.t = t
    self.image = image
    self.q_meas = q_meas
    self.leader_qpos = leader_qpos
    self.vectors = vectors


# Modalities logged as rr.Scalars when present on the observation.
_VECTOR_MODALITIES = (EE, OBJECT_XY, OBJECT_UV)


class RerunSink:
  """Logger thread draining control telemetry + policy observations to one .rrd."""

  def __init__(
    self,
    rrd_dir: str | Path,
    telemetry: TelemetryQueue,
    *,
    n_joints: int,
    application_id: str = "chuck_dreamer",
    obs_queue_maxsize: int = 256,
  ) -> None:
    self._rrd_dir = Path(rrd_dir)
    self._telemetry = telemetry
    self._n_joints = n_joints
    self._application_id = application_id
    self._obs_q: "queue.Queue[_ObsSnapshot]" = queue.Queue(maxsize=obs_queue_maxsize)
    self._dropped = 0
    self._dropped_lock = threading.Lock()
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None
    self._rec: Any = None
    self._path: Path | None = None

  @property
  def path(self) -> Path | None:
    """The ``.rrd`` path of the current recording (set by :meth:`start`)."""
    return self._path

  @property
  def dropped(self) -> int:
    with self._dropped_lock:
      return self._dropped

  def start(self, recording_id: str) -> None:
    import rerun as rr

    self._rrd_dir.mkdir(parents=True, exist_ok=True)
    self._path = self._rrd_dir / f"{recording_id}.rrd"
    self._rec = rr.RecordingStream(
      application_id=self._application_id, recording_id=recording_id)
    self._rec.save(str(self._path))
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, name="runtime-rerun", daemon=True)
    self._thread.start()

  def stop(self, join_timeout: float = 5.0) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=join_timeout)
      self._thread = None
    # Drain anything still queued so shutdown never loses tail records, then
    # force chunks to disk (the "flushed logs on shutdown" guarantee).
    if self._rec is not None:
      self._drain_telemetry()
      self._drain_observations()
      self._rec.flush(timeout_sec=30.0)
      self._rec = None

  def log_observation(self, obs: "RuntimeObservation", *, step: int, t: float) -> None:
    """Enqueue one observation for the logger thread (policy-thread, non-blocking)."""
    image = obs.image
    snap = _ObsSnapshot(
      step, t,
      image=None if image is None else np.asarray(image, dtype=np.uint8),
      q_meas=np.asarray(obs.q_meas, dtype=np.float32),
      leader_qpos=None if obs.leader_qpos is None
      else np.asarray(obs.leader_qpos, dtype=np.float32),
      vectors={
        name: np.asarray(obs.modalities[name], dtype=np.float32)
        for name in _VECTOR_MODALITIES if name in obs.modalities
      },
    )
    try:
      self._obs_q.put_nowait(snap)
    except queue.Full:
      with self._dropped_lock:
        self._dropped += 1

  # -- logger thread --------------------------------------------------------

  def _run(self) -> None:
    while not self._stop.is_set():
      # Block briefly on the higher-volume control queue, then opportunistically
      # drain observations — both go to the same recording on a shared timeline.
      rec = self._telemetry.get(timeout=0.1)
      if rec is not None:
        self._log_telemetry(rec)
      self._drain_observations()

  def _drain_telemetry(self) -> None:
    for rec in self._telemetry.drain():
      self._log_telemetry(rec)

  def _drain_observations(self) -> None:
    while True:
      try:
        snap = self._obs_q.get_nowait()
      except queue.Empty:
        break
      self._log_observation_snapshot(snap)

  def _log_telemetry(self, rec: TelemetryRecord) -> None:
    import rerun as rr

    self._rec.set_time("step", sequence=int(rec.tick))
    self._rec.set_time("time", duration=float(rec.t_mono))
    if rec.event:
      self._rec.log("events", rr.TextLog(f"{rec.event}: {rec.detail}"))
      return
    self._rec.log("control/mode", rr.TextLog(rec.mode))
    self._rec.log("control/clamped", rr.Scalars(int(rec.clamped)))
    self._rec.log("control/stale", rr.Scalars(int(rec.stale)))
    self._rec.log("control/dt", rr.Scalars(float(rec.dt)))
    self._rec.log("control/seq", rr.Scalars(int(rec.seq)))
    self._rec.log("control/backend_s", rr.Scalars(float(rec.backend_s)))
    for name, arr in (("q_meas", rec.q_meas), ("q_cmd", rec.q_cmd), ("target", rec.target)):
      if arr is not None:
        self._rec.log(f"control/{name}", rr.Scalars(np.asarray(arr, dtype=float).tolist()))

  def _log_observation_snapshot(self, snap: _ObsSnapshot) -> None:
    import rerun as rr

    self._rec.set_time("step", sequence=int(snap.step))
    self._rec.set_time("time", duration=float(snap.t))
    if snap.image is not None:
      self._rec.log("camera/image", rr.Image(snap.image))
    self._rec.log("obs/joint_qpos", rr.Scalars(snap.q_meas.tolist()))
    if snap.leader_qpos is not None:
      self._rec.log("obs/leader_qpos", rr.Scalars(snap.leader_qpos.tolist()))
    for name, arr in snap.vectors.items():
      self._rec.log(f"obs/{name}", rr.Scalars(arr.tolist()))
