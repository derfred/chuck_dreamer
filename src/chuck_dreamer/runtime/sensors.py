"""Camera sensors — latest-value producer slots sampled at policy-step time.

A :class:`Sensor` is the observation-side analog of the M2 ``LeaderReader``
(``teleop.py``): a producer with a non-blocking ``latest()`` the policy loop
samples each step. Each sensor declares the modalities it ``produces`` so the
startup composition (``modalities.compose_modalities``) can fail fast.

``SimCameraSensor`` renders the MuJoCo scene through the backend's model/data —
the exact ``mujoco.Renderer`` recipe used by ``sim/pushing_env.py`` — and must
hold the backend's ``physics_lock`` while rendering (the renderer touches the
shared mjData stack that the physics thread writes via ``mj_step``).

``WebcamSensor`` reads a real camera on a background poll thread so ``latest()``
never blocks the policy loop; ``cv2`` is imported lazily in ``start`` so the
module stays import-safe on a machine without OpenCV.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .modalities import IMAGE

__all__ = ["Sensor", "SimCameraSensor", "WebcamSensor"]


def _parse_hw(render_size, *, default: tuple[int, int]) -> tuple[int, int]:
  """Coerce a ``"HxW"`` string (or ``(H, W)`` pair) to an ``(H, W)`` int tuple."""
  if render_size is None:
    return default
  if isinstance(render_size, str):
    h, w = (int(x) for x in render_size.lower().split("x"))
    return h, w
  h, w = (int(x) for x in render_size)
  return h, w


@runtime_checkable
class Sensor(Protocol):
  name: str                       # entity-path stem, e.g. "camera/front"
  produces: tuple[str, ...]       # modalities emitted, e.g. ("image",)

  def start(self) -> None: ...
  def stop(self) -> None: ...
  def latest(self) -> dict[str, np.ndarray] | None: ...   # None before first sample


class SimCameraSensor:
  """MuJoCo render of the backend's scene as the ``image`` modality.

  Renders synchronously inside ``latest()`` so it needs no thread of its own;
  ``start`` / ``stop`` manage the ``mujoco.Renderer`` and a private scratch
  ``MjData``.

  Crucially, the render does **not** hold the backend's physics lock for the
  slow GL call. The live ``mjData`` state is snapshotted into a private scratch
  buffer under the lock (a ~0.1 ms copy + ``mj_forward``), then the lock is
  released and ``render()`` runs lock-free against the scratch buffer. Holding
  the lock across the full ~5 ms render starved the control loop (which needs
  the same lock every tick) badly enough to trip the watchdog — exactly the
  "slow perception must not affect the control loop" invariant (spec §3.2/§4.1).
  The scratch buffer mirrors ``MujocoBackend._fk_ee``'s pattern.

  The ``mujoco.Renderer`` owns a GL context that is bound to the thread that
  creates it: constructing it on one thread and calling ``render()`` on another
  crashes (SIGTRAP on macOS). ``start`` runs on the main thread but ``latest``
  runs on the policy thread, so the renderer is built **lazily on first
  ``latest``** — on the policy thread that uses it — and torn down only from
  that same thread (``stop``, called from the main thread, just drops the
  reference; process teardown reclaims the context).
  """

  produces: tuple[str, ...] = (IMAGE,)

  def __init__(
    self,
    backend,
    *,
    camera: str = "main_camera",
    height: int = 240,
    width: int = 320,
    name: str = "camera/front",
  ) -> None:
    if not (hasattr(backend, "model") and hasattr(backend, "data")
            and hasattr(backend, "physics_lock")):
      raise TypeError(
        f"SimCameraSensor needs a MuJoCo-style backend (model/data/physics_lock); "
        f"{type(backend).__name__} does not")
    self.name                     = name
    self._backend                 = backend
    self._camera                  = camera
    self._height                  = int(height)
    self._width                   = int(width)
    self._renderer: Any           = None
    self._scratch: Any            = None
    self._started                 = False
    self._owner_ident: int | None = None  # thread that built the GL context

  @classmethod
  def from_config(cls, cfg, *, backend, **params) -> "SimCameraSensor":
    render_size   = params.pop("render_size", None)
    height, width = _parse_hw(render_size, default=(240, 320))
    return cls(backend, height=height, width=width, **params)

  def start(self) -> None:
    # The renderer + its GL context are built lazily in latest() on the policy
    # thread (the GL context is thread-bound). start() only arms the sensor.
    self._started = True

  def stop(self) -> None:
    # Called from the main thread; only close the GL context if we happen to be
    # on the thread that created it. Otherwise drop the reference and let process
    # teardown reclaim it (closing a GL context cross-thread crashes).
    if self._renderer is not None and threading.get_ident() == self._owner_ident:
      self._renderer.close()
    self._renderer = None
    self._scratch = None
    self._started = False

  def latest(self) -> dict[str, np.ndarray] | None:
    import mujoco  # type: ignore[import-untyped]

    if not self._started:
      return None
    if self._renderer is None:
      # Build the renderer + scratch on this (policy) thread so the GL context
      # is created and used on the same thread.
      self._renderer    = mujoco.Renderer(self._backend.model, self._height, self._width)
      self._scratch     = mujoco.MjData(self._backend.model)
      self._owner_ident = threading.get_ident()
    # Snapshot the live state into the scratch buffer under the lock (cheap),
    # then render lock-free so the control loop is never starved by the GL call.
    with self._backend.physics_lock():
      self._scratch.qpos[:] = self._backend.data.qpos
      self._scratch.qvel[:] = self._backend.data.qvel
      mujoco.mj_forward(self._backend.model, self._scratch)
    self._renderer.update_scene(self._scratch, camera=self._camera)
    img = np.asarray(self._renderer.render(), dtype=np.uint8)
    return {IMAGE: img}


class WebcamSensor:
  """Real camera via ``cv2.VideoCapture`` on a background poll thread."""

  produces: tuple[str, ...] = (IMAGE,)

  def __init__(
    self,
    *,
    device_index: int = 0,
    height: int = 480,
    width: int = 640,
    name: str = "camera/front",
    poll_rate_hz: float = 30.0,
  ) -> None:
    self.name                             = name
    self._device_index                    = int(device_index)
    self._height                          = int(height)
    self._width                           = int(width)
    self._period                          = 1.0 / float(poll_rate_hz)
    self._cap: Any                        = None  # cv2.VideoCapture (untyped import)
    self._lock                            = threading.Lock()
    self._latest: np.ndarray | None       = None
    self._stop                            = threading.Event()
    self._thread: threading.Thread | None = None

  @classmethod
  def from_config(cls, cfg, **params) -> "WebcamSensor":
    render_size = params.pop("render_size", None)
    if render_size is not None:
      params["height"], params["width"] = _parse_hw(render_size, default=(480, 640))
    return cls(**params)

  def start(self) -> None:
    import cv2  # type: ignore[import-not-found]

    self._cap = cv2.VideoCapture(self._device_index)
    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, name="runtime-webcam", daemon=True)
    self._thread.start()

  def stop(self, join_timeout: float = 5.0) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=join_timeout)
      self._thread = None
    if self._cap is not None:
      self._cap.release()
      self._cap = None

  def latest(self) -> dict[str, np.ndarray] | None:
    with self._lock:
      img = None if self._latest is None else self._latest.copy()
    return None if img is None else {IMAGE: img}

  def _run(self) -> None:
    import cv2  # type: ignore[import-not-found]

    next_deadline = time.monotonic()
    while not self._stop.is_set():
      ok, frame = self._cap.read()
      if ok and frame is not None:
        # cv2 yields BGR; the rest of the pipeline expects RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._lock:
          self._latest = np.asarray(rgb, dtype=np.uint8)
      next_deadline += self._period
      sleep_for = next_deadline - time.monotonic()
      self._stop.wait(sleep_for if sleep_for > 0 else 0)
