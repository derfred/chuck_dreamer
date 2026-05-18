"""Polymorphic observation wrapper.

The codebase supports three observation modes (``state``, ``image``,
``image_proprio``) which previously diverged in shape — flat ndarray
vs. dict-of-arrays — and forced ``isinstance(obs, dict)`` branches
through the buffer, evaluator, CEM policy, and the model. This module
provides one dataclass that carries whichever fields the mode populates
and exposes the image-side transformations (resize, normalize) as
methods.

The dataclass is intentionally generic over the leading axes *and* over
the array library: fields can hold ``np.ndarray``, ``mx.array``, or
``torch.Tensor``. Methods that need a specific library are explicit:
:meth:`resize_image` and :meth:`normalize_image` require numpy (used at
the loader boundary); :meth:`map_arrays` lifts everything through a
caller-supplied function (used by backend ``coerce``); generic ops
(:meth:`__getitem__`, :meth:`leading_shape`, :meth:`add_batch_axis`)
work on any array library because they only touch shapes / indexing.

A ``focus_mask`` field carries an optional per-frame scalar weight used
by the reconstruction loss to up-weight image regions of interest. It
only makes sense for image modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

import cv2  # type: ignore[import-not-found]
import numpy as np


ObsMode = Literal["state", "image", "image_proprio"]


def _as_array(x: Any, *, dtype=None) -> Any:
  """Pass-through for backend tensors (mx.array, torch.Tensor); np.asarray otherwise.

  The constructors are used at two boundaries: the loader (where inputs
  are numpy or list-of-numbers and need ``np.asarray`` to materialize)
  and the model (where inputs are already mx.array / torch.Tensor and
  must not be coerced via ``__array__``, which would lose the backend
  type). We sniff for the latter by checking ``hasattr(x, "shape")`` and
  not being a numpy array.
  """
  if isinstance(x, np.ndarray):
    return x if dtype is None else x.astype(dtype, copy=False)
  if hasattr(x, "shape") and hasattr(x, "ndim"):
    return x  # backend tensor — leave as-is
  return np.asarray(x, dtype=dtype)


def _resize_stack(arr: np.ndarray, size: int, interpolation: int) -> np.ndarray:
  """Resize a ``(leading..., H, W, C)`` numpy stack so H = W = ``size``.

  Flattens the leading axes, resizes each frame with the given cv2
  interpolation mode, then restores the leading shape. Used for both
  RGB images (bilinear/area) and focus masks (nearest-neighbor).
  cv2.resize drops a trailing C=1 axis on output, so we reshape back.
  """
  C    = arr.shape[-1]
  flat = arr.reshape((-1,) + arr.shape[-3:])
  out  = np.empty((flat.shape[0], size, size, C), dtype=arr.dtype)
  for i in range(flat.shape[0]):
    resized = cv2.resize(flat[i], (size, size), interpolation=interpolation)
    out[i] = resized.reshape(size, size, C)
  return out.reshape(arr.shape[:-3] + (size, size, C))


@dataclass(frozen=True, slots=True)
class Observation:
  """One mode-tagged observation (per-step, sequence, or batched).

  Exactly the fields the mode needs are non-None:
    * ``state``        — flat float32 vector, modes: ``state``
    * ``image``        — (..., H, W, 3) uint8 or float32, modes: ``image``, ``image_proprio``
    * ``proprio``      — (..., P) float32,                modes: ``image_proprio``
    * ``focus_mask``   — (...,) float32 (optional),       modes: image-having

  Fields hold whichever array type the caller put in (numpy at the
  loader/buffer boundary, mx.array / torch.Tensor after :meth:`map_arrays`
  coerces them to a backend). ``leading_shape`` returns the axes before
  the per-frame trailing axes — what callers used to extract by hand
  from ``obs.shape[:-1]`` or ``next(iter(obs.values())).shape[:K]``.
  """

  mode: ObsMode
  state: Any = None
  image: Any = None
  proprio: Any = None
  focus_mask: Any = None
  # Number of trailing axes that describe one frame's content (1 for
  # state/proprio, 3 for an image). The first ``ndim - trailing_ndim``
  # axes are leading (steps / batch / time). Populated at construction.
  _trailing_ndim: int = field(default=0, repr=False)

  # ------------------------------------------------------------------
  # Constructors
  # ------------------------------------------------------------------

  @classmethod
  def state_vector(cls, vec) -> "Observation":
    return cls(mode="state", state=_as_array(vec, dtype=np.float32), _trailing_ndim=1)

  @classmethod
  def image_only(cls, image, focus_mask=None) -> "Observation":
    return cls(
      mode="image",
      image=_as_array(image),
      focus_mask=None if focus_mask is None else _as_array(focus_mask, dtype=np.float32),
      _trailing_ndim=3,
    )

  @classmethod
  def image_proprio(cls, image, proprio, focus_mask=None) -> "Observation":
    img = _as_array(image)
    pro = _as_array(proprio, dtype=np.float32)
    # Leading axes must match across leaves — proprio has 1 trailing axis,
    # image has 3, so the prefixes are img.shape[:-3] and pro.shape[:-1].
    img_lead = tuple(img.shape[:-3])
    pro_lead = tuple(pro.shape[:-1])
    if img_lead != pro_lead:
      raise ValueError(
        f"image and proprio leading shapes mismatch: image={img_lead} proprio={pro_lead}"
      )
    return cls(
      mode="image_proprio",
      image=img,
      proprio=pro,
      focus_mask=None if focus_mask is None else _as_array(focus_mask, dtype=np.float32),
      _trailing_ndim=3,
    )

  @classmethod
  def from_buffer_value(
    cls,
    value,
    mode: ObsMode,
    *,
    focus_mask=None,
  ) -> "Observation":
    """Wrap the legacy buffer-side obs value (ndarray or dict) back into an Observation.

    Used when ingesting episodes that arrive in the pre-wrapper schema
    (``episode["obs"]`` is an ndarray for state/image, a dict for
    image_proprio). Backend tensors (mx.array / torch.Tensor) pass
    through unchanged so the same wrap call works at the buffer boundary
    and at the model boundary (after backend coerce). Idempotent on
    already-wrapped inputs: if ``value`` is an :class:`Observation`,
    it's returned unchanged (a passed ``focus_mask`` overrides the
    wrapper's own only when non-None).
    """
    if isinstance(value, cls):
      if focus_mask is not None:
        return replace(value, focus_mask=focus_mask)
      return value
    if mode == "state":
      return cls.state_vector(value)
    if mode == "image":
      return cls.image_only(value, focus_mask=focus_mask)
    if mode == "image_proprio":
      if not (isinstance(value, dict) and "image" in value and "proprio" in value):
        raise ValueError(
          f"image_proprio expects a dict with 'image' and 'proprio' keys; got {type(value).__name__}"
        )
      return cls.image_proprio(value["image"], value["proprio"], focus_mask=focus_mask)
    raise ValueError(f"unknown obs mode: {mode!r}")

  @classmethod
  def stack(cls, observations: list["Observation"], axis: int = 0) -> "Observation":
    """Stack a list of like-shaped Observations along a new leading axis.

    Every observation must share ``mode``. The new axis is inserted at
    ``axis`` (default 0); per-field arrays are stacked independently.
    Replaces the dict-or-array fan-out used by the buffer's sample loop.
    """
    if not observations:
      raise ValueError("stack() requires at least one Observation")
    mode = observations[0].mode
    for o in observations:
      if o.mode != mode:
        raise ValueError(f"stack() requires uniform mode; got {mode!r} and {o.mode!r}")

    def stack_field(name: str):
      vals = [getattr(o, name) for o in observations]
      if vals[0] is None:
        if any(v is not None for v in vals):
          raise ValueError(f"stack(): field {name!r} mixed None and non-None across batch")
        return None
      return np.stack(vals, axis=axis)

    return cls(
      mode=mode,
      state=stack_field("state"),
      image=stack_field("image"),
      proprio=stack_field("proprio"),
      focus_mask=stack_field("focus_mask"),
      _trailing_ndim=observations[0]._trailing_ndim,
    )

  # ------------------------------------------------------------------
  # Image transformations (no-op when no image is present)
  # ------------------------------------------------------------------

  def resize_image(self, size: int) -> "Observation":
    """Resize the (leading..., H, W, 3) image stack so H = W = ``size``.

    Leaves a float32 image alone if it already matches ``size`` and
    only iterates per-frame when a resize is needed. Pass-through for
    ``state`` mode. When a ``focus_mask`` is present (shape
    ``(leading..., H, W)``) it's resized too, with nearest-neighbor
    interpolation so a bool mask stays bool-equivalent (0/1 only).
    """
    if self.image is None:
      return self
    img = self.image
    if img.shape[-3] == size and img.shape[-2] == size:
      return self
    out = _resize_stack(img, size, cv2.INTER_AREA)
    fm = self.focus_mask
    if fm is not None:
      fm = _resize_stack(fm[..., None], size, cv2.INTER_NEAREST).squeeze(-1)
    return replace(self, image=out, focus_mask=fm)

  def normalize_image(self) -> "Observation":
    """Map uint8 image → float32 in ``[-0.5, 0.5]``. Float images pass through."""
    if self.image is None:
      return self
    if self.image.dtype == np.uint8:
      img = self.image.astype(np.float32) / 255.0 - 0.5
      return replace(self, image=img)
    return replace(self, image=self.image.astype(np.float32, copy=False))

  # ------------------------------------------------------------------
  # Sequence / batch ops — the dict-vs-ndarray branching this replaces
  # ------------------------------------------------------------------

  @property
  def leading_shape(self) -> tuple[int, ...]:
    """Shape of the axes before the per-frame trailing axes."""
    arr = self._any_array()
    return tuple(arr.shape[: arr.ndim - self._trailing_ndim])

  def _any_array(self) -> np.ndarray:
    for v in (self.state, self.image, self.proprio):
      if v is not None:
        return v
    raise RuntimeError(f"Observation(mode={self.mode!r}) has no populated array")

  def __getitem__(self, idx) -> "Observation":
    """Index/slice along the leading axes uniformly across all fields."""
    def take(arr):
      return None if arr is None else arr[idx]
    return replace(
      self,
      state=take(self.state),
      image=take(self.image),
      proprio=take(self.proprio),
      focus_mask=take(self.focus_mask),
    )

  def map_arrays(self, fn: Callable[[Any], Any]) -> "Observation":
    """Apply ``fn`` to every populated field. Use this to coerce to a backend.

    Field metadata (mode, trailing_ndim) carries through unchanged. Used
    by the backends' ``coerce()`` to lift a numpy-side ``Observation``
    into one carrying mx.array / torch.Tensor leaves.
    """
    return replace(
      self,
      state=None if self.state is None else fn(self.state),
      image=None if self.image is None else fn(self.image),
      proprio=None if self.proprio is None else fn(self.proprio),
      focus_mask=None if self.focus_mask is None else fn(self.focus_mask),
    )

  def add_batch_axis(self) -> "Observation":
    """Prepend a length-1 axis to every populated field.

    Implemented via indexing (``arr[None]``) so it works for numpy and
    backend tensor types alike — used by single-step inference paths
    (CEM, evaluator's per-episode batching).
    """
    return self[None]

  # ------------------------------------------------------------------
  # Adapters to the legacy episode/buffer schema
  # ------------------------------------------------------------------

  def to_buffer_value(self):
    """Return the value stored under ``episode["obs"]`` today.

    ``state``/``image`` modes return a single ndarray; ``image_proprio``
    returns a ``{"image", "proprio"}`` dict. Focus masks ride in a
    separate ``Episode`` field — not part of ``obs``.
    """
    if self.mode == "state":
      assert self.state is not None
      return self.state
    if self.mode == "image":
      assert self.image is not None
      return self.image
    if self.mode == "image_proprio":
      assert self.image is not None and self.proprio is not None
      return {"image": self.image, "proprio": self.proprio}
    raise ValueError(f"unknown obs mode: {self.mode!r}")

  def model_shape(self):
    """Return the trailing-shape descriptor a model encoder expects.

    Tuple for ``state``/``image``; dict for ``image_proprio``. Matches
    what :attr:`PushingEnv.model_obs_shape` produces.
    """
    if self.mode == "state":
      assert self.state is not None
      return tuple(self.state.shape[-1:])
    if self.mode == "image":
      assert self.image is not None
      return tuple(self.image.shape[-3:])
    if self.mode == "image_proprio":
      assert self.image is not None and self.proprio is not None
      return {
        "image":   tuple(self.image.shape[-3:]),
        "proprio": tuple(self.proprio.shape[-1:]),
      }
    raise ValueError(f"unknown obs mode: {self.mode!r}")
