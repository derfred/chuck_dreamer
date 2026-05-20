"""Polymorphic multi-modal observation wrapper.

``obs_mode`` is a list of component names (a non-empty subset of
``image``, ``state``, ``proprio``, ``ee``). Each component is encoded
by its own branch (CNN for ``image``, MLP for the others); embeddings
are concatenated and fused into a single ``embed_size`` vector. The
decoder is symmetric — one reconstruction head per component, summed
into the recon loss.

Component vocabulary:
  * ``image``   — raw RGB frame, processed by the CNN branch.
  * ``state``   — ``object_xy ‖ joint_qpos`` (the privileged ground-truth
                  scene state). End-effector pose deliberately lives in
                  ``ee`` so a config can ablate them independently.
  * ``proprio`` — ``joint_qpos`` only (what a proprioceptive sensor sees
                  on the real arm).
  * ``ee``      — ``ee_pos ‖ ee_quat`` (7-D end-effector pose).

The :class:`Observation` dataclass carries a uniform
``components: dict[str, ndarray]`` mapping. A single-mode config like
``obs_mode: ["image"]`` produces a 1-key dict; multi-mode configs add
more keys. ``focus_mask`` is a per-frame scalar image weight used by
the reconstruction loss to up-weight image regions of interest — only
populated when ``"image"`` is among the components.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

import cv2  # type: ignore[import-not-found]
import numpy as np


# Component names. ``image`` is the only spatial component; the rest are
# 1-D vectors. Keep this list in sync with the encoder/decoder branches
# in the dreamer backends.
ComponentName = str
IMAGE = "image"
STATE = "state"
PROPRIO = "proprio"
EE = "ee"
VALID_COMPONENTS: frozenset[str] = frozenset({IMAGE, STATE, PROPRIO, EE})

# Convenience: number of trailing axes that describe one frame of each
# component. ``image`` is (H, W, C); others are flat vectors.
_TRAILING_NDIM: dict[str, int] = {IMAGE: 3, STATE: 1, PROPRIO: 1, EE: 1}


# An ``obs_mode`` value as it travels through the codebase: a tuple of
# component names. The config-load layer normalizes string and list
# inputs into this canonical form.
ObsMode = tuple[ComponentName, ...]


def normalize_obs_mode(value: Any) -> ObsMode:
  """Coerce a config-shaped ``obs_mode`` into a canonical tuple.

  Accepts a string (legacy single-mode form), a list/tuple of strings,
  or an OmegaConf ``ListConfig``. Rejects empty inputs and unknown
  component names so a typo trips at config load rather than at the
  encoder. Duplicates are rejected too — two heads for the same
  component would silently double-count its loss.
  """
  if isinstance(value, str):
    items: list[str] = [value]
  else:
    try:
      items = [str(v) for v in value]
    except TypeError as exc:
      raise ValueError(f"obs_mode must be a string or sequence of strings; got {value!r}") from exc
  if not items:
    raise ValueError("obs_mode must contain at least one component")
  unknown = [c for c in items if c not in VALID_COMPONENTS]
  if unknown:
    raise ValueError(
      f"unknown obs_mode component(s) {unknown!r}; expected subset of {sorted(VALID_COMPONENTS)!r}"
    )
  if len(set(items)) != len(items):
    raise ValueError(f"obs_mode has duplicate components: {items!r}")
  return tuple(items)


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
  """One observation across one or more components.

  ``components`` is a dict mapping component name → array. Arrays may be
  numpy at the loader/buffer boundary or backend tensors
  (mx.array / torch.Tensor) after :meth:`map_arrays` coerces them.

  ``focus_mask`` (per-frame scalar weight, shape ``(..., H, W)``) is
  only populated when ``"image"`` is among the components — it weights
  image-pixel reconstruction error.

  The leading axes (per-step / per-sequence / per-batch) are shared
  across all components and the focus mask; their trailing per-frame
  shapes differ by component (image: ``(H, W, 3)``; vectors: ``(D,)``).
  """

  components: dict[str, Any]
  focus_mask: Any = None
  # Stored once at construction so generic ops don't have to look it up
  # on every call. Each entry is the number of trailing axes for that
  # component (1 for vectors, 3 for images).
  _trailing_ndims: dict[str, int] = field(default_factory=dict, repr=False)

  # ------------------------------------------------------------------
  # Constructors
  # ------------------------------------------------------------------

  @classmethod
  def from_components(
    cls,
    components: dict[str, Any],
    *,
    focus_mask: Any = None,
  ) -> "Observation":
    """Build an Observation from a dict of populated component arrays.

    Validates that every key is a known component name and that the
    leading axes match across components. Vector components are cast to
    float32 (matches buffer storage); the image component preserves its
    incoming dtype so uint8 frames survive until the model normalizes
    them.
    """
    if not components:
      raise ValueError("from_components(): components dict is empty")
    out: dict[str, Any] = {}
    leads: dict[str, tuple[int, ...]] = {}
    trailing: dict[str, int] = {}
    for name, arr in components.items():
      if name not in VALID_COMPONENTS:
        raise ValueError(f"unknown component {name!r}; expected {sorted(VALID_COMPONENTS)!r}")
      tn = _TRAILING_NDIM[name]
      a  = _as_array(arr) if name == IMAGE else _as_array(arr, dtype=np.float32)
      lead = tuple(a.shape[: a.ndim - tn])
      out[name]      = a
      leads[name]    = lead
      trailing[name] = tn
    # All components must share leading axes — that's what makes one
    # Observation cover the same step/sequence/batch.
    ref_name, ref_lead = next(iter(leads.items()))
    for n, l in leads.items():
      if l != ref_lead:
        raise ValueError(
          f"leading shapes mismatch across components: {ref_name}={ref_lead}, {n}={l}"
        )
    fm = None if focus_mask is None else _as_array(focus_mask, dtype=np.float32)
    if fm is not None and IMAGE not in out:
      raise ValueError("focus_mask passed but 'image' is not among the components")
    return cls(components=out, focus_mask=fm, _trailing_ndims=trailing)

  @classmethod
  def from_buffer_value(
    cls,
    value,
    obs_mode: Any,
    *,
    focus_mask=None,
  ) -> "Observation":
    """Wrap a buffer-side obs value back into an Observation.

    Buffer values are always dicts keyed by component name (even for
    single-component configs — uniform schema simplifies downstream
    code). Idempotent on already-wrapped inputs: an :class:`Observation`
    passes through; a passed ``focus_mask`` overrides only when non-None.
    """
    if isinstance(value, cls):
      if focus_mask is not None:
        return replace(value, focus_mask=_as_array(focus_mask, dtype=np.float32))
      return value
    mode = normalize_obs_mode(obs_mode)
    if not isinstance(value, dict):
      raise ValueError(
        f"buffer value must be a dict keyed by component name; got {type(value).__name__}"
      )
    missing = [c for c in mode if c not in value]
    if missing:
      raise ValueError(f"buffer value missing components {missing!r} (mode={mode!r})")
    # Restrict to the configured mode — extra keys (legacy "proprio"
    # under a state-only mode, for example) are ignored.
    selected = {c: value[c] for c in mode}
    return cls.from_components(selected, focus_mask=focus_mask)

  @classmethod
  def stack(cls, observations: list["Observation"], axis: int = 0) -> "Observation":
    """Stack like-shaped Observations along a new leading axis.

    Every observation must share the same component keys. The new axis
    is inserted at ``axis`` (default 0); each component is stacked
    independently; ``focus_mask`` follows the same rule.
    """
    if not observations:
      raise ValueError("stack() requires at least one Observation")
    ref_keys = set(observations[0].components.keys())
    for o in observations:
      if set(o.components.keys()) != ref_keys:
        raise ValueError(
          f"stack() requires uniform components; got {ref_keys!r} and {set(o.components.keys())!r}"
        )
    stacked: dict[str, Any] = {}
    for name in observations[0].components:
      stacked[name] = np.stack([o.components[name] for o in observations], axis=axis)
    fms = [o.focus_mask for o in observations]
    if fms[0] is None:
      if any(fm is not None for fm in fms):
        raise ValueError("stack(): focus_mask mixed None and non-None across batch")
      fm: Any = None
    else:
      fm = np.stack(fms, axis=axis)
    return cls(
      components=stacked,
      focus_mask=fm,
      _trailing_ndims=dict(observations[0]._trailing_ndims),
    )

  # ------------------------------------------------------------------
  # Image transformations (no-ops when no image is present)
  # ------------------------------------------------------------------

  def resize_image(self, size: int) -> "Observation":
    """Resize the (leading..., H, W, 3) image stack so H = W = ``size``.

    Leaves a float32 image alone if it already matches ``size`` and
    only iterates per-frame when a resize is needed. Pass-through when
    ``"image"`` is not among the components. When a ``focus_mask`` is
    present it's resized too, with nearest-neighbor interpolation so a
    bool mask stays bool-equivalent (0/1 only).
    """
    img = self.components.get(IMAGE)
    if img is None:
      return self
    if img.shape[-3] == size and img.shape[-2] == size:
      return self
    out = _resize_stack(img, size, cv2.INTER_AREA)
    new_components = {**self.components, IMAGE: out}
    fm = self.focus_mask
    if fm is not None:
      fm = _resize_stack(fm[..., None], size, cv2.INTER_NEAREST).squeeze(-1)
    return replace(self, components=new_components, focus_mask=fm)

  def normalize_image(self) -> "Observation":
    """Map uint8 image → float32 in ``[-0.5, 0.5]``. Float images pass through."""
    img = self.components.get(IMAGE)
    if img is None:
      return self
    if img.dtype == np.uint8:
      img = img.astype(np.float32) / 255.0 - 0.5
    else:
      img = img.astype(np.float32, copy=False)
    return replace(self, components={**self.components, IMAGE: img})

  # ------------------------------------------------------------------
  # Sequence / batch ops
  # ------------------------------------------------------------------

  @property
  def leading_shape(self) -> tuple[int, ...]:
    """Shape of the axes before the per-frame trailing axes.

    Every component shares this prefix (verified at construction), so
    we can read it off whichever is convenient — pick the first.
    """
    name, arr = next(iter(self.components.items()))
    tn = self._trailing_ndims[name]
    return tuple(arr.shape[: arr.ndim - tn])

  def __getitem__(self, idx) -> "Observation":
    """Index/slice along the leading axes uniformly across all fields."""
    new_components = {n: a[idx] for n, a in self.components.items()}
    fm = None if self.focus_mask is None else self.focus_mask[idx]
    return replace(self, components=new_components, focus_mask=fm)

  def map_arrays(self, fn: Callable[[Any], Any]) -> "Observation":
    """Apply ``fn`` to every populated field.

    Used by the backends' ``coerce()`` to lift a numpy-side
    :class:`Observation` into one carrying mx.array / torch.Tensor
    leaves. The trailing-ndim metadata carries through unchanged.
    """
    new_components = {n: fn(a) for n, a in self.components.items()}
    fm = None if self.focus_mask is None else fn(self.focus_mask)
    return replace(self, components=new_components, focus_mask=fm)

  def add_batch_axis(self) -> "Observation":
    """Prepend a length-1 axis to every populated field.

    Implemented via indexing (``arr[None]``) so it works for numpy and
    backend tensor types alike — used by single-step inference paths
    (CEM, evaluator's per-episode batching).
    """
    return self[None]

  # ------------------------------------------------------------------
  # Adapters to the episode/buffer schema
  # ------------------------------------------------------------------

  def to_buffer_value(self) -> dict[str, Any]:
    """Return the value stored under ``episode["obs"]``.

    Always a dict keyed by component name — uniform schema independent
    of how many components are present. Focus masks ride in a separate
    ``Episode`` field, not in this dict.
    """
    return dict(self.components)

  def model_shape(self) -> dict[str, tuple[int, ...]]:
    """Return a per-component trailing-shape descriptor.

    Always a dict; matches what :attr:`PushingEnv.model_obs_shape`
    produces and what the model's multi-modal encoder consumes.
    """
    out: dict[str, tuple[int, ...]] = {}
    for name, arr in self.components.items():
      tn = self._trailing_ndims[name]
      out[name] = tuple(arr.shape[arr.ndim - tn:])
    return out
