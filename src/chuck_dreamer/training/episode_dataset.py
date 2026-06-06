"""Typed episode handle + in-memory dataset over a directory of episodes.

Two layers:

  * :class:`Episode` — one episode's arrays plus its source path and
    optional sidecar metadata (``act_mode``, ``goal_xy``, …). Replaces
    the bare raw-dict episodes that used to flow around the codebase.
    Use :meth:`Episode.from_file` to round-trip a single file (handy in
    one-off scripts and importer tests).
  * :class:`EpisodeDataset` — the set of episodes living at a directory.
    Encapsulates two access modes: ``stream()`` for one-pass ingestion
    that never holds the full set in RAM (replay-buffer warmup), and
    ``load()`` for materializing the set into memory once for repeated
    re-iteration (the evaluator's cached set).

Discovery (``len(ds)`` / ``ds.paths``) is cheap and works without
loading. ``load()`` is idempotent so callers can call it from a
warmup hook without checking ``loaded`` first.

This module owns the format-specific readers. The two reader
implementations (HDF5 / Rerun) are private; round-trip a single file
through :meth:`Episode.from_file`.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, TypeAlias, Union

import h5py  # type: ignore[import-untyped]
import numpy as np

from chuck_dreamer.common.episode import Episode as EpisodeData


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


ProgressCallback: TypeAlias = Callable[[int, int, Path], None]
Progress: TypeAlias = Union[bool, ProgressCallback]

# A loaded episode's array container: the field-attributed
# :class:`~chuck_dreamer.common.episode.Episode` shared with the write side.
# It is a ``Mapping`` returning values, so every consumer that indexes it by
# name (``raw["image"]``, ``raw.get``, ``"x" in raw``) is unchanged from when
# this was a bare dict; loading just tags each field with its kind/persist.
RawEpisode: TypeAlias = EpisodeData
# Processor type — applied inside the loader worker by stream_processed()
# so raw images never travel back to the main process. Kept as a generic
# callable on RawEpisode to avoid pulling in episode_processor here.
RawEpisodeTransform: TypeAlias = Callable[[RawEpisode], Any]


SUPPORTED_FORMATS = ("hdf5", "rerun")

# Sidecar fields that live alongside the per-step arrays in the writer
# format but are properties of the whole episode. We surface them on
# ``Episode.metadata`` so call sites can read them without sniffing the
# raw arrays dict.
_METADATA_KEYS: tuple[str, ...] = ("goal_xy", "mat_centre", "tags")


# ---------------------------------------------------------------------------
# Format-specific readers (private — go through Episode / EpisodeDataset)
# ---------------------------------------------------------------------------


# Maps RawEpisode key -> HDF5 dataset name. Action datasets are resolved
# dynamically since exactly one of joint_action / ee_action is present.
_HDF5_DATASET = {
  "image": "images",
  "reward": "rewards",
  "timestamp": "timestamps",
  "joint_qpos": "joint_qpos",
  "ee_pos": "ee_pos",
  "ee_quat": "ee_quat",
  "object_xy": "object_xy",
}


def _load_hdf5_episode(path: str | Path) -> RawEpisode:
  """Read one HDF5 episode written by ``HDF5EpisodeWriter`` into an
  :class:`~chuck_dreamer.common.episode.Episode`.

  The arrays are gathered into a plain dict first (one h5py read per
  dataset), then wrapped via :meth:`Episode.from_arrays`, which tags each
  field with its kind/persist from the registry — the symmetric inverse of
  the field-kind-driven writer."""
  raw: dict[str, Any] = {}
  with h5py.File(path, "r") as f:
    for key, dataset in _HDF5_DATASET.items():
      raw[key] = np.asarray(f[dataset][()])

    if "object_uv" in f:
      raw["object_uv"] = np.asarray(f["object_uv"][()], dtype=np.float32)

    has_joint = "joint_action" in f
    has_ee    = "ee_action"    in f
    if has_joint:
      raw["joint_action"] = np.asarray(f["joint_action"][()])
    if has_ee:
      raw["ee_action"] = np.asarray(f["ee_action"][()])
    if not (has_joint or has_ee):
      raise KeyError(f"{path}: missing action dataset (joint_action and/or ee_action)")

    if "segmentation" in f:
      seg_grp = f["segmentation"]
      for name in seg_grp.keys():
        raw[f"segmentation_{name}"] = np.asarray(seg_grp[name][()], dtype=bool)

    if "metadata" in f:
      meta = f["metadata"]
      if "goal_xy" in meta:
        raw["goal_xy"] = np.asarray(meta["goal_xy"][()], dtype=np.float32)
      if "mat_centre" in meta:
        raw["mat_centre"] = np.asarray(meta["mat_centre"][()], dtype=np.float32)
      if "tags" in meta:
        raw_tags = meta["tags"][()]
        raw["tags"] = tuple(
          t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in raw_tags
        )
  return EpisodeData.from_arrays(raw)


def _collect_rerun_chunks(path: str | Path) -> tuple[dict[str, list[Any]], dict[str, dict]]:
  """Open a ``.rrd`` and bucket its chunks as Arrow ``RecordBatch`` lists.

  Returns ``(by_entity, static)`` where:

    * ``by_entity[path]`` is a list of ``pyarrow.RecordBatch`` for each
      time-indexed entity. Keeping the Arrow form (rather than
      ``to_pydict()`` materializing it as Python lists) lets the readers
      pull raw uint8 / float buffers in bulk — image extraction drops
      from ``T`` per-row reshapes to a single ``flatten().to_numpy()``.
    * ``static[path]`` carries the small metadata chunks as plain
      pydicts; they're tiny and read once.

  Uses the experimental ``RrdReader`` (added in rerun 0.32). The
  per-file speedup (~50× on real recordings) comes from extracting
  pixels via Arrow's ``ListArray.values.values`` instead of materializing
  every row through ``to_pydict()``.
  """
  from rerun.experimental import RrdReader

  reader = RrdReader(str(path))
  # ``reader.stream()`` (not ``reader.store()``) so footerless recordings
  # still read: the streaming file sink writes the chunks but no footer, and
  # ``store()`` rejects footerless RRDs. ``stream()`` scans the chunks
  # directly with the same per-chunk interface.
  by_entity: dict[str, list[Any]] = {}
  static:    dict[str, dict]      = {}
  for chunk in reader.stream():
    entity = str(chunk.entity_path)
    if chunk.is_static:
      static[entity] = chunk.to_record_batch().to_pydict()
      continue
    if entity.startswith("/__"):
      continue
    by_entity.setdefault(entity, []).append(chunk.to_record_batch())
  return by_entity, static


def _ordered_scalar_column(record_batches: list[Any], step_key: str = "step") -> np.ndarray:
  """Flatten per-chunk scalar rows into a single ``(N, d)`` float32 array sorted by step.

  ``Scalars:scalars`` is logged as ``list<float64>``; ``.values`` on the
  combined ListArray gives one contiguous float buffer, which numpy
  reshapes in O(1). The previous per-row ``zip``/``stack`` is gone.
  """
  import pyarrow as pa

  step_parts: list[np.ndarray] = []
  scalar_cols: list[Any] = []
  for rb in record_batches:
    step_parts.append(np.asarray(rb.column(step_key)))
    scalar_cols.append(rb.column("Scalars:scalars"))
  steps = np.concatenate(step_parts) if len(step_parts) > 1 else step_parts[0]
  combined = pa.concat_arrays(scalar_cols) if len(scalar_cols) > 1 else scalar_cols[0]
  flat = np.asarray(combined.values, dtype=np.float32)
  values: np.ndarray = flat.reshape(steps.size, -1)
  if steps.size > 1 and not np.all(steps[1:] >= steps[:-1]):
    order = np.argsort(steps, kind="stable")
    values = values[order]
  return values


def _ordered_image_stack(
  record_batches: list[Any], buf_name: str = "Image:buffer", fmt_name: str = "Image:format",
) -> tuple[np.ndarray, np.ndarray]:
  """Flatten per-chunk image rows into a single ``(T, H, W, 3)`` uint8 stack + sorted steps.

  ``Image:buffer`` is logged as ``list<list<uint8>>`` — flattening twice
  yields one contiguous uint8 buffer with the whole pixel stream, which
  numpy reshapes to ``(T, H, W, 3)`` in O(1). Cross-chunk we
  ``concat_arrays`` the buffer columns once (Arrow zero-copies them)
  and stable-sort the result by step.

  All frames in a recording share H/W (the writer always logs the same
  camera resolution); the first chunk's format wins and any divergent
  row would fail the final reshape.
  """
  import pyarrow as pa

  step_parts: list[np.ndarray] = []
  buf_cols: list[Any] = []
  for rb in record_batches:
    step_parts.append(np.asarray(rb.column("step")))
    buf_cols.append(rb.column(buf_name))
  steps = np.concatenate(step_parts) if len(step_parts) > 1 else step_parts[0]

  fmt0 = record_batches[0].column(fmt_name)[0].as_py()
  fmt0 = fmt0[0] if isinstance(fmt0, list) else fmt0
  h, w = int(fmt0["height"]), int(fmt0["width"])

  combined = pa.concat_arrays(buf_cols) if len(buf_cols) > 1 else buf_cols[0]
  pixels = np.asarray(combined.values.values)  # list<list<uint8>> → uint8[]
  imgs = pixels.reshape(steps.size, h, w, 3)
  order = np.argsort(steps, kind="stable")
  return imgs[order], steps[order]


def _ordered_image_times(record_batches: list[Any]) -> np.ndarray:
  """Pull the ``time`` (Timedelta) column out of image chunks, sorted by step.

  Kept separate from :func:`_ordered_image_stack` so the image extractor
  is reusable for segmentation (which doesn't need a time axis).
  """
  step_parts: list[np.ndarray] = []
  time_parts: list[np.ndarray] = []
  for rb in record_batches:
    step_parts.append(np.asarray(rb.column("step")))
    time_parts.append(_durations_to_seconds(rb.column("time")))
  steps = np.concatenate(step_parts) if len(step_parts) > 1 else step_parts[0]
  times = np.concatenate(time_parts) if len(time_parts) > 1 else time_parts[0]
  order = np.argsort(steps, kind="stable")
  return times[order]


def _durations_to_seconds(arrow_col: Any) -> np.ndarray:
  """Convert a rerun ``time`` Arrow column to ``(N,)`` float32 seconds.

  Rerun's time column is a duration array; ``.to_numpy()`` yields
  ``timedelta64`` which divides cleanly into seconds.
  """
  arr = np.asarray(arrow_col)
  if np.issubdtype(arr.dtype, np.timedelta64):
    return (arr / np.timedelta64(1, "s")).astype(np.float32)
  if arr.dtype == object:
    return np.asarray(
      [t.total_seconds() if hasattr(t, "total_seconds") else float(t) for t in arr],
      dtype=np.float32,
    )
  return arr.astype(np.float32, copy=False)


def _load_rerun_episode(path: str | Path) -> RawEpisode:
  """Read one Rerun ``.rrd`` episode written by ``RerunEpisodeWriter`` into an
  :class:`~chuck_dreamer.common.episode.Episode`.

  Fields are gathered into a plain dict (the ``_load_rerun_*`` helpers mutate
  it by name), then wrapped via :meth:`Episode.from_arrays` so each field is
  tagged with its kind/persist — symmetric with the writer."""
  by_entity, static = _collect_rerun_chunks(path)

  def _scalars(entity: str) -> np.ndarray:
    if entity not in by_entity:
      raise KeyError(f"entity {entity!r} not found in {path}")
    return _ordered_scalar_column(by_entity[entity])

  raw: dict[str, Any] = {}
  has_joint = "/joint_action" in by_entity
  has_ee    = "/ee_action"    in by_entity
  if has_joint:
    raw["joint_action"] = _scalars("/joint_action")
  if has_ee:
    raw["ee_action"] = _scalars("/ee_action")
  if not (has_joint or has_ee):
    raise KeyError(f"{path}: missing action entity (/joint_action and/or /ee_action)")

  raw["reward"] = _scalars("/reward").reshape(-1)
  raw["joint_qpos"] = _scalars("/joint_qpos")
  raw["ee_pos"] = _scalars("/ee_pos")
  raw["ee_quat"] = _scalars("/ee_quat")
  raw["object_xy"] = _scalars("/object_xy")
  if "/object_uv" in by_entity:
    raw["object_uv"] = _scalars("/object_uv").astype(np.float32, copy=False)

  _load_rerun_camera(by_entity, static, raw, path)
  _load_rerun_segmentation(by_entity, raw)
  _load_rerun_metadata(static, raw)

  return EpisodeData.from_arrays(raw)


def _load_rerun_camera(
  by_entity: dict[str, list[Any]], static: dict[str, dict],
  raw: dict[str, Any], path: str | Path,
) -> None:
  """Decode the camera RGB stack + per-step timestamps.

  The writer stores RGB as one ``rr.AssetVideo`` blob on ``/camera/video``
  (static) plus a per-step ``rr.VideoFrameReference``; decode the blob and
  select the episode's frames by their reference timestamps. Older
  recordings that pre-date the video switch logged a raw ``rr.Image`` per
  frame on ``/camera/image`` — fall back to that path for them.
  """
  from chuck_dreamer.sim.rerun_video import decode_frames, video_frame_timestamps

  if "/camera/video" in by_entity or "/camera/video" in static:
    vchunks = by_entity.get("/camera/video", [])
    frame_pts = video_frame_timestamps(vchunks)
    raw["timestamp"] = _ordered_video_times(vchunks)
    video_bytes = _asset_video_blob(static, "/camera/video", path)
    raw["image"] = decode_frames(video_bytes, list(frame_pts))
    return

  img_chunks = by_entity["/camera/image"]
  images, _steps = _ordered_image_stack(img_chunks)
  raw["image"] = images
  raw["timestamp"] = _ordered_image_times(img_chunks)


def _asset_video_blob(static: dict[str, dict], entity: str, path: str | Path) -> bytes:
  """Pull the raw video bytes out of an ``AssetVideo`` static chunk.

  ``AssetVideo:buffer`` is logged as a single ``list<uint8>`` cell; the
  experimental reader surfaces it as a one-element column whose first row
  is the byte blob."""
  chunk = static.get(entity)
  cells = chunk.get("AssetVideo:blob") if chunk else None
  if not cells:
    raise KeyError(f"{path}: no AssetVideo:blob on {entity}")
  return np.asarray(cells[0], dtype=np.uint8).tobytes()


def _ordered_video_times(record_batches: list[Any]) -> np.ndarray:
  """Per-step ``time`` (Timedelta) column off the ``/camera/video`` chunks,
  sorted by step — the episode timestamps, same as the old image path."""
  if not record_batches:
    return np.empty((0,), dtype=np.float32)
  step_parts: list[np.ndarray] = []
  time_parts: list[np.ndarray] = []
  for rb in record_batches:
    if "time" not in rb.schema.names:
      continue
    step_parts.append(np.asarray(rb.column("step")))
    time_parts.append(_durations_to_seconds(rb.column("time")))
  if not step_parts:
    return np.empty((0,), dtype=np.float32)
  steps = np.concatenate(step_parts) if len(step_parts) > 1 else step_parts[0]
  times = np.concatenate(time_parts) if len(time_parts) > 1 else time_parts[0]
  order = np.argsort(steps, kind="stable")
  return times[order]


def _load_rerun_metadata(static: dict[str, dict], raw: dict[str, Any]) -> None:
  """Pull whole-episode sidecar fields out of static ``/metadata/*`` entities.

  Metadata is logged as static :class:`rr.TextDocument` cells on
  ``/metadata/<key>`` and surfaces in the ``static`` dict from
  :func:`_collect_rerun_chunks` keyed by full entity path.
  """
  def _text(entity: str) -> str | None:
    chunk = static.get(entity)
    if chunk is None:
      return None
    cells = chunk.get("TextDocument:text")
    if not cells:
      return None
    cell = cells[0]
    if isinstance(cell, list):
      cell = cell[0] if cell else None
    if cell is None:
      return None
    return cell.decode("utf-8") if isinstance(cell, bytes) else str(cell)

  tags = _text("/metadata/tags")
  if tags:
    raw["tags"] = tuple(t for t in tags.split(",") if t)


# Single-entity masks the writer emits alongside the per-piece clutter masks.
# The ``/camera/seg/clutter`` union is intentionally excluded — it's reconstructable
# from the per-piece masks and would be redundant in the round-tripped dict.
_RERUN_SEG_SINGLE = ("target", "goal", "arm", "background")


def _ordered_seg_masks(record_batches: list[Any]) -> np.ndarray:
  """Flatten segmentation chunks into ``(T, H, W)`` bool sorted by step.

  The writer stores masks as PNG ``rr.EncodedImage`` blobs; decode each and
  cast to bool. Older recordings logged raw ``rr.SegmentationImage`` buffers
  — fall back to bulk-flattening those when present.
  """
  if "EncodedImage:blob" in record_batches[0].schema.names:
    return _ordered_png_masks(record_batches)
  return _ordered_raw_seg_masks(record_batches)


def _ordered_png_masks(record_batches: list[Any]) -> np.ndarray:
  """Decode PNG ``EncodedImage`` mask chunks into ``(T, H, W)`` bool by step."""
  import io as _io

  from PIL import Image

  steps: list[int] = []
  masks: list[np.ndarray] = []
  for rb in record_batches:
    step_col = np.asarray(rb.column("step"))
    blob_col = rb.column("EncodedImage:blob")
    for i in range(len(step_col)):
      # Cell is list<list<uint8>>; flatten twice to the PNG byte stream.
      png = np.asarray(blob_col[i].values.values, dtype=np.uint8).tobytes()
      arr = np.asarray(Image.open(_io.BytesIO(png))).astype(bool)
      # Masks are written as transparent RGBA (colour where set, all-zero
      # elsewhere). Collapse any colour channels back to a 2D boolean — a
      # pixel is foreground iff any channel is non-zero. Grayscale masks
      # from older recordings are already 2D.
      mask = np.asarray(arr.any(axis=-1)) if arr.ndim == 3 else arr
      steps.append(int(step_col[i]))
      masks.append(mask)
  order = np.argsort(np.asarray(steps), kind="stable")
  return np.stack([masks[i] for i in order], axis=0)


def _ordered_raw_seg_masks(record_batches: list[Any]) -> np.ndarray:
  """Legacy path: flatten raw ``SegmentationImage`` buffers into
  ``(T, H, W)`` bool sorted by step (recordings predating the PNG switch)."""
  import pyarrow as pa

  step_parts: list[np.ndarray] = []
  buf_cols: list[Any] = []
  for rb in record_batches:
    step_parts.append(np.asarray(rb.column("step")))
    buf_cols.append(rb.column("SegmentationImage:buffer"))
  steps = np.concatenate(step_parts) if len(step_parts) > 1 else step_parts[0]

  fmt0 = record_batches[0].column("SegmentationImage:format")[0].as_py()
  fmt0 = fmt0[0] if isinstance(fmt0, list) else fmt0
  h, w = int(fmt0["height"]), int(fmt0["width"])

  combined = pa.concat_arrays(buf_cols) if len(buf_cols) > 1 else buf_cols[0]
  pixels = np.asarray(combined.values.values)
  masks = pixels.reshape(steps.size, h, w).astype(bool)
  order = np.argsort(steps, kind="stable")
  return masks[order]


def _load_rerun_segmentation(by_entity: dict[str, list[Any]], raw: dict[str, Any]) -> None:
  """Pull segmentation masks (if present) out of a rerun recording's entities.

  Mirrors :class:`RerunEpisodeWriter`'s log layout: ``/camera/seg/{name}`` for
  each named mask plus ``/camera/seg/clutter_{k}`` per clutter piece. Restacks
  the per-piece clutter masks into ``segmentation_clutter`` with shape
  ``(T, K, H, W)`` to match ``_stack_image_obs``.
  """
  for name in _RERUN_SEG_SINGLE:
    entity = f"/camera/seg/{name}"
    if entity in by_entity:
      raw[f"segmentation_{name}"] = _ordered_seg_masks(by_entity[entity])

  clutter_pieces: list[tuple[int, np.ndarray]] = []
  for entity, chunks in by_entity.items():
    if entity.startswith("/camera/seg/clutter_"):
      k = int(entity.rsplit("_", 1)[1])
      clutter_pieces.append((k, _ordered_seg_masks(chunks)))
  if clutter_pieces:
    clutter_pieces.sort(key=lambda kv: kv[0])
    raw["segmentation_clutter"] = np.stack([m for _, m in clutter_pieces], axis=1)


_FORMAT_SUFFIX = {"hdf5": ".hdf5", "rerun": ".rrd"}
_FORMAT_LOADER = {"hdf5": _load_hdf5_episode, "rerun": _load_rerun_episode}
_FORMAT_POOL: dict[str, type[ThreadPoolExecutor] | type[ProcessPoolExecutor]] = {
  # h5py reads release the GIL → threads suffice. The rerun reader runs
  # heavy per-step Python parsing → only a process pool actually
  # parallelizes it.
  "hdf5":  ThreadPoolExecutor,
  "rerun": ProcessPoolExecutor,
}


def _load_and_transform(
  path: str | Path,
  fmt: str,
  transform: RawEpisodeTransform | None,
) -> Any:
  """Worker entry-point: load one episode, optionally apply ``transform`` in-process.

  Top-level so it can be pickled for ``ProcessPoolExecutor``. When a
  transform is supplied (typically the buffer's :class:`EpisodeProcessor`)
  the raw arrays — including the multi-hundred-MiB image stack — never
  travel back to the main process; only the small processed result does.
  """
  raw = _FORMAT_LOADER[fmt](path)
  if transform is None:
    return raw
  return transform(raw)


# ---------------------------------------------------------------------------
# Progress helper (shared by load() and stream())
# ---------------------------------------------------------------------------


def _resolve_progress(progress: Progress) -> ProgressCallback | None:
  """Turn a ``Progress`` value into a per-file callback (or None)."""
  if progress is False:
    return None
  if callable(progress):
    return progress

  try:
    from tqdm import tqdm  # type: ignore[import-not-found]
  except ImportError:
    def _print_cb(i: int, total: int, path: Path) -> None:
      print(f"[{i}/{total}] {path.name}")

    return _print_cb

  bar: dict[str, Any] = {"tqdm": None}

  def _tqdm_cb(i: int, total: int, path: Path) -> None:
    if bar["tqdm"] is None:
      bar["tqdm"] = tqdm(total=total, unit="ep", desc="episodes")
    bar["tqdm"].set_postfix_str(path.name, refresh=False)
    bar["tqdm"].update(1)
    if i == total:
      bar["tqdm"].close()

  return _tqdm_cb


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


def _extract_metadata(raw: RawEpisode) -> dict[str, Any]:
  """Copy whole-episode sidecar fields out of a raw dict into a metadata dict.

  Non-destructive: the values stay in ``raw`` so processors that look
  them up there keep working. The metadata dict is for callers that
  want to read episode-level fields without sniffing ``data``.
  """
  return {k: raw[k] for k in _METADATA_KEYS if k in raw}


@dataclass(frozen=True, slots=True)
class Episode:
  """One episode loaded from disk: arrays + provenance.

  ``data`` is the field-attributed
  :class:`~chuck_dreamer.common.episode.Episode` the readers return — the same
  container the write side fills. It is a ``Mapping`` returning values, so the
  processors in :mod:`.episode_processor` (and every other ``data[...]``
  consumer) read it exactly as they read the old flat dict. ``metadata``
  carries whole-episode sidecar fields the writers serialize separately.

  This handle is the read-side *provenance wrapper* (path / format / metadata);
  the array container itself is :class:`EpisodeData`. They stay separate: a
  handle has a source, a container is just fields.

  ``Episode`` instances are read-only — load once, share freely.
  """

  path: Path
  format: str
  data: RawEpisode
  metadata: dict[str, Any] = field(default_factory=dict)

  @classmethod
  def from_file(cls, path: str | Path, *, format: str | None = None) -> "Episode":
    """Round-trip a single episode file into an :class:`Episode`.

    ``format`` is inferred from the file extension when omitted
    (``.hdf5`` → ``"hdf5"``, ``.rrd`` → ``"rerun"``); pass it explicitly
    when the extension doesn't match the format you want to read.
    """
    p = Path(path)
    fmt = format or _format_from_path(p)
    if fmt not in SUPPORTED_FORMATS:
      raise ValueError(f"unsupported format {fmt!r}; use one of {SUPPORTED_FORMATS}")
    raw = _FORMAT_LOADER[fmt](p)
    return cls(path=p, format=fmt, data=raw, metadata=_extract_metadata(raw))

  @property
  def stem(self) -> str:
    return self.path.stem

  @property
  def num_steps(self) -> int:
    """Number of recorded steps (length of the action / reward axis)."""
    if "reward" in self.data:
      return int(np.asarray(self.data["reward"]).shape[0])
    for key in ("joint_action", "ee_action"):
      if key in self.data:
        return int(np.asarray(self.data[key]).shape[0])
    raise KeyError(f"episode {self.path.name} has neither reward nor action")

  def has(self, key: str) -> bool:
    return key in self.data

  def get(self, key: str, default: Any = None) -> Any:
    return self.data.get(key, default)


def _format_from_path(path: Path) -> str:
  suffix = path.suffix.lower()
  for fmt, ext in _FORMAT_SUFFIX.items():
    if suffix == ext:
      return fmt
  raise ValueError(
    f"cannot infer episode format from suffix {suffix!r} on {path.name}; "
    f"pass format= explicitly"
  )


# ---------------------------------------------------------------------------
# EpisodeDataset
# ---------------------------------------------------------------------------


class EpisodeDataset:
  """The set of episodes recorded under one directory.

  Cheap to construct — ``__init__`` only records the path, doesn't read
  any file. Call :meth:`load` (or :meth:`stream`) to actually decode.
  ``len(ds)`` and :attr:`paths` work without loading.
  """

  def __init__(self, directory: str | Path, *, format: str = "hdf5"):
    if format not in SUPPORTED_FORMATS:
      raise ValueError(f"unsupported format {format!r}; use one of {SUPPORTED_FORMATS}")
    self.directory = Path(directory)
    self.format    = format
    self._paths: list[Path] | None = None
    self._loaded: list[Episode] | None = None

  # ---------- discovery (cheap) ----------

  @property
  def paths(self) -> list[Path]:
    """Sorted list of episode file paths in the directory.

    Cached after the first scan — call sites can reuse without paying
    for repeated globs. Returns an empty list when the directory is
    missing.
    """
    if self._paths is None:
      if self.directory.exists():
        suffix = _FORMAT_SUFFIX[self.format]
        self._paths = sorted(self.directory.glob(f"episode-*{suffix}"))
      else:
        self._paths = []
    return self._paths

  def __len__(self) -> int:
    if self._loaded is not None:
      return len(self._loaded)
    return len(self.paths)

  @property
  def loaded(self) -> bool:
    return self._loaded is not None

  # ---------- loading ----------

  def load(
    self,
    *,
    num_workers: int | None = None,
    progress: Progress = True,
    num_episodes: int | None = None,
    rng: np.random.Generator | None = None,
  ) -> "EpisodeDataset":
    """Materialize every episode into memory. Idempotent.

    Returns ``self`` so callers can chain (``EpisodeDataset(...).load()``).
    On a missing directory the set stays empty (``len == 0``) — call
    sites that care should check ``len(dataset)``.

    ``num_workers=None`` picks ``max(1, cpu_count() // 2)`` — the same
    default the replay-buffer warmup uses.
    """
    if self._loaded is not None:
      return self
    self._loaded = list(self._stream(
      paths=self._pick_paths(num_episodes, rng),
      num_workers=self._resolve_workers(num_workers),
      progress=progress,
    ))
    return self

  def stream(
    self,
    *,
    num_workers: int | None = None,
    progress: Progress = False,
    num_episodes: int | None = None,
    rng: np.random.Generator | None = None,
  ) -> Iterator[Episode]:
    """Yield :class:`Episode`\\ s without retaining them.

    Use this for one-pass ingestion (replay-buffer warmup) where the
    caller folds each episode into another structure and doesn't need
    to revisit. Does not populate :attr:`loaded`.
    """
    yield from self._stream(
      paths=self._pick_paths(num_episodes, rng),
      num_workers=self._resolve_workers(num_workers),
      progress=progress,
    )

  def stream_processed(
    self,
    transform: RawEpisodeTransform,
    *,
    num_workers: int | None = None,
    progress: Progress = False,
    num_episodes: int | None = None,
    rng: np.random.Generator | None = None,
  ) -> Iterator[tuple[Path, Any]]:
    """Yield ``(path, transform(raw))`` pairs, applying ``transform`` in-worker.

    The transform runs in the loader worker (process or thread) so its
    output — typically a buffer-shaped episode with downsampled images —
    is what travels back to the main process, not the full raw arrays.
    For real recordings with native 512×512 frames this cuts in-flight
    RAM by ~60× versus :meth:`stream` (which yields the raw image stack
    intact).

    ``transform`` must be picklable for the rerun (process-pool) path.
    The plain :class:`~.episode_processor.EpisodeProcessor` is.
    """
    yield from self._stream_processed(
      paths=self._pick_paths(num_episodes, rng),
      num_workers=self._resolve_workers(num_workers),
      progress=progress,
      transform=transform,
    )

  # ---------- access after load() ----------

  def __iter__(self) -> Iterator[Episode]:
    if self._loaded is None:
      raise RuntimeError(
        "EpisodeDataset not loaded — call .load() first or use .stream()."
      )
    return iter(self._loaded)

  def __getitem__(self, key: int | str) -> Episode:
    """Index by integer position or by file stem (``"episode-00007"``)."""
    if self._loaded is None:
      raise RuntimeError(
        "EpisodeDataset not loaded — call .load() first or use .stream()."
      )
    if isinstance(key, int):
      return self._loaded[key]
    # By-stem lookup. Linear is fine: eval sets are O(100); building a
    # dict on every access would be wasted bookkeeping.
    for ep in self._loaded:
      if ep.stem == key:
        return ep
    raise KeyError(f"no episode with stem {key!r} in {self.directory}")

  # ---------- internals ----------

  def _resolve_workers(self, num_workers: int | None) -> int:
    if num_workers is None:
      import os
      return max(1, (os.cpu_count() or 2) // 2)
    return int(num_workers)

  def _pick_paths(
    self, num_episodes: int | None, rng: np.random.Generator | None,
  ) -> list[Path]:
    paths = list(self.paths)
    if num_episodes is not None and num_episodes < len(paths):
      rng = rng if rng is not None else np.random.default_rng()
      idx = rng.choice(len(paths), size=int(num_episodes), replace=False)
      paths = [paths[i] for i in sorted(idx.tolist())]
    return paths

  def _stream(
    self,
    *,
    paths: list[Path],
    num_workers: int,
    progress: Progress,
  ) -> Iterator[Episode]:
    """Shared driver for ``load`` and ``stream`` — yields Episode objects."""
    fmt = self.format
    for path, raw, i, total in self._drive_worker_pool(
      paths=paths,
      num_workers=num_workers,
      progress=progress,
      transform=None,
    ):
      metadata = _extract_metadata(raw)
      del i, total
      yield Episode(path=path, format=fmt, data=raw, metadata=metadata)

  def _stream_processed(
    self,
    *,
    paths: list[Path],
    num_workers: int,
    progress: Progress,
    transform: RawEpisodeTransform,
  ) -> Iterator[tuple[Path, Any]]:
    """Driver for :meth:`stream_processed` — yields ``(path, transform(raw))``.

    The transform runs in the worker so the (potentially large) raw dict
    is never returned across the process boundary.
    """
    for path, processed, i, total in self._drive_worker_pool(
      paths=paths,
      num_workers=num_workers,
      progress=progress,
      transform=transform,
    ):
      del i, total
      yield path, processed

  def _drive_worker_pool(
    self,
    *,
    paths: list[Path],
    num_workers: int,
    progress: Progress,
    transform: RawEpisodeTransform | None,
  ) -> Iterator[tuple[Path, Any, int, int]]:
    """Submit ``(load_and_optionally_transform)`` per path; yield results in order.

    When ``transform`` is None the yielded payload is the raw episode dict;
    otherwise it's ``transform(raw)`` computed inside the worker so the
    raw arrays never travel back. Maintains an in-flight window of size
    ``num_workers`` for steady throughput. Calls ``progress`` once per
    yielded result.
    """
    callback = _resolve_progress(progress)
    total    = len(paths)
    fmt      = self.format
    pool_cls = _FORMAT_POOL[fmt]

    def _submit(pool, idx: int):
      return (idx, paths[idx], pool.submit(_load_and_transform, paths[idx], fmt, transform))

    if num_workers <= 0 or total <= 1:
      for i, p in enumerate(paths, start=1):
        payload = _load_and_transform(p, fmt, transform)
        if callback is not None:
          callback(i, total, p)
        yield p, payload, i, total
      return

    window = min(num_workers, total)
    with pool_cls(max_workers=num_workers) as pool:
      in_flight: deque = deque()
      next_submit = 0
      for _ in range(window):
        in_flight.append(_submit(pool, next_submit))
        next_submit += 1

      yielded = 0
      while in_flight:
        _idx, path, fut = in_flight.popleft()
        payload = fut.result()
        yielded += 1
        if callback is not None:
          callback(yielded, total, path)
        yield path, payload, yielded, total
        if next_submit < total:
          in_flight.append(_submit(pool, next_submit))
          next_submit += 1
