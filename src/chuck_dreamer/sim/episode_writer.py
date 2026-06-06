"""Episode writers — persist sim and eval episodes to HDF5 or Rerun ``.rrd`` files.

Two kinds of episodes share these writers:

  * **Sim collect** — full per-step record from a real rollout (image,
    action, reward, joint state, …). Written via :meth:`write_episode`.
  * **Eval** — what the world model saw vs. what it produced under a
    burn-in / open-loop split (raw obs, processed obs, posterior + prior
    reconstructions, latent ``h`` / ``s``). Written via
    :meth:`write_eval_episode`.

Each writer class supports both: the format and on-disk layout differ,
but the entry-point + metadata handling are shared.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np

from chuck_dreamer.common.episode import Episode, FieldKind

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("hdf5", "rerun")

# Per-step fields whose HDF5 dataset name differs from the field name. Every
# other persisted SCALAR/ACTION field is written to a dataset of its own name.
# (Mirrors the reader's ``_HDF5_DATASET`` map so the round-trip lines up.)
_HDF5_DATASET_ALIAS = {
    "image":     "images",
    "reward":    "rewards",
    "timestamp": "timestamps",
}


def _as_episode(episode: Episode | dict[str, Any]) -> Episode:
    """Coerce a writer input to an :class:`Episode`.

    Already-:class:`Episode` inputs pass through (keeping their per-field
    kind/persist metadata); a plain dict is wrapped, inferring each field's
    disposition from its name. Lets the writers be field-kind-driven while every
    legacy caller that still passes a bare dict keeps working unchanged."""
    return episode if isinstance(episode, Episode) else Episode.from_arrays(episode)


# Filename prefixes — different kinds of episodes live alongside one
# another in the same dump dir, so :class:`EpisodeDataset` can find sim
# episodes without picking up eval dumps and vice versa.
EPISODE_FILENAME_PREFIX      = "episode"
EVAL_EPISODE_FILENAME_PREFIX = "eval"


def EpisodeWriter(output_dir: str, format: str = "hdf5"):
    """Factory that returns the concrete writer for the requested ``format``.

    The returned writer supports both :meth:`write_episode` (sim) and
    :meth:`write_eval_episode` (eval), so a single instance can persist
    both kinds.
    """
    if format == "hdf5":
        return HDF5EpisodeWriter(output_dir)
    if format == "rerun":
        return RerunEpisodeWriter(output_dir)
    raise ValueError(
        f"Unsupported format '{format}'. Supported formats: {SUPPORTED_FORMATS}.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


# Per-step columns shared by all action modes for sim writes.
# ``joint_action`` and ``ee_action`` may both ride along — the file is
# action-space-agnostic and the training pipeline picks which to read.
_REQUIRED_KEYS = (
    "image", "reward", "timestamp",
    "joint_qpos", "ee_pos", "ee_quat", "object_xy",
)


def _serialize_metadata_config(metadata: dict[str, Any] | None) -> str | None:
    """Return the ``config`` field of metadata as a JSON string, or None."""
    if metadata is None:
        return None
    cfg = metadata.get("config")
    if cfg is None:
        return None
    if isinstance(cfg, (str, bytes)):
        return cfg if isinstance(cfg, str) else cfg.decode("utf-8")
    return json.dumps(cfg if isinstance(cfg, dict) else asdict(cfg))  # type: ignore[arg-type]


def _collect_actions(episode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return ``{kind: array}`` for whichever of joint_action / ee_action are present.

    Recordings may carry both: the file is action-space-agnostic and the
    training pipeline picks which view to use at load time based on
    ``config.env.act_mode``. At least one must be present.
    """
    out: dict[str, np.ndarray] = {}
    for key in ("joint_action", "ee_action"):
        if key in episode:
            out[key] = np.asarray(episode[key], dtype=np.float32)
    if not out:
        raise KeyError(
            "episode is missing an action field (expected joint_action and/or ee_action)")
    return out


def _denormalize_recon_image(recon: np.ndarray) -> np.ndarray:
    """Map a decoder image output (float in ``[-0.5, 0.5]``) to uint8 RGB.

    Out-of-range values are clipped so we get a viewable image even when
    the model has not converged yet.
    """
    arr: np.ndarray = np.asarray(recon, dtype=np.float32)
    arr = np.clip(arr + 0.5, 0.0, 1.0) * 255.0
    return np.asarray(arr.astype(np.uint8))


def _coerce_image_for_log(img: np.ndarray) -> np.ndarray:
    """Accept uint8 [0,255] or float images in [-0.5, 0.5] / [0, 1] and return uint8."""
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.min() < 0.0:
        arr = arr + 0.5
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def _to_unit_float(img: np.ndarray) -> np.ndarray:
    """Map an obs/recon image to float32 in ``[0, 1]`` for diffing.

    Mirrors :func:`_coerce_image_for_log` / :func:`_denormalize_recon_image`
    but keeps the result as floats so per-pixel differences keep their sign
    range and don't underflow at uint8.
    """
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    if arr.min() < 0.0:
        arr = arr + 0.5
    return np.clip(arr, 0.0, 1.0)


def _pixel_diff(obs: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """Return signed per-pixel difference ``obs - recon`` in ``[-1, 1]``."""
    return np.asarray(_to_unit_float(obs) - _to_unit_float(recon))


def _diff_to_uint8(diff: np.ndarray) -> np.ndarray:
    """Map a signed diff in ``[-1, 1]`` to a viewable uint8 image centred at 128."""
    arr = np.clip(diff, -1.0, 1.0)
    arr = (arr * 0.5 + 0.5) * 255.0
    return arr.astype(np.uint8)


def _video_fps(metadata: dict[str, Any] | None, images: np.ndarray, T: int) -> float:
    """Frame rate for the re-encode fallback.

    Only the spacing of timeline timestamps matters for playback; pixels are
    indexed by step regardless. Prefer an ``fps`` carried in
    ``metadata['source_video']``, else estimate from the episode timestamps,
    else default to 30. The chosen value is just a denominator for the
    fallback clip's PTS, which the reader matches back exactly.
    """
    src = (metadata or {}).get("source_video")
    if isinstance(src, dict) and src.get("fps"):
        try:
            fps = float(src["fps"])
            if fps > 0:
                return fps
        except (TypeError, ValueError):
            pass
    return 30.0


def _rerun_metadata_props(metadata: dict[str, Any] | None, extra_keys: tuple[str, ...] = ()) -> dict[str, str]:
    """Project an episode-metadata dict onto Rerun-loggable string props.

    Always picks up the JSON-serialized ``config``, ``seed``, ``source``,
    and ``outcome`` fields when present. ``extra_keys`` adds further
    scalar fields (looked up with ``metadata.get(k)``, str-cast,
    none-filtered) so eval can surface ``iteration`` / ``episode_index``
    / ``burn_in`` without forking a second helper.
    """
    props: dict[str, str] = {}
    if metadata is None:
        return props
    cfg_json = _serialize_metadata_config(metadata)
    if cfg_json is not None:
        props["config"] = cfg_json
    if "seed" in metadata:
        props["seed"] = str(int(metadata["seed"]))
    if metadata.get("source") is not None:
        props["source"] = str(metadata["source"])
    if metadata.get("outcome") is not None:
        props["outcome"] = str(metadata["outcome"])
    for k in extra_keys:
        v = metadata.get(k) if metadata else None
        if v is not None:
            props[k] = str(v)
    return props


class _BaseEpisodeWriter:
    """Shared base — owns the output directory and per-recording paths."""

    file_extension: str = ""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, prefix: str, name_suffix: str) -> Path:
        # Re-mkdir on every lookup: long-running training shares `data/`
        # with cleanup scripts and other runs, so the dir created in
        # __init__ may be gone by the time we write the next episode.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"{prefix}-{name_suffix}.{self.file_extension}"


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------


class HDF5EpisodeWriter(_BaseEpisodeWriter):
    """Writes both sim and eval episodes to HDF5 files.

    Sim episodes go to ``output_dir/episode-{suffix}.hdf5`` with the
    structure::

        images          (T, H, W, 3)    uint8
        joint_action    (T, n_joints)   float32   — optional, present when the caller supplies it
        ee_action       (T, 7)          float32   — optional, present when the caller supplies it
        rewards         (T,)            float32
        timestamps      (T,)            float32
        joint_qpos      (T, n_joints)   float32
        ee_pos          (T, 3)          float32
        ee_quat         (T, 4)          float32
        object_xy       (T, 2)          float32
        object_uv       (T, 2)          float32   — optional, pixel centroid of the target mask
        segmentation/                              — optional, present iff env produced masks
            target      (T, H, W)       bool
            goal        (T, H, W)       bool
            arm         (T, H, W)       bool      — optional, union of robot-arm geoms
            background  (T, H, W)       bool      — optional
        metadata/
            config      scalar string (JSON)
            seed        scalar int64
            source      scalar string
            outcome     scalar string
            goal_xy     (2,) float32
            mat_centre  (2,) float32     — optional, populated by the "real" preset

    Files may carry one or both action streams. The on-disk recording is
    action-space-agnostic; :class:`EpisodeProcessor` picks which view to
    use at load time based on ``config.env.act_mode``.

    Eval episodes go to ``output_dir/eval-{suffix}.hdf5`` with::

        obs                (T, ...)            float32 — encoder input
        recon_posterior    (T, ...)
        recon_prior        (T, ...)
        diff_posterior     (T, ...) float32 — obs - recon_posterior (image only)
        diff_prior         (T, ...) float32 — obs - recon_prior     (image only)
        h_posterior        (T, deter_dim)
        s_posterior        (T, stoch_dim)
        h_prior            (T, deter_dim)
        s_prior            (T, stoch_dim)
        metadata/
            burn_in        int
            iteration      int (if present)
            episode_index  int (if present)
            seed           int (if present)
    """

    file_extension = "hdf5"

    @staticmethod
    def _write_fields_hdf5(f, episode: Episode) -> None:
        """Persist the episode's per-step fields, dispatching on each field's
        :class:`FieldKind` to reproduce the historical layout exactly:

          * ``IMAGE`` / ``MASK`` → gzip-compressed (``images`` / under
            ``segmentation/<name>`` with the ``segmentation_`` prefix stripped),
          * ``SCALAR`` / ``ACTION`` → an uncompressed dataset named by the field
            (``reward``→``rewards`` etc. via :data:`_HDF5_DATASET_ALIAS`),
          * ``OVERLAY`` (Rerun-only) and ``SCRATCH`` → not written.

        Field kinds replace the old hardcoded key lists, so a new persisted
        field needs no edit here — it's written per its declared kind.
        """
        seg_grp = None
        for name, fld in episode.persisted():
            if fld.kind is FieldKind.OVERLAY:
                continue  # Rerun-only; HDF5 has never carried the mesh overlay.
            if fld.kind is FieldKind.IMAGE:
                f.create_dataset(
                    _HDF5_DATASET_ALIAS.get(name, name),
                    data=np.asarray(fld.value, dtype=np.uint8),
                    compression="gzip", compression_opts=4)
            elif fld.kind is FieldKind.MASK:
                if seg_grp is None:
                    seg_grp = f.create_group("segmentation")
                seg_grp.create_dataset(
                    name[len("segmentation_"):],
                    data=np.asarray(fld.value, dtype=bool),
                    compression="gzip", compression_opts=4)
            else:  # SCALAR / ACTION
                f.create_dataset(
                    _HDF5_DATASET_ALIAS.get(name, name),
                    data=np.asarray(fld.value))

    def write_episode(
        self,
        episode: Episode | dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None,
        *,
        name_suffix: str,
    ) -> Path:
        episode = _as_episode(episode)
        actions = _collect_actions(episode)
        T = next(iter(actions.values())).shape[0]
        if T == 0:
            raise ValueError("episode must not be empty")

        ep_path = self._path_for(EPISODE_FILENAME_PREFIX, name_suffix)
        with h5py.File(ep_path, "w") as f:
            self._write_fields_hdf5(f, episode)

            meta_grp = f.create_group("metadata")
            if metadata is not None:
                cfg = _serialize_metadata_config(metadata)
                if cfg is not None:
                    meta_grp.create_dataset("config", data=cfg)
                seed = metadata.get("seed", -1)
                meta_grp.create_dataset("seed", data=int(seed))
                source = metadata.get("source", "sim")
                meta_grp.create_dataset("source", data=str(source))
                outcome = metadata.get("outcome")
                if outcome is not None:
                    meta_grp.create_dataset("outcome", data=str(outcome))
                goal_xy = metadata.get("goal_xy")
                if goal_xy is not None:
                    meta_grp.create_dataset(
                        "goal_xy",
                        data=np.asarray(goal_xy, dtype=np.float32))
                mat_centre = metadata.get("mat_centre")
                if mat_centre is not None:
                    meta_grp.create_dataset(
                        "mat_centre",
                        data=np.asarray(mat_centre, dtype=np.float32))
                tags = metadata.get("tags")
                if tags:
                    meta_grp.create_dataset(
                        "tags",
                        data=np.asarray([str(t) for t in tags], dtype=h5py.string_dtype()))
                # Arm-to-world calibration block (stored as a single JSON
                # string for forward-compatibility with the brief's data
                # contract). Written by import-lerobot when an episode-
                # config supplies T_world_arm for the dataset.
                t_world_arm = metadata.get("T_world_arm")
                if t_world_arm is not None:
                    blob = {
                        "T_world_arm":     t_world_arm,
                        "arm_diagnostics": metadata.get("arm_diagnostics"),
                        "arm_metadata":    metadata.get("arm_metadata"),
                    }
                    meta_grp.create_dataset(
                        "T_world_arm", data=json.dumps(blob))

        return ep_path

    def write_eval_episode(
        self,
        episode: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        *,
        name_suffix: str,
    ) -> Path:
        ep_path = self._path_for(EVAL_EPISODE_FILENAME_PREFIX, name_suffix)
        obs             = episode["obs"]
        recon_posterior = episode["recon_posterior"]
        recon_prior     = episode["recon_prior"]
        with h5py.File(ep_path, "w") as f:
            # Obs and reconstructions are per-component dicts: write each
            # component as its own dataset under ``obs/<name>``,
            # ``recon_posterior/<name>``, ``recon_prior/<name>``. Image
            # components additionally get a per-pixel diff dataset.
            obs_grp     = f.create_group("obs")
            rpost_grp   = f.create_group("recon_posterior")
            rprior_grp  = f.create_group("recon_prior")
            for name, arr in obs.items():
                obs_arr   = np.asarray(arr,                  dtype=np.float32)
                rpost_arr = np.asarray(recon_posterior[name], dtype=np.float32)
                rprior_arr = np.asarray(recon_prior[name],   dtype=np.float32)
                obs_grp.create_dataset(name, data=obs_arr)
                rpost_grp.create_dataset(name, data=rpost_arr,
                                         compression="gzip", compression_opts=4)
                rprior_grp.create_dataset(name, data=rprior_arr,
                                          compression="gzip", compression_opts=4)
                if obs_arr.ndim == 4:  # image component
                    f.create_dataset(f"diff_posterior/{name}",
                                     data=_pixel_diff(obs_arr, rpost_arr).astype(np.float32),
                                     compression="gzip", compression_opts=4)
                    f.create_dataset(f"diff_prior/{name}",
                                     data=_pixel_diff(obs_arr, rprior_arr).astype(np.float32),
                                     compression="gzip", compression_opts=4)
            f.create_dataset("h_posterior", data=np.asarray(episode["h_posterior"], dtype=np.float32))
            f.create_dataset("s_posterior", data=np.asarray(episode["s_posterior"], dtype=np.float32))
            f.create_dataset("h_prior",     data=np.asarray(episode["h_prior"],     dtype=np.float32))
            f.create_dataset("s_prior",     data=np.asarray(episode["s_prior"],     dtype=np.float32))

            meta_grp = f.create_group("metadata")
            if metadata is not None:
                cfg = _serialize_metadata_config(metadata)
                if cfg is not None:
                    meta_grp.create_dataset("config", data=cfg)
                for k in ("burn_in", "iteration", "episode_index", "seed"):
                    if metadata.get(k) is not None:
                        meta_grp.create_dataset(k, data=int(metadata[k]))
                if metadata.get("source") is not None:
                    meta_grp.create_dataset("source", data=str(metadata["source"]))
        return ep_path


# ---------------------------------------------------------------------------
# Rerun writer
# ---------------------------------------------------------------------------


class RerunEpisodeWriter(_BaseEpisodeWriter):
    """Writes both sim and eval episodes to Rerun ``.rrd`` files.

    See :class:`HDF5EpisodeWriter` for a description of the two payload
    schemas. Sim recordings log ``camera/image``, ``reward``, the action
    column, and per-step joint/EE/object signals; eval recordings log
    raw + processed obs, posterior + prior reconstructions, and the
    latent trajectories.

    Metadata is logged as static text docs on the ``metadata/`` entity
    so it surfaces in the viewer.
    """

    file_extension = "rrd"

    def __init__(self, output_dir: str) -> None:
        import rerun as rr  # noqa: F401 — fail fast if rerun is missing
        super().__init__(output_dir)

    def _new_recording(self, recording_id: str):
        import rerun as rr
        return rr.RecordingStream(
            application_id="chuck_dreamer",
            recording_id=recording_id,
        )

    @staticmethod
    def _log_metadata(rec, props: dict[str, str]) -> None:
        import rerun as rr
        for key, value in props.items():
            rec.log(f"metadata/{key}", rr.TextDocument(value), static=True)

    @staticmethod
    def _build_camera_video(
        images: np.ndarray, metadata: dict[str, Any] | None, T: int,
    ) -> tuple[bytes, str, list[float]]:
        """Encode this episode's RGB frames as a single video blob.

        Prefers a lossless *stream copy* of the source LeRobot MP4 window
        when ``metadata['source_video']`` names one (set by the importer):
        ``{"path", "from_ts", "to_ts"}``. That reuses the original encoded
        packets, so it adds no second generation of lossy compression and is
        the smallest option. Falls back to a bit-exact re-encode of
        ``images`` when there's no source video (sim episodes) or the copied
        window doesn't line up with the episode's ``T`` frames.

        Returns ``(video_bytes, media_type, frame_pts)`` — the per-step
        clip-relative presentation timestamps the caller logs as
        ``VideoFrameReference`` cells.
        """
        from .rerun_video import reencode_mp4, stream_copy_window

        src = (metadata or {}).get("source_video")
        if isinstance(src, dict) and src.get("path"):
            try:
                vbytes, media, pts = stream_copy_window(
                    src["path"], float(src["from_ts"]), float(src["to_ts"]))
                if len(pts) == T:
                    return vbytes, media, pts
                logger.warning(
                    "source video window yielded %d frames, episode has %d; "
                    "re-encoding instead", len(pts), T)
            except Exception as exc:  # noqa: BLE001 — any decode/remux failure → re-encode
                logger.warning("stream copy of %s failed (%s); re-encoding",
                               src.get("path"), exc)

        fps = _video_fps(metadata, images, T)
        return reencode_mp4(images[:T], fps=fps)

    # Distinct translucent colour per overlay layer, so several masks layered
    # over the video stay visually separable. RGBA, alpha ~43%.
    _SEG_RGBA = {
        "target":     (0, 200, 255, 110),
        "goal":       (255, 0, 200, 110),
        "arm":        (255, 160, 0, 110),
        "background": (120, 120, 120, 90),
        "mesh":       (0, 255, 0, 110),
    }

    @staticmethod
    def _png_bytes(image: np.ndarray) -> bytes:
        """PNG-encode one ``(H, W)`` grayscale or ``(H, W, 4)`` RGBA image
        (lossless) for an ``rr.EncodedImage``."""
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
        return buf.getvalue()

    @classmethod
    def _mask_rgba(cls, mask: np.ndarray, name: str) -> np.ndarray:
        """Colour a binary ``(H, W)`` mask into a transparent ``(H, W, 4)``
        RGBA frame: the target's colour where the mask is set, fully
        transparent elsewhere — so it composites over the video as a tinted
        overlay rather than rendering as an all-black ``{0,1}`` image."""
        m = np.asarray(mask) > 0
        rgba = np.zeros((*m.shape[:2], 4), dtype=np.uint8)
        rgba[m] = cls._SEG_RGBA.get(name, (0, 200, 255, 110))
        return rgba

    # Per-step fields the Rerun writer consumes *structurally* rather than as
    # generic scalar entities: ``image`` becomes the video, ``timestamp`` drives
    # the time axis. Everything else SCALAR/ACTION is logged as ``rr.Scalars``.
    _RERUN_STRUCTURAL = ("image", "timestamp")

    def write_episode(
        self,
        episode: Episode | dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None,
        *,
        name_suffix: str,
    ) -> Path:
        episode = _as_episode(episode)
        actions = _collect_actions(episode)
        T = next(iter(actions.values())).shape[0]
        if T == 0:
            raise ValueError("episode must not be empty")

        import rerun as rr

        ep_path = self._path_for(EPISODE_FILENAME_PREFIX, name_suffix)
        rec = self._new_recording(f"{EPISODE_FILENAME_PREFIX}-{name_suffix}")
        rec.save(str(ep_path))

        props = _rerun_metadata_props(metadata)
        if metadata is not None and metadata.get("goal_xy") is not None:
            goal = np.asarray(metadata["goal_xy"], dtype=np.float32)
            props["goal_xy"] = f"[{float(goal[0])}, {float(goal[1])}]"
        if metadata is not None and metadata.get("mat_centre") is not None:
            mat = np.asarray(metadata["mat_centre"], dtype=np.float32)
            props["mat_centre"] = f"[{float(mat[0])}, {float(mat[1])}]"
        if metadata is not None and metadata.get("tags"):
            # Tags ride as a comma-joined string since Rerun metadata props are
            # dict[str, str]. The reader splits on commas to recover the tuple.
            props["tags"] = ",".join(str(t) for t in metadata["tags"])
        if metadata is not None and metadata.get("T_world_arm") is not None:
            props["T_world_arm"] = json.dumps({
                "T_world_arm":     metadata["T_world_arm"],
                "arm_diagnostics": metadata.get("arm_diagnostics"),
                "arm_metadata":    metadata.get("arm_metadata"),
            })
        self._log_metadata(rec, props)

        timestamps = np.asarray(episode["timestamp"], dtype=np.float32)
        images     = np.asarray(episode["image"],     dtype=np.uint8)

        # Bucket the persisted per-step fields by kind. SCALAR/ACTION fields
        # (minus the structural ones) become ``rr.Scalars`` entities named by
        # the field; MASK fields become tinted PNG overlays under camera/seg/;
        # the OVERLAY field becomes camera/mesh_overlay. This dispatch replaces
        # the old hardcoded per-field blocks, so a new field is logged per its
        # declared kind with no edit here.
        scalar_fields: dict[str, np.ndarray] = {}
        seg_masks: dict[str, np.ndarray] = {}
        mesh_overlay: np.ndarray | None = None
        for name, fld in episode.persisted():
            if fld.kind in (FieldKind.SCALAR, FieldKind.ACTION):
                if name in self._RERUN_STRUCTURAL:
                    continue
                scalar_fields[name] = np.asarray(fld.value)
            elif fld.kind is FieldKind.MASK:
                seg_masks[name[len("segmentation_"):]] = np.asarray(fld.value)
            elif fld.kind is FieldKind.OVERLAY:
                mesh_overlay = np.asarray(fld.value)

        # RGB rides as one encoded video blob (logged once, static) plus a
        # per-step VideoFrameReference, instead of a raw rr.Image per frame —
        # this is the bulk of the old multi-GB .rrd size.
        video_bytes, media_type, frame_pts = self._build_camera_video(
            images, metadata, T)
        rec.log("camera/video",
                rr.AssetVideo(contents=video_bytes, media_type=media_type),
                static=True)

        for i in range(T):
            rec.set_time("step", sequence=i)
            rec.set_time("time", duration=float(timestamps[i]))

            rec.log("camera/video", rr.VideoFrameReference(seconds=frame_pts[i]))
            for name, arr in scalar_fields.items():
                cell = arr[i]
                if cell.ndim:
                    value: Any = cell.tolist()
                elif cell.dtype == np.bool_:
                    value = int(cell)  # match the old int() cast for gap flags
                else:
                    value = cell.item()
                rec.log(name, rr.Scalars(value))

            # Masks ride as transparent RGBA PNGs (tinted where set) so they
            # composite over the video, rather than raw SegmentationImage. Masks
            # whose name has no dedicated colour fall back to the target tint.
            for name, mask in seg_masks.items():
                rec.log(f"camera/seg/{name}",
                        rr.EncodedImage(
                            contents=self._png_bytes(self._mask_rgba(mask[i], name)),
                            media_type="image/png"))

            # Fitted-mesh silhouette tinted into a transparent PNG over video.
            if mesh_overlay is not None:
                rec.log("camera/mesh_overlay",
                        rr.EncodedImage(
                            contents=self._png_bytes(
                                self._mask_rgba(mesh_overlay[i], "mesh")),
                            media_type="image/png"))

        # Force all chunks to disk before returning. The file sink flushes on
        # a background thread; without this an immediately-following read (or
        # the next episode's heavy video re-encode starving the flush) can
        # see a half-written, entity-less recording.
        rec.flush(timeout_sec=30.0)
        return ep_path

    def write_eval_episode(
        self,
        episode: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        *,
        name_suffix: str,
    ) -> Path:
        import rerun as rr

        recording_id = f"{EVAL_EPISODE_FILENAME_PREFIX}-{name_suffix}"
        ep_path      = self._path_for(EVAL_EPISODE_FILENAME_PREFIX, name_suffix)
        rec          = self._new_recording(recording_id)
        rec.save(str(ep_path))

        self._log_metadata(rec, _rerun_metadata_props(
            metadata, extra_keys=("iteration", "episode_index", "burn_in"),
        ))

        obs             = episode["obs"]
        recon_posterior = episode["recon_posterior"]
        recon_prior     = episode["recon_prior"]
        h_posterior     = np.asarray(episode["h_posterior"],     dtype=np.float32)
        s_posterior     = np.asarray(episode["s_posterior"],     dtype=np.float32)
        h_prior         = np.asarray(episode["h_prior"],         dtype=np.float32)
        s_prior         = np.asarray(episode["s_prior"],         dtype=np.float32)

        # Materialize each component as float32 ndarrays up front so the
        # per-step loop stays tight. Image components are logged with the
        # de-normalize + diff dance; vector components fan out into
        # per-dim Scalars on the ``<name>/`` entity prefix.
        component_arrays: dict[str, dict[str, np.ndarray]] = {}
        for name, arr in obs.items():
            component_arrays[name] = {
                "obs":             np.asarray(arr,                  dtype=np.float32),
                "recon_posterior": np.asarray(recon_posterior[name], dtype=np.float32),
                "recon_prior":     np.asarray(recon_prior[name],     dtype=np.float32),
            }
        T = next(iter(component_arrays.values()))["obs"].shape[0]

        for i in range(T):
            rec.set_time("step", sequence=i)

            for name, arrays in component_arrays.items():
                obs_i    = arrays["obs"][i]
                rpost_i  = arrays["recon_posterior"][i]
                rprior_i = arrays["recon_prior"][i]
                if obs_i.ndim == 3:  # image: (H, W, C)
                    rec.log(f"{name}/obs",             rr.Image(_coerce_image_for_log(obs_i)))
                    rec.log(f"{name}/recon/posterior", rr.Image(_denormalize_recon_image(rpost_i)))
                    rec.log(f"{name}/recon/prior",     rr.Image(_denormalize_recon_image(rprior_i)))
                    rec.log(f"{name}/recon/diff_posterior",
                            rr.Image(_diff_to_uint8(_pixel_diff(obs_i, rpost_i))))
                    rec.log(f"{name}/recon/diff_prior",
                            rr.Image(_diff_to_uint8(_pixel_diff(obs_i, rprior_i))))
                else:
                    rec.log(f"{name}/obs",             rr.Scalars(obs_i.tolist()))
                    rec.log(f"{name}/recon/posterior", rr.Scalars(rpost_i.tolist()))
                    rec.log(f"{name}/recon/prior",     rr.Scalars(rprior_i.tolist()))

            rec.log("latent/h",       rr.Scalars(h_posterior[i].tolist()))
            rec.log("latent/s",       rr.Scalars(s_posterior[i].tolist()))
            rec.log("latent/h_prior", rr.Scalars(h_prior[i].tolist()))
            rec.log("latent/s_prior", rr.Scalars(s_prior[i].tolist()))

        return ep_path
