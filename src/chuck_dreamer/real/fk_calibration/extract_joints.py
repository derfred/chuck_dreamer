"""Extract one representative joint vector per touch episode.

Each LeRobot episode in our touch-recordings is a single fiducial-mat
touch: the arm settles on one labelled point and holds (state is
near-constant for the whole episode). For every (dataset, episode) we
emit one resting joint vector ``q`` to feed the FK calibration step.

Output shape — one JSON per episode, written to
``<cache_dir>/<dataset_slug>/joint_extraction/episode_<NNN>.json``::

    {
      "dataset_id":    "user/dataset",
      "episode_index": 5,
      "touch_label":   "left-middle",
      "n_frames":      174,
      "joint_unit":    "deg",
      "stat":          "median",
      "joints":        {"shoulder_pan": -37.36, ...},
      "spread":        {"shoulder_pan": 0.0, ...},
      "flags":         ["short_segment", "drift:wrist_flex"]
    }

A sibling ``metadata.json`` in the same ``joint_extraction/`` folder
records the run-level summary (config used, joint order, total touches,
warnings). This module owns the data-side logic; CLI wiring lives in
``chuck_dreamer.real.fk_calibration.cli``.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..object_localization.types import dataset_cache_dir, dataset_slug

logger = logging.getLogger(__name__)


POSITIONING_JOINTS: tuple[str, ...] = (
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
)

# SO-101 LeRobot recordings store ``observation.state`` in degrees. The
# brief talks in radians (0.05 rad ~ 2.9 deg); we surface both so the
# warning threshold semantics match the brief.
JOINT_UNIT = "deg"
DRIFT_THRESHOLD_RAD = 0.05
DRIFT_THRESHOLD_DEG = math.degrees(DRIFT_THRESHOLD_RAD)
SHORT_SEGMENT_FRAMES = 10


@dataclass
class EpisodeExtraction:
  """One touch's resting joint vector + per-joint diagnostics."""
  dataset_id:    str
  episode_index: int
  touch_label:   str
  n_frames:      int
  joints:        dict[str, float]   # median per positioning joint, in deg
  spread:        dict[str, float]   # (max-min) per positioning joint, in deg
  flags:         list[str] = field(default_factory=list)

  def to_json(self) -> dict[str, Any]:
    return {
      "dataset_id":    self.dataset_id,
      "episode_index": int(self.episode_index),
      "touch_label":   self.touch_label,
      "n_frames":      int(self.n_frames),
      "joint_unit":    JOINT_UNIT,
      "stat":          "median",
      "joints":        {k: float(v) for k, v in self.joints.items()},
      "spread":        {k: float(v) for k, v in self.spread.items()},
      "flags":         list(self.flags),
    }


@dataclass
class DatasetExtraction:
  """All per-episode extractions for one dataset + run metadata."""
  dataset_id: str
  episodes:   list[EpisodeExtraction]
  run_info:   dict[str, Any]


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def load_episode_config(path: Path | str) -> dict[str, dict[int, str]]:
  """Read an episode-config file (JSON) and normalize to
  ``{dataset_id: {episode_index: touch_label}}``.

  Schema (one JSON object)::

      {
        "prototypes": {
          "three_touch": ["left-middle", "origin", "right-middle"],
          "seven_touch": ["left-middle", "origin", "right-middle",
                          "left-far", "left-near",
                          "right-near", "right-far"]
        },
        "datasets": [
          {
            "dataset_id": "user/foo",
            "prototype":  "three_touch",
            "episodes":   [5, 6, 7]
          },
          {
            "dataset_id": "user/odd",
            "prototype":  "three_touch",
            "episodes":   [2, 3, 9],
            "overrides":  {"9": "origin"}
          }
        ]
      }

  ``episodes`` must have the same length as the referenced prototype
  (paired positionally). ``overrides`` patches individual episode
  labels post-prototype. Dataset order is preserved.
  """
  p = Path(path)
  blob = json.loads(p.read_text())
  if not isinstance(blob, dict):
    raise ValueError(f"{p}: top-level must be a JSON object")
  protos = blob.get("prototypes")
  datasets = blob.get("datasets")
  if not isinstance(protos, dict) or not protos:
    raise ValueError(f"{p}: 'prototypes' must be a non-empty object.")
  if not isinstance(datasets, list) or not datasets:
    raise ValueError(f"{p}: 'datasets' must be a non-empty array.")

  proto_norm: dict[str, list[str]] = {}
  for name, labels in protos.items():
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
      raise ValueError(
        f"{p}: prototype {name!r} must be a list of label strings.")
    proto_norm[name] = list(labels)

  out: dict[str, dict[int, str]] = {}
  for i, entry in enumerate(datasets):
    if not isinstance(entry, dict):
      raise ValueError(f"{p}: datasets[{i}] must be an object.")
    did = entry.get("dataset_id")
    proto_name = entry.get("prototype")
    eps = entry.get("episodes")
    if not isinstance(did, str) or not did:
      raise ValueError(f"{p}: datasets[{i}].dataset_id missing.")
    if proto_name not in proto_norm:
      raise ValueError(
        f"{p}: datasets[{i}] ({did}) references unknown prototype "
        f"{proto_name!r}; known: {sorted(proto_norm)}")
    if not isinstance(eps, list) or not all(isinstance(e, int) for e in eps):
      raise ValueError(
        f"{p}: datasets[{i}] ({did}).episodes must be a list of ints.")
    labels = proto_norm[proto_name]
    if len(eps) != len(labels):
      raise ValueError(
        f"{p}: datasets[{i}] ({did}) has {len(eps)} episodes but "
        f"prototype {proto_name!r} has {len(labels)} labels.")
    mapping = {int(ep): lab for ep, lab in zip(eps, labels)}
    for ep_key, lab in entry.get("overrides", {}).items():
      try:
        ep_i = int(ep_key)
      except (TypeError, ValueError):
        raise ValueError(
          f"{p}: datasets[{i}] ({did}).overrides key {ep_key!r} is not an int.")
      if ep_i not in mapping:
        raise ValueError(
          f"{p}: datasets[{i}] ({did}).overrides references episode "
          f"{ep_i} which is not in episodes={eps}.")
      mapping[ep_i] = str(lab)
    if did in out:
      raise ValueError(f"{p}: dataset_id {did!r} appears more than once.")
    out[did] = mapping
  return out


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_episodes_table(dataset_id: str) -> pd.DataFrame:
  """Frames table for ``dataset_id``, sourced from cached LeRobot
  parquet without spinning up the video pipeline.

  Returns a dataframe with columns at least: ``observation.state``,
  ``episode_index``, ``frame_index``.
  """
  from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore

  meta = LeRobotDatasetMetadata(dataset_id)
  root = Path(meta.root)
  data_files = sorted((root / "data").rglob("*.parquet"))
  if not data_files:
    raise FileNotFoundError(
      f"{dataset_id}: no parquet files under {root / 'data'}; is the "
      f"dataset fully downloaded?")
  return pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)


def _state_joint_names(dataset_id: str) -> list[str]:
  """Return the canonical joint-name list from the dataset's
  ``observation.state`` feature spec, with ``.pos`` suffixes stripped.
  """
  from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore

  meta = LeRobotDatasetMetadata(dataset_id)
  feat = meta.features.get("observation.state", {})
  names: list[str] = list(feat.get("names", []))
  return [n[:-4] if n.endswith(".pos") else n for n in names]


# ---------------------------------------------------------------------------
# Per-episode extraction
# ---------------------------------------------------------------------------


def _episode_state_array(df: pd.DataFrame, episode_index: int) -> np.ndarray:
  """Stack ``observation.state`` for one episode into ``(T, D) float64``.

  ``observation.state`` is stored as a per-row object (list/ndarray);
  ``np.stack`` collapses it into a clean 2-D array.
  """
  sub = df[df["episode_index"] == episode_index]
  if len(sub) == 0:
    return np.zeros((0, 0), dtype=np.float64)
  return np.stack([np.asarray(s, dtype=np.float64)
                   for s in sub["observation.state"].values], axis=0)


def extract_episode(
  df: pd.DataFrame,
  dataset_id: str,
  episode_index: int,
  touch_label: str,
  joint_names: list[str],
) -> EpisodeExtraction:
  """Compute median joints + spreads for one episode. Pure function."""
  state = _episode_state_array(df, episode_index)
  if state.size == 0:
    raise ValueError(
      f"{dataset_id} episode {episode_index}: no frames found in dataset")

  # Index into the full state vector by name. The dataset's state is
  # (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
  # gripper); we only keep the 5 positioning joints.
  try:
    idxs = [joint_names.index(name) for name in POSITIONING_JOINTS]
  except ValueError as e:
    raise ValueError(
      f"{dataset_id}: state feature names {joint_names} are missing one "
      f"of the required positioning joints {POSITIONING_JOINTS}: {e}")
  positioning = state[:, idxs]   # (T, 5)

  med = np.median(positioning, axis=0)
  spread = positioning.max(axis=0) - positioning.min(axis=0)

  joints = {name: float(med[i]) for i, name in enumerate(POSITIONING_JOINTS)}
  spread_d = {name: float(spread[i]) for i, name in enumerate(POSITIONING_JOINTS)}

  flags: list[str] = []
  if positioning.shape[0] < SHORT_SEGMENT_FRAMES:
    flags.append("short_segment")
  for name, s in spread_d.items():
    if s > DRIFT_THRESHOLD_DEG:
      flags.append(f"drift:{name}")

  return EpisodeExtraction(
    dataset_id    = dataset_id,
    episode_index = int(episode_index),
    touch_label   = touch_label,
    n_frames      = int(positioning.shape[0]),
    joints        = joints,
    spread        = spread_d,
    flags         = flags,
  )


# ---------------------------------------------------------------------------
# Dataset-level orchestration
# ---------------------------------------------------------------------------


def extract_dataset(
  dataset_id: str,
  episode_to_label: Mapping[int, str],
) -> DatasetExtraction:
  """Process every (episode, label) in ``episode_to_label``. Raises
  ``ValueError`` on missing-episode config bugs.
  """
  df = _load_episodes_table(dataset_id)
  joint_names = _state_joint_names(dataset_id)
  if not joint_names:
    raise ValueError(f"{dataset_id}: state feature has no joint names")

  available = set(df["episode_index"].astype(int).unique().tolist())
  requested = set(int(k) for k in episode_to_label.keys())
  missing = sorted(requested - available)
  if missing:
    raise ValueError(
      f"{dataset_id}: episode_config references episodes {missing} "
      f"that don't exist in the dataset (available: {sorted(available)})")

  extractions: list[EpisodeExtraction] = []
  for ep in sorted(requested):
    label = episode_to_label[ep]
    ext = extract_episode(df, dataset_id, ep, label, joint_names)
    extractions.append(ext)
    logger.info(
      "  ep %3d  label=%-13s  n=%d  spread_deg=%s%s",
      ep, label, ext.n_frames,
      ", ".join(f"{k}={v:.3f}" for k, v in ext.spread.items()),
      f"  flags={ext.flags}" if ext.flags else "",
    )

  run_info = {
    "dataset_id":            dataset_id,
    "joint_names":           list(POSITIONING_JOINTS),
    "state_feature_names":   joint_names,
    "stat":                  "median",
    "joint_unit":            JOINT_UNIT,
    "drift_threshold_rad":   DRIFT_THRESHOLD_RAD,
    "drift_threshold_deg":   DRIFT_THRESHOLD_DEG,
    "short_segment_frames":  SHORT_SEGMENT_FRAMES,
    "episode_count":         len(extractions),
    "episode_indices":       [e.episode_index for e in extractions],
    "touch_labels":          sorted({e.touch_label for e in extractions}),
    "timestamp":             datetime.now(timezone.utc).isoformat(),
  }
  return DatasetExtraction(
    dataset_id = dataset_id,
    episodes   = extractions,
    run_info   = run_info,
  )


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def joint_extraction_dir(cache_dir: Path | str, dataset_id: str) -> Path:
  return dataset_cache_dir(cache_dir, dataset_id) / "joint_extraction"


def episode_json_path(cache_dir: Path | str, dataset_id: str,
                      episode_index: int) -> Path:
  return (joint_extraction_dir(cache_dir, dataset_id)
          / f"episode_{int(episode_index):03d}.json")


def write_dataset_extraction(
  cache_dir: Path | str,
  result: DatasetExtraction,
) -> list[Path]:
  """Write one JSON per episode + a ``metadata.json`` sidecar. Returns
  the list of files written (episodes first, metadata last).
  """
  out_dir = joint_extraction_dir(cache_dir, result.dataset_id)
  out_dir.mkdir(parents=True, exist_ok=True)
  written: list[Path] = []
  for ext in result.episodes:
    p = episode_json_path(cache_dir, result.dataset_id, ext.episode_index)
    p.write_text(json.dumps(ext.to_json(), indent=2))
    written.append(p)
  meta_p = out_dir / "metadata.json"
  meta_p.write_text(json.dumps(result.run_info, indent=2))
  written.append(meta_p)
  return written


def read_dataset_extraction(
  cache_dir: Path | str,
  dataset_id: str,
) -> DatasetExtraction:
  """Reverse of ``write_dataset_extraction`` — for the visualizer."""
  out_dir = joint_extraction_dir(cache_dir, dataset_id)
  meta_p = out_dir / "metadata.json"
  if not meta_p.exists():
    raise FileNotFoundError(
      f"{meta_p} not found — run `extract-joints {dataset_id}` first.")
  run_info = json.loads(meta_p.read_text())
  episodes: list[EpisodeExtraction] = []
  for ep_idx in run_info.get("episode_indices", []):
    p = episode_json_path(cache_dir, dataset_id, int(ep_idx))
    blob = json.loads(p.read_text())
    episodes.append(EpisodeExtraction(
      dataset_id    = blob["dataset_id"],
      episode_index = int(blob["episode_index"]),
      touch_label   = blob["touch_label"],
      n_frames      = int(blob["n_frames"]),
      joints        = {k: float(v) for k, v in blob["joints"].items()},
      spread        = {k: float(v) for k, v in blob["spread"].items()},
      flags         = list(blob.get("flags", [])),
    ))
  return DatasetExtraction(
    dataset_id = dataset_id,
    episodes   = episodes,
    run_info   = run_info,
  )
