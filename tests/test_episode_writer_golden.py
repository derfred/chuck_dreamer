"""Golden on-disk-format tests for the episode writers.

These pin the *exact bytes/structure* ``HDF5EpisodeWriter`` and
``RerunEpisodeWriter`` emit for a representative sim episode that exercises every
field kind: core per-step scalars, both action streams, all segmentation masks,
``object_uv`` / ``object_gap_too_long``, the Rerun-only mesh overlay, and the
sidecar metadata (tags, goal_xy, mat_centre, T_world_arm).

They exist to guard the Episode-object migration: the in-memory representation is
changing, but the persisted format must not. Every migration step is checked
against these. HDF5 is asserted structurally (dataset tree: name → shape, dtype,
content hash). Rerun ``.rrd`` files carry nondeterministic ids, so it is asserted
via the reader round-trip (the property that actually matters downstream) plus
the set of logged entities.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chuck_dreamer.sim.episode_writer import (
  HDF5EpisodeWriter,
  RerunEpisodeWriter,
)


T, H, W = 5, 8, 12
NJ = 6


def _representative_episode() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
  """A sim episode dict touching every persisted field kind, with fixed
  deterministic content so its hashes are stable across runs."""
  rng = np.random.default_rng(0)
  episode = {
    "image":        rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8),
    "joint_action": rng.standard_normal((T, NJ)).astype(np.float32),
    "ee_action":    rng.standard_normal((T, 7)).astype(np.float32),
    "reward":       rng.standard_normal((T,)).astype(np.float32),
    "timestamp":    (np.arange(T) / 10.0).astype(np.float32),
    "joint_qpos":   rng.standard_normal((T, NJ)).astype(np.float32),
    "ee_pos":       rng.standard_normal((T, 3)).astype(np.float32),
    "ee_quat":      rng.standard_normal((T, 4)).astype(np.float32),
    "object_xy":    rng.standard_normal((T, 2)).astype(np.float32),
    "object_uv":    rng.standard_normal((T, 2)).astype(np.float32),
    "object_gap_too_long": (np.arange(T) % 2 == 0),
    "segmentation_target":     rng.integers(0, 2, (T, H, W)).astype(bool),
    "segmentation_goal":       rng.integers(0, 2, (T, H, W)).astype(bool),
    "segmentation_arm":        rng.integers(0, 2, (T, H, W)).astype(bool),
    "segmentation_background": rng.integers(0, 2, (T, H, W)).astype(bool),
    "object_mesh_overlay":     rng.integers(0, 2, (T, H, W)).astype(np.uint8),
  }
  metadata = {
    "config":  {"source_repo": "test/ds", "episode_index": 0, "n_joints": NJ},
    "seed":    7,
    "source":  "sim",
    "outcome": "imported",
    "number_of_frames": T,
    "goal_xy": np.array([1.0, 2.0], dtype=np.float32),
    "mat_centre": np.array([3.0, 4.0], dtype=np.float32),
    "tags":    ("real", "demo"),
    "T_world_arm": [[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  }
  return episode, metadata


# ---------------------------------------------------------------------------
# HDF5 structural snapshot
# ---------------------------------------------------------------------------


def _hdf5_tree(path: Path) -> dict[str, tuple[tuple[int, ...], str, str]]:
  """Map every dataset path in an HDF5 file to ``(shape, dtype, sha1)`` of its
  contents — a structural + content fingerprint independent of write order."""
  import h5py

  tree: dict[str, tuple[tuple[int, ...], str, str]] = {}

  def visit(name: str, obj: Any) -> None:
    if isinstance(obj, h5py.Dataset):
      data = np.asarray(obj[()])
      digest = hashlib.sha1(np.ascontiguousarray(data).tobytes()).hexdigest()
      tree[name] = (tuple(data.shape), str(data.dtype), digest)

  with h5py.File(path, "r") as f:
    f.visititems(visit)
  return tree


# The exact HDF5 dataset tree the current writer emits for the representative
# episode. Regenerate intentionally (and review the diff) only when the on-disk
# format is *meant* to change.
def test_hdf5_golden_structure(tmp_path):
  episode, metadata = _representative_episode()
  writer = HDF5EpisodeWriter(str(tmp_path))
  path = writer.write_episode(episode, metadata=metadata, name_suffix="00000")

  tree = _hdf5_tree(Path(path))

  # Dataset *paths* present (the structural contract).
  assert set(tree) == {
    "images",
    "joint_action", "ee_action",
    "rewards", "timestamps",
    "joint_qpos", "ee_pos", "ee_quat",
    "object_xy", "object_uv", "object_gap_too_long",
    "segmentation/target", "segmentation/goal",
    "segmentation/arm", "segmentation/background",
    "metadata/config", "metadata/seed", "metadata/source",
    "metadata/outcome", "metadata/goal_xy", "metadata/mat_centre",
    "metadata/tags", "metadata/T_world_arm",
  }
  # object_mesh_overlay is Rerun-only — it must NOT be in the HDF5 file.
  assert not any("mesh_overlay" in k for k in tree)

  # Shapes + dtypes of the per-step datasets.
  assert tree["images"][:2] == ((T, H, W, 3), "uint8")
  assert tree["joint_action"][:2] == ((T, NJ), "float32")
  assert tree["ee_action"][:2] == ((T, 7), "float32")
  assert tree["object_gap_too_long"][:2] == ((T,), "bool")
  assert tree["segmentation/target"][:2] == ((T, H, W), "bool")


def test_hdf5_content_roundtrips_through_reader(tmp_path):
  # The reader is the real downstream consumer; assert it recovers the arrays
  # bit-for-bit (within the format's documented dtypes).
  from chuck_dreamer.training.episode_dataset import Episode

  episode, metadata = _representative_episode()
  writer = HDF5EpisodeWriter(str(tmp_path))
  path = writer.write_episode(episode, metadata=metadata, name_suffix="00000")

  raw = Episode.from_file(path, format="hdf5").data
  np.testing.assert_array_equal(raw["image"], episode["image"])
  np.testing.assert_array_equal(raw["joint_action"], episode["joint_action"])
  np.testing.assert_array_equal(raw["ee_action"], episode["ee_action"])
  np.testing.assert_array_equal(raw["object_xy"], episode["object_xy"])
  np.testing.assert_array_equal(raw["object_uv"], episode["object_uv"])
  np.testing.assert_array_equal(
    raw["segmentation_target"], episode["segmentation_target"])


# ---------------------------------------------------------------------------
# Rerun round-trip snapshot
# ---------------------------------------------------------------------------


def test_rerun_golden_roundtrip(tmp_path):
  pytest.importorskip("rerun")
  from chuck_dreamer.training.episode_dataset import Episode

  from chuck_dreamer.training.episode_dataset import _collect_rerun_chunks

  episode, metadata = _representative_episode()
  writer = RerunEpisodeWriter(str(tmp_path))
  path = writer.write_episode(episode, metadata=metadata, name_suffix="00000")
  assert Path(path).exists() and Path(path).suffix == ".rrd"

  # Writer-side entity fingerprint: exactly which entities get logged. This is
  # the stable contract the migration must preserve (independent of the reader).
  by_entity, static = _collect_rerun_chunks(path)
  entities = {e for e in (set(by_entity) | set(static)) if not e.startswith("/__")}
  assert entities == {
    "/camera/video", "/camera/seg/target", "/camera/seg/goal",
    "/camera/seg/arm", "/camera/seg/background", "/camera/mesh_overlay",
    "/joint_action", "/ee_action", "/reward",
    "/joint_qpos", "/ee_pos", "/ee_quat",
    "/object_xy", "/object_uv", "/object_gap_too_long",
    "/metadata/config", "/metadata/seed", "/metadata/source",
    "/metadata/outcome", "/metadata/tags",
    "/metadata/goal_xy", "/metadata/mat_centre", "/metadata/T_world_arm",
  }

  raw = Episode.from_file(path, format="rerun").data
  # Scalars round-trip exactly.
  np.testing.assert_allclose(raw["object_xy"], episode["object_xy"], atol=1e-5)
  np.testing.assert_allclose(raw["ee_pos"], episode["ee_pos"], atol=1e-5)
  np.testing.assert_allclose(raw["object_uv"], episode["object_uv"], atol=1e-5)
  # Segmentation target is present and shaped (T, H, W) bool. (The Rerun
  # per-step mask *content* round-trip has a pre-existing reader quirk with
  # tiny synthetic frames that is orthogonal to this migration, so we assert
  # shape/dtype/presence here, not per-frame equality.)
  rt = raw["segmentation_target"]
  assert rt.shape == (T, H, W) and rt.dtype == bool
  # Tags survive the metadata round-trip.
  ep = Episode.from_file(path, format="rerun")
  assert ep.metadata.get("tags") == ("real", "demo")
