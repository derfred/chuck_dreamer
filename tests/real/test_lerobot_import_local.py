"""Exercise the local-path branch of :func:`chuck_dreamer.sim.lerobot_import.import_dataset`.

Reuses the synthetic-dataset writers from ``tests/test_lerobot_import.py``
but skips the HF monkey-patching: when ``repo_id`` is a directory the
importer uses the local resolver and never touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

av = pytest.importorskip("av")

from chuck_dreamer.sim import lerobot_import as li  # noqa: E402


VIDEO_KEY = "observation.images.wrist"
FPS = 10
H, W = 32, 32


def _write_mp4(path: Path, n_frames: int) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  container = av.open(str(path), mode="w")
  stream = container.add_stream("mpeg4", rate=FPS)
  stream.width = W
  stream.height = H
  stream.pix_fmt = "yuv420p"
  for i in range(n_frames):
    img = np.full((H, W, 3), i * 8, dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    for pkt in stream.encode(frame):
      container.mux(pkt)
  for pkt in stream.encode():
    container.mux(pkt)
  container.close()


def _write_data_parquet(path: Path, n_frames: int) -> None:
  actions = [[float(i + 0.1 * j) for j in range(6)] for i in range(n_frames)]
  states  = [[float(i + 0.01 * j) for j in range(6)] for i in range(n_frames)]
  ts      = [float(i) / FPS for i in range(n_frames)]
  tbl = pa.table({
    "action":            pa.array(actions, type=pa.list_(pa.float32(), 6)),
    "observation.state": pa.array(states,  type=pa.list_(pa.float32(), 6)),
    "timestamp":         pa.array(ts,      type=pa.float32()),
    "frame_index":       pa.array(list(range(n_frames)), type=pa.int64()),
    "episode_index":     pa.array([0] * n_frames, type=pa.int64()),
  })
  path.parent.mkdir(parents=True, exist_ok=True)
  pq.write_table(tbl, path)


def _write_episodes_parquet(path: Path, episodes: list[dict]) -> None:
  cols: dict[str, list] = {
    "episode_index":                       [e["episode_index"] for e in episodes],
    "tasks":                               [e["tasks"]         for e in episodes],
    "length":                              [e["length"]        for e in episodes],
    "data/chunk_index":                    [0] * len(episodes),
    "data/file_index":                     [0] * len(episodes),
    "dataset_from_index":                  [e["data_from"]     for e in episodes],
    "dataset_to_index":                    [e["data_to"]       for e in episodes],
    f"videos/{VIDEO_KEY}/chunk_index":     [0] * len(episodes),
    f"videos/{VIDEO_KEY}/file_index":      [0] * len(episodes),
    f"videos/{VIDEO_KEY}/from_timestamp":  [e["v_from"]        for e in episodes],
    f"videos/{VIDEO_KEY}/to_timestamp":    [e["v_to"]          for e in episodes],
  }
  schema = pa.schema([
    ("episode_index",                       pa.int64()),
    ("tasks",                               pa.list_(pa.string())),
    ("length",                              pa.int64()),
    ("data/chunk_index",                    pa.int64()),
    ("data/file_index",                     pa.int64()),
    ("dataset_from_index",                  pa.int64()),
    ("dataset_to_index",                    pa.int64()),
    (f"videos/{VIDEO_KEY}/chunk_index",     pa.int64()),
    (f"videos/{VIDEO_KEY}/file_index",      pa.int64()),
    (f"videos/{VIDEO_KEY}/from_timestamp",  pa.float32()),
    (f"videos/{VIDEO_KEY}/to_timestamp",    pa.float32()),
  ])
  path.parent.mkdir(parents=True, exist_ok=True)
  pq.write_table(pa.table(cols, schema=schema), path)


def _write_info(path: Path) -> None:
  info = {
    "codebase_version": "v3.0",
    "fps": FPS,
    "features": {
      "action":            {"dtype": "float32", "shape": [6]},
      "observation.state": {"dtype": "float32", "shape": [6]},
      "timestamp":         {"dtype": "float32", "shape": [1]},
      VIDEO_KEY: {
        "dtype": "video", "shape": [H, W, 3],
        "info":  {"video.height": H, "video.width": W, "video.fps": FPS},
      },
    },
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as f:
    json.dump(info, f)


@pytest.fixture
def local_dataset(tmp_path: Path) -> Path:
  root = tmp_path / "ds"
  root.mkdir()
  _write_mp4(root / f"videos/{VIDEO_KEY}/chunk-000/file-000.mp4", 6)
  _write_data_parquet(root / "data/chunk-000/file-000.parquet", 6)
  _write_episodes_parquet(
    root / "meta/episodes/chunk-000/file-000.parquet",
    [{
      "episode_index": 0, "tasks": ["push"], "length": 6,
      "data_from": 0, "data_to": 6, "v_from": 0.0, "v_to": 6 / FPS,
    }],
  )
  _write_info(root / "meta/info.json")
  return root


def test_import_local_dataset(local_dataset, tmp_path, monkeypatch):
  # Hard-fail if hf_hub_download is called — the local path must not need it.
  def bomb(*_a, **_kw):
    raise AssertionError("hf_hub_download must not be called for a local dataset")
  monkeypatch.setattr(li, "hf_hub_download", bomb)

  out_dir = tmp_path / "episodes"
  results = list(li.import_dataset(str(local_dataset), str(out_dir), format="hdf5"))
  assert [r[0] for r in results] == [0]
  out_path = results[0][1]
  assert out_path.exists()
  assert out_path.name == "episode-00000.hdf5"


def test_import_local_dataset_missing_file_raises(local_dataset, tmp_path):
  # Remove the data parquet — the importer should raise via the local resolver.
  (local_dataset / "data" / "chunk-000" / "file-000.parquet").unlink()
  out_dir = tmp_path / "episodes"
  with pytest.raises(FileNotFoundError):
    list(li.import_dataset(str(local_dataset), str(out_dir), format="hdf5"))
