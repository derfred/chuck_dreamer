"""Sanity-check plots for joint extractions.

This stage does no FK and no transforms. The visualizer's job is to
make obvious mistakes obvious: a mis-labelled touch, an episode with
heavy drift, or a touch whose joint vector lands far from the rest of
its label group across recordings.

Outputs three figures to ``<cache>/<slug>/joint_extraction/``:

- ``visualize_joints.png``    — per-joint stripplot, x=touch label,
                                y=joint value (one panel per joint).
                                Each dot = one episode.
- ``visualize_spread.png``    — per-joint spread (max-min) bar chart,
                                with the drift threshold line.
- ``visualize_consistency.png`` — pairwise distance heatmap between
                                episodes (in joint-vector space).

The figures cover all datasets passed to ``render`` so cross-recording
consistency is visible at a glance.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from .extract_joints import (
  DRIFT_THRESHOLD_DEG,
  DatasetExtraction,
  POSITIONING_JOINTS,
  joint_extraction_dir,
)

logger = logging.getLogger(__name__)


def render(
  results: list[DatasetExtraction],
  out_dir: Path | str,
) -> list[Path]:
  """Render all three figures into ``out_dir``. Returns the file paths
  in render order.
  """
  import matplotlib.pyplot as plt

  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  flat = _flatten(results)
  if not flat:
    logger.warning("visualize: no episodes to plot.")
    return []

  out: list[Path] = []
  out.append(_plot_joint_stripplot(flat, out_dir / "visualize_joints.png"))
  out.append(_plot_spread_bars(flat, out_dir / "visualize_spread.png"))
  out.append(_plot_consistency_heatmap(flat, out_dir / "visualize_consistency.png"))
  return out


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------


def _flatten(results: list[DatasetExtraction]) -> list[dict]:
  """Flatten a list of DatasetExtractions into per-episode dicts."""
  flat: list[dict] = []
  for r in results:
    for ext in r.episodes:
      flat.append({
        "dataset_id":    r.dataset_id,
        "dataset_short": _short(r.dataset_id),
        "episode_index": ext.episode_index,
        "touch_label":   ext.touch_label,
        "joints":        ext.joints,
        "spread":        ext.spread,
        "flags":         ext.flags,
      })
  return flat


def _short(dataset_id: str) -> str:
  """Compact label for legends: keep the last path segment."""
  return dataset_id.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Plot 1: per-joint stripplot
# ---------------------------------------------------------------------------


def _plot_joint_stripplot(flat: list[dict], path: Path) -> Path:
  """One subplot per positioning joint; x=touch label, y=joint value.

  Episodes from the same touch label should cluster vertically;
  outliers point at mislabels or drift.
  """
  import matplotlib.pyplot as plt
  import matplotlib.patches as mpatches

  labels = sorted({e["touch_label"] for e in flat}, key=_label_sort_key)
  label_to_x = {lab: i for i, lab in enumerate(labels)}
  datasets = sorted({e["dataset_short"] for e in flat})
  ds_to_color = {ds: c for ds, c in zip(datasets, _palette(len(datasets)))}

  n_j = len(POSITIONING_JOINTS)
  fig, axes = plt.subplots(n_j, 1, figsize=(max(7, 1.2 * len(labels) + 2), 2.0 * n_j),
                           sharex=True)
  if n_j == 1:
    axes = [axes]

  rng = np.random.default_rng(0)
  for ax, joint in zip(axes, POSITIONING_JOINTS):
    for e in flat:
      x = label_to_x[e["touch_label"]] + rng.uniform(-0.12, 0.12)
      y = e["joints"][joint]
      ax.scatter(x, y, color=ds_to_color[e["dataset_short"]],
                 s=60, edgecolor="black", linewidth=0.4,
                 marker="x" if e["flags"] else "o", zorder=3)
      if e["flags"]:
        ax.annotate(f"ep{e['episode_index']}", (x, y), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color="firebrick")
    ax.set_ylabel(f"{joint}\n(deg)", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

  axes[-1].set_xticks(list(label_to_x.values()))
  axes[-1].set_xticklabels(list(label_to_x.keys()), rotation=20, ha="right")
  axes[-1].set_xlabel("touch label")

  handles = [mpatches.Patch(color=ds_to_color[ds], label=ds) for ds in datasets]
  handles.append(mpatches.Patch(color="firebrick", label="× = flagged (drift / short)"))
  fig.legend(handles=handles, fontsize=7, loc="upper right",
             bbox_to_anchor=(0.995, 0.99))
  fig.suptitle("Per-joint median across touches (one dot = one episode)",
               y=0.995, fontsize=10)
  fig.tight_layout(rect=(0, 0, 1, 0.96))
  fig.savefig(path, dpi=140)
  plt.close(fig)
  return path


# ---------------------------------------------------------------------------
# Plot 2: spread bars
# ---------------------------------------------------------------------------


def _plot_spread_bars(flat: list[dict], path: Path) -> Path:
  """Bar chart of (max-min) per joint across each episode, with the
  drift threshold marked.
  """
  import matplotlib.pyplot as plt

  rows = sorted(flat, key=lambda e: (e["dataset_short"], e["episode_index"]))
  x = np.arange(len(rows))
  width = 0.16
  n_j = len(POSITIONING_JOINTS)

  fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(rows) + 2), 4.5))
  for j, joint in enumerate(POSITIONING_JOINTS):
    vals = [r["spread"][joint] for r in rows]
    ax.bar(x + (j - n_j / 2) * width + width / 2, vals,
           width=width, label=joint)

  ax.axhline(DRIFT_THRESHOLD_DEG, color="firebrick", linestyle="--",
             linewidth=1.0,
             label=f"drift threshold ({DRIFT_THRESHOLD_DEG:.2f} deg)")
  ax.set_xticks(x)
  ax.set_xticklabels(
    [f"{r['dataset_short']}\nep{r['episode_index']} ({r['touch_label']})"
     for r in rows],
    rotation=35, ha="right", fontsize=7,
  )
  ax.set_ylabel("joint spread within episode (deg)")
  ax.set_title("Episode-level joint spread (max-min)")
  ax.grid(True, alpha=0.3, axis="y")
  ax.legend(fontsize=7, ncols=3, loc="upper left")
  fig.tight_layout()
  fig.savefig(path, dpi=140)
  plt.close(fig)
  return path


# ---------------------------------------------------------------------------
# Plot 3: consistency heatmap
# ---------------------------------------------------------------------------


def _plot_consistency_heatmap(flat: list[dict], path: Path) -> Path:
  """L2 distance heatmap between every pair of episode joint vectors.

  Episodes labelled the same should be near each other; bright off-block
  cells flag mismatches.
  """
  import matplotlib.pyplot as plt

  rows = sorted(flat, key=lambda e: (e["touch_label"],
                                      e["dataset_short"],
                                      e["episode_index"]))
  vectors = np.stack([
    np.array([r["joints"][j] for j in POSITIONING_JOINTS], dtype=np.float64)
    for r in rows
  ], axis=0)
  diff = vectors[:, None, :] - vectors[None, :, :]
  dist = np.linalg.norm(diff, axis=-1)

  fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(rows) + 2),
                                  max(6, 0.45 * len(rows) + 2)))
  im = ax.imshow(dist, cmap="viridis_r")
  ticks = [f"{r['touch_label']}: {r['dataset_short']}/ep{r['episode_index']}"
           for r in rows]
  ax.set_xticks(np.arange(len(rows)))
  ax.set_yticks(np.arange(len(rows)))
  ax.set_xticklabels(ticks, rotation=80, ha="right", fontsize=7)
  ax.set_yticklabels(ticks, fontsize=7)

  # Group separators between adjacent rows of different touch labels.
  for i in range(1, len(rows)):
    if rows[i]["touch_label"] != rows[i - 1]["touch_label"]:
      ax.axhline(i - 0.5, color="white", linewidth=0.8)
      ax.axvline(i - 0.5, color="white", linewidth=0.8)

  fig.colorbar(im, ax=ax, label="L2 joint-vector distance (deg)",
               fraction=0.046, pad=0.04)
  fig.suptitle("Inter-episode joint-vector distance (sorted by touch label)",
               y=0.995, fontsize=11)
  fig.tight_layout()
  fig.savefig(path, dpi=140)
  plt.close(fig)
  return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LABEL_ORDER = (
  "left-far", "left-near", "left-middle",
  "origin",
  "right-middle", "right-near", "right-far",
)


def _label_sort_key(label: str) -> tuple[int, str]:
  """Sort touch labels in a sensible left->right order, unknowns last."""
  try:
    return (_LABEL_ORDER.index(label), label)
  except ValueError:
    return (len(_LABEL_ORDER), label)


def _palette(n: int) -> list:
  import matplotlib.pyplot as plt
  cmap = plt.get_cmap("tab10")
  return [cmap(i % 10) for i in range(n)]
