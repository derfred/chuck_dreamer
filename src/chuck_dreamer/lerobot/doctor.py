"""The import doctor (spec §10.3): declaration- and store-driven.

Walks the same node lists ``import-lerobot run`` would execute and derives
each node's *leaf* inputs from its declarations — an input is a leaf iff no
earlier node in the list produces it. Every leaf is then checked against
its actual source: annotation and cached artifacts against the artifact
store (per resolved scope key), assets against the filesystem / manifest.
There are no per-node ``requirements()`` lists to maintain: adding a node
with declared inputs surfaces its prerequisites here automatically, and a
passing doctor means a run cannot fail on a missing input (only on node
internals).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import click

if TYPE_CHECKING:
  from .pipeline import Run


@dataclass(frozen=True)
class LeafStatus:
  """One resolved leaf input: where it was looked up and how to produce it."""
  artifact: str
  scope_desc: str            # human-readable resolved scope key
  ok: bool
  remediation: str = ""


def _check_resolver(resolve: Callable[[], Any], artifact: str,
                    scope_desc: str) -> LeafStatus:
  """Leaf check for a store-backed input: satisfied iff the RunContext
  resolver returns; the resolver's MissingArtifact message *is* the
  remediation (it names the producing command)."""
  try:
    resolve()
    return LeafStatus(artifact, scope_desc, ok=True)
  except Exception as e:  # noqa: BLE001 - missing/corrupt both mean "missing"
    return LeafStatus(artifact, scope_desc, ok=False, remediation=str(e))


def _check_leaf(name: str, ctx: Any, episode_index: int) -> LeafStatus | None:
  """Check one declared leaf input at its resolved scope key.

  Returns ``None`` for dataset-supplied inputs (base tracks and the video
  slice) — those exist by construction of the run."""
  from chuck_dreamer.common import FK_MODEL_PATH

  ds = ctx.source_repo
  if name == "fk_model":
    return LeafStatus(
      name, str(FK_MODEL_PATH), ok=FK_MODEL_PATH.exists(),
      remediation="restore assets/urdf/SO101/so101_new_calib.urdf from git")
  if name == "object_mesh":
    mesh = Path(ctx.ol_cfg.mesh_path)
    return LeafStatus(
      name, str(mesh), ok=mesh.exists(),
      remediation="set object_localization.mesh_path in configs/default.yaml "
                  "to a valid .obj/.ply")
  if name == "sam2_weights":
    from chuck_dreamer.common import asset_manifest_entry
    try:
      asset_manifest_entry("sam2_weights")
      return LeafStatus(name, "assets/manifest.json", ok=True)
    except Exception as e:  # noqa: BLE001 - any failure means unpinned
      return LeafStatus(name, "assets/manifest.json", ok=False,
                        remediation=f"fix assets/manifest.json: {e}")
  if name == "intrinsics":
    cam = (ctx.dataset_config() or {}).get("camera_id") or "?"
    return _check_resolver(ctx.resolve_intrinsics, name, f"camera/{cam}")
  if name == "extrinsics":
    return _check_resolver(ctx.resolve_extrinsics, name, f"dataset/{ds}")
  if name == "table_to_arm":
    return _check_resolver(ctx.resolve_table_to_arm, name, f"dataset/{ds}")
  if name == "object_prompts":
    ok = 0 in ctx.keyframe_prompts(episode_index)
    return LeafStatus(
      name, f"dataset/{ds} episode {episode_index}", ok=ok,
      remediation=f"uv run python main.py import-lerobot prompt-episodes {ds}")
  # Dataset-supplied: base tracks (image / timestamp / joint_qpos, …) and
  # the episode's own video slice.
  return None


def doctor_run(run: "Run") -> bool:
  """Report every missing leaf input for the run's selected episodes.

  Returns ``True`` iff everything the enabled pipeline needs is present.
  """
  params = run.params
  click.echo(f"Doctor check for import-lerobot {run.spec.dataset_id}")
  click.echo(
    f"  with_ee_pos={params.get('with_ee_pos', False)}  "
    f"with_object_pose={params.get('with_object_pose', False)}  "
    f"with_table_frame={params.get('with_table_frame', False)}")

  try:
    slices = run.slices
  except Exception as e:  # noqa: BLE001 - surface metadata-read failures as doctor output
    click.echo(click.style(f"  ERROR  could not read episodes: {e}", fg="red"))
    return False
  episode_indices = [sl.episode_index for sl in slices]
  click.echo(f"  episodes: {episode_indices or 'none'}")

  missing: list[LeafStatus] = []
  seen: set[tuple[str, str]] = set()

  for ep_idx in episode_indices:
    nodes = run.pipeline(ep_idx)
    produced: set[str] = set()
    for node in nodes:
      for decl in node.inputs:
        if decl.name in produced:
          continue
        status = _check_leaf(decl.name, run.ctx, ep_idx)
        if status is None:
          continue
        key = (status.artifact, status.scope_desc)
        if key in seen:
          continue
        seen.add(key)
        if status.ok:
          click.echo(click.style("  OK    ", fg="green")
                     + f"{status.artifact} @ {status.scope_desc}")
        else:
          click.echo(click.style("  MISS  ", fg="red")
                     + f"{status.artifact} @ {status.scope_desc}")
          click.echo(f"        {status.remediation}")
          missing.append(status)
      produced.update(spec.name for spec in node.outputs)

  if missing:
    click.echo(click.style(
      f"\nDoctor: {len(missing)} missing artifact(s).", fg="red"))
    return False
  click.echo(click.style("\nDoctor: all required artifacts present.", fg="green"))
  return True
