"""Per-episode stage pipeline and the import-run it belongs to.

A :class:`Run` ties an :class:`EpisodeSpec` to one set of importer flags.
It owns the shared, run-level :class:`RunContext` (so the FK evaluator,
object-localization runtime, estimator and smoother stay cached across
episodes), resolves the dataset's on-disk root + selected episode slices,
and is the single factory for per-episode :class:`Pipeline` objects. Both
the importer (which calls ``stage.apply``) and ``import-lerobot --doctor``
(which calls ``stage.requirements``) drive a :class:`Run` so the
context/root/slice/pipeline wiring lives in exactly one place.

A :class:`Pipeline` is the ordered, dependency-resolved stage list for a
single episode. Each stage is bound to the context at construction
(``ctx`` is a stage init param, not an ``apply``/``requirements``
argument), so iterating the pipeline and calling
``stage.apply(episode, metadata)`` is all the importer does per episode.
The doctor iterates the same pipeline per episode and prints each stage's
:meth:`Stage.requirements`, which may vary per episode (e.g. the frame-0
prompt a segmentation stage needs is episode-specific).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from .base import RunContext, Stage
from .registry import build_registry, resolve_stages

if TYPE_CHECKING:
  from chuck_dreamer.common.episode_spec import EpisodeSlice, EpisodeSpec


class Run:
  """One import run: an :class:`EpisodeSpec` plus the importer flags.

  Holds the shared :class:`RunContext` and is the sole factory for the
  per-episode :class:`Pipeline`, so the importer and the doctor share the
  same context/root/slice resolution and flag→stage wiring.
  """

  def __init__(self, spec: EpisodeSpec, *,
               with_ee_pos: bool = True, with_object_pose: bool = True) -> None:
    self.spec = spec
    self.with_ee_pos = with_ee_pos
    self.with_object_pose = with_object_pose
    self.ctx = RunContext(source_repo=spec.dataset_id)
    # A directory dataset_id is an on-disk LeRobot root; HF repo ids aren't.
    self.local_root: Path | None = (
      Path(spec.dataset_id) if Path(spec.dataset_id).is_dir() else None)
    self._registry = build_registry(self.ctx)

  def read_slices(self, video_key: str | None = None) -> tuple[list[EpisodeSlice], str]:
    """Resolve this spec's selected episode slices + the video key, reading
    only the offline ``meta/`` sidecars (no video touch)."""
    return self.spec.read_episodes(video_key=video_key, root=self.local_root)

  def pipeline(self, episode_index: int) -> Pipeline:
    """The dependency-ordered :class:`Pipeline` for one episode, sharing
    this run's context and stage registry."""
    return Pipeline(
      self.ctx, episode_index,
      with_ee_pos=self.with_ee_pos, with_object_pose=self.with_object_pose,
      registry=self._registry)


class Pipeline:
  """The ordered stage list for a single episode.

  ``ctx`` is the run-level shared context (built once, reused across every
  episode's pipeline so its caches survive). ``episode_index`` identifies
  the episode this pipeline runs on, letting stages surface episode-
  specific requirements. The per-episode mask cache is reset on
  construction so a stage never sees a prior episode's masks.
  """

  def __init__(self, ctx: RunContext, episode_index: int = 0, *,
               with_ee_pos: bool = True, with_object_pose: bool = True,
               registry: dict[str, Stage] | None = None) -> None:
    self.ctx = ctx
    self.episode_index = episode_index
    ctx.episode_index = episode_index
    ctx.masks.clear()

    enabled: set[str] = set()
    if with_ee_pos:
      enabled.add("ee_pos")
    if with_object_pose:
      enabled.add("object_pose")
    self._stages = resolve_stages(
      enabled, ctx,
      registry=registry if registry is not None else build_registry(ctx))

  def __iter__(self) -> Iterator[Stage]:
    return iter(self._stages)

  def __len__(self) -> int:
    return len(self._stages)
