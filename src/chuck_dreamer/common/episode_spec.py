"""Episode-selection spec: the ``dataset#episodes:frames`` CLI syntax.

A single :class:`EpisodeSpec` both *parses* the CLI token and *applies*
the selection to a list of per-episode metadata objects, so any command
that takes a ``DATASET[#EPISODES[:FRAMES]]`` argument can share one
primitive. Each consumer supplies its own metadata reader (the importer
reads rich per-episode slices; the calibration CLIs read episode bounds)
and uses :meth:`EpisodeSpec.select_episodes` to keep only the requested
episodes — the spec never knows the concrete metadata type.

Spec grammar (ABNF-ish):

  source     = dataset-id [ "#" episodes [ ":" frames ] ]
  episodes   = "all" / range-list   ; default: all
  frames     = range-list           ; default: auto-sample by length
  range-list = range *("," range)
  range      = INT [ "-" INT ] [ ":" INT ]   ; stride only valid with N-M

Both episode and frame ranges are **inclusive on both ends**. Strides
default to 1.

**Frame indices are episode-relative.** ``5-80`` means "frames 5 through
80 of each selected episode" (offset 5 from that episode's start, not
global indices 5..80). Ranges that exceed an episode's length are
silently clipped to its bounds, so ``0-10000`` is a safe "all frames"
spelling.

Examples:

  user/cal                    -> all episodes, frames auto-sampled
  user/cal#0                  -> just episode 0, frames auto-sampled
  user/cal#0,2-4              -> episodes 0, 2, 3, 4 (auto-sampled)
  user/cal#all:0-200          -> all eps, frames 0..200 of each
  user/cal#1:0-200:5          -> episode 1, frames 0,5,...,200
  user/cal#0-1:0,50-100:10    -> eps 0+1, frame 0 + frames 50,60,...,100 of each
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

import click

_T = TypeVar("_T")


@dataclass
class EpisodeSpec:
  """A parsed ``dataset#episodes[:frames]`` selection.

  ``episodes`` is None when the user didn't specify an episode filter
  (= "all"); :meth:`select_episodes` then keeps every episode. ``frames``
  is None when no frame filter was given (the resolver auto-samples);
  otherwise it holds the episode-relative frame indices.
  """
  dataset_id: str
  episodes:   list[int] | None = None
  frames:     list[int] | None = None

  # ---- parsing -----------------------------------------------------------
  @classmethod
  def parse(cls, raw: str, *,
            allow_frames: bool = True,
            command: str | None = None) -> "EpisodeSpec":
    """Parse one CLI token, surfacing errors as ``click.ClickException``.

    ``allow_frames=False`` rejects any spec carrying a ``:FRAMES`` portion
    (for commands that operate on whole episodes — import-lerobot,
    augment-keyframes). ``command`` only names the offending command in
    the rejection message.
    """
    try:
      spec = _parse(raw)
    except ValueError as e:
      raise click.ClickException(str(e))
    if not allow_frames and spec.frames is not None:
      who = command or "this command"
      raise click.ClickException(
        f"{who} doesn't accept the :FRAMES portion of the spec "
        f"(only #EPISODES is meaningful). Got: {raw!r}")
    return spec

  @classmethod
  def parse_many(cls, raws: tuple[str, ...] | list[str], *,
                 allow_frames: bool = True,
                 command: str | None = None) -> list["EpisodeSpec"]:
    return [cls.parse(r, allow_frames=allow_frames, command=command)
            for r in raws]

  # ---- selection ---------------------------------------------------------
  @property
  def episode_filter(self) -> set[int] | None:
    """``episodes`` as a set, or ``None`` for "all episodes"."""
    return set(self.episodes) if self.episodes is not None else None

  def select_episodes(
    self, items: list[_T], *, key: Callable[[_T], int],
  ) -> list[_T]:
    """Keep only ``items`` whose ``key(item)`` is in this spec's
    ``episodes`` selection, preserving input order.

    Returns the list unchanged when ``episodes`` is None ("all"). ``key``
    maps each item to its episode index — e.g.
    ``lambda s: s.episode_index`` for the importer's episode slices, or
    ``lambda t: t[0]`` for ``(episode_idx, bounds)`` tuples.
    """
    if self.episodes is None:
      return list(items)
    keep = set(self.episodes)
    return [it for it in items if key(it) in keep]


# ---------------------------------------------------------------------------
# Grammar parsing (raises ValueError; EpisodeSpec.parse wraps as ClickException)
# ---------------------------------------------------------------------------
def _parse(raw: str) -> EpisodeSpec:
  raw = (raw or "").strip()
  if not raw:
    raise ValueError("empty episode spec.")
  if "#" not in raw:
    return EpisodeSpec(dataset_id=raw)

  dataset_id, _, rest = raw.partition("#")
  if not dataset_id:
    raise ValueError(f"{raw!r}: empty dataset id before '#'.")
  episodes_str, sep, frames_str = rest.partition(":")
  episodes_str = episodes_str.strip()
  frames_str   = frames_str.strip() if sep else None

  episodes = _parse_episodes(raw, episodes_str)
  frames   = _parse_range_list(raw, frames_str, what="frames") if frames_str else None
  return EpisodeSpec(dataset_id=dataset_id, episodes=episodes, frames=frames)


def _parse_episodes(raw: str, s: str) -> list[int] | None:
  s = s.strip().lower()
  if s in ("", "all"):
    return None
  return _parse_range_list(raw, s, what="episodes")


def _parse_range_list(raw: str, s: str, what: str) -> list[int]:
  """Parse 'N' / 'N-M' / 'N-M:STRIDE' tokens (comma-separated)."""
  out: list[int] = []
  seen: set[int] = set()
  for tok in s.split(","):
    tok = tok.strip()
    if not tok:
      continue
    out.extend(_parse_one_range(raw, tok, what=what, seen=seen))
  if not out:
    raise ValueError(f"{raw!r}: no usable {what} after parsing.")
  return out


def _parse_one_range(raw: str, tok: str, what: str, seen: set[int]) -> list[int]:
  # Optional ':STRIDE' suffix.
  stride = 1
  if ":" in tok:
    range_part, _, stride_part = tok.partition(":")
    try:
      stride = int(stride_part.strip())
    except ValueError:
      raise ValueError(
        f"{raw!r}: stride {stride_part!r} after {range_part!r} in {what} "
        f"is not an integer.")
    if stride < 1:
      raise ValueError(f"{raw!r}: stride {stride} in {what} must be >= 1.")
    tok = range_part.strip()
  if "-" in tok:
    a_str, _, b_str = tok.partition("-")
    try:
      a, b = int(a_str.strip()), int(b_str.strip())
    except ValueError:
      raise ValueError(f"{raw!r}: range {tok!r} in {what} is not integers.")
    if a < 0 or b < 0:
      raise ValueError(f"{raw!r}: {what} range {a}-{b} has a negative value.")
    if b < a:
      raise ValueError(f"{raw!r}: {what} range {a}-{b} is reversed.")
    new = list(range(a, b + 1, stride))
  else:
    if stride != 1:
      raise ValueError(f"{raw!r}: stride only allowed with a N-M range, not {tok!r}.")
    try:
      n = int(tok)
    except ValueError:
      raise ValueError(f"{raw!r}: {what} token {tok!r} is not an integer.")
    if n < 0:
      raise ValueError(f"{raw!r}: {what} value {n} is negative.")
    new = [n]
  out: list[int] = []
  for v in new:
    if v in seen:
      continue
    seen.add(v)
    out.append(v)
  return out
