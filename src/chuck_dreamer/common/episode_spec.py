"""Episode-selection spec: the ``dataset#episodes:frames`` CLI syntax.

A single :class:`EpisodeSpec` parses the CLI token and resolves it to the
dataset's episodes: :meth:`EpisodeSpec.read_episodes` reads the LeRobot
metadata and returns one :class:`EpisodeSlice` per *selected* episode, so
any command taking a ``DATASET[#EPISODES[:FRAMES]]`` argument shares one
primitive for both parsing and resolution.

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
from pathlib import Path

import click


@dataclass
class EpisodeSlice:
  """One episode's metadata: frame bounds, video-decode window, and the
  chunk/file coordinates for both the data parquet and the video file."""
  episode_index: int
  data_from: int
  data_to: int                    # exclusive
  video_from_ts: float
  video_to_ts: float              # exclusive
  data_chunk: int
  data_file: int
  video_chunk: int
  video_file: int
  task: str
  length: int

  @property
  def bounds(self) -> tuple[int, int]:
    """``(from_idx, to_idx_exclusive)`` for this episode."""
    return self.data_from, self.data_to


@dataclass
class EpisodeSpec:
  """A parsed ``dataset#episodes[:frames]`` selection.

  ``episodes`` is None when the user didn't specify an episode filter
  (= "all"); :meth:`read_episodes` then returns every episode. ``frames``
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

  def _select_episodes(self, slices: list[EpisodeSlice]) -> list[EpisodeSlice]:
    """Keep only ``slices`` whose ``episode_index`` is in this spec's
    ``episodes`` selection, preserving input order. Returns the list
    unchanged when ``episodes`` is None ("all"). Used by
    :meth:`read_episodes`."""
    if self.episodes is None:
      return slices
    keep = set(self.episodes)
    return [s for s in slices if s.episode_index in keep]

  # ---- resolution to LeRobot episode metadata ----------------------------
  def read_episodes(
    self, *,
    video_key: str | None = None,
    root: str | Path | None = None,
  ) -> tuple[list[EpisodeSlice], str]:
    """Read this spec's dataset and return ``(slices, resolved_video_key)``:
    one :class:`EpisodeSlice` per **selected** episode (sorted by index,
    filtered by this spec's ``episodes``) plus the video key the slices'
    video fields were populated from.

    Reads only the parquet sidecars under ``meta/`` via
    ``LeRobotDatasetMetadata`` (no video touch). ``video_key`` selects
    which ``observation.images.*`` feature's chunk/file/timestamp columns
    populate the video fields (defaults to the dataset's first video key);
    ``root`` points at an on-disk dataset directory.

    The LeRobot import is deferred to call time so that merely parsing a
    spec string (the CLIs' common case) doesn't pull in the LeRobot stack.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore

    meta = LeRobotDatasetMetadata(self.dataset_id, root=root)
    vkey = _resolve_video_key(meta, self.dataset_id, video_key)

    vcol_chunk = f"videos/{vkey}/chunk_index"
    vcol_file = f"videos/{vkey}/file_index"
    vcol_from = f"videos/{vkey}/from_timestamp"
    vcol_to = f"videos/{vkey}/to_timestamp"

    slices: list[EpisodeSlice] = []
    for row in meta.episodes:
      tasks = row.get("tasks") or []
      slices.append(EpisodeSlice(
        episode_index=int(row["episode_index"]),
        data_from=int(row["dataset_from_index"]),
        data_to=int(row["dataset_to_index"]),
        video_from_ts=float(row[vcol_from]),
        video_to_ts=float(row[vcol_to]),
        data_chunk=int(row["data/chunk_index"]),
        data_file=int(row["data/file_index"]),
        video_chunk=int(row[vcol_chunk]),
        video_file=int(row[vcol_file]),
        task=str(tasks[0]) if tasks else "",
        length=int(row["length"]),
      ))
    slices.sort(key=lambda s: s.episode_index)
    return self._select_episodes(slices), vkey


def _resolve_video_key(meta, dataset_id: str, video_key: str | None) -> str:
  video_keys = [str(k) for k in meta.video_keys]
  if video_key is not None:
    if video_key not in video_keys:
      raise ValueError(
        f"{dataset_id}: video key {video_key!r} not among {video_keys}")
    return video_key
  if not video_keys:
    raise ValueError(f"{dataset_id}: no video features found")
  return video_keys[0]


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
  episodes_str, sep, frames_raw = rest.partition(":")
  frames_str: str | None = frames_raw.strip() if sep else None

  episodes = _parse_episodes(raw, episodes_str.strip())
  frames = (_parse_range_list(raw, frames_str, what="frames")
            if frames_str else None)
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
