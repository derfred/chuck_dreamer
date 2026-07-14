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
  range      = INT                            ; a single index
             / INT "-" INT [ ":" INT ]        ; closed range, optional stride
             / INT "-"                         ; open upper: INT to the last
             / "-" INT                         ; open lower: first to INT

Both episode and frame ranges are **inclusive on both ends**. Strides
default to 1 and are only valid on a closed ``N-M`` range.

**Open-ended ranges** leave one bound implicit and are resolved against
the dataset at selection time: ``N-`` means "from N to the last episode
(or frame) that exists", and ``-M`` means "from the first (episode 0 /
frame 0) up to and including M". They are the ergonomic spelling for
"everything from here on" without having to know the upper index.

**Frame indices are episode-relative.** ``5-80`` means "frames 5 through
80 of each selected episode" (offset 5 from that episode's start, not
global indices 5..80). Closed ranges that exceed an episode's length are
silently clipped to its bounds, so ``0-10000`` is a safe "all frames"
spelling — as is the open-ended ``0-``.

Examples:

  user/cal                    -> all episodes, frames auto-sampled
  user/cal#0                  -> just episode 0, frames auto-sampled
  user/cal#0,2-4              -> episodes 0, 2, 3, 4 (auto-sampled)
  user/cal#8-                 -> episodes 8 through the last in the dataset
  user/cal#-20                -> episodes 0 through 20
  user/cal#all:0-200          -> all eps, frames 0..200 of each
  user/cal#1:0-200:5          -> episode 1, frames 0,5,...,200
  user/cal#1:100-             -> episode 1, frames 100 through its last
  user/cal#0-1:0,50-100:10    -> eps 0+1, frame 0 + frames 50,60,...,100 of each
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click


@dataclass
class EpisodeSlice:
  """One episode's metadata: frame bounds, video-decode window, and the
  chunk/file coordinates for both the data parquet and the video file.

  ``video_path`` is the absolute path to the episode's MP4, resolved at
  :meth:`EpisodeSpec.read_episodes` time from the dataset root + the
  chunk/file coordinates. It is ``None`` only on slices built outside
  ``read_episodes`` (e.g. test fixtures), so a stage decoding the video
  window doesn't have to re-open ``LeRobotDatasetMetadata`` to find it."""
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
  video_path: Path | None = None

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

  An **open-upper** range (``N-``: "N through the last") can't be expanded
  to concrete indices at parse time, so it's held separately in
  ``episode_open_from`` / ``frame_open_from`` as the smallest ``N`` seen,
  and unioned with the finite picks at selection time. (Open-*lower*
  ``-M`` is just ``0-M`` and is expanded immediately.) When a spec carries
  *only* an open range, the finite list is empty but non-None, so it is
  still a filter rather than "all".
  """
  dataset_id: str
  episodes:   list[int] | None = None
  frames:     list[int] | None = None
  # Smallest N from any open-upper ``N-`` token, or None if there were none.
  episode_open_from: int | None = None
  frame_open_from:   int | None = None

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
    """``episodes`` as a set, or ``None`` for "all episodes".

    Returns ``None`` when the spec carries an open-upper ``N-`` range: the
    selected set can only be known once the dataset's episode list is read,
    so callers using this for a progress total should treat it as unknown.
    """
    if self.episode_open_from is not None:
      return None
    return set(self.episodes) if self.episodes is not None else None

  def _select_episodes(self, slices: list[EpisodeSlice]) -> list[EpisodeSlice]:
    """Keep only ``slices`` whose ``episode_index`` is selected by this
    spec, preserving input order. Returns the list unchanged when no
    episode filter was given ("all"). An open-upper ``N-`` range keeps
    every available episode index ``>= N``. Used by :meth:`read_episodes`."""
    if self.episodes is None and self.episode_open_from is None:
      return slices
    keep = set(self.episodes or ())
    lo = self.episode_open_from
    return [s for s in slices
            if s.episode_index in keep
            or (lo is not None and s.episode_index >= lo)]

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

    # ``meta.video_path`` is a formattable template (videos/{video_key}/
    # chunk-{:03d}/file-{:03d}.mp4) relative to ``meta.root``; resolve each
    # slice's absolute MP4 here so consumers needn't re-open the metadata.
    root_dir  = Path(meta.root)
    vpath_tpl = meta.video_path

    slices: list[EpisodeSlice] = []
    for row in meta.episodes:
      tasks = row.get("tasks") or []
      vchunk = int(row[vcol_chunk])
      vfile  = int(row[vcol_file])
      vpath  = root_dir / vpath_tpl.format(
        video_key=vkey, chunk_index=vchunk, file_index=vfile)
      slices.append(EpisodeSlice(
        episode_index=int(row["episode_index"]),
        data_from=int(row["dataset_from_index"]),
        data_to=int(row["dataset_to_index"]),
        video_from_ts=float(row[vcol_from]),
        video_to_ts=float(row[vcol_to]),
        data_chunk=int(row["data/chunk_index"]),
        data_file=int(row["data/file_index"]),
        video_chunk=vchunk,
        video_file=vfile,
        task=str(tasks[0]) if tasks else "",
        length=int(row["length"]),
        video_path=vpath,
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

  episodes, ep_open = _parse_episodes(raw, episodes_str.strip())
  if frames_str:
    frames, fr_open = _parse_range_list(raw, frames_str, what="frames")
  else:
    frames, fr_open = None, None
  return EpisodeSpec(
    dataset_id=dataset_id,
    episodes=episodes,
    frames=frames,
    episode_open_from=ep_open,
    frame_open_from=fr_open,
  )


def _parse_episodes(raw: str, s: str) -> tuple[list[int] | None, int | None]:
  s = s.strip().lower()
  if s in ("", "all"):
    return None, None
  return _parse_range_list(raw, s, what="episodes")


def _parse_range_list(raw: str, s: str, what: str
                      ) -> tuple[list[int], int | None]:
  """Parse a comma-separated range list, returning ``(finite_indices,
  open_from)``. ``open_from`` is the smallest ``N`` across any open-upper
  ``N-`` token (or None if there were none); the finite indices cover
  every other token. A spec may mix the two (``0,8-`` → ``{0}`` ∪ ``>=8``).
  """
  out: list[int] = []
  seen: set[int] = set()
  open_from: int | None = None
  for tok in s.split(","):
    tok = tok.strip()
    if not tok:
      continue
    indices, tok_open = _parse_one_range(raw, tok, what=what, seen=seen)
    out.extend(indices)
    if tok_open is not None:
      open_from = tok_open if open_from is None else min(open_from, tok_open)
  if not out and open_from is None:
    raise ValueError(f"{raw!r}: no usable {what} after parsing.")
  return out, open_from


def _parse_one_range(raw: str, tok: str, what: str, seen: set[int]
                     ) -> tuple[list[int], int | None]:
  """Parse one token into ``(finite_indices, open_from)``. ``open_from`` is
  set (and ``finite_indices`` empty) only for an open-upper ``N-`` token."""
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
    a_str, b_str = a_str.strip(), b_str.strip()
    # Open-upper ``N-``: from N to the (yet-unknown) last index.
    if b_str == "":
      if a_str == "":
        raise ValueError(f"{raw!r}: {what} range '-' has neither bound.")
      if stride != 1:
        raise ValueError(
          f"{raw!r}: stride only allowed with a closed N-M range, not {tok!r}-.")
      try:
        a = int(a_str)
      except ValueError:
        raise ValueError(f"{raw!r}: range {tok!r}- in {what} is not an integer.")
      if a < 0:
        raise ValueError(f"{raw!r}: {what} range {a}- has a negative value.")
      return [], a
    # Open-lower ``-M``: from the first index (0) to M — fully finite.
    if a_str == "":
      a_str = "0"
    try:
      a, b = int(a_str), int(b_str)
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
  return out, None
