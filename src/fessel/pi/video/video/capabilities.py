"""Camera capability probing (V1.3).

On Linux, parse `v4l2-ctl --list-formats-ext` (MJPG modes only — USB
webcams output MJPG). On macOS (dev) supply a static set. Each usable
mode becomes a (resolution, fps, bitrate) triplet; the bitrate is a
sensible default per pixel-rate, tunable later.
"""

from __future__ import annotations

import logging
import platform
import re
import subprocess

from fessel_schemas import ModeTriplet

log = logging.getLogger(__name__)

# Dev fallback when no real camera is probed (macOS, or test-pattern source).
_DEV_MODES = [
  ("640x480", 30),
  ("1280x720", 30),
  ("1280x720", 15),
  ("1920x1080", 10),
]


def default_bitrate_bps(width: int, height: int, fps: int) -> int:
  """A starting-point bitrate from pixel rate.

  ~0.07 bits per pixel-per-second, clamped to a sane range. This is a
  tunable starting point, not a tuned value (see plan notes).
  """
  bps = int(width * height * fps * 0.07)
  return max(500_000, min(bps, 8_000_000))


def _mode(resolution: str, fps: int) -> ModeTriplet:
  w, h = (int(x) for x in resolution.split("x"))
  return ModeTriplet(resolution=resolution, fps=fps, bitrate_bps=default_bitrate_bps(w, h, fps))


def _parse_v4l2(text: str) -> list[ModeTriplet]:
  """Parse the MJPG section of `v4l2-ctl --list-formats-ext` output.

  We collect (WxH, fps) pairs that appear under an MJPG pixelformat block.
  """
  modes: list[ModeTriplet] = []
  in_mjpg = False
  current_size: str | None = None
  size_re = re.compile(r"Size:\s+\w+\s+(\d+)x(\d+)")
  fps_re = re.compile(r"\(([\d.]+)\s*fps\)")
  fmt_re = re.compile(r"\[\d+\]:\s*'(\w+)'")

  for line in text.splitlines():
    fmt = fmt_re.search(line)
    if fmt:
      in_mjpg = fmt.group(1) in ("MJPG", "JPEG")
      continue
    if not in_mjpg:
      continue
    size = size_re.search(line)
    if size:
      current_size = f"{size.group(1)}x{size.group(2)}"
      continue
    fps_m = fps_re.search(line)
    if fps_m and current_size:
      fps = int(round(float(fps_m.group(1))))
      if fps > 0:
        modes.append(_mode(current_size, fps))
  return modes


def probe(device: str = "/dev/video0") -> list[ModeTriplet]:
  """Probe the camera for usable modes.

  Falls back to the dev set on macOS or if v4l2-ctl is unavailable/empty.
  Deduplicates while preserving order.
  """
  modes: list[ModeTriplet] = []
  if platform.system() == "Linux":
    try:
      out = subprocess.run(
        ["v4l2-ctl", "--device", device, "--list-formats-ext"],
        capture_output=True,
        text=True,
        check=True,
      )
      modes = _parse_v4l2(out.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
      log.warning("v4l2-ctl probe failed (%s); using dev modes", e)

  if not modes:
    modes = [_mode(res, fps) for res, fps in _DEV_MODES]

  seen: set[tuple[str, int]] = set()
  deduped: list[ModeTriplet] = []
  for m in modes:
    key = (m.resolution, m.fps)
    if key not in seen:
      seen.add(key)
      deduped.append(m)
  return deduped
