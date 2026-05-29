"""Run the mat-annotation UI on a single JPEG and solve extrinsics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from chuck_dreamer.config import load_config
from chuck_dreamer.lerobot.object_localization.calibration.extrinsics import (
  render_extrinsics_overlay,
  render_world_axes,
  solve_extrinsics,
)
from chuck_dreamer.lerobot.object_localization.calibration.interactive import (
  annotate_mat_cv2,
  close_popup_window,
  reset_popup_state,
)
from chuck_dreamer.lerobot.object_localization.runtime import init_from_config


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("image", type=Path)
  ap.add_argument("--calib", type=Path,
                  default=Path("calibration_cache/intrinsics.json"),
                  help="intrinsics.json from calibration_cache (top-level 'K','dist').")
  ap.add_argument("--config", type=str, default=None,
                  help="Optional config overlay yaml; defaults to package defaults.")
  ap.add_argument("--out", type=Path, default=Path("annotate_mat_out"))
  args = ap.parse_args()

  bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
  if bgr is None:
    raise SystemExit(f"could not read {args.image}")
  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

  calib = json.loads(args.calib.read_text())
  K = np.asarray(calib["K"], dtype=np.float64)
  dist = np.asarray(calib["dist"], dtype=np.float64)

  cfg = load_config(args.config)
  ol_cfg = init_from_config(cfg)

  reset_popup_state()
  try:
    detection = annotate_mat_cv2(rgb)
  finally:
    close_popup_window()
  if detection is None:
    raise SystemExit("annotation aborted")

  result = solve_extrinsics(detection, K, dist, ol_cfg)
  print(f"rms={result.refined_rms_px:.3f}px  "
        f"phase_offset={result.phase_offset}  reverse={result.reverse}")

  args.out.mkdir(parents=True, exist_ok=True)
  overlay = render_extrinsics_overlay(rgb, result.extrinsics, K, dist, ol_cfg,
                                      detection, result)
  axes = render_world_axes(rgb, result.extrinsics, K, dist)
  cv2.imwrite(str(args.out / "overlay.png"), overlay)
  cv2.imwrite(str(args.out / "world_axes.png"), axes)
  (args.out / "extrinsics.json").write_text(json.dumps({
    "R": result.extrinsics.R.tolist(),
    "t": result.extrinsics.t.tolist(),
    "rms_px": float(result.refined_rms_px),
    "phase_offset": int(result.phase_offset),
    "reverse": bool(result.reverse),
  }, indent=2))
  print(f"wrote {args.out}/")


if __name__ == "__main__":
  main()
