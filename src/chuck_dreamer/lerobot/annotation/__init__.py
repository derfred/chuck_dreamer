"""Annotation tooling for the import pipeline.

The commands and helpers that produce the pipeline's annotation artifacts
(``extrinsics`` via annotate-mat, ``object_prompts`` via prompt-episodes) and
their supporting code: LeRobot frame access, prompt sidecars, the scene
background model, and the interactive extrinsics fit.
"""
from .annotate_mat import annotate_mat_cmd, annotate_mat_jpeg_cmd
from .calibrate_intrinsics import calibrate_intrinsics_cmd
from .prompt_episodes import prompt_episodes_cmd

__all__ = [
  "annotate_mat_cmd",
  "annotate_mat_jpeg_cmd",
  "calibrate_intrinsics_cmd",
  "prompt_episodes_cmd",
]
