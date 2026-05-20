"""Decorators shared across the object_localization CLIs.

Mirrors the ``override_option`` in ``main.py`` so the OL commands accept
the same ``-o KEY=VALUE`` syntax. We re-declare it here rather than
importing from ``main`` to avoid an import cycle (main registers these
commands at the bottom of the file).
"""
from __future__ import annotations

import click


override_option = click.option(
  "-o", "--override", "overrides", multiple=True, metavar="KEY=VALUE",
  help="Dotted-path config override, repeatable (e.g. "
       "-o object_localization.intrinsics.n_frames_target=128). "
       "Values are parsed as YAML scalars.",
)
