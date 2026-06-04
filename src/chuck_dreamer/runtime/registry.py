"""Import-path-plus-params construction (spec §3.2, §8).

The runtime constructs backends, scripted sources, and (later) the policy
from config by import path. A spec is either a bare dotted path string —
``"pkg.mod:ClassName"`` or ``"pkg.mod.ClassName"`` — or a mapping
``{"target": <path>, "params": {...}}``. Extra keyword arguments passed by
the caller (e.g. a resolved ``cfg``) are merged on top of ``params``.
"""

from __future__ import annotations

import importlib
from typing import Any, Mapping


def import_symbol(path: str) -> Any:
  """Resolve a dotted ``"pkg.mod:name"`` or ``"pkg.mod.name"`` to an object."""
  if ":" in path:
    module_name, _, attr = path.partition(":")
  else:
    module_name, _, attr = path.rpartition(".")
  if not module_name or not attr:
    raise ValueError(
      f"invalid import path {path!r}; expected 'pkg.mod:Name' or 'pkg.mod.Name'")
  module = importlib.import_module(module_name)
  try:
    return getattr(module, attr)
  except AttributeError as exc:
    raise ValueError(f"{module_name!r} has no attribute {attr!r}") from exc


def construct(spec: Mapping[str, Any] | str, **extra_kwargs: Any) -> Any:
  """Construct an object from a registry ``spec`` plus caller kwargs.

  ``spec`` is a path string or a mapping with ``target`` (required) and
  optional ``params``. ``extra_kwargs`` override matching keys in ``params``.
  """
  if isinstance(spec, str):
    target, params = spec, {}
  elif isinstance(spec, Mapping):
    if "target" not in spec:
      raise ValueError(f"registry spec is missing 'target': {dict(spec)!r}")
    target = spec["target"]
    params = dict(spec.get("params") or {})
  else:
    raise TypeError(f"registry spec must be a string or mapping, got {type(spec)}")

  factory = import_symbol(target)
  return factory(**{**params, **extra_kwargs})
