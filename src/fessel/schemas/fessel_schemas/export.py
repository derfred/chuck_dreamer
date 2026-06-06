"""Emit JSON Schema for the wire models.

Used by tools/generate-types.sh, which converts the output into:
  - TypeScript types     (json-schema-to-typescript, bundled-root form)
  - runtime Zod schemas  (json-schema-to-zod, per-model form)

Two emitters:
  - `json_schema()`        — one bundled document; the TS generator walks its
                             properties and emits one interface per model.
  - `per_model_schemas()`  — one self-contained document per model (root = the
                             model, dependencies kept in $defs). The Zod
                             generator emits one named schema per document and
                             resolves inter-model refs (e.g. StateResponse ->
                             PlugState). Run: python -m fessel_schemas.export --per-model
"""

import argparse
import json
import sys

from pydantic import TypeAdapter

from .models import (
  AnomalyLogEntry,
  AudioState,
  CameraState,
  Capabilities,
  LiveActivate,
  LiveDeactivate,
  LiveState,
  ModeTriplet,
  PlugState,
  RecordingListItem,
  RecordingMetadata,
  RecordingState,
  StateResponse,
  UploadBacklog,
  UploadProgress,
  VisionState,
)

_MODELS = {
  "ModeTriplet": ModeTriplet,
  "Capabilities": Capabilities,
  "LiveActivate": LiveActivate,
  "LiveDeactivate": LiveDeactivate,
  "LiveState": LiveState,
  "PlugState": PlugState,
  "CameraState": CameraState,
  "RecordingState": RecordingState,
  "RecordingMetadata": RecordingMetadata,
  "UploadProgress": UploadProgress,
  "UploadBacklog": UploadBacklog,
  "RecordingListItem": RecordingListItem,
  # Slice 5 sensing shapes the dashboard reads (via /api/state and /api/anomalies).
  "VisionState": VisionState,
  "AudioState": AudioState,
  "AnomalyLogEntry": AnomalyLogEntry,
  # StateResponse last: it embeds several of the above; per-model inlining wants
  # the dependencies defined first is not required (refs resolve), but keeping
  # the aggregate last matches the Slice 3 ordering convention.
  "StateResponse": StateResponse,
}


def json_schema() -> dict:
  """Bundle every wire model into one document.

  A top-level object references each model as a property so the TypeScript
  generator emits one named interface per model (it walks properties, not
  bare $defs). The root interface itself is incidental.
  """
  defs: dict = {}
  properties: dict = {}
  for name, model in _MODELS.items():
    schema = TypeAdapter(model).json_schema(ref_template="#/$defs/{model}")
    defs.update(schema.pop("$defs", {}))
    defs[name] = schema
    properties[name] = {"$ref": f"#/$defs/{name}"}
  return {
    "title": "FesselSchemas",
    "type": "object",
    "properties": properties,
    "$defs": defs,
  }


def _inline_refs(node: object, defs: dict[str, dict]) -> object:
  """Recursively replace every local `$ref: #/$defs/<Name>` with a copy of the
  referenced definition, so the result has NO `$ref`s left.

  json-schema-to-zod does not resolve local `$ref`s (it falls back to
  z.any()), so we dereference them here, in Python, before handing it the
  schema. Our models form a DAG (no recursive types), so a straight recursive
  inline terminates.
  """
  if isinstance(node, dict):
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
      target = defs[ref.split("/")[-1]]
      inlined = _inline_refs(target, defs)
      # Preserve sibling keywords (e.g. a `default` alongside the $ref).
      extra = {k: v for k, v in node.items() if k != "$ref"}
      if isinstance(inlined, dict) and extra:
        return {**inlined, **extra}
      return inlined
    return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
  if isinstance(node, list):
    return [_inline_refs(v, defs) for v in node]
  return node


def per_model_schemas() -> dict[str, dict]:
  """One self-contained, fully ref-inlined JSON Schema per model.

  Each value is a standalone schema with the model as its root and every
  cross-model `$ref` dereferenced in place (no `$defs`, no `$ref`), so the Zod
  generator emits real validators for nested models (StateResponse embeds a
  validated PlugState/CameraState/SafetyState) instead of z.any().
  """
  out: dict[str, dict] = {}
  for name, model in _MODELS.items():
    schema = TypeAdapter(model).json_schema(ref_template="#/$defs/{model}")
    defs = schema.get("$defs", {})
    out[name] = _inline_refs(schema, defs)
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--per-model",
    action="store_true",
    help="emit one self-contained schema per model (JSON object name->schema)",
  )
  args = parser.parse_args()
  if args.per_model:
    json.dump(per_model_schemas(), sys.stdout, indent=2)
  else:
    print(json.dumps(json_schema(), indent=2))


if __name__ == "__main__":
  main()
