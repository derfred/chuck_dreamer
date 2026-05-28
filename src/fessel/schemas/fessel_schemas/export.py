"""Emit a JSON Schema document for the wire models.

Used by tools/generate-types.sh, which pipes the output into a
TypeScript generator. Run: python -m fessel_schemas.export
"""

import json

from pydantic import TypeAdapter

from .models import Capabilities, LiveActivate, LiveDeactivate, LiveState, ModeTriplet

_MODELS = {
  "ModeTriplet": ModeTriplet,
  "Capabilities": Capabilities,
  "LiveActivate": LiveActivate,
  "LiveDeactivate": LiveDeactivate,
  "LiveState": LiveState,
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


def main() -> None:
  print(json.dumps(json_schema(), indent=2))


if __name__ == "__main__":
  main()
