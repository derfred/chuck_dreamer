"""Native pipeline nodes (spec §7).

One module per node family; every node declares typed inputs/outputs and is
bound to a :class:`~chuck_dreamer.lerobot.context.RunContext`.
``NODE_TYPES`` is the name → class registry the parallel importer's worker
processes rebuild their lane from.
"""
from .ee import (
  EeActionNode,
  EePosArmNode,
  EeRotationNode,
  NormalizeJointsNode,
  build_ee_nodes,
)
from .object_pose import ObjectPoseNode
from .segmentation import ObjUvExtractionNode, SegmentationNode
from .table_frame import (
  EeActionTableNode,
  EePosTableNode,
  EeQuatTableNode,
  ObjPosArmNode,
  build_ee_table_nodes,
  build_object_table_nodes,
)

NODE_TYPES = {cls.name: cls for cls in (
  NormalizeJointsNode, EePosArmNode, EeRotationNode, EeActionNode,
  EePosTableNode, EeQuatTableNode, EeActionTableNode,
  SegmentationNode, ObjUvExtractionNode, ObjectPoseNode, ObjPosArmNode,
)}


def build_object_nodes(ctx) -> list:
  """The dependency-ordered object chain (spec §7.4/§7.5/§7.10)."""
  return [SegmentationNode(ctx), ObjUvExtractionNode(ctx), ObjectPoseNode(ctx)]


__all__ = [
  "NormalizeJointsNode",
  "EePosArmNode",
  "EeRotationNode",
  "EeActionNode",
  "EePosTableNode",
  "EeQuatTableNode",
  "EeActionTableNode",
  "ObjPosArmNode",
  "SegmentationNode",
  "ObjUvExtractionNode",
  "ObjectPoseNode",
  "build_ee_nodes",
  "build_ee_table_nodes",
  "build_object_nodes",
  "build_object_table_nodes",
  "NODE_TYPES",
]
