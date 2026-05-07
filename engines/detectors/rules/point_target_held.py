"""
point_target_held — index finger pointing at a named screen region, held for hold_ms.
Used for Scene 1 point-quilt-block OI, Scene 4 North Star OI.

Params:
  target_region (str): named region key, resolved from context["target_regions"].
  hold_ms (int): ms the point must stay on target. Default 500.

Approach: project wrist→fingertip ray onto screen plane; check if intersection
falls within target region bounding box (supplied by render engine via context).

Context keys: target_regions (set by render engine), point_on_target_since
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    # TODO: project finger ray, look up target bounding box from
    # context["target_regions"], hold-gate.
    if "point_on_target_since" not in context:
        context["point_on_target_since"] = None
    return False
