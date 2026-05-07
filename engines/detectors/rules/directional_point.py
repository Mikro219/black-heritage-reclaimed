"""
directional_point — single hand pointing in a direction, held for hold_ms.

Params:
  directions (list[str]): accepted directions — "left", "right", "up", "down".
  hold_ms (int): ms the point must be held. Default 500.

Approach: wrist (lm 0) → index fingertip (lm 8) vector; classify dominant axis.

Context keys: point_direction_since, point_direction
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    # TODO: compute wrist→tip direction vector, classify axis,
    # match against params["directions"], hold-gate.
    return False
