"""
directional_head_or_hand — head-tilt OR hand raised in a direction.
Used for "look up", "scan left/right", "cup ear".

Params:
  direction (str): "left" | "right" | "up" | "ear"

Approach:
  "up"    — any wrist above 30% frame height.
  "left"  — dominant hand moves past left third of frame.
  "right" — dominant hand moves past right third of frame.
  "ear"   — fingertip within ~15% normalised distance of estimated ear position.
  Head-turn fallback: hand-position heuristic (no face landmarks required).

Context keys: directional_since
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    # TODO: implement per-direction heuristics described above.
    return False
