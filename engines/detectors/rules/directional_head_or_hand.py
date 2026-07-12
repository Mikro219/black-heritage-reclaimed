"""
directional_head_or_hand — a pose wrist raised/positioned in a direction.
Used for Scene 3 `look_up` OI, Scene 6 `scan_left`/`scan_right` OIs,
Scene 10 `cup_ear_listen` OI.

POSE-ONLY (July 2026): reads Pose wrists (15/16); ear positions default to the
live Pose ear landmarks (7/8). The old require_curl finger check is gone.

Params:
  direction (str): "left" | "right" | "up" | "brow" | "ear". Required.
  hold_ms (int): ms the pose must be maintained. Default 500.
  ear_positions (list): [(x, y), ...] override; defaults to live pose ears,
                        then to fixed estimates.
  ear_radius (float): wrist distance that counts as "near ear". Default 0.22.
  brow_y (float): Y threshold for "brow" (overridden live by
                  _inject_pose_params). Default 0.38.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Direction semantics (raw camera space; display is mirrored):
  "up"   — any wrist y < 0.35        "brow" — any wrist y < brow_y
  "left" — any wrist raw x > 0.70 (player's left)
  "right"— any wrist raw x < 0.30
  "ear"  — wrist within ear_radius of either ear

Context keys: directional_since  (reads: _pose_lm)
"""

import math
import time

from . import pose_helpers

_EAR_POSITIONS = [(0.15, 0.35), (0.85, 0.35)]


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if not wrists:
        context["directional_since"] = None
        return False

    direction = params.get("direction", "up")
    hold_ms = params.get("hold_ms", 500)
    ear_radius = params.get("ear_radius", 0.22)
    brow_y = params.get("brow_y", 0.38)

    ear_positions = params.get("ear_positions")
    if not ear_positions:
        ears = [pose_helpers.lm(pose_lm, 7), pose_helpers.lm(pose_lm, 8)]
        ear_positions = [(e.x, e.y) for e in ears if e is not None] or _EAR_POSITIONS

    matched = False
    for wrist in wrists.values():
        if direction == "up":
            matched = wrist.y < 0.35
        elif direction == "left":
            matched = wrist.x > 0.70       # player's left = camera right
        elif direction == "right":
            matched = wrist.x < 0.30
        elif direction == "brow":
            matched = wrist.y < brow_y
        elif direction == "ear":
            matched = any(math.hypot(wrist.x - ex, wrist.y - ey) < ear_radius
                          for ex, ey in ear_positions)
        if matched:
            break

    now = time.monotonic()
    if matched:
        if context.get("directional_since") is None:
            context["directional_since"] = now
        return (now - context["directional_since"]) * 1000 >= hold_ms

    context["directional_since"] = None
    return False
