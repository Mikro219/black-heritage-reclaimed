"""
trail_follow — extended hand follows a waypoint sequence at ≥ threshold match.
Used for Scene 7 `trace_trail` CG (AL-07-006).

POSE-ONLY (July 2026): the tracked point is the Pose hand point per side —
index landmark (19/20) when visible, else the wrist. Either hand can follow
the trail; the extension gate is arm reach instead of finger extension.

Params:
  threshold (float): fraction of waypoints that must be visited. Default 0.60.
  proximity (float): normalised radius for a waypoint "visited". Default 0.08.
  min_reach_frac (float): arm extension required. Default 0.55.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Waypoints via context["trail_waypoints"]; a 5-point dev trail is the fallback.

Context keys: trail_waypoints (set by scene), trail_cursor, trail_visited
  (reads: _pose_lm)
"""

import math

from . import pose_helpers

_DEV_WAYPOINTS = [
    (0.2, 0.5), (0.35, 0.45), (0.5, 0.5), (0.65, 0.55), (0.8, 0.5),
]


def detect(landmarks, params: dict, context: dict) -> bool:
    if "trail_cursor" not in context:
        context["trail_cursor"] = 0
        context["trail_visited"] = 0

    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if not wrists:
        return False

    threshold = params.get("threshold", 0.60)
    proximity = params.get("proximity", 0.08)
    min_reach = params.get("min_reach_frac", 0.55)

    waypoints = context.get("trail_waypoints") or _DEV_WAYPOINTS
    cursor = context["trail_cursor"]
    if cursor >= len(waypoints):
        return False

    target = waypoints[cursor]
    hit = False
    for side in wrists:
        if pose_helpers.arm_reach_frac(pose_lm, side) < min_reach:
            continue
        tip = pose_helpers.hand_point(pose_lm, side)
        if tip is None:
            continue
        if math.hypot(tip.x - target[0], tip.y - target[1]) <= proximity:
            hit = True
            break

    if hit:
        context["trail_cursor"] += 1
        context["trail_visited"] += 1
        if context["trail_cursor"] >= len(waypoints):
            score = context["trail_visited"] / len(waypoints)
            context["trail_cursor"] = 0
            context["trail_visited"] = 0
            return score >= threshold

    return False
