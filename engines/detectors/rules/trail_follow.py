"""
trail_follow — extended hand follows a waypoint sequence at ≥ threshold match.
Used for Scene 7 `trace_trail` CG (AL-07-006).

Params:
  threshold (float): fraction of waypoints that must be visited. Default 0.60.
  proximity (float): normalised radius for waypoint "visited" check. Default 0.08.

Waypoints supplied via context["trail_waypoints"] (list of (x, y) normalised tuples),
set by the render engine when the trail is displayed. If no waypoints are set, a
default 5-waypoint horizontal trail is used for dev/testing purposes.

Context keys: trail_waypoints (set by scene), trail_cursor, trail_visited
"""

import math

_DEV_WAYPOINTS = [
    (0.2, 0.5), (0.35, 0.45), (0.5, 0.5), (0.65, 0.55), (0.8, 0.5),
]


def detect(landmarks, params: dict, context: dict) -> bool:
    if "trail_cursor" not in context:
        context["trail_cursor"] = 0
        context["trail_visited"] = 0

    if not landmarks:
        return False

    threshold = params.get("threshold", 0.60)
    proximity = params.get("proximity", 0.08)

    waypoints = context.get("trail_waypoints") or _DEV_WAYPOINTS
    cursor = context["trail_cursor"]

    if cursor >= len(waypoints):
        return False

    # Use index fingertip position
    tip = None
    for hand in landmarks:
        lm = hand.landmark
        if lm[8].y < lm[6].y:  # extended finger
            tip = lm[8]
            break

    if tip is None:
        return False

    target = waypoints[cursor]
    dist = math.hypot(tip.x - target[0], tip.y - target[1])
    if dist <= proximity:
        context["trail_cursor"] += 1
        context["trail_visited"] += 1

        if context["trail_cursor"] >= len(waypoints):
            # All waypoints reached — check threshold
            score = context["trail_visited"] / len(waypoints)
            context["trail_cursor"] = 0
            context["trail_visited"] = 0
            return score >= threshold

    return False
