"""
trail_follow — extended hand follows a waypoint sequence at ≥ threshold match.
Used for Scene 7 trace-trail CG (≥60% waypoint match per config).

Params:
  threshold (float): fraction of waypoints that must be visited in order.
      Defaults to config trail_waypoint_match (0.60).
  proximity (float): normalised radius for waypoint "visited" check. Default 0.08.

Waypoints are supplied by the scene render layer via context["trail_waypoints"].

Context keys: trail_waypoints (set by scene), trail_cursor
"""


def detect(landmarks, params: dict, context: dict) -> bool:
    # TODO: retrieve waypoints from context["trail_waypoints"],
    # advance cursor as fingertip visits each in order, gate on threshold.
    if "trail_cursor" not in context:
        context["trail_cursor"] = 0
    return False
