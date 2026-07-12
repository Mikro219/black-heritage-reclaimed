"""
forward_point — arm reaching toward the camera, hand over a named target region.

Used for:
  AL-01-010  point_quilt_block   target_region="top_left_quadrant"
  AL-03-006  point_river_marker  target_region="lower_third"
  AL-04-008  point_dipper_corner target_region="top_right_quadrant"

POSE-ONLY (July 2026): what used to be the "pose+depth fallback" is now THE
detector — it was already the reliable path, because a hand pointing straight
at the camera is exactly the pose the Hands model failed on (foreshortening).
A visible Pose wrist counts as a forward point when the arm is reaching toward
the camera AND the wrist sits inside the target region. "Reaching" is judged
by real depth when the Orbbec sampler is available
(shoulder_depth - wrist_depth >= min_reach_depth_mm), else by the Pose z delta.

Params:
  target_region (str): one of the named regions below. Required.
  hold_ms (int): ms the pose must be maintained within the region. Default 500.
  min_reach_depth_mm (float): required depth reach (mm) with the sampler.
                  Default 300.
  min_pose_z_delta (float): required shoulder-to-wrist Pose z delta without
                  depth (Pose z is hip-normalised). Default 0.25.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Target regions (player-mirrored screen space, x=0 left, y=0 top):
  "top_left_quadrant"   — player_x < 0.5, y < 0.5
  "top_right_quadrant"  — player_x > 0.5, y < 0.5
  "lower_third"         — y > 0.67
  "center"              — 0.25 < player_x < 0.75, 0.25 < y < 0.75

Context keys: forward_point_since  (reads: _pose_lm, _depth_mm_at)
"""

import time

# (pose wrist index, same-side pose shoulder index)
_ARMS = [(15, 11), (16, 12)]

_REGIONS: dict[str, dict] = {
    "top_left_quadrant":  {"px_max": 0.5, "y_max": 0.5},
    "top_right_quadrant": {"px_min": 0.5, "y_max": 0.5},
    "lower_third":        {"y_min": 0.67},
    "center":             {"px_min": 0.25, "px_max": 0.75, "y_min": 0.25, "y_max": 0.75},
}


def _point_in_region(px: float, py: float, region_key: str) -> bool:
    """Region test on a player-mirrored screen-space point."""
    region = _REGIONS.get(region_key)
    if region is None:
        return True
    if "px_min" in region and px < region["px_min"]:
        return False
    if "px_max" in region and px > region["px_max"]:
        return False
    if "y_min" in region and py < region["y_min"]:
        return False
    if "y_max" in region and py > region["y_max"]:
        return False
    return True


def detect(landmarks, params: dict, context: dict) -> bool:
    region_key: str = params.get("target_region", "")
    hold_ms: int = params.get("hold_ms", 500)
    min_visibility = params.get("min_visibility", 0.5)
    min_reach_mm = params.get("min_reach_depth_mm", 300)
    min_z_delta = params.get("min_pose_z_delta", 0.25)

    pose_lm = context.get("_pose_lm")
    depth_at = context.get("_depth_mm_at")

    on_target = False
    if pose_lm is not None:
        for wrist_i, shoulder_i in _ARMS:
            try:
                wrist, shoulder = pose_lm[wrist_i], pose_lm[shoulder_i]
            except (IndexError, TypeError):
                continue
            if getattr(wrist, "visibility", 1.0) < min_visibility:
                continue

            # Reach check: real depth when both samples are valid, else Pose z.
            wrist_mm = shoulder_mm = None
            if callable(depth_at):
                wrist_mm = depth_at(wrist.x, wrist.y)
                shoulder_mm = depth_at(shoulder.x, shoulder.y)
            if wrist_mm is not None and shoulder_mm is not None:
                reaching = (shoulder_mm - wrist_mm) >= min_reach_mm
            else:
                z_delta = getattr(shoulder, "z", 0.0) - getattr(wrist, "z", 0.0)
                reaching = z_delta >= min_z_delta

            if reaching and _point_in_region(1.0 - wrist.x, wrist.y, region_key):
                on_target = True
                break

    now = time.monotonic()
    if on_target:
        if context.get("forward_point_since") is None:
            context["forward_point_since"] = now
        return (now - context["forward_point_since"]) * 1000 >= hold_ms

    context["forward_point_since"] = None
    return False
