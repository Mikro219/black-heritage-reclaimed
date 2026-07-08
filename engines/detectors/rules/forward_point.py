"""
forward_point — index finger extended toward a specific on-screen target region.

Distinct from directional_point, which evaluates which screen half the finger indicates
(for left/right branching). forward_point evaluates whether the extended finger is aimed
at a small named target region and whether the fingertip is the leading point of the hand
(i.e., the finger is pointing toward the camera, not just pointing sideways).

Used for:
  AL-01-010  point_quilt_block   target_region="top_left_quadrant"
  AL-03-006  point_river_marker  target_region="lower_third"
  AL-04-008  point_dipper_corner target_region="top_right_quadrant"

Params:
  target_region (str): one of the named regions below. Required.
  hold_ms (int): ms the pose must be maintained within the region. Default 500.

Target regions (evaluated in player-mirrored screen space, x=0 left, y=0 top):
  "top_left_quadrant"   — player_x < 0.5, y < 0.5
  "top_right_quadrant"  — player_x > 0.5, y < 0.5
  "lower_third"         — y > 0.67
  "center"              — 0.25 < player_x < 0.75, 0.25 < y < 0.75

Detection approach:
  1. Index finger must be extended: tip (lm8) is farther from wrist (lm0) than the MCP
     joint (lm5), measured in 2D. This is the same extension check as directional_point
     and correctly fires for all pointing directions including downward.
  2. The fingertip's player-space position must fall within the declared target_region.
  3. Both conditions must hold continuously for hold_ms.

POSE+DEPTH FALLBACK (July 2026): a hand pointing STRAIGHT AT the camera is heavily
foreshortened — MediaPipe Hands usually fails to detect it at all, and even when it
does, the 2D extension check fails. When the Hands path doesn't produce a hit, the
detector falls back to Pose: a visible Pose wrist counts as a forward point when the
arm is reaching toward the camera AND the wrist sits inside the target region.
"Reaching" is judged by real depth when the Orbbec sampler is available
(shoulder_depth - wrist_depth >= min_reach_depth_mm), else by the Pose z delta.
Both paths share one hold timer, so Hands flickering in/out mid-hold doesn't reset it.

Params:
  target_region / hold_ms — as above.
  min_reach_depth_mm (float): required depth reach (mm) for the fallback when the
                  depth sampler is present. Default 300.
  min_pose_z_delta (float): required shoulder-to-wrist Pose z delta for the fallback
                  without depth (Pose z is hip-normalised). Default 0.25.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: forward_point_since  (reads: _pose_lm, _depth_mm_at)
"""

import math
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


def _tip_in_region(lm, region_key: str) -> bool:
    tip = lm[8]  # index fingertip
    # Mirror x to player-perspective space (same convention as render engine)
    return _point_in_region(1.0 - tip.x, tip.y, region_key)


def _pose_forward_fallback(pose_lm, params: dict, context: dict,
                           region_key: str) -> bool:
    """Foreshortened-hand fallback: a visible Pose wrist reaching toward the
    camera, positioned inside the target region. Depth preferred; Pose z else."""
    if pose_lm is None:
        return False
    min_visibility = params.get("min_visibility", 0.5)
    min_reach_mm = params.get("min_reach_depth_mm", 300)
    min_z_delta = params.get("min_pose_z_delta", 0.25)
    depth_at = context.get("_depth_mm_at")

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
            return True
    return False


def detect(landmarks, params: dict, context: dict) -> bool:
    region_key: str = params.get("target_region", "")
    hold_ms: int = params.get("hold_ms", 500)

    on_target = False
    for hand in landmarks or []:
        lm = hand.landmark
        wrist = lm[0]
        tip = lm[8]    # index fingertip
        mcp = lm[5]    # index MCP (knuckle)

        # Finger extension check: tip farther from wrist than MCP in 2D
        wrist_tip_sq = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
        wrist_mcp_sq = (mcp.x - wrist.x) ** 2 + (mcp.y - wrist.y) ** 2
        if wrist_tip_sq <= wrist_mcp_sq:
            continue  # finger not extended

        # Region check on player-mirrored fingertip position
        if _tip_in_region(lm, region_key):
            on_target = True
            break

    # Hands couldn't resolve a pointing hand — a hand aimed straight at the
    # camera usually can't be detected at all. Fall back to pose+depth.
    if not on_target:
        on_target = _pose_forward_fallback(context.get("_pose_lm"), params,
                                           context, region_key)

    now = time.monotonic()
    if on_target:
        if context.get("forward_point_since") is None:
            context["forward_point_since"] = now
        return (now - context["forward_point_since"]) * 1000 >= hold_ms
    else:
        context["forward_point_since"] = None
        return False
