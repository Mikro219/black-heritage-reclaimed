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

Context keys: forward_point_since
"""

import math
import time

_REGIONS: dict[str, dict] = {
    "top_left_quadrant":  {"px_max": 0.5, "y_max": 0.5},
    "top_right_quadrant": {"px_min": 0.5, "y_max": 0.5},
    "lower_third":        {"y_min": 0.67},
    "center":             {"px_min": 0.25, "px_max": 0.75, "y_min": 0.25, "y_max": 0.75},
}


def _tip_in_region(lm, region_key: str) -> bool:
    region = _REGIONS.get(region_key)
    if region is None:
        return True

    tip = lm[8]  # index fingertip
    # Mirror x to player-perspective space (same convention as render engine)
    px = 1.0 - tip.x
    py = tip.y

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
    if not landmarks:
        context["forward_point_since"] = None
        return False

    region_key: str = params.get("target_region", "")
    hold_ms: int = params.get("hold_ms", 500)

    on_target = False
    for hand in landmarks:
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

    now = time.monotonic()
    if on_target:
        if context.get("forward_point_since") is None:
            context["forward_point_since"] = now
        return (now - context["forward_point_since"]) * 1000 >= hold_ms
    else:
        context["forward_point_since"] = None
        return False
