"""
point_target_held — hand held on a named screen region for hold_ms.
Used for the point OIs across Scenes 1/3/4/5/6 and the tutorial target step.

POSE-ONLY (July 2026): the tracked point per side is the Pose hand point —
index landmark (19/20) when visible, else the wrist. The old pointing-finger
classification (index out, others curled) needed the Hands model and is gone;
the deliberate-gesture gate is arm extension (min_reach_frac), which is what
made the pose fallback trustworthy before. Direction filters use the pose
point vector (wrist→index, else forearm).

Params:
  target_region (str):   key into context["target_regions"] rect dict.
  region_rect (dict):    inline rect {"x","y","w","h"} in RAW (unmirrored)
                         normalised coords, same as before.
  directions (list[str]): optional 8-way filter on the hand's point vector.
  hold_ms (int):         ms the point must stay on target. Default 800.
  min_reach_frac (float): arm extension required. Default 0.55.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

If neither region_rect nor directions is given, fires on any extended arm with
its hand point in the lower half (legacy fallback).

Context keys: point_on_target_since  (reads: _pose_lm)
"""

import time

from . import pose_helpers


def _point_in_rect(px: float, py: float, rect: dict) -> bool:
    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
    return x <= px <= x + w and y <= py <= y + h


def _resolve_rect(params: dict, context: dict):
    inline_rect = params.get("region_rect")
    if inline_rect:
        return inline_rect
    target_key = params.get("target_region")
    target_regions: dict = context.get("target_regions", {})
    if target_key and target_key in target_regions:
        return target_regions[target_key]
    return None


def detect(landmarks, params: dict, context: dict) -> bool:
    hold_ms = params.get("hold_ms", 800)
    directions = params.get("directions")
    rect = _resolve_rect(params, context)
    min_reach = params.get("min_reach_frac", 0.55)
    min_visibility = params.get("min_visibility", 0.5)

    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context, min_visibility)

    on_target = False
    for side in wrists:
        if pose_helpers.arm_reach_frac(pose_lm, side) < min_reach:
            continue
        tip = pose_helpers.hand_point(pose_lm, side, min_visibility)
        if tip is None:
            continue

        if directions:
            vec = pose_helpers.point_vector(pose_lm, side)
            if vec is None or pose_helpers.classify_direction(*vec) not in directions:
                continue
        if rect is not None:
            if not _point_in_rect(tip.x, tip.y, rect):
                continue
        elif not directions:
            if tip.y <= 0.4:          # legacy no-filter fallback: lower half
                continue
        on_target = True
        break

    now = time.monotonic()
    if on_target:
        if context.get("point_on_target_since") is None:
            context["point_on_target_since"] = now
        return (now - context["point_on_target_since"]) * 1000 >= hold_ms

    context["point_on_target_since"] = None
    return False
