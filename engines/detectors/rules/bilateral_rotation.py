"""
bilateral_rotation — both hands making circular motions, ≥ min_cycles.
Used as an alternative unravel model (Scene 11).

POSE-ONLY (July 2026): the rotating vector per side is wrist→index (Pose
landmarks 19/20) when the index is visible, else elbow→wrist — either way the
accumulated angle sweep measures the winding motion.

Params:
  min_cycles (int): complete rotations required per hand. Default 2.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: rotation_cycle_count, rotation_prev_angle, rotation_total_angle
  (reads: _pose_lm)
"""

import math

from . import pose_helpers


def _side_angle(pose_lm, side: str):
    vec = pose_helpers.point_vector(pose_lm, side)
    if vec is None:
        return None
    return math.atan2(vec[1], vec[0])


def _angle_delta(prev: float, curr: float) -> float:
    d = curr - prev
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if len(wrists) < 2:
        return False

    if "rotation_cycle_count" not in context:
        context["rotation_cycle_count"] = {}
        context["rotation_prev_angle"] = {}
        context["rotation_total_angle"] = {}

    min_cycles = params.get("min_cycles", 2)

    for side in wrists:
        angle = _side_angle(pose_lm, side)
        if angle is None:
            continue
        prev = context["rotation_prev_angle"].get(side)
        if prev is None:
            context["rotation_prev_angle"][side] = angle
            context["rotation_total_angle"][side] = 0.0
            context["rotation_cycle_count"][side] = 0
            continue

        delta = _angle_delta(prev, angle)
        context["rotation_total_angle"][side] = (
            context["rotation_total_angle"].get(side, 0.0) + abs(delta))
        context["rotation_prev_angle"][side] = angle
        context["rotation_cycle_count"][side] = int(
            context["rotation_total_angle"][side] / (2 * math.pi))

    counts = context["rotation_cycle_count"]
    if len(counts) >= 2 and all(v >= min_cycles for v in counts.values()):
        context["rotation_cycle_count"] = {}
        context["rotation_prev_angle"] = {}
        context["rotation_total_angle"] = {}
        return True

    return False
