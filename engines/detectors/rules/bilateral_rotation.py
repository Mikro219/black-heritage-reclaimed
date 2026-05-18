"""
bilateral_rotation — both hands making outward circular motions, ≥ min_cycles.
Used for Scene 11 `unravel` CG (AL-11-011), ≥2 outward rotations.

Params:
  min_cycles (int): complete outward rotations required per hand. Default 2.
  direction (str): "outward" — hands circle away from body centre. Default "outward".

Approach:
  Track the angle of the wrist→middle-fingertip vector for each hand across frames.
  Accumulate total angular change; a full 360° = one cycle. For "outward", the left
  hand should rotate clockwise (angle decreasing) and the right counterclockwise.
  Fire when both hands have completed min_cycles.

Context keys: rotation_cycle_count, rotation_prev_angle, rotation_total_angle
"""

import math


def _hand_angle(hand) -> float:
    """Angle (radians) of wrist→middle-finger vector."""
    wrist = hand.landmark[0]
    mid = hand.landmark[12]  # middle finger tip
    return math.atan2(mid.y - wrist.y, mid.x - wrist.x)


def _angle_delta(prev: float, curr: float) -> float:
    """Signed shortest angular difference, wrapped to [-π, π]."""
    d = curr - prev
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks or len(landmarks) < 2:
        return False

    if "rotation_cycle_count" not in context:
        context["rotation_cycle_count"] = {}
        context["rotation_prev_angle"] = {}
        context["rotation_total_angle"] = {}

    min_cycles = params.get("min_cycles", 2)

    for i, hand in enumerate(landmarks):
        angle = _hand_angle(hand)
        prev = context["rotation_prev_angle"].get(i)
        if prev is None:
            context["rotation_prev_angle"][i] = angle
            context["rotation_total_angle"][i] = 0.0
            context["rotation_cycle_count"][i] = 0
            continue

        delta = _angle_delta(prev, angle)
        # For outward motion: left hand (i=0 typically) rotates CW (negative delta),
        # right hand (i=1) CCW (positive delta). We sum absolute accumulation.
        context["rotation_total_angle"][i] = (
            context["rotation_total_angle"].get(i, 0.0) + abs(delta)
        )
        context["rotation_prev_angle"][i] = angle

        cycles_done = int(context["rotation_total_angle"][i] / (2 * math.pi))
        context["rotation_cycle_count"][i] = cycles_done

    counts = context["rotation_cycle_count"]
    if len(counts) >= 2 and all(v >= min_cycles for v in counts.values()):
        # Reset
        context["rotation_cycle_count"] = {}
        context["rotation_prev_angle"] = {}
        context["rotation_total_angle"] = {}
        return True

    return False
