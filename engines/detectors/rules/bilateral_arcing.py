"""
bilateral_arcing — bilateral high-to-low forward arc (paddle stroke), N times.
Used as an alternative paddle model (Scene 11).

POSE-ONLY (July 2026): tracks Pose wrists (15/16) by side.

Params:
  strokes (int): complete down-then-up cycles required. Default 4.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: arc_stroke_count, arc_phase, arc_prev_y  (reads: _pose_lm)
"""

from . import pose_helpers


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if len(wrists) < 2:
        return False

    if "arc_stroke_count" not in context:
        context["arc_stroke_count"] = 0
        context["arc_phase"] = {}     # side → "up" | "down"
        context["arc_prev_y"] = {}

    strokes = params.get("strokes", 4)
    phase = context["arc_phase"]
    prev_y = context["arc_prev_y"]

    completed = []
    for side, wrist in wrists.items():
        y = wrist.y
        if phase.get(side) is None:
            phase[side] = "up" if y < 0.5 else "down"
        elif phase[side] == "up" and y > 0.65:
            phase[side] = "down"
        elif phase[side] == "down" and y < 0.45:
            phase[side] = "up"
            completed.append(side)
        prev_y[side] = y

    if len(completed) == 2:
        context["arc_stroke_count"] += 1
        if context["arc_stroke_count"] >= strokes:
            context["arc_stroke_count"] = 0
            return True

    context["arc_prev_y"] = prev_y
    return False
