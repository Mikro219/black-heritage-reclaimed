"""
bilateral_arcing — bilateral high-to-low forward arc (paddle stroke), N times.
Used as an alternative paddle model (Scene 11).

POSE-ONLY (July 2026): tracks Pose wrists (15/16) by side.

Strokes are counted inside a rolling window and phase state resets on wrist
dropout (review fix, July 2026): the count previously never decayed, so
arbitrarily slow, unrelated up/down drifts spread across a long window could
sum to a fire, and a stale phase could complete a phantom stroke when
tracking returned.

Params:
  strokes (int): complete down-then-up cycles required. Default 4.
  stroke_window_ms (int): rolling window the strokes must fall within.
                          Default 12000.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: arc_strokes, arc_phase, arc_prev_y  (reads: _pose_lm)
"""

import time

from . import pose_helpers


def _reset(context: dict) -> None:
    context["arc_strokes"] = []
    context["arc_phase"] = {}     # side → "up" | "down"
    context["arc_prev_y"] = {}


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if len(wrists) < 2:
        _reset(context)   # dropout: a stale phase must not finish a stroke
        return False

    if "arc_strokes" not in context:
        _reset(context)

    strokes = params.get("strokes", 4)
    window_s = params.get("stroke_window_ms", 12000) / 1000.0
    phase = context["arc_phase"]
    prev_y = context["arc_prev_y"]
    now = time.monotonic()

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
        context["arc_strokes"].append(now)

    # Rolling window: only recent strokes count toward the target.
    cutoff = now - window_s
    context["arc_strokes"] = [t for t in context["arc_strokes"] if t >= cutoff]
    if len(context["arc_strokes"]) >= strokes:
        _reset(context)
        return True

    context["arc_prev_y"] = prev_y
    return False
