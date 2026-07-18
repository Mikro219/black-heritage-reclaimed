"""
bilateral_alternating — alternating wrist motion for ≥ min_cycles.
Used for Scene 9 Path B walk CG (slow, ≥3 cycles).

POSE-ONLY (July 2026): tracks Pose wrists (15/16) by side; the crossing line
is the live body midline from the pose shoulders instead of fixed x=0.5.

Strokes are counted inside a rolling window and all state resets on wrist
dropout (review fix, July 2026): an unbounded count let crossings spread over
a whole OI window sum to a fire, and a stale pre-dropout `prev_x` could
register a phantom crossing when tracking returned.

Params:
  min_cycles (int): L/R alternating strokes required. Default 3.
  cadence (str | None): "walk" (inter-stroke > 0.35s) | "run" (< 0.55s) | None.
  stroke_window_ms (int): rolling window the strokes must fall within.
                          Default 8000.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: alt_strokes, alt_last_hand, alt_last_time, alt_prev_x
  (reads: _pose_lm)
"""

import time

from . import pose_helpers


def _reset(context: dict) -> None:
    context["alt_strokes"] = []
    context["alt_last_hand"] = None
    context["alt_last_time"] = None
    context["alt_prev_x"] = {}


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    geom = pose_helpers.geometry(pose_lm)
    if len(wrists) < 2 or geom is None:
        _reset(context)   # dropout: stale prev_x must not fake a crossing
        return False

    if "alt_strokes" not in context:
        _reset(context)

    min_cycles = params.get("min_cycles", 3)
    window_s = params.get("stroke_window_ms", 8000) / 1000.0
    cadence = params.get("cadence")
    midline = geom["midline_x"]

    now = time.monotonic()
    prev_x = context["alt_prev_x"]

    for side, wrist in wrists.items():
        x = wrist.x
        p = prev_x.get(side)
        if p is None:
            prev_x[side] = x
            continue

        crossed = (p < midline <= x) or (x < midline <= p)
        if crossed and context["alt_last_hand"] != side:
            last_t = context["alt_last_time"]
            dt = (now - last_t) if last_t is not None else None

            ok = True
            if cadence == "walk" and dt is not None and dt < 0.35:
                ok = False
            elif cadence == "run" and dt is not None and dt > 0.55:
                ok = False

            if ok:
                context["alt_strokes"].append(now)
                context["alt_last_hand"] = side
                context["alt_last_time"] = now

        prev_x[side] = x

    # Rolling window: only recent strokes count toward the cycle target.
    cutoff = now - window_s
    context["alt_strokes"] = [t for t in context["alt_strokes"] if t >= cutoff]
    if len(context["alt_strokes"]) >= min_cycles * 2:   # each hand counts
        _reset(context)
        return True

    context["alt_prev_x"] = prev_x
    return False
