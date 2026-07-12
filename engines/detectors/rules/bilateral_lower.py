"""
bilateral_lower — both pose wrists moving downward together, then held low.
Used for Scene 7 `duck_down` CG (AL-07-011).

POSE-ONLY (July 2026): reads Pose wrists (15/16), keyed by side so the peaks
track the same physical wrist across frames.

Params:
  min_delta (float): minimum downward Y movement per wrist from its peak.
                     Default 0.06.
  hold_ms (int): ms both wrists must stay low after the drop. Default 400.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: peak_y, low_since  (reads: _pose_lm)
"""

import time

from . import pose_helpers


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if len(wrists) < 2:
        context["peak_y"] = None
        context["low_since"] = None
        return False

    min_delta = params.get("min_delta", 0.06)
    hold_ms = params.get("hold_ms", 400)

    current = {s: w.y for s, w in wrists.items()}
    peaks = context.get("peak_y")
    if not isinstance(peaks, dict) or set(peaks) != set(current):
        peaks = dict(current)

    # Track the highest position each wrist reached (lowest y value)
    peaks = {s: min(peaks[s], current[s]) for s in current}
    context["peak_y"] = peaks

    both_low = all(current[s] > peaks[s] + min_delta and current[s] > 0.45
                   for s in current)

    now = time.monotonic()
    if both_low:
        if context.get("low_since") is None:
            context["low_since"] = now
        return (now - context["low_since"]) * 1000 >= hold_ms

    context["low_since"] = None
    return False
