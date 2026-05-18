"""
bilateral_lower — both hands moving downward together, then held low.
Used for Scene 7 `duck_down` CG (AL-07-011).

Params:
  min_delta (float): minimum downward Y movement per hand to count as "lowering". Default 0.06.
  hold_ms (int): ms both wrists must stay below mid-screen after the downward motion. Default 400.

Approach:
  Record wrist Y positions each frame. When both hands' Y increases (moves down) by
  min_delta from their peak and are both below 0.55 (mid-to-lower screen), start hold timer.

Context keys: prev_wrist_y, peak_y, low_since
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks or len(landmarks) < 2:
        context["prev_wrist_y"] = []
        context["peak_y"] = []
        context["low_since"] = None
        return False

    min_delta = params.get("min_delta", 0.06)
    hold_ms = params.get("hold_ms", 400)

    current_y = [hand.landmark[0].y for hand in landmarks]

    prev_y = context.get("prev_wrist_y", current_y[:])
    peak_y = context.get("peak_y", current_y[:])

    # Update peaks (track highest position = lowest y value)
    new_peaks = [min(p, c) for p, c in zip(peak_y, current_y)]

    # Both hands moved down from peak by min_delta and are in lower half
    both_low = all(
        cy > py + min_delta and cy > 0.45
        for cy, py in zip(current_y, new_peaks)
    )

    now = time.monotonic()
    if both_low:
        if context.get("low_since") is None:
            context["low_since"] = now
        fired = (now - context["low_since"]) * 1000 >= hold_ms
    else:
        context["low_since"] = None
        fired = False

    context["prev_wrist_y"] = current_y
    context["peak_y"] = new_peaks
    return fired
