"""
presence_bilateral — both hands present, optional Y-position constraint, held for hold_ms.

Params:
  y_min_normalized (float): wrist must be above this normalised Y (0=top, 1=bottom). Default 0.0.
  hold_ms (int): ms both hands must be continuously present. Default 500.

Context keys: bilateral_present_since
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks or len(landmarks) < 2:
        context["bilateral_present_since"] = None
        return False

    y_min = params.get("y_min_normalized", 0.0)
    for hand in landmarks:
        if hand.landmark[0].y > (1.0 - y_min):
            context["bilateral_present_since"] = None
            return False

    hold_ms = params.get("hold_ms", 500)
    now = time.monotonic()
    if context.get("bilateral_present_since") is None:
        context["bilateral_present_since"] = now
    return (now - context["bilateral_present_since"]) * 1000 >= hold_ms
