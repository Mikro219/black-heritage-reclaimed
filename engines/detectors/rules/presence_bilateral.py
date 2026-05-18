"""
presence_bilateral — both hands present, optional Y-position constraint, held for hold_ms.

Params:
  y_threshold (str|float): named threshold ("shoulder" ≈ 0.5, "head" ≈ 0.3) or
                            a raw normalised Y value (0=top, 1=bottom).
                            Wrist landmark must be ABOVE (lower Y) this threshold.
                            Default: no constraint (0.0 = anywhere is fine).
  hold_ms (int): ms both hands must be continuously present. Default 500.

Context keys: bilateral_present_since
"""

import time

_NAMED_THRESHOLDS = {
    "shoulder": 0.5,   # wrist Y must be < 0.5 (upper half)
    "head":     0.3,   # wrist Y must be < 0.3 (top third)
    "waist":    0.7,   # wrist Y must be < 0.7
}


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks or len(landmarks) < 2:
        context["bilateral_present_since"] = None
        return False

    raw = params.get("y_threshold", params.get("y_min_normalized", 0.0))
    if isinstance(raw, str):
        y_max = _NAMED_THRESHOLDS.get(raw, 0.5)
    else:
        y_max = float(raw)

    if y_max > 0.0:
        for hand in landmarks:
            if hand.landmark[0].y > y_max:  # wrist too low
                context["bilateral_present_since"] = None
                return False

    hold_ms = params.get("hold_ms", 500)
    now = time.monotonic()
    if context.get("bilateral_present_since") is None:
        context["bilateral_present_since"] = now
    return (now - context["bilateral_present_since"]) * 1000 >= hold_ms
