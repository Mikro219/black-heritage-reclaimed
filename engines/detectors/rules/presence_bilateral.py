"""
presence_bilateral — both pose wrists present (trusted), optional Y-position
constraint, held for hold_ms. Used for AL-01-007 raise_hands and the tutorial's
"raise both hands" step.

POSE-ONLY (July 2026): reads Pose wrists (15/16) from context["_pose_lm"];
the Hands model is gone from the runtime.

Params:
  y_threshold (str|float): named threshold ("shoulder"/"head" — resolved to the
                            live pose line by _inject_pose_params) or a raw
                            normalised Y value (0=top). Wrists must be ABOVE
                            (smaller Y than) this. Default: no constraint.
  hold_ms (int): ms both wrists must continuously qualify. Default 500.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: bilateral_present_since  (reads: _pose_lm)
"""

import time

from . import pose_helpers

_NAMED_THRESHOLDS = {
    "shoulder": 0.5,   # fallback when _inject_pose_params found no pose
    "head":     0.3,
    "waist":    0.7,
}


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if len(wrists) < 2:
        context["bilateral_present_since"] = None
        return False

    raw = params.get("y_threshold", params.get("y_min_normalized", 0.0))
    if isinstance(raw, str):
        y_max = _NAMED_THRESHOLDS.get(raw, 0.5)
    else:
        y_max = float(raw)

    if y_max > 0.0 and any(w.y > y_max for w in wrists.values()):
        context["bilateral_present_since"] = None
        return False

    hold_ms = params.get("hold_ms", 500)
    now = time.monotonic()
    if context.get("bilateral_present_since") is None:
        context["bilateral_present_since"] = now
    return (now - context["bilateral_present_since"]) * 1000 >= hold_ms
