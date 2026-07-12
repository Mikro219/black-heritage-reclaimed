"""
mouth_proximity_tip — hand raised to the mouth in a drinking tip.
Used for Scene 1 drink OI (AL-01-002).

POSE-ONLY (July 2026): the "tip" is the Pose hand point — the index landmark
(19/20) when visible, else the wrist. Mouth position comes from Pose mouth
corners (9/10). Tipping validation is unchanged in spirit: the wrist must sit
AT OR BELOW the hand point (y grows downward) — the natural orientation when a
hand wraps a flask and tips it up to the mouth.

Params:
  hold_ms (int): ms the hand must remain near the mouth. Default 400.
  proximity_threshold (float): max normalised distance hand-point→mouth.
                  Default 0.14 (slightly wider than the old fingertip check —
                  the pose index landmark sits at the knuckle, not the tip).
  mouth_x / mouth_y (float): fallback mouth position when Pose mouth landmarks
                  are unavailable. Defaults 0.50 / 0.78.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: tip_near_since  (reads: _pose_lm)
"""

import math
import time

from . import pose_helpers

_MOUTH_X = 0.50
_MOUTH_Y = 0.78


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if not wrists:
        context["tip_near_since"] = None
        return False

    threshold = params.get("proximity_threshold", 0.14)
    hold_ms = params.get("hold_ms", 400)

    mouth_l = pose_helpers.lm(pose_lm, 9)
    mouth_r = pose_helpers.lm(pose_lm, 10)
    if mouth_l is not None and mouth_r is not None:
        mouth_x = (mouth_l.x + mouth_r.x) / 2
        mouth_y = (mouth_l.y + mouth_r.y) / 2
    else:
        mouth_x = params.get("mouth_x", _MOUTH_X)
        mouth_y = params.get("mouth_y", _MOUTH_Y)

    near = False
    for side, wrist in wrists.items():
        tip = pose_helpers.hand_point(pose_lm, side) or wrist
        dist = math.hypot(tip.x - mouth_x, tip.y - mouth_y)
        if dist > threshold:
            continue
        # Tipping: wrist at or below the hand point — the drink orientation.
        if wrist.y >= tip.y - 0.03:
            near = True
            break

    now = time.monotonic()
    if near:
        if context.get("tip_near_since") is None:
            context["tip_near_since"] = now
        return (now - context["tip_near_since"]) * 1000 >= hold_ms

    context["tip_near_since"] = None
    return False
