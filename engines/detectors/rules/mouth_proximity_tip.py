"""
mouth_proximity_tip — hand tip near mouth with tipping/tilting wrist motion.
Used for Scene 1 drink OI (AL-01-002).

Params:
  hold_ms (int): ms tip must remain near mouth region. Default 400.
  proximity_threshold (float): max normalised Euclidean distance tip→mouth. Default 0.12.
  mouth_x (float): normalised x of mouth centre. Default 0.50. Override with live pose
                   value for accurate tracking (tuner injects pose landmark 9/10 midpoint).
  mouth_y (float): normalised y of mouth centre. Default 0.78. Override with live pose.
  require_curl (bool): if True, require 2+ fingers curled (not extended past PIP).
                       Distinguishes a cupped/tipping hand from an open flat hand.

Approach:
  We check the index fingertip (lm 8) and thumb tip (lm 4) on any hand.
  Tipping validation: wrist (lm 0) must be above the fingertip (y-wise), indicating
  an upward tilt consistent with bringing a cup to the mouth.

Context keys: tip_near_since
"""

import math
import time

_MOUTH_X = 0.50
_MOUTH_Y = 0.78

_TIP_PIP = [(8, 6), (12, 10), (16, 14), (20, 18)]


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks:
        context["tip_near_since"] = None
        return False

    threshold = params.get("proximity_threshold", 0.12)
    hold_ms = params.get("hold_ms", 400)
    mouth_x = params.get("mouth_x", _MOUTH_X)
    mouth_y = params.get("mouth_y", _MOUTH_Y)
    require_curl = params.get("require_curl", False)

    near = False
    for hand in landmarks:
        lm = hand.landmark
        wrist = lm[0]

        if require_curl:
            curl_count = sum(1 for tip_i, pip_i in _TIP_PIP
                             if lm[tip_i].y >= lm[pip_i].y)
            if curl_count < 2:
                continue

        for tip_idx in (8, 4):  # index tip, thumb tip
            tip = lm[tip_idx]
            dist = math.hypot(tip.x - mouth_x, tip.y - mouth_y)
            if dist <= threshold:
                # Tipping check: wrist above tip (wrist.y < tip.y)
                if wrist.y < tip.y:
                    near = True
                    break
        if near:
            break

    now = time.monotonic()
    if near:
        if context.get("tip_near_since") is None:
            context["tip_near_since"] = now
        return (now - context["tip_near_since"]) * 1000 >= hold_ms
    else:
        context["tip_near_since"] = None
        return False
