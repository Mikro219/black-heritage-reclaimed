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
  Tipping validation: wrist (lm 0) must be at or below the fingertip (y-wise) —
  the natural orientation when a hand wraps a flask and tips it to the mouth.

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
    require_curl = params.get("require_curl", False)

    # Use live pose mouth landmarks (9=left, 10=right) when available;
    # fall back to hardcoded defaults or explicit param overrides.
    # Use live pose mouth landmarks (9=left, 10=right) when available.
    # Both hand and pose landmarks are raw MediaPipe coordinates (same space),
    # so no mirroring needed here — the flip only applies in the render engine.
    pose_lm = context.get("_pose_lm")
    if pose_lm is not None:
        mouth_x = (pose_lm[9].x + pose_lm[10].x) / 2
        mouth_y = (pose_lm[9].y + pose_lm[10].y) / 2
    else:
        mouth_x = params.get("mouth_x", _MOUTH_X)
        mouth_y = params.get("mouth_y", _MOUTH_Y)

    near = False
    best_dist = 9.0
    best_fail = ""
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
            if dist < best_dist:
                best_dist = dist
                if dist <= threshold:
                    # Tipping validation: drinking wraps the fingers around the
                    # flask at mouth level with the wrist AT OR BELOW them (y grows
                    # downward). The pre-playtest check demanded the opposite
                    # orientation and rejected every natural drink.
                    if wrist.y >= tip.y - 0.03:
                        near = True
                        best_fail = "ok"
                    else:
                        best_fail = f"wrist above tips (wrist.y={wrist.y:.2f} tip.y={tip.y:.2f})"
                else:
                    best_fail = f"too far (dist={dist:.3f} > threshold={threshold})"
        if near:
            break

    # Debug: print once per second when a hand is reasonably close
    _last_print = context.get("_debug_print_t", 0.0)
    now_debug = time.monotonic()
    if best_dist < threshold * 3 and now_debug - _last_print > 1.0:
        context["_debug_print_t"] = now_debug
        print(f"[mouth_proximity_tip] dist={best_dist:.3f}  mouth=({mouth_x:.2f},{mouth_y:.2f})  "
              f"threshold={threshold}  status={best_fail or 'no hand close enough'}")

    now = time.monotonic()
    if near:
        if context.get("tip_near_since") is None:
            context["tip_near_since"] = now
        return (now - context["tip_near_since"]) * 1000 >= hold_ms
    else:
        context["tip_near_since"] = None
        return False
