"""
directional_head_or_hand — head-tilt OR hand raised/positioned in a direction.
Used for Scene 3 `look_up` OI, Scene 6 `scan_left` / `scan_right` OIs.

Params:
  direction (str): "left" | "right" | "up" | "brow" | "ear". Required.
  hold_ms (int): ms the pose must be maintained. Default 500.
  require_curl (bool): if True, require 2+ fingers to be curled (not extended past PIP).
                       Used for cup_ear_listen to distinguish a cupped hand from a point.
  ear_positions (list): list of (x, y) tuples for ear proximity check. Defaults to
                        estimated positions [(0.15, 0.35), (0.85, 0.35)]. Override with
                        live pose landmarks for accurate per-player tracking.
  brow_y (float): absolute Y threshold for "brow" direction. Default 0.38. Override with
                  live pose eye-level Y for accurate per-player tracking.

Approach (hand-position heuristics):
  "up"    — any wrist is in the upper 35% of frame (y < 0.35).
  "left"  — any wrist is in the left 30% of frame (x < 0.30).
  "right" — any wrist is in the right 30% of frame (x > 0.70).
  "brow"  — any wrist is at brow/forehead level (y < brow_y, default 0.38). shade_eyes.
  "ear"   — index or middle fingertip within 0.15 normalised distance of an ear position.

Context keys: directional_since
"""

import math
import time

_EAR_POSITIONS = [(0.15, 0.35), (0.85, 0.35)]  # left ear, right ear (normalised)

_TIP_PIP = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (tip_idx, pip_idx) per finger


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks:
        context["directional_since"] = None
        return False

    direction = params.get("direction", "up")
    hold_ms = params.get("hold_ms", 500)
    require_curl = params.get("require_curl", False)
    ear_positions = params.get("ear_positions", _EAR_POSITIONS)
    brow_y = params.get("brow_y", 0.38)
    matched = False

    for hand in landmarks:
        lm = hand.landmark
        wrist = lm[0]

        if direction == "up":
            matched = wrist.y < 0.35
        elif direction == "left":
            # Player's left = camera's right (high raw x), display is mirrored
            matched = wrist.x > 0.70
        elif direction == "right":
            # Player's right = camera's left (low raw x), display is mirrored
            matched = wrist.x < 0.30
        elif direction == "brow":
            matched = wrist.y < brow_y
        elif direction == "ear":
            for tip_idx in (8, 12):  # index tip, middle tip
                tip = lm[tip_idx]
                for ex, ey in ear_positions:
                    if math.hypot(tip.x - ex, tip.y - ey) < 0.15:
                        matched = True
                        break
                if matched:
                    break

        if matched and require_curl:
            # Require 2+ fingers curled (tip not extended past PIP joint)
            curl_count = sum(1 for tip_i, pip_i in _TIP_PIP
                             if lm[tip_i].y >= lm[pip_i].y)
            if curl_count < 2:
                matched = False

        if matched:
            break

    now = time.monotonic()
    if matched:
        if context.get("directional_since") is None:
            context["directional_since"] = now
        return (now - context["directional_since"]) * 1000 >= hold_ms
    else:
        context["directional_since"] = None
        return False
