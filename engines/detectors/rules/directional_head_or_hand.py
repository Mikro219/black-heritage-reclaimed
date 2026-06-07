"""
directional_head_or_hand — head-tilt OR hand raised/positioned in a direction.
Used for Scene 3 `look_up` OI, Scene 6 `scan_left` / `scan_right` OIs,
Scene 2 `cup_ear_listen` OI.

Params:
  direction (str): "left" | "right" | "up" | "brow" | "ear". Required.
  hold_ms (int): ms the pose must be maintained. Default 500.
  require_curl (bool): if True, require 2+ fingers to be curled (not extended past PIP).
                       Rarely needed — no longer used for cup_ear_listen (see below).
  ear_positions (list): list of (x, y) tuples for ear proximity check. Defaults to
                        estimated positions [(0.15, 0.35), (0.85, 0.35)]. Override with
                        live pose landmarks for accurate per-player tracking.
  ear_radius (float): normalised distance from ear position that counts as "near ear."
                      Default 0.22. Wider than the original fingertip check (was 0.15)
                      because we now use the wrist landmark — which is farther from the
                      ear than the fingertips when the hand is cupped to the ear.
  brow_y (float): absolute Y threshold for "brow" direction. Default 0.38. Override with
                  live pose eye-level Y for accurate per-player tracking.

"ear" direction (cup-ear gesture):
  Uses wrist proximity rather than fingertip proximity. Either hand, either ear.
  No palm-orientation or curl check required — any hand near the ear counts.
  This is intentionally loose (public installation, untrained visitors).

Approach (hand-position heuristics):
  "up"    — any wrist is in the upper 35% of frame (y < 0.35).
  "left"  — any wrist is in the left 30% of frame (x < 0.30).
  "right" — any wrist is in the right 30% of frame (x > 0.70).
  "brow"  — any wrist is at brow/forehead level (y < brow_y, default 0.38). shade_eyes.
  "ear"   — wrist within ear_radius (default 0.22) of either ear position. Either hand.

Context keys: directional_since
"""

import math
import time

_EAR_POSITIONS = [(0.15, 0.35), (0.85, 0.35)]  # left ear, right ear (normalised)
_EAR_RADIUS_DEFAULT = 0.22  # wrist proximity; wider than original 0.15 fingertip check

_TIP_PIP = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (tip_idx, pip_idx) per finger


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks:
        context["directional_since"] = None
        return False

    direction = params.get("direction", "up")
    hold_ms = params.get("hold_ms", 500)
    require_curl = params.get("require_curl", False)
    ear_positions = params.get("ear_positions", _EAR_POSITIONS)
    ear_radius = params.get("ear_radius", _EAR_RADIUS_DEFAULT)
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
            # Wrist proximity to either ear — loose check, no palm orientation required.
            wrist = lm[0]
            for ex, ey in ear_positions:
                if math.hypot(wrist.x - ex, wrist.y - ey) < ear_radius:
                    matched = True
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
