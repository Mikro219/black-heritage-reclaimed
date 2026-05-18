"""
directional_point — single hand pointing in a direction, held for hold_ms.
Used for Scene 2 `point_path` CG, Scene 7 `point_fork` CG, Scene 9 `choose_path` CG.

Params:
  directions (list[str]): accepted directions — any subset of the 8 compass points:
                          "left" | "right" | "up" | "down" |
                          "up_left" | "up_right" | "down_left" | "down_right"
                          If absent or empty, all 8 directions are accepted.
  hold_ms (int): ms the point must be held. Default 500.
  target_x (float): optional screen-space x (0–1, player perspective) of the target
                    marker. If provided along with target_y, the fingertip must be
                    within proximity_threshold of the marker to register.
  target_y (float): optional screen-space y (0–1) of the target marker.
  proximity_threshold (float): max normalised distance from marker. Default 0.30.

Approach:
  Wrist (lm 0) → index fingertip (lm 8) vector; snap to the nearest of 8 compass
  directions using angle sectors of 45° each.
  Extension check: tip must be farther from wrist than PIP joint (distance-based,
  works for all directions including down).

Context keys: point_direction_since, point_direction
"""

import math
import time

_ALL_DIRECTIONS = ["left", "right", "up", "down",
                   "up_left", "up_right", "down_left", "down_right"]

# 8 compass directions in counterclockwise order starting from +x (right)
_DIRS_CW = ["right", "down_right", "down", "down_left",
            "left", "up_left", "up", "up_right"]


def _classify_direction(dx: float, dy: float) -> str:
    """Snap the (dx, dy) vector to the nearest of 8 compass directions.

    dx/dy are already in player screen-space (x flipped, y down=positive).
    Each direction occupies a 45° sector centred on the named angle.
    """
    angle = math.degrees(math.atan2(dy, dx)) % 360
    return _DIRS_CW[int((angle + 22.5) / 45) % 8]


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks:
        context["point_direction_since"] = None
        context["point_direction"] = None
        return False

    accepted = set(params.get("directions") or _ALL_DIRECTIONS)
    hold_ms = params.get("hold_ms", 500)
    target_x = params.get("target_x")
    target_y = params.get("target_y")
    proximity_threshold = params.get("proximity_threshold", 0.30)

    matched_dir = None
    for hand in landmarks:
        lm = hand.landmark
        wrist = lm[0]
        tip = lm[8]   # index fingertip
        pip = lm[6]   # index PIP joint

        # Require extended index finger: tip must be farther from wrist than PIP.
        # Distance-based check works for all directions including downward pointing.
        wrist_tip_sq = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
        wrist_pip_sq = (pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2
        if wrist_tip_sq <= wrist_pip_sq:
            continue

        # Optional proximity check: fingertip must be near the on-screen target marker.
        if target_x is not None and target_y is not None:
            screen_tip_x = 1.0 - tip.x   # mirror to player space
            screen_tip_y = tip.y
            if math.hypot(screen_tip_x - target_x, screen_tip_y - target_y) > proximity_threshold:
                continue

        # Negate x so directions match the player's mirrored perspective:
        # player points RIGHT → cursor moves right on screen → -raw_dx > 0 → "right"
        dx = -(tip.x - wrist.x)
        dy = tip.y - wrist.y
        direction = _classify_direction(dx, dy)
        if direction in accepted:
            matched_dir = direction
            break

    now = time.monotonic()
    if matched_dir:
        prev_dir = context.get("point_direction")
        if prev_dir != matched_dir:
            # Direction changed — reset timer
            context["point_direction"] = matched_dir
            context["point_direction_since"] = now
        return (now - context["point_direction_since"]) * 1000 >= hold_ms
    else:
        context["point_direction_since"] = None
        context["point_direction"] = None
        return False
