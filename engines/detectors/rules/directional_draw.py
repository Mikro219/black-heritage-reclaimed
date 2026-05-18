"""
directional_draw — accumulate index-fingertip motion over a sliding window and
fire when the dominant direction matches the declared direction within ±30°.

Used for AL-04-007 trace_dipper and AL-04-009 trace_star stroke chains.
Each stroke is its own detector invocation; the narration engine re-arms with a
fresh context dict between strokes so draw_history resets automatically.

Params:
  direction (str):       target direction. One of:
                           left | right | up | down |
                           up_left | up_right | down_left | down_right
  window_ms (float):     accumulation window in ms. Default 400.
  min_displacement (float): minimum displacement (0..1 screen fraction) before
                           evaluating angle. Default 0.04.

Coordinate convention — directions are in SCREEN / player space:
  The render engine displays landmarks at (1 - lm.x)*w, lm.y*h, which mirrors
  the x-axis so the player's left hand appears on the left of the screen.
  This detector applies the same flip so "left" means the cursor moves left,
  matching what the player sees and what the storyboard labels.

  Raw-to-screen mapping:
    screen_dx = -(lm.x_now - lm.x_past)   ← x flipped
    screen_dy =   lm.y_now - lm.y_past

Context keys: draw_history (list of [x, y, t]), draw_fired (bool)
"""

import math
import time

_DIRECTION_ANGLES: dict[str, float] = {
    "right":      0.0,
    "down_right": math.pi * 0.25,
    "down":       math.pi * 0.5,
    "down_left":  math.pi * 0.75,
    "left":       math.pi,           # also matches -math.pi
    "up_left":   -math.pi * 0.75,
    "up":        -math.pi * 0.5,
    "up_right":  -math.pi * 0.25,
}

_TOLERANCE_RAD = math.pi / 6   # ±30°
_MIN_DURATION_MS = 100          # need at least this span of history before evaluating


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return abs(d)


def detect(landmarks, params: dict, context: dict) -> bool:
    direction = params.get("direction", "right")
    window_ms = params.get("window_ms", 400)
    min_disp = params.get("min_displacement", 0.04)

    target_angle = _DIRECTION_ANGLES.get(direction, 0.0)

    if "draw_history" not in context:
        context["draw_history"] = []
        context["draw_fired"] = False

    # Stroke already fired — hold until narration engine resets context
    if context.get("draw_fired"):
        return False

    now = time.monotonic()

    # Record current index-fingertip position from the first visible hand
    if landmarks:
        for hand in landmarks:
            tip = hand.landmark[8]
            context["draw_history"].append([tip.x, tip.y, now])
            break

    # Prune history older than window
    cutoff = now - window_ms / 1000.0
    context["draw_history"] = [p for p in context["draw_history"] if p[2] >= cutoff]

    history = context["draw_history"]
    if len(history) < 2:
        return False

    span_ms = (history[-1][2] - history[0][2]) * 1000
    if span_ms < _MIN_DURATION_MS:
        return False

    # Displacement in raw MediaPipe coords, then flip x to screen space
    raw_dx = history[-1][0] - history[0][0]
    raw_dy = history[-1][1] - history[0][1]
    screen_dx = -raw_dx
    screen_dy = raw_dy

    magnitude = math.hypot(screen_dx, screen_dy)
    if magnitude < min_disp:
        return False

    angle = math.atan2(screen_dy, screen_dx)
    if _angle_diff(angle, target_angle) <= _TOLERANCE_RAD:
        context["draw_fired"] = True
        context["draw_history"] = []
        return True

    return False
