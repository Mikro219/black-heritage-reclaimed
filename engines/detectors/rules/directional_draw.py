"""
directional_draw — accumulate index-fingertip motion over a sliding window and
fire when the dominant direction matches the declared direction within ±30°.

Used for AL-04-007 trace_dipper and AL-04-009 trace_star stroke chains.
Each stroke is its own detector invocation; the narration engine re-arms with a
fresh context dict between strokes so draw_history resets automatically.

POSE-WRIST PRIMARY (July 2026 playtest): tracing with an outstretched arm
routinely defeats the Hands model (foreshortening, face-adjacent hand, motion
blur), which is why the stroke chains stalled mid-star — no fingertip samples,
no stroke, 30s fallback timeout. Motion direction doesn't need a fingertip:
the detector now tracks the most visible Pose WRIST by default and only falls
back to the index fingertip when Pose is unavailable. It runs in the gesture
engine's _NO_HANDS_DETECTORS set. Set use_pose: false for fingertip-only.

Params:
  direction (str):       target direction. One of:
                           left | right | up | down |
                           up_left | up_right | down_left | down_right
  window_ms (float):     accumulation window in ms. Default 400.
  min_displacement (float): minimum displacement (0..1 screen fraction) before
                           evaluating angle. Default 0.04.
  tolerance_deg (float): angular tolerance around the target direction.
                           Default 30 (matches detection_thresholds.
                           directional_draw_tolerance_deg).
  use_pose (bool):       track Pose wrists instead of the fingertip. Default True.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

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

from ...depth.fusion import trusted_landmark

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

_MIN_DURATION_MS = 100          # need at least this span of history before evaluating

_POSE_WRISTS = (15, 16)


def _tracked_points(landmarks, params: dict, context: dict) -> dict:
    """{track_key: (x, y)} of every point whose motion we accumulate this tick.
    Pose mode tracks BOTH visible wrists independently — either hand can draw
    the stroke, and a static second hand must not mask the moving one."""
    points: dict = {}
    if params.get("use_pose", True):
        pose_lm = context.get("_pose_lm")
        if pose_lm is not None:
            min_visibility = params.get("min_visibility", 0.5)
            for idx in _POSE_WRISTS:
                try:
                    w = pose_lm[idx]
                except (IndexError, TypeError):
                    continue
                if trusted_landmark(context, idx, w, min_visibility):
                    points[f"wrist_{idx}"] = (w.x, w.y)
            if points:
                return points
    for hand in landmarks or []:
        tip = hand.landmark[8]
        points["fingertip"] = (tip.x, tip.y)
        break
    return points


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
    tolerance_rad = math.radians(params.get("tolerance_deg", 30))

    target_angle = _DIRECTION_ANGLES.get(direction, 0.0)

    if "draw_history" not in context or not isinstance(context["draw_history"], dict):
        context["draw_history"] = {}
        context["draw_fired"] = False

    # Stroke already fired — hold until narration engine resets context
    if context.get("draw_fired"):
        return False

    now = time.monotonic()
    histories: dict = context["draw_history"]

    for key, (px, py) in _tracked_points(landmarks, params, context).items():
        histories.setdefault(key, []).append([px, py, now])

    # Prune each track's history to the window, then evaluate independently.
    cutoff = now - window_ms / 1000.0
    for key in list(histories):
        histories[key] = [p for p in histories[key] if p[2] >= cutoff]
        history = histories[key]
        if len(history) < 2:
            continue
        span_ms = (history[-1][2] - history[0][2]) * 1000
        if span_ms < _MIN_DURATION_MS:
            continue

        # Displacement in raw MediaPipe coords, then flip x to screen space
        screen_dx = -(history[-1][0] - history[0][0])
        screen_dy = history[-1][1] - history[0][1]

        magnitude = math.hypot(screen_dx, screen_dy)
        if magnitude < min_disp:
            continue

        angle = math.atan2(screen_dy, screen_dx)
        if _angle_diff(angle, target_angle) <= tolerance_rad:
            context["draw_fired"] = True
            context["draw_history"] = {}
            return True

    return False
