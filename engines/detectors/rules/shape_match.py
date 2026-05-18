"""
shape_match — path-recorded hand trace evaluated against a reference template.
Used for Scene 4: dipper trace (≥50%) and star trace (≥65%).

Params:
  shape (str): "dipper" | "star"
  threshold (float): override match score (defaults from config thresholds).

Recording:
  Captures index-fingertip XY while hand is active (wrist above y=0.6).
  Evaluates when hand drops below y=0.6 for PAUSE_THRESHOLD_MS.

Matching:
  The recorded path is resampled to N points and compared against a reference
  template using a directional similarity score — fraction of consecutive
  segments whose direction is within ±45° of the template direction.

  This is intentionally permissive (per CLAUDE.md: "detect intent, not precision").

Context keys: shape_recording, shape_hand_down_since
"""

import math
import time

PAUSE_THRESHOLD_MS = 500
_ACTIVE_Y = 0.6  # wrist below this = hand lowered

# Reference templates as normalised direction sequences (angle in radians)
# Dipper: bowl (curved CCW ~4 pts) + handle (3 pts angling lower-right)
# Values are approximate — match is directional, not positional
_DIPPER_DIRECTIONS = [
    0.0, math.pi * 0.5, math.pi, math.pi * 1.5,  # bowl quadrants
    math.pi * 0.25, math.pi * 0.35, math.pi * 0.45,  # handle
]

# Star: 5 segments of a 5-pointed star (skip-every-other-point draw order)
_STAR_DIRECTIONS = [
    -math.pi * 0.5,       # up
    math.pi * 0.7,        # lower-right
    -math.pi * 0.1,       # right-upper
    math.pi * 1.1,        # lower-left
    math.pi * 0.3,        # upper-right close
]

_TEMPLATES = {
    "dipper": _DIPPER_DIRECTIONS,
    "star": _STAR_DIRECTIONS,
}

_THRESHOLDS = {
    "dipper": 0.50,
    "star": 0.65,
}


def _resample(path: list[tuple], n: int) -> list[tuple]:
    """Resample a list of (x, y) points to exactly n uniformly-spaced points."""
    if len(path) < 2:
        return path
    total = sum(
        math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        for i in range(len(path) - 1)
    )
    if total == 0:
        return [path[0]] * n
    step = total / (n - 1)
    result = [path[0]]
    acc = 0.0
    i = 0
    while len(result) < n and i < len(path) - 1:
        seg = math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        while acc + seg >= step and len(result) < n:
            t = (step - acc) / seg
            x = path[i][0] + t * (path[i+1][0] - path[i][0])
            y = path[i][1] + t * (path[i+1][1] - path[i][1])
            result.append((x, y))
            acc -= step
        acc += seg
        i += 1
    while len(result) < n:
        result.append(path[-1])
    return result


def _direction_match(path: list[tuple], template_dirs: list[float]) -> float:
    """Fraction of path segments within ±60° of the corresponding template direction."""
    n = len(template_dirs) + 1
    pts = _resample(path, n)
    tol = math.pi / 3  # 60°
    matches = 0
    for i, tdir in enumerate(template_dirs):
        dx = pts[i+1][0] - pts[i][0]
        dy = pts[i+1][1] - pts[i][1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        angle = math.atan2(dy, dx)
        diff = abs(_wrapped_diff(angle, tdir))
        if diff <= tol:
            matches += 1
    return matches / len(template_dirs) if template_dirs else 0.0


def _wrapped_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def detect(landmarks, params: dict, context: dict) -> bool:
    if "shape_recording" not in context:
        context["shape_recording"] = []
        context["shape_hand_down_since"] = None

    shape = params.get("shape", "star")
    template = _TEMPLATES.get(shape)
    if template is None:
        return False

    threshold = params.get("threshold", _THRESHOLDS.get(shape, 0.60))

    hand_active = False
    if landmarks:
        for hand in landmarks:
            if hand.landmark[0].y < _ACTIVE_Y:  # wrist in upper portion
                tip = hand.landmark[8]
                context["shape_recording"].append((tip.x, tip.y))
                hand_active = True
                break

    now = time.monotonic()
    if not hand_active:
        if context["shape_hand_down_since"] is None:
            context["shape_hand_down_since"] = now
        elif (now - context["shape_hand_down_since"]) * 1000 >= PAUSE_THRESHOLD_MS:
            # Evaluate
            path = context["shape_recording"]
            context["shape_recording"] = []
            context["shape_hand_down_since"] = None
            if len(path) >= len(template) + 1:
                score = _direction_match(path, template)
                if score >= threshold:
                    return True
    else:
        context["shape_hand_down_since"] = None

    return False
