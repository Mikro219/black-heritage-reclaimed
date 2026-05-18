"""
point_target_held — index finger pointing at a named screen region, held for hold_ms.
Used for Scene 1 point-quilt-block OI (AL-01-010), Scene 4 North Star OI (AL-04-012).

Params:
  target_region (str): key into context["target_regions"] rect dict.
                       If absent or context key missing, fires on any point in lower half.
  hold_ms (int): ms the point must stay on target. Default 800.

Target region rect format (set by render engine):
  { "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0 }  — normalised screen coords.

Approach:
  Extend a ray from wrist (lm 0) through index fingertip (lm 8) and project
  forward to the screen plane (z=0). The projected point is computed as:
      pt = tip + (tip - wrist) * k
  where k extends the ray to the screen. We approximate by using the
  fingertip position directly (k=0) — good enough for a fixed-distance screen.

Context keys: point_on_target_since
"""

import time


def _point_in_rect(px: float, py: float, rect: dict) -> bool:
    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
    return x <= px <= x + w and y <= py <= y + h


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks:
        context["point_on_target_since"] = None
        return False

    hold_ms = params.get("hold_ms", 800)
    target_key = params.get("target_region")
    target_regions: dict = context.get("target_regions", {})

    # Pick best hand: prefer the one with lowest index-finger curl
    best_tip = None
    for hand in landmarks:
        lm = hand.landmark
        tip = lm[8]   # index fingertip
        pip = lm[6]   # index PIP
        # Extended if tip is above (lower y) pip — finger is pointing
        if tip.y < pip.y:
            best_tip = tip
            break

    if best_tip is None:
        context["point_on_target_since"] = None
        return False

    # Check against named region or fall back to lower-screen quadrant
    if target_key and target_key in target_regions:
        on_target = _point_in_rect(best_tip.x, best_tip.y, target_regions[target_key])
    else:
        # Fallback: any point in the lower 60% of screen (the "scene" area)
        on_target = best_tip.y > 0.4

    now = time.monotonic()
    if on_target:
        if context.get("point_on_target_since") is None:
            context["point_on_target_since"] = now
        return (now - context["point_on_target_since"]) * 1000 >= hold_ms
    else:
        context["point_on_target_since"] = None
        return False
