"""
point_target_held — index finger pointing at a named screen region, held for hold_ms.
Used for Scene 1 point-quilt-block OI (AL-01-010), Scene 4 North Star OI (AL-04-012),
Scene 3 point-river-marker OI (AL-03-006).

Params:
  target_region (str):   key into context["target_regions"] rect dict.
  region_rect (dict):    inline rect {"x","y","w","h"} in normalised screen coords.
  directions (list[str]): if set, the wrist→tip vector must classify into one of these
                          8-way directions: "up","down","left","right",
                          "up_left","up_right","down_left","down_right".
                          Checked in addition to region_rect when both are present.
  hold_ms (int):         ms the point must stay on target. Default 800.

If neither region_rect nor directions is specified, fires on any point in the lower half.

Context keys: point_on_target_since
"""

import math
import time

_DIRS_CW = ["right", "down_right", "down", "down_left",
            "left",  "up_left",   "up",   "up_right"]


def _point_in_rect(px: float, py: float, rect: dict) -> bool:
    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
    return x <= px <= x + w and y <= py <= y + h


def _classify_direction(wrist, tip) -> str:
    """Classify the wrist→tip vector into one of 8 compass directions.
    X is flipped (dx = -(tip.x - wrist.x)) to match player-perspective
    mirrored display, consistent with directional_point.py."""
    dx = -(tip.x - wrist.x)
    dy =   tip.y - wrist.y
    angle = math.degrees(math.atan2(dy, dx)) % 360
    return _DIRS_CW[int((angle + 22.5) / 45) % 8]


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks:
        context["point_on_target_since"] = None
        return False

    hold_ms    = params.get("hold_ms", 800)
    target_key = params.get("target_region")
    directions = params.get("directions")          # e.g. ["down","down_left","down_right"]
    target_regions: dict = context.get("target_regions", {})

    # Pick best hand: finger is extended when tip is farther from wrist than PIP.
    # Direction-agnostic — works for pointing up, down, sideways, etc.
    best_hand  = None
    best_tip   = None
    best_wrist = None
    for hand in landmarks:
        lm = hand.landmark
        wrist = lm[0]
        tip   = lm[8]   # index fingertip
        pip   = lm[6]   # index PIP
        wrist_tip_sq = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
        wrist_pip_sq = (pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2
        if wrist_tip_sq > wrist_pip_sq:
            best_tip   = tip
            best_wrist = wrist
            break

    if best_tip is None:
        context["point_on_target_since"] = None
        return False

    # Direction check
    if directions:
        dir_ok = _classify_direction(best_wrist, best_tip) in directions
    else:
        dir_ok = True

    # Region check
    inline_rect = params.get("region_rect")
    if inline_rect:
        rect_ok = _point_in_rect(best_tip.x, best_tip.y, inline_rect)
    elif target_key and target_key in target_regions:
        rect_ok = _point_in_rect(best_tip.x, best_tip.y, target_regions[target_key])
    else:
        rect_ok = True   # no region filter when directions are the only constraint

    on_target = dir_ok and rect_ok

    # Fallback: no filter at all — any point in lower half
    if not directions and not inline_rect and target_key not in target_regions:
        on_target = best_tip.y > 0.4

    now = time.monotonic()
    if on_target:
        if context.get("point_on_target_since") is None:
            context["point_on_target_since"] = now
        return (now - context["point_on_target_since"]) * 1000 >= hold_ms
    else:
        context["point_on_target_since"] = None
        return False
