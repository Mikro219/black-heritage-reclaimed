"""
point_target_held — index finger pointing at a named screen region, held for hold_ms.
Used for Scene 1 point-quilt-block OI (AL-01-010), Scene 4 North Star OI (AL-04-012),
Scene 3 point-river-marker OI (AL-03-006), Scene 5/6 point OIs.

POINTING-FINGER, NOT OPEN HAND (July 2026 playtest): the old extension check
(index tip farther from wrist than PIP) is also true of a fully open palm, so
any open hand waving through the region counted as a "point" — that's the
scene 2/3 pointing bug from Mike's punch list. The detector now requires an
actual pointing pose: index extended with the other fingers (mostly) curled
(hand_pose.is_pointing). Set require_pointing_pose: false to restore the loose
behaviour for a specific shot.

POSE FALLBACK: an arm extended at a target often defeats the Hands model
(foreshortening / distance). When no pointing hand resolves, a visible Pose
wrist inside the target region counts instead, sharing the same hold timer so
Hands flicker doesn't reset the hold. Disable with pose_fallback: false.

Params:
  target_region (str):   key into context["target_regions"] rect dict.
  region_rect (dict):    inline rect {"x","y","w","h"} in normalised screen coords.
  directions (list[str]): if set, the wrist→tip vector must classify into one of these
                          8-way directions: "up","down","left","right",
                          "up_left","up_right","down_left","down_right".
                          Checked in addition to region_rect when both are present.
  hold_ms (int):         ms the point must stay on target. Default 800.
  require_pointing_pose (bool): reject open palms. Default True.
  pose_fallback (bool):  allow a visible Pose wrist in-region when no pointing
                         hand resolves. Default True.
  min_visibility (float): Pose wrist visibility gate for the fallback. Default 0.5.

If neither region_rect nor directions is specified, fires on any point in the lower half.

Context keys: point_on_target_since  (reads: _pose_lm)
"""

import math
import time

from . import hand_pose

_DIRS_CW = ["right", "down_right", "down", "down_left",
            "left",  "up_left",   "up",   "up_right"]

_POSE_WRISTS = (15, 16)


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


def _resolve_rect(params: dict, context: dict):
    inline_rect = params.get("region_rect")
    if inline_rect:
        return inline_rect
    target_key = params.get("target_region")
    target_regions: dict = context.get("target_regions", {})
    if target_key and target_key in target_regions:
        return target_regions[target_key]
    return None


def detect(landmarks, params: dict, context: dict) -> bool:
    hold_ms    = params.get("hold_ms", 800)
    directions = params.get("directions")          # e.g. ["down","down_left","down_right"]
    rect       = _resolve_rect(params, context)
    require_pointing = params.get("require_pointing_pose", True)

    # ── Hands path: an actual pointing finger ────────────────────────────────
    best_tip   = None
    best_wrist = None
    for hand in landmarks or []:
        lm = hand.landmark
        wrist = lm[0]
        tip   = lm[8]   # index fingertip
        pip   = lm[6]   # index PIP
        wrist_tip_sq = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
        wrist_pip_sq = (pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2
        if wrist_tip_sq <= wrist_pip_sq:
            continue   # index not extended
        if require_pointing and not hand_pose.is_pointing(hand):
            continue   # open palm / indeterminate — not a point
        best_tip   = tip
        best_wrist = wrist
        break

    on_target = False
    if best_tip is not None:
        dir_ok = _classify_direction(best_wrist, best_tip) in directions if directions else True
        rect_ok = _point_in_rect(best_tip.x, best_tip.y, rect) if rect else True
        if not directions and not rect:
            on_target = best_tip.y > 0.4   # legacy no-filter fallback: lower half
        else:
            on_target = dir_ok and rect_ok

    # ── Pose fallback: visible wrist inside the region when no pointing hand
    #    resolves. Direction filters can't be evaluated without a finger, so the
    #    fallback only applies to region-defined targets. ──────────────────────
    if not on_target and best_tip is None and rect is not None \
            and params.get("pose_fallback", True):
        pose_lm = context.get("_pose_lm")
        min_visibility = params.get("min_visibility", 0.5)
        if pose_lm is not None:
            for idx in _POSE_WRISTS:
                try:
                    w = pose_lm[idx]
                except (IndexError, TypeError):
                    continue
                if getattr(w, "visibility", 1.0) < min_visibility:
                    continue
                if _point_in_rect(w.x, w.y, rect):
                    on_target = True
                    break

    now = time.monotonic()
    if on_target:
        if context.get("point_on_target_since") is None:
            context["point_on_target_since"] = now
        return (now - context["point_on_target_since"]) * 1000 >= hold_ms
    else:
        context["point_on_target_since"] = None
        return False
