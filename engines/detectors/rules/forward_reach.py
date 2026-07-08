"""
forward_reach — hand moves toward the screen (Z-approach), used for AL-01-002 reach_flask.

Unlike directional_point, no specific finger pose is required — open palm, loose fist,
or pointing hand all count. The signal is "hand moving closer to camera": with the
Gemini 335 depth sampler present (context["_depth_mm_at"]) the wrist's REAL depth
must drop by min_depth_delta_mm within the window (authoritative while flowing);
on a plain webcam, bounding-box area growth is the monocular fallback. Optionally
constrained to a target screen region.

Params:
  area_growth_threshold (float): fraction of area growth required over the window. Default 0.30.
  window_frames (int): sliding window depth (frames) for area comparison. Default 12 (~400ms @30fps).
  min_depth_delta_mm (float): required real wrist-depth decrease within the
                  window when the depth sampler is present. Default 180.
  target_region (str | None): screen region the wrist must occupy. One of:
      "upper_center"   — upper 40% of frame, middle 40% of width (the flask region in Scene 1)
      "upper_left"     — upper 40%, left 50%
      "upper_right"    — upper 40%, right 50%
      "center"         — middle third in both axes
      None             — no spatial constraint; anywhere in frame counts.

Context keys: bbox_area_history, reach_depth_history, forward_reach_fired
"""

from . import hand_pose

_REGIONS = {
    "upper_center": {"x_min": 0.30, "x_max": 0.70, "y_max": 0.40},
    "upper_left":   {"x_max": 0.50, "y_max": 0.40},
    "upper_right":  {"x_min": 0.50, "y_max": 0.40},
    "center":       {"x_min": 0.33, "x_max": 0.67, "y_min": 0.33, "y_max": 0.67},
}


def _wrist_in_region(hand, region_key: str) -> bool:
    """Return True if the wrist falls within the named region (in player-mirrored space)."""
    region = _REGIONS.get(region_key)
    if region is None:
        return True
    wrist = hand.landmark[0]
    # Mirror x to player's perspective (same convention as the rest of the engine)
    px = 1.0 - wrist.x
    py = wrist.y
    if region.get("x_min") is not None and px < region["x_min"]:
        return False
    if region.get("x_max") is not None and px > region["x_max"]:
        return False
    if region.get("y_min") is not None and py < region["y_min"]:
        return False
    if region.get("y_max") is not None and py > region["y_max"]:
        return False
    return True


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks:
        context.pop("bbox_area_history", None)
        context.pop("reach_depth_history", None)
        context["forward_reach_fired"] = False
        return False

    # Once fired in this window, don't re-fire until the context is reset
    if context.get("forward_reach_fired"):
        return False

    threshold = params.get("area_growth_threshold", 0.30)
    window_frames = params.get("window_frames", 12)
    region_key = params.get("target_region")
    min_depth_delta = params.get("min_depth_delta_mm", 180)

    # Use the largest hand (most likely the dominant/extended one)
    best_hand = max(landmarks, key=hand_pose.bbox_area)

    # ── Preferred: real wrist depth (Gemini 335) — fire when the wrist has
    #    approached the camera by min_depth_delta_mm within the window. ───────
    depth_at = context.get("_depth_mm_at")
    if callable(depth_at):
        hw = best_hand.landmark[0]
        d = depth_at(hw.x, hw.y)
        if d is not None:
            dh: list = context.setdefault("reach_depth_history", [])
            dh.append(d)
            if len(dh) > window_frames:
                dh.pop(0)
            if len(dh) >= 2 and (max(dh) - d) >= min_depth_delta:
                if region_key and not _wrist_in_region(best_hand, region_key):
                    return False
                context["forward_reach_fired"] = True
                return True
            if len(dh) >= 2:
                # Depth is flowing and says "no approach yet" — it's
                # authoritative; skip the bbox proxy this frame.
                return False

    history: list = context.setdefault("bbox_area_history", [])
    current_area = hand_pose.bbox_area(best_hand)
    history.append(current_area)
    if len(history) > window_frames:
        history.pop(0)

    if len(history) < window_frames:
        return False  # not enough history yet

    oldest_area = history[0]
    if oldest_area < 1e-6:
        return False  # degenerate bbox

    growth = current_area / oldest_area
    if growth < 1.0 + threshold:
        return False

    # Spatial constraint: wrist must be in the target region
    if region_key and not _wrist_in_region(best_hand, region_key):
        return False

    context["forward_reach_fired"] = True
    return True
