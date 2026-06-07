"""
forward_reach — hand moves toward the screen (Z-approach), used for AL-01-002 reach_flask.

Unlike directional_point, no specific finger pose is required — open palm, loose fist,
or pointing hand all count. The signal is "hand moving closer to camera," detected via
bounding-box area growth, optionally constrained to a target screen region.

Params:
  area_growth_threshold (float): fraction of area growth required over the window. Default 0.30.
  window_frames (int): sliding window depth (frames) for area comparison. Default 12 (~400ms @30fps).
  target_region (str | None): screen region the wrist must occupy. One of:
      "upper_center"   — upper 40% of frame, middle 40% of width (the flask region in Scene 1)
      "upper_left"     — upper 40%, left 50%
      "upper_right"    — upper 40%, right 50%
      "center"         — middle third in both axes
      None             — no spatial constraint; anywhere in frame counts.

Detection approach:
  Each frame, compute the axis-aligned bounding box of all 21 landmarks.
  The box's normalised area (w * h) is pushed onto a circular buffer.
  When current_area / oldest_area >= (1 + area_growth_threshold) the visitor's hand
  is growing on screen, i.e., approaching the camera. If a target_region is configured,
  the wrist landmark must also fall within that region.

Context keys: bbox_area_history, forward_reach_fired
"""

import time

_REGIONS = {
    "upper_center": {"x_min": 0.30, "x_max": 0.70, "y_max": 0.40},
    "upper_left":   {"x_max": 0.50, "y_max": 0.40},
    "upper_right":  {"x_min": 0.50, "y_max": 0.40},
    "center":       {"x_min": 0.33, "x_max": 0.67, "y_min": 0.33, "y_max": 0.67},
}


def _hand_bbox_area(hand) -> float:
    """Return the normalised bounding-box area (w * h) for a single hand."""
    lm = hand.landmark
    xs = [l.x for l in lm]
    ys = [l.y for l in lm]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


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
        context["forward_reach_fired"] = False
        return False

    # Once fired in this window, don't re-fire until the context is reset
    if context.get("forward_reach_fired"):
        return False

    threshold = params.get("area_growth_threshold", 0.30)
    window_frames = params.get("window_frames", 12)
    region_key = params.get("target_region")

    history: list = context.setdefault("bbox_area_history", [])

    # Use the largest hand (most likely the dominant/extended one)
    best_hand = max(landmarks, key=_hand_bbox_area)
    current_area = _hand_bbox_area(best_hand)
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
