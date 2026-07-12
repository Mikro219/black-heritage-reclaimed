"""
forward_reach — hand moves toward the screen (Z-approach), used for AL-01-002
reach_flask.

POSE-ONLY (July 2026): the signal is a Pose wrist approaching the camera —
its REAL depth drops by min_depth_delta_mm within the window when the Gemini
335 sampler is present (authoritative), else its Pose z drops by
min_pose_z_delta (hip-normalised, more negative = closer). The old hand
bounding-box growth proxy needed the Hands model and is gone. Optionally
constrained to a target screen region (wrist position, player-mirrored).

Params:
  min_depth_delta_mm (float): required real wrist-depth decrease within the
                  window when the depth sampler is present. Default 180.
  min_pose_z_delta (float): required Pose-z decrease without depth. Default 0.18.
  window_ms (int): sliding window for the approach. Default 900.
  target_region (str | None): screen region the wrist must occupy:
      "upper_center" | "upper_left" | "upper_right" | "center" | None.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: reach_depth_history, forward_reach_fired  (reads: _pose_lm, _depth_mm_at)
"""

import time

from . import pose_helpers

_REGIONS = {
    "upper_center": {"x_min": 0.30, "x_max": 0.70, "y_max": 0.40},
    "upper_left":   {"x_max": 0.50, "y_max": 0.40},
    "upper_right":  {"x_min": 0.50, "y_max": 0.40},
    "center":       {"x_min": 0.33, "x_max": 0.67, "y_min": 0.33, "y_max": 0.67},
}


def _wrist_in_region(wrist, region_key: str) -> bool:
    region = _REGIONS.get(region_key)
    if region is None:
        return True
    px, py = 1.0 - wrist.x, wrist.y   # player-mirrored space
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
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if not wrists:
        context.pop("reach_depth_history", None)
        context["forward_reach_fired"] = False
        return False

    if context.get("forward_reach_fired"):
        return False

    min_mm = params.get("min_depth_delta_mm", 180)
    min_z = params.get("min_pose_z_delta", 0.18)
    window_s = params.get("window_ms", 900) / 1000.0
    region_key = params.get("target_region")

    now = time.monotonic()
    hist: dict = context.setdefault("reach_depth_history", {})

    for side, wrist in wrists.items():
        kind, val = pose_helpers.wrist_depth_or_z(context, pose_lm, side)
        if kind is None:
            continue
        h = hist.setdefault(side, [])
        h.append((now, kind, val))
        h[:] = [(t, k, v) for t, k, v in h if now - t <= window_s]
        same = [v for t, k, v in h if k == kind]
        if len(same) < 2:
            continue
        drop = max(same) - val
        if (kind == "mm" and drop >= min_mm) or (kind == "z" and drop >= min_z):
            if region_key and not _wrist_in_region(wrist, region_key):
                continue
            context["forward_reach_fired"] = True
            return True

    return False
