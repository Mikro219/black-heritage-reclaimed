"""
bilateral_sweep — one arm sweeping horizontally across ≥ min_screen_fraction.
Used for Scene 3 arm sweep CG.

POSE-ONLY (July 2026): tracks the most-extended trusted Pose wrist; the old
finger-extension gate is replaced by arm extension (min_reach_frac).

Params:
  min_screen_fraction (float): minimum X travel during the sweep. Default 0.33.
  direction (str): "left_to_right" | "right_to_left" | "any". Default "left_to_right".
  min_reach_frac (float): arm extension required (shoulder→wrist over full arm
                          length). Default 0.6.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: sweep_start_x, sweep_direction_sign  (reads: _pose_lm)
"""

from . import pose_helpers


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if not wrists:
        context["sweep_start_x"] = None
        return False

    min_frac = params.get("min_screen_fraction", 0.33)
    min_reach = params.get("min_reach_frac", 0.6)
    direction = params.get("direction", "left_to_right")
    sign = -1 if direction == "right_to_left" else 1

    best_wrist_x = None
    best_reach = min_reach
    for side, wrist in wrists.items():
        reach = pose_helpers.arm_reach_frac(pose_lm, side)
        if reach >= best_reach:
            best_reach = reach
            best_wrist_x = wrist.x

    if best_wrist_x is None:
        return False

    start_x = context.get("sweep_start_x")
    if start_x is None:
        context["sweep_start_x"] = best_wrist_x
        context["sweep_direction_sign"] = sign
        return False

    traveled = (best_wrist_x - start_x) * sign
    if traveled >= min_frac:
        context["sweep_start_x"] = None   # reset for next sweep
        return True

    if traveled < -0.10:                  # reversed — restart from here
        context["sweep_start_x"] = best_wrist_x

    return False
