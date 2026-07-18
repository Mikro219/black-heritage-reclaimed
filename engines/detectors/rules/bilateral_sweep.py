"""
bilateral_sweep — one arm sweeping horizontally across ≥ min_screen_fraction.
Used for Scene 3 arm sweep CG.

POSE-ONLY (July 2026): a sweep is started by the most-extended trusted Pose
wrist and then LOCKED to that side — measuring travel between two different
physical hands (dominance flipping mid-sweep) produced spurious jumps
(review fix, July 2026). The old finger-extension gate is replaced by arm
extension (min_reach_frac).

Params:
  min_screen_fraction (float): minimum X travel during the sweep. Default 0.33.
  direction (str): "left_to_right" | "right_to_left" | "any". Default "left_to_right".
  min_reach_frac (float): arm extension required (shoulder→wrist over full arm
                          length). Default 0.6.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: sweep_start_x, sweep_side, sweep_direction_sign  (reads: _pose_lm)
"""

from . import pose_helpers


def _reset(context: dict) -> None:
    context["sweep_start_x"] = None
    context["sweep_side"] = None


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if not wrists:
        _reset(context)
        return False

    min_frac = params.get("min_screen_fraction", 0.33)
    min_reach = params.get("min_reach_frac", 0.6)
    direction = params.get("direction", "left_to_right")
    sign = -1 if direction == "right_to_left" else 1

    side = context.get("sweep_side")
    if side is None or context.get("sweep_start_x") is None:
        # Start a sweep on the most-extended wrist and lock to that side.
        best_side = None
        best_reach = min_reach
        for s in wrists:
            reach = pose_helpers.arm_reach_frac(pose_lm, s)
            if reach >= best_reach:
                best_reach = reach
                best_side = s
        if best_side is None:
            return False
        context["sweep_side"] = best_side
        context["sweep_start_x"] = wrists[best_side].x
        context["sweep_direction_sign"] = sign
        return False

    # Continue only with the sweep's own hand — extended and still tracked.
    wrist = wrists.get(side)
    if wrist is None or pose_helpers.arm_reach_frac(pose_lm, side) < min_reach:
        _reset(context)
        return False

    traveled = (wrist.x - context["sweep_start_x"]) * sign
    if traveled >= min_frac:
        _reset(context)                   # done — ready for the next sweep
        return True

    if traveled < -0.10:                  # reversed — restart from here
        context["sweep_start_x"] = wrist.x

    return False
