"""
presence_bilateral_still — both pose wrists present AND nearly motionless for
stillness_ms. Used for AL-06-008 hold_still_palms.

POSE-ONLY (July 2026): reads Pose wrists (15/16). The old palm_orientation
check needed finger landmarks and is gone — stillness alone is the signal
(detect intent, not precision).

Params:
  stillness_ms (int): ms both wrists must stay below the velocity threshold.
                      Default 1000.
  velocity_threshold (float): max summed wrist delta/frame (normalised).
                      Default 0.08.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: bilateral_still_since, prev_wrist_xy  (reads: _pose_lm)
"""

import time

from . import pose_helpers


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if len(wrists) < 2:
        context["bilateral_still_since"] = None
        context["prev_wrist_xy"] = None
        return False

    stillness_ms = params.get("stillness_ms", 1000)
    vel_thresh = params.get("velocity_threshold", 0.08)

    now = time.monotonic()
    current_xy = {side: (w.x, w.y) for side, w in wrists.items()}

    prev_xy = context.get("prev_wrist_xy")
    if prev_xy is None or set(prev_xy) != set(current_xy):
        context["prev_wrist_xy"] = current_xy
        context["bilateral_still_since"] = None
        return False

    total_delta = sum(
        ((current_xy[s][0] - prev_xy[s][0]) ** 2 +
         (current_xy[s][1] - prev_xy[s][1]) ** 2) ** 0.5
        for s in current_xy
    )
    context["prev_wrist_xy"] = current_xy

    if total_delta < vel_thresh:
        if context.get("bilateral_still_since") is None:
            context["bilateral_still_since"] = now
        return (now - context["bilateral_still_since"]) * 1000 >= stillness_ms

    context["bilateral_still_since"] = None
    return False
