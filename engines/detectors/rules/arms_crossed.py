"""
arms_crossed — both wrists cross the body midline simultaneously.

Used by: AL-08-006 arms_crossed (put on coat, CG).

Uses Pose landmarks exclusively for wrist and body geometry. MediaPipe Hands
landmarks are not needed — Pose labels left/right correctly from the player's
perspective and gives us the shoulder + hip reference lines.

Body geometry (Pose landmarks, read from context["_pose_lm"]):
  11 (left_shoulder),  12 (right_shoulder) → midline_x, shoulder_y, shoulder_width
  15 (left_wrist),     16 (right_wrist)    → tracked wrists
  23 (left_hip),       24 (right_hip)      → hip_y (torso-height lower bound)

Detection logic:
  midline_x    = (sh_l.x + sh_r.x) / 2
  dead_zone    = 0.05 * shoulder_width      (prevents jitter near centreline)
  shoulder_y   = (sh_l.y + sh_r.y) / 2
  hip_y        = (hip_l.y + hip_r.y) / 2

  left_crossed  = pose_lm[15].x < (midline_x - dead_zone)
  right_crossed = pose_lm[16].x > (midline_x + dead_zone)
  at_torso      = shoulder_y <= wrist.y <= hip_y  (for both wrists)

  All four conditions → hold for hold_ms.

Note on raw vs. player-space X: Pose provides raw camera-space coordinates.
In the raw frame the player's left body side appears on the right (high raw x),
so "left wrist crossing to the right side of the body" means raw x DECREASES
past the midline. The dead_zone is subtracted/added symmetrically.

Params:
  hold_ms (int): ms hold required. Default 500.
  dead_zone_frac (float): fraction of shoulder_width for dead zone. Default 0.05.

Fallback (no Pose): return False — this detector cannot fire without body geometry.

Context keys: arms_crossed_since
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    if pose_lm is None:
        context["arms_crossed_since"] = None
        return False

    hold_ms        = params.get("hold_ms", 500)
    dz_frac        = params.get("dead_zone_frac", 0.05)

    sh_l, sh_r     = pose_lm[11], pose_lm[12]
    hip_l, hip_r   = pose_lm[23], pose_lm[24]
    lw, rw         = pose_lm[15], pose_lm[16]

    midline_x      = (sh_l.x + sh_r.x) / 2
    shoulder_width = abs(sh_l.x - sh_r.x)
    dead_zone      = dz_frac * max(shoulder_width, 0.10)

    shoulder_y     = (sh_l.y + sh_r.y) / 2
    hip_y          = (hip_l.y + hip_r.y) / 2

    # In raw camera coords, player's left wrist crosses over when its raw x
    # falls below (midline_x - dead_zone), and vice versa for right wrist.
    left_crossed   = lw.x < (midline_x - dead_zone)
    right_crossed  = rw.x > (midline_x + dead_zone)

    left_at_torso  = shoulder_y <= lw.y <= hip_y
    right_at_torso = shoulder_y <= rw.y <= hip_y

    matched = left_crossed and right_crossed and left_at_torso and right_at_torso

    now = time.monotonic()
    if matched:
        if context.get("arms_crossed_since") is None:
            context["arms_crossed_since"] = now
        return (now - context["arms_crossed_since"]) * 1000 >= hold_ms
    else:
        context["arms_crossed_since"] = None
        return False
