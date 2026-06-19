"""
touch_head — one or both wrists touch the crown of the head.

Used by:
  AL-05-011 touch_temple   → hands_required: 1  (knowledge-code touch)
  AL-08-005 hands_to_head  → hands_required: 2  (put on hat)

The cue AL-05-011 is still called "touch_temple" for audio-file compatibility, but
the detection target is the crown of the head, matching the script line
"touch your own head." The cue name is preserved intentionally; see session summary.

Params:
  hands_required (int): 1 or 2 wrists must be in the head region. Default 1.
  hold_ms (int): ms the condition must be maintained. Default 500.
  target (str): "crown" (default) targets the top of the head (offset upward) for
                the hat gesture; "temple"/"head" targets the head centre at ear
                level with a generous radius, so a hand at the temple/side of the
                head registers (lenient public-installation touch).

Body geometry (Pose landmarks, read from context["_pose_lm"]):
  7  (left_ear),  8 (right_ear)  → head centre X/Y
  11 (left_shoulder), 12 (right_shoulder) → for radius + crown-offset computation

Crown target:
  head_cx = (ear_l.x + ear_r.x) / 2
  head_cy = (ear_l.y + ear_r.y) / 2
  ear_shoulder_dist = shoulder_y - ear_mid_y        (positive: ears are above shoulders)
  crown_y = head_cy - 0.4 * ear_shoulder_dist       (offset upward toward actual crown)
  crown_x = head_cx
  shoulder_width = abs(shoulder_l.x - shoulder_r.x)
  radius = 0.15 * shoulder_width

Fallback (no Pose): use fixed estimates (head centre ~(0.5, 0.12), radius ~0.10).

Context keys: touch_head_since
"""

import math
import time

# Fixed fallback geometry when Pose is unavailable
_FIXED_CROWN_X = 0.50
_FIXED_CROWN_Y = 0.10
_FIXED_RADIUS  = 0.10


def _target_and_radius(pose_lm, target: str):
    """Return (target_x, target_y, radius) from Pose landmarks, or fixed fallback."""
    if pose_lm is None:
        if target == "crown":
            return _FIXED_CROWN_X, _FIXED_CROWN_Y, _FIXED_RADIUS
        return 0.50, 0.13, 0.15   # head-centre fallback, generous radius

    ear_l, ear_r     = pose_lm[7],  pose_lm[8]
    sh_l,  sh_r      = pose_lm[11], pose_lm[12]

    head_cx = (ear_l.x + ear_r.x) / 2
    head_cy = (ear_l.y + ear_r.y) / 2
    shoulder_y = (sh_l.y + sh_r.y) / 2
    ear_shoulder_dist = max(shoulder_y - head_cy, 0.05)  # ears above shoulders
    shoulder_width = abs(sh_l.x - sh_r.x)

    if target == "crown":
        target_y = head_cy - 0.4 * ear_shoulder_dist     # offset up to the crown
        radius = max(0.15 * shoulder_width, 0.07)
    else:  # "temple" / "head": ear level, generous radius covers side-of-head touches
        target_y = head_cy
        radius = max(0.22 * shoulder_width, 0.13)

    return head_cx, target_y, radius


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    hold_ms = params.get("hold_ms", 500)
    hands_required = params.get("hands_required", 1)
    target = params.get("target", "crown")

    crown_x, crown_y, radius = _target_and_radius(pose_lm, target)

    in_region = 0
    if landmarks:
        for hand in landmarks:
            wrist = hand.landmark[0]
            dist = math.hypot(wrist.x - crown_x, wrist.y - crown_y)
            if dist < radius:
                in_region += 1

    now = time.monotonic()
    if in_region >= hands_required:
        if context.get("touch_head_since") is None:
            context["touch_head_since"] = now
        return (now - context["touch_head_since"]) * 1000 >= hold_ms
    else:
        context["touch_head_since"] = None
        return False
