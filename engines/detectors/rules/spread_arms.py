"""
spread_arms — both arms held wide out to the sides (wingspan pose).

Used by: AL-11-019 spread_arms (Scene 11 OI flourish, act 11 one-shot).

Pose-based: both wrists must be horizontally OUTSIDE the shoulder line by a
margin scaled to shoulder width, within a vertical band around shoulder height
(arms out to the side — not hanging, not raised overhead), held for hold_ms.

Written orientation-agnostic: instead of assigning left/right (mirroring
pitfalls), it requires the outermost wrist on EACH side to clear that side's
shoulder outward. Phantom low-visibility wrists are ignored (same rule as
point_region — Pose reports positions for occluded landmarks too).

Body geometry (Pose landmarks, read from context["_pose_lm"]):
  11 (left_shoulder), 12 (right_shoulder) → shoulder line + width
  15 (left_wrist),    16 (right_wrist)    → tracked wrists
  23 (left_hip),      24 (right_hip)      → torso height for the vertical band

Params:
  spread_margin_frac (float): how far beyond the shoulder each wrist must reach,
                          as a fraction of shoulder width. Default 0.5 (half a
                          shoulder-width past the shoulder — a clear, open reach).
  band_frac (float): vertical tolerance around shoulder height, as a fraction of
                          shoulder→hip torso height. Default 0.6 (roughly chest
                          band; generous for a public installation).
  hold_ms (int): ms the pose must be maintained. Default 400.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Fallback (no Pose): return False.

Context keys: spread_since
"""

import time

from ...depth.fusion import trusted_landmark


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    if pose_lm is None:
        context["spread_since"] = None
        return False

    margin_frac    = params.get("spread_margin_frac", 0.5)
    band_frac      = params.get("band_frac", 0.6)
    hold_ms        = params.get("hold_ms", 400)
    min_visibility = params.get("min_visibility", 0.5)

    try:
        sh_l, sh_r   = pose_lm[11], pose_lm[12]
        lw, rw       = pose_lm[15], pose_lm[16]
        hip_l, hip_r = pose_lm[23], pose_lm[24]
    except (IndexError, TypeError):
        context["spread_since"] = None
        return False

    matched = False
    # trusted_landmark = visibility gate + depth phantom veto (the point_region
    # parity the docstring promises — a raw visibility check misses the veto).
    if (trusted_landmark(context, 15, lw, min_visibility)
            and trusted_landmark(context, 16, rw, min_visibility)):
        shoulder_width = abs(sh_l.x - sh_r.x)
        shoulder_y = (sh_l.y + sh_r.y) / 2
        hip_y = (hip_l.y + hip_r.y) / 2
        torso_h = max(hip_y - shoulder_y, 0.05)
        margin = margin_frac * max(shoulder_width, 0.10)

        # Orientation-agnostic spread check: the outermost wrist on each side must
        # clear the outermost shoulder on that side by the margin.
        outer_hi = max(sh_l.x, sh_r.x)   # higher-x shoulder edge
        outer_lo = min(sh_l.x, sh_r.x)   # lower-x shoulder edge
        wrist_hi = max(lw.x, rw.x)
        wrist_lo = min(lw.x, rw.x)
        spread_ok = (wrist_hi > outer_hi + margin) and (wrist_lo < outer_lo - margin)

        # Both wrists roughly at chest/shoulder height (arms out, not hanging/overhead).
        band = band_frac * torso_h
        height_ok = (abs(lw.y - shoulder_y) <= band) and (abs(rw.y - shoulder_y) <= band)

        matched = spread_ok and height_ok

    now = time.monotonic()
    if matched:
        if context.get("spread_since") is None:
            context["spread_since"] = now
        return (now - context["spread_since"]) * 1000 >= hold_ms

    context["spread_since"] = None
    return False
