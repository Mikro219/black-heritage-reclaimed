"""
paddle — both wrists cross a horizontal midline, counted as one stroke each time
both hands have crossed since the last count. Fires after min_strokes.

Used by: AL-11-013 paddle (Scene 11 CG, 4 strokes).

A stroke is counted each time BOTH wrists have individually crossed the midline
(at shoulder/hip midpoint) at least once since the previous count. Order and
direction don't matter — the constraint is just that both hands participate
before the counter advances.

Params:
  min_strokes (int):      crossings required. Default 2.
  waist_y_offset (float): shifts the midline up (negative) or down (positive)
                          in normalised screen coords. Default 0.0.

Fallback (no Pose): return False.

Context keys:
  paddle_l_above, paddle_r_above (bool)  — last known side of each wrist
  paddle_l_crossed, paddle_r_crossed (bool) — crossed since last count
  paddle_stroke_count (int)
"""


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    if pose_lm is None:
        return False

    min_strokes = params.get("min_strokes", 2)

    sh_l, sh_r   = pose_lm[11], pose_lm[12]
    hip_l, hip_r = pose_lm[23], pose_lm[24]
    lw, rw       = pose_lm[15], pose_lm[16]

    shoulder_y = (sh_l.y + sh_r.y) / 2
    hip_y      = (hip_l.y + hip_r.y) / 2 + params.get("waist_y_offset", 0.0)
    midline_y  = (shoulder_y + hip_y) / 2

    l_above = lw.y < midline_y
    r_above = rw.y < midline_y

    # First frame — initialise, nothing to compare yet
    if "paddle_l_above" not in context:
        context["paddle_l_above"]    = l_above
        context["paddle_r_above"]    = r_above
        context["paddle_l_crossed"]  = False
        context["paddle_r_crossed"]  = False
        context["paddle_stroke_count"] = 0
        return False

    # Detect per-hand crossings
    if l_above != context["paddle_l_above"]:
        context["paddle_l_crossed"] = True
    if r_above != context["paddle_r_above"]:
        context["paddle_r_crossed"] = True

    context["paddle_l_above"] = l_above
    context["paddle_r_above"] = r_above

    # Count a stroke once both hands have crossed since the last count
    if context["paddle_l_crossed"] and context["paddle_r_crossed"]:
        context["paddle_stroke_count"] += 1
        context["paddle_l_crossed"] = False
        context["paddle_r_crossed"] = False

        if context["paddle_stroke_count"] >= min_strokes:
            context["paddle_stroke_count"] = 0
            return True

    return False
