"""
paddle — both wrists arc together on the same side of the body midline, ≥4 strokes.

Used by: AL-11-013 paddle (Scene 11 CG, 4 strokes).

Models rowing a boat: both hands on the same side (left or right of body centre),
making synchronized up-down strokes that span from above shoulder to below hip.

Body geometry (Pose landmarks, read from context["_pose_lm"]):
  11 (left_shoulder),  12 (right_shoulder)  → midline_x, shoulder_y
  15 (left_wrist),     16 (right_wrist)     → tracked wrists
  23 (left_hip),       24 (right_hip)       → hip_y

Same-side constraint:
  On the first valid frame, record which side of midline_x the centroid of both
  wrists is on ("left" or "right"). Lock that as paddle_side.
  On each subsequent frame, if EITHER wrist crosses to the opposite side of
  midline_x, reset the stroke count and re-detect side.

Stroke definition (bilateral-synchronized):
  Both wrists must travel from above shoulder_y to below hip_y (or vice versa).
  A stroke is counted when BOTH wrists complete the full Y range in the same
  direction within a synchrony window (sync_window_ms).

Params:
  min_strokes (int): complete strokes required. Default 4.
  sync_window_ms (float): both wrists must reach range boundary within this window. Default 400.

Fallback (no Pose): return False.

Context keys:
  paddle_side ("left" | "right" | None)
  paddle_stroke_count (int)
  paddle_phase_l ("up" | "down")  — which half of stroke left wrist is in
  paddle_phase_r ("up" | "down")
  paddle_l_completed (bool)       — left completed the current half-stroke
  paddle_r_completed (bool)
  paddle_phase_start_time (float)
"""

import time


def _wrist_side(lw_x: float, rw_x: float, midline_x: float) -> str | None:
    """Both wrists on same side?  Return 'left' / 'right' / None."""
    l_left = lw_x < midline_x
    r_left = rw_x < midline_x
    if l_left == r_left:
        return "left" if l_left else "right"
    return None   # wrists are on opposite sides


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    if pose_lm is None:
        return False

    min_strokes    = params.get("min_strokes", 4)
    sync_window_s  = params.get("sync_window_ms", 400) / 1000.0

    sh_l, sh_r     = pose_lm[11], pose_lm[12]
    hip_l, hip_r   = pose_lm[23], pose_lm[24]
    lw, rw         = pose_lm[15], pose_lm[16]

    midline_x      = (sh_l.x + sh_r.x) / 2
    shoulder_y     = (sh_l.y + sh_r.y) / 2
    hip_y          = (hip_l.y + hip_r.y) / 2 + params.get("waist_y_offset", 0.0)

    # Same-side check
    current_side = _wrist_side(lw.x, rw.x, midline_x)
    locked_side  = context.get("paddle_side")

    if current_side is None:
        # Wrists on opposite sides — reset
        context["paddle_side"] = None
        context["paddle_stroke_count"] = 0
        context["paddle_phase_l"] = None
        context["paddle_phase_r"] = None
        return False

    if locked_side is None:
        # First valid frame — lock the side
        context["paddle_side"] = current_side
        locked_side = current_side
    elif current_side != locked_side:
        # Side changed — reset
        context["paddle_side"] = current_side
        context["paddle_stroke_count"] = 0
        context["paddle_phase_l"] = None
        context["paddle_phase_r"] = None
        return False

    # Stroke phase tracking per wrist
    # Phase "up"   = wrist has been below hip_y and is now returning upward
    # Phase "down" = wrist has been above shoulder_y and is now moving downward
    # A stroke completes when BOTH wrists finish the same phase simultaneously
    # (within sync_window_ms of each other).

    now = time.monotonic()

    def _update_phase(wrist_y, phase_key, completed_key):
        phase = context.get(phase_key)
        if phase is None:
            context[phase_key] = "up" if wrist_y < (shoulder_y + hip_y) / 2 else "down"
            context[completed_key] = False
            return

        if phase == "down" and wrist_y > hip_y:
            context[phase_key] = "up"
            context[completed_key] = True
            if context.get("paddle_phase_start_time") is None:
                context["paddle_phase_start_time"] = now
        elif phase == "up" and wrist_y < shoulder_y:
            context[phase_key] = "down"
            context[completed_key] = True
            if context.get("paddle_phase_start_time") is None:
                context["paddle_phase_start_time"] = now

    _update_phase(lw.y, "paddle_phase_l", "paddle_l_completed")
    _update_phase(rw.y, "paddle_phase_r", "paddle_r_completed")

    l_done = context.get("paddle_l_completed", False)
    r_done = context.get("paddle_r_completed", False)
    phase_start = context.get("paddle_phase_start_time") or now

    if l_done and r_done and (now - phase_start) <= sync_window_s:
        context["paddle_stroke_count"] = context.get("paddle_stroke_count", 0) + 1
        context["paddle_l_completed"] = False
        context["paddle_r_completed"] = False
        context.pop("paddle_phase_start_time", None)

        if context["paddle_stroke_count"] >= min_strokes:
            context["paddle_stroke_count"] = 0
            context["paddle_side"] = None
            return True

    return False
