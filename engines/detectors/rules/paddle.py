"""
paddle — both wrists cross a horizontal midline, counted as one stroke each time
both hands have crossed since the last count. Fires after min_strokes.

Used by: AL-11-013 paddle (Scene 11 CG, 4 strokes).

A stroke is counted each time BOTH wrists have individually crossed the midline
(at shoulder/hip midpoint) at least once since the previous count. Order and
direction don't matter — the constraint is just that both hands participate
before the counter advances.

TIMING FIX (July 2026 playtest): the old version had no hysteresis and no time
constraints, so hands hovering AT the midline jittered across it and racked up
"strokes" instantly — the detection timing felt disconnected from actual rowing.
Now a wrist only changes side after clearing the midline by ±band_frac of the
torso height, counted strokes are refractory-limited (two strokes can't land
within min_stroke_interval_ms), and strokes must accumulate within
stroke_window_ms of each other or the count decays back to zero.

Params:
  min_strokes (int):      crossings required. Default 2.
  waist_y_offset (float): shifts the midline up (negative) or down (positive)
                          in normalised screen coords. Default 0.0.
  band_frac (float):      hysteresis half-band as a fraction of shoulder→hip
                          torso height. Default 0.12.
  min_stroke_interval_ms (int): refractory between counted strokes. Default 250.
  stroke_window_ms (int): all min_strokes strokes must land within this rolling
                          window. Default 6000.

Fallback (no Pose): return False.

Context keys:
  paddle_l_above, paddle_r_above (bool)  — last confirmed side of each wrist
  paddle_l_crossed, paddle_r_crossed (bool) — crossed since last count
  paddle_stroke_times (list[float])
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    if pose_lm is None:
        return False

    min_strokes  = params.get("min_strokes", 2)
    band_frac    = params.get("band_frac", 0.12)
    min_interval = params.get("min_stroke_interval_ms", 250) / 1000.0
    window_s     = params.get("stroke_window_ms", 6000) / 1000.0

    sh_l, sh_r   = pose_lm[11], pose_lm[12]
    hip_l, hip_r = pose_lm[23], pose_lm[24]
    lw, rw       = pose_lm[15], pose_lm[16]

    shoulder_y = (sh_l.y + sh_r.y) / 2
    hip_y      = (hip_l.y + hip_r.y) / 2 + params.get("waist_y_offset", 0.0)
    midline_y  = (shoulder_y + hip_y) / 2
    band       = band_frac * max(hip_y - shoulder_y, 0.05)

    def _side(wrist_y: float, prev_above):
        """Hysteresis: only flip side after clearing the midline by ±band."""
        if wrist_y < midline_y - band:
            return True
        if wrist_y > midline_y + band:
            return False
        return prev_above   # inside the band — hold previous state

    # First frame — initialise, nothing to compare yet
    if "paddle_l_above" not in context:
        context["paddle_l_above"]    = lw.y < midline_y
        context["paddle_r_above"]    = rw.y < midline_y
        context["paddle_l_crossed"]  = False
        context["paddle_r_crossed"]  = False
        context["paddle_stroke_times"] = []
        return False

    l_above = _side(lw.y, context["paddle_l_above"])
    r_above = _side(rw.y, context["paddle_r_above"])

    if l_above != context["paddle_l_above"]:
        context["paddle_l_crossed"] = True
    if r_above != context["paddle_r_above"]:
        context["paddle_r_crossed"] = True

    context["paddle_l_above"] = l_above
    context["paddle_r_above"] = r_above

    now = time.monotonic()
    times: list = context.setdefault("paddle_stroke_times", [])
    times[:] = [t for t in times if now - t <= window_s]

    # Count a stroke once both hands have crossed since the last count
    if context["paddle_l_crossed"] and context["paddle_r_crossed"]:
        if not times or (now - times[-1]) >= min_interval:
            times.append(now)
        context["paddle_l_crossed"] = False
        context["paddle_r_crossed"] = False

        if len(times) >= min_strokes:
            context["paddle_stroke_times"] = []
            return True

    return False
