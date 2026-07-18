"""
point_region — pose-based LEFT/RIGHT region selection from wrist position.

Used by Scene 2 `choose_path` CG (act_02_crossroads shot 09). Instead of requiring
a precise index-finger point at an on-screen marker (directional_point), the player
simply raises a hand to the LEFT or RIGHT of their body and holds it there. Region
membership is read from MediaPipe Pose wrist landmarks relative to the body midline
(shoulder centre), so it works even when MediaPipe Hands cannot resolve a finger
pose out at the side of the body.

This computes its thresholds live from the player's own body each frame (midline
and shoulder width from the pose), rather than using fixed screen coordinates — the
dead zone scales with how wide the player's shoulders appear to the camera.

Geometry (Pose landmarks, read from context["_pose_lm"]):
  11 (left_shoulder), 12 (right_shoulder) → body midline X + shoulder width
  15 (left_wrist),    16 (right_wrist)    → candidate hands
  23 (left_hip),      24 (right_hip)      → "raised" gate (wrist above hip line)

All X comparisons are done in player-screen space (x flipped, matching the mirrored
display and directional_point), so "left" = the player's left = the left path on
screen.

Params:
  directions (list[str]): accepted regions, subset of {"left", "right"}.
                          Default ["left", "right"].
  hold_ms (int): ms the wrist must stay in one region. Default 500.
  dead_zone_frac (float): half-width of the centre dead zone as a fraction of
                          shoulder width. A wrist within this band of the midline
                          is "centre" and selects nothing, so only a deliberate
                          reach to one side registers. Default 0.45 (July 2026 —
                          raised from 0.15: a raised hand barely off-centre was
                          selecting a side; now the wrist must reach out to
                          roughly its own shoulder line).
  require_raised (bool): if True (default), only wrists above the raise gate are
                          considered, so a relaxed arm hanging at the side (which
                          sits far from the midline) cannot trigger a selection.
  raise_gate_frac (float): how far above the hip line the raise gate sits, as a
                          fraction of the hip→shoulder torso height. 0.0 = the old
                          hip-line behaviour; 0.25 (default) puts the gate at the
                          lower chest, so hands resting at the waist — which hover
                          right at the hip line and jitter across it — can't select.
  reject_open_palm (bool): ACCEPTED AND IGNORED since the pose-only rework —
                          finger shape is invisible to the Pose model. The raise
                          gate + dead zone are the deliberateness filters.
  allow_low_reach (bool) / low_reach_frac (float): accept a wrist below the raise
                          gate when it reaches at least low_reach_frac shoulder
                          widths from the midline — natural pointing at LOW
                          on-screen targets (Scene 9 hidden compartment) happens
                          at waist height. Defaults True / 0.9.
  min_visibility (float): minimum Pose visibility for a wrist to be considered.
                          Pose always reports positions for all 33 landmarks, even
                          occluded/out-of-frame ones; those phantom estimates can sit
                          stably in one region and fire unintended selections (and
                          then unintended SWITCHES, ping-ponging against the real
                          hand). Default 0.5.

Context keys: point_direction, point_direction_since, dominant_direction
  (named to match directional_point so the gesture engine reads `choice`
   identically for both detectors.)
"""

import time

from ...depth.fusion import trusted_landmark

_ALL = ["left", "right"]


def _pose_geometry(pose_lm):
    """Return (left_wrist, right_wrist, midline_x, shoulder_width, shoulder_y, hip_y)
    or None."""
    try:
        sh_l, sh_r   = pose_lm[11], pose_lm[12]
        lw, rw       = pose_lm[15], pose_lm[16]
        hip_l, hip_r = pose_lm[23], pose_lm[24]
    except (IndexError, TypeError):
        return None
    midline_x = (sh_l.x + sh_r.x) / 2
    shoulder_width = abs(sh_l.x - sh_r.x)
    shoulder_y = (sh_l.y + sh_r.y) / 2
    hip_y = (hip_l.y + hip_r.y) / 2
    return lw, rw, midline_x, shoulder_width, shoulder_y, hip_y


def detect(landmarks, params: dict, context: dict) -> bool:
    accepted = set(params.get("directions") or _ALL)
    hold_ms = params.get("hold_ms", 500)
    dead_zone_frac = params.get("dead_zone_frac", 0.45)
    require_raised = params.get("require_raised", True)
    raise_gate_frac = params.get("raise_gate_frac", 0.25)
    min_visibility = params.get("min_visibility", 0.5)

    pose_lm = context.get("_pose_lm")
    geom = _pose_geometry(pose_lm) if pose_lm is not None else None

    best_offset = 0.0
    best_direction = None

    if geom is not None:
        lw, rw, midline_x, shoulder_width, shoulder_y, hip_y = geom
        screen_midline = 1.0 - midline_x
        dead = dead_zone_frac * max(shoulder_width, 0.10)
        # Raise gate sits raise_gate_frac of the torso height above the hip line
        # (y grows downward, so "above" = smaller y).
        gate_y = hip_y - raise_gate_frac * max(hip_y - shoulder_y, 0.0)

        # Consider each wrist; keep the one reaching farthest past the dead zone so a
        # resting hand near centre can't win over a clear reach by the other hand.
        for wrist_idx, wrist in ((15, lw), (16, rw)):
            # Visibility gate + Gemini depth veto: a "wrist" whose sampled depth
            # is the back wall is a phantom, whatever visibility claims.
            if not trusted_landmark(context, wrist_idx, wrist, min_visibility):
                continue
            screen_x = 1.0 - wrist.x
            offset = screen_x - screen_midline    # <0 = screen-left, >0 = screen-right
            if require_raised and wrist.y > gate_y:
                # Below the raise gate. A fork target low on screen (Scene 9's
                # hidden compartment) draws a natural point at waist height —
                # accept it anyway when the reach is unambiguous: well past the
                # dead zone, at low_reach_frac of a shoulder width from midline.
                low_reach = params.get("low_reach_frac", 0.9) * max(shoulder_width, 0.10)
                if not (params.get("allow_low_reach", True) and abs(offset) >= low_reach):
                    continue
            if abs(offset) < dead:
                continue   # inside centre dead zone — selects nothing
            direction = "left" if offset < 0 else "right"
            if direction not in accepted:
                continue
            # (POSE-ONLY, July 2026: the old reject_open_palm finger check
            # needed the Hands model and is gone — the raise gate + dead zone
            # are the deliberateness filters now. The param is accepted and
            # ignored so existing metadata stays valid.)
            if abs(offset) > abs(best_offset):
                best_offset = offset
                best_direction = direction

    context["dominant_direction"] = best_direction   # for debug overlay

    now = time.monotonic()
    if best_direction:
        if context.get("point_direction") != best_direction:
            # Region changed (or first entry) — restart the hold timer.
            context["point_direction"] = best_direction
            context["point_direction_since"] = now
        since = context.get("point_direction_since", now)
        return (now - since) * 1000 >= hold_ms

    context["point_direction"] = None
    context["point_direction_since"] = None
    return False
