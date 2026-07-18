"""
unravel — both hands circling in front of the torso (winding/unwinding rope).

Used by: AL-11-011 unravel (Scene 11, ≥2 cycles).

ELBOW-LINE CROSSING MODEL (July 2026): the old model tracked the wrist-vs-
wrist diff signal (left.y - right.y), which only oscillates when the hands
wind in OPPOSITE phase — the scripted "both hands rotating outward" is
parallel, in-phase circles, so the diff stayed nearly flat and a hidden
amplitude gate blocked firing forever while the zero-crossing count (riding
on jitter) climbed. Now each wrist is tracked INDEPENDENTLY against the line
drawn between the two ELBOW landmarks (13/14): a circling hand crosses that
line twice per cycle whatever the phase relationship, and the line moves
with the body so the check survives camera angle and posture drift. A
hysteresis band (band_frac of shoulder width around the line) rejects
landmark jitter, crossings older than the rolling window age away, and
wrist/elbow dropout resets all state.

Constraints kept from the old model:
  * Proximity — wrists within prox_frac * shoulder_width of each other
    (winding around each other, not independent arm waving).
  * Torso band — both wrists between shoulder and hip height (loose).
Either failing clears the accumulated state, same as before.

Params:
  min_cycles (int): complete cycles (2 elbow-line crossings each) required
                    PER WRIST. Default 2.
  window_ms (float): sliding window for crossing accumulation. Default 3000.
  prox_frac (float): max wrist-to-wrist distance as fraction of
                     shoulder_width. Default 0.25.
  band_frac (float): hysteresis half-band around the elbow line as a
                     fraction of shoulder width — a wrist must clear it on
                     BOTH sides for a crossing to count. Default 0.08.
  min_visibility (float): Pose wrist/elbow visibility gate. Default 0.5.
  min_amplitude_frac: ACCEPTED AND IGNORED (legacy diff-signal amplitude
                     gate — superseded by the hysteresis band).

Fallback (no Pose): return False.

Context keys: unravel_zone, unravel_crossings, unravel_fired
"""

import math
import time

from ...depth.fusion import trusted_landmark


def _reset(context: dict) -> None:
    context["unravel_zone"] = {}
    context["unravel_crossings"] = {}


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    if pose_lm is None:
        return False

    if context.get("unravel_fired"):
        return False

    min_cycles = params.get("min_cycles", 2)
    window_s   = params.get("window_ms", 3000) / 1000.0
    prox_frac  = params.get("prox_frac", 0.25)
    band_frac  = params.get("band_frac", 0.08)
    min_vis    = params.get("min_visibility", 0.5)

    try:
        sh_l, sh_r   = pose_lm[11], pose_lm[12]
        el_l, el_r   = pose_lm[13], pose_lm[14]
        hip_l, hip_r = pose_lm[23], pose_lm[24]
        lw, rw       = pose_lm[15], pose_lm[16]
    except (IndexError, TypeError):
        return False

    # Wrists + elbows must be real (visibility gate + Gemini phantom veto) —
    # a stale zone against a phantom line must not register crossings.
    for idx, lm_ in ((15, lw), (16, rw), (13, el_l), (14, el_r)):
        if not trusted_landmark(context, idx, lm_, min_vis):
            _reset(context)
            return False

    shoulder_width = abs(sh_l.x - sh_r.x)
    shoulder_y     = (sh_l.y + sh_r.y) / 2
    hip_y          = (hip_l.y + hip_r.y) / 2

    # Proximity check: wrists winding around each other, not waving apart.
    wrist_dist = math.hypot(lw.x - rw.x, lw.y - rw.y)
    if wrist_dist > prox_frac * max(shoulder_width, 0.10):
        _reset(context)
        return False

    # Torso height check (loose — include shoulder height level).
    for w_ in (lw, rw):
        if not (shoulder_y - 0.05 <= w_.y <= hip_y + 0.05):
            _reset(context)
            return False

    # Elbow line: direction + length; collapsed line (body sideways) is
    # useless geometry.
    dx, dy = el_r.x - el_l.x, el_r.y - el_l.y
    line_len = math.hypot(dx, dy)
    if line_len < 0.03:
        _reset(context)
        return False

    band = band_frac * max(shoulder_width, 0.10)
    now = time.monotonic()
    if "unravel_crossings" not in context:
        _reset(context)

    for side, w_ in (("L", lw), ("R", rw)):
        # signed perpendicular offset of the wrist from the elbow line
        off = (dx * (w_.y - el_l.y) - dy * (w_.x - el_l.x)) / line_len
        zone = context["unravel_zone"].get(side)
        new_zone = 1 if off > band else (-1 if off < -band else zone)
        if zone is not None and new_zone == -zone:
            context["unravel_crossings"].setdefault(side, []).append(now)
        context["unravel_zone"][side] = new_zone

    cutoff = now - window_s
    counts = {}
    for side, hist in context["unravel_crossings"].items():
        while hist and hist[0] < cutoff:
            hist.pop(0)
        counts[side] = len(hist) // 2      # 2 crossings = 1 cycle

    if counts.get("L", 0) >= min_cycles and counts.get("R", 0) >= min_cycles:
        context["unravel_fired"] = True
        _reset(context)
        return True

    return False
