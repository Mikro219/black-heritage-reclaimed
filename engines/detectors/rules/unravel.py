"""
unravel — both wrists wind around each other in opposite phase (winding rope).

Used by: AL-11-011 unravel (Scene 11 CG, ≥2 cycles).

Tracks the difference signal  diff(t) = left_wrist.y - right_wrist.y.
When the wrists rotate around each other the diff signal oscillates,
crossing zero as one wrist rises above the other and back.

Wrist positions come from hand landmarks (index 0 of each hand) when both
hands are visible — more accurate than Pose wrists for fine motion. Falls
back to Pose wrists (landmarks 15/16) when fewer than 2 hands are detected.
Hands are sorted by x so left/right assignment is stable across frames.

Body geometry (Pose landmarks, read from context["_pose_lm"]):
  11 (left_shoulder),  12 (right_shoulder)  → shoulder_width, shoulder_y
  23 (left_hip),       24 (right_hip)       → hip_y

Proximity constraint: wrists must stay within prox_frac * shoulder_width
  of each other throughout. This confirms they're winding around each other,
  not just making independent arm movements.

Torso constraint: both wrists between shoulder_y and hip_y. Loose — wrists
  at shoulder height or just below hip are included (the action is roughly
  in front of the torso).

Amplitude constraint: peak-to-peak of the diff signal over the window must
  be >= min_amplitude_frac * shoulder_to_hip_dist. Rejects tiny twitches.

Zero-crossing counting: fire when zero_crossings >= min_cycles * 2
  within the last window_ms and amplitude condition is met.

Params:
  min_cycles (int): complete winding cycles required. Default 2.
  window_ms (float): sliding window for zero-crossing accumulation. Default 3000.
  min_amplitude_frac (float): min peak-to-peak as fraction of shoulder-hip dist. Default 0.10.
  prox_frac (float): max wrist-to-wrist distance as fraction of shoulder_width. Default 0.25.

Fallback (no Pose): return False.

Context keys: unravel_diff_history (list[(timestamp, diff)]), unravel_fired
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    if pose_lm is None:
        return False

    if context.get("unravel_fired"):
        return False

    min_cycles     = params.get("min_cycles", 2)
    window_s       = params.get("window_ms", 3000) / 1000.0
    min_amp_frac   = params.get("min_amplitude_frac", 0.10)
    prox_frac      = params.get("prox_frac", 0.25)

    sh_l, sh_r     = pose_lm[11], pose_lm[12]
    hip_l, hip_r   = pose_lm[23], pose_lm[24]

    shoulder_width = abs(sh_l.x - sh_r.x)
    shoulder_y     = (sh_l.y + sh_r.y) / 2
    hip_y          = (hip_l.y + hip_r.y) / 2
    sh_hip_dist    = max(hip_y - shoulder_y, 0.10)

    # POSE-ONLY (July 2026): wrists come from the Pose skeleton (15/16).
    lw, rw = pose_lm[15], pose_lm[16]

    import math as _math
    # Proximity check: wrists close to each other
    wrist_dist = _math.hypot(lw.x - rw.x, lw.y - rw.y)
    if wrist_dist > prox_frac * max(shoulder_width, 0.10):
        context["unravel_diff_history"] = []
        return False

    # Torso height check (loose — include shoulder height level)
    if not (shoulder_y - 0.05 <= lw.y <= hip_y + 0.05):
        context["unravel_diff_history"] = []
        return False
    if not (shoulder_y - 0.05 <= rw.y <= hip_y + 0.05):
        context["unravel_diff_history"] = []
        return False

    diff = lw.y - rw.y
    now  = time.monotonic()

    history: list = context.setdefault("unravel_diff_history", [])
    history.append((now, diff))
    # Prune old entries
    context["unravel_diff_history"] = [(t, d) for t, d in history if now - t <= window_s]
    history = context["unravel_diff_history"]

    if len(history) < 4:
        return False

    diffs = [d for _, d in history]

    # Amplitude check
    peak_to_peak = max(diffs) - min(diffs)
    min_amplitude = min_amp_frac * sh_hip_dist
    if peak_to_peak < min_amplitude:
        return False

    # Count zero-crossings
    crossings = 0
    for i in range(1, len(diffs)):
        if (diffs[i - 1] > 0) != (diffs[i] > 0):
            crossings += 1

    if crossings >= min_cycles * 2:
        context["unravel_fired"] = True
        context["unravel_diff_history"] = []
        return True

    return False
