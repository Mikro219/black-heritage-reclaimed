"""
push_out — both hands thrust forward simultaneously (Z-approach).

Used by:
  AL-10-009 forward_push  → high velocity: window_ms 250
  AL-11-012 launch_push   → moderate velocity: window_ms 400

POSE-ONLY (July 2026): both Pose wrists must approach the camera within the
window — measured in REAL millimetres when the Gemini 335 sampler is present
(context["_depth_mm_at"]), else by the Pose z drop per wrist. The old hand
bounding-box growth proxy needed the Hands model and is gone.

Params:
  window_ms (float): sliding window duration in milliseconds. Default 250.
  min_depth_delta_mm (float): required real depth decrease per wrist within
                  the window when the sampler is present. Default 200.
  min_pose_z_delta (float): required Pose-z decrease per wrist without depth.
                  Default 0.15.
  min_growth_pct (float): ACCEPTED AND IGNORED (legacy bbox param still present
                  in shot metadata).
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

No starting-pose constraint — any position counts, as long as both wrists
move forward together. Forgiving for visitors who start extended.

Context keys: push_depth_history, push_fired  (reads: _pose_lm, _depth_mm_at)
"""

import time

from . import pose_helpers


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if len(wrists) < 2:
        return False

    if context.get("push_fired"):
        return False

    window_s = params.get("window_ms", 250) / 1000.0
    min_mm = params.get("min_depth_delta_mm", 200)
    min_z = params.get("min_pose_z_delta", 0.15)

    now = time.monotonic()
    hist: dict = context.setdefault("push_depth_history", {})

    pushed = {}
    for side in ("L", "R"):
        kind, val = pose_helpers.wrist_depth_or_z(context, pose_lm, side)
        if kind is None:
            return False
        h = hist.setdefault(side, [])
        h.append((now, kind, val))
        h[:] = [(t, k, v) for t, k, v in h if now - t <= window_s]
        same = [v for t, k, v in h if k == kind]
        if len(same) < 2:
            pushed[side] = False
            continue
        drop = max(same) - val
        pushed[side] = ((kind == "mm" and drop >= min_mm)
                        or (kind == "z" and drop >= min_z))

    if pushed and all(pushed.values()):
        context["push_fired"] = True
        return True
    return False
