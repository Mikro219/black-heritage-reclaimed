"""
throw — overhand throwing motion: one hand winds up above the shoulder line
(arming stays active for as long as the hand is up there), then fires when the
hand snaps below the shoulder line within the stroke window.

POSE-ONLY (July 2026): phase tracking uses Pose wrists (it always did — that's
what kept the stroke alive through motion blur). The optional z-approach check
(`require_growth`) now uses REAL wrist depth (Gemini 335) with the Pose wrist
z delta as fallback; the old hand-bbox growth proxy needed the Hands model and
is gone. Transient wrist-visibility loss mid-stroke does NOT disarm — motion
blur is exactly when a real throw is happening.

Either arm can throw. Both are tracked independently; the first to complete a
valid stroke fires.

Params:
  max_stroke_ms (float): max time between leaving the above-shoulder zone and
                          crossing below the shoulder line. Default 600.
  shoulder_margin (float): normalised y hysteresis margin around the shoulder
                          line. Default 0.03.
  release_y_offset (float): shifts the RELEASE line up (negative) or down
                          (positive), independent of the wind-up line. Tuned
                          with W/S in the gesture tuner. Default 0.0.
  require_growth (bool): if True, the release must also pass the z-approach
                          check. Default False (crossing suffices).
  min_depth_delta_mm (float): required wrist depth decrease between wind-up
                          and release (Gemini sampler). Default 250.
  min_pose_z_delta (float): required Pose wrist z decrease without depth
                          (more negative = closer). Default 0.2.
  min_visibility (float): Pose wrist visibility gate (the Gemini depth veto
                          stacks on top). Default 0.5.

Context keys: throw_armed (dict per side)  (reads: _pose_lm, _pose_depth)
"""

import time

from ...depth.fusion import trusted_landmark

# (side, pose wrist index, pose shoulder index)
_ARMS = [("L", 15, 11), ("R", 16, 12)]


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    if pose_lm is None:
        context["throw_armed"] = {}
        return False

    require_growth  = params.get("require_growth", False)
    min_z_delta     = params.get("min_pose_z_delta", 0.2)
    min_depth_delta = params.get("min_depth_delta_mm", 250)
    max_stroke_s    = params.get("max_stroke_ms", 600) / 1000.0
    margin          = params.get("shoulder_margin", 0.03)
    release_offset  = params.get("release_y_offset", 0.0)
    min_visibility  = params.get("min_visibility", 0.5)

    fusion = context.get("_pose_depth")
    armed: dict = context.setdefault("throw_armed", {})
    now = time.monotonic()

    for side, wrist_i, shoulder_i in _ARMS:
        try:
            wrist    = pose_lm[wrist_i]
            shoulder = pose_lm[shoulder_i]
        except (IndexError, TypeError):
            continue
        if not trusted_landmark(context, wrist_i, wrist, min_visibility):
            # Motion blur mid-stroke routinely tanks wrist visibility for a few
            # frames — exactly when a real throw is happening. Keep the armed
            # state and skip the frame; the stroke window still expires a stale
            # arm on its own clock.
            continue

        # Wind-up is judged at the true shoulder line; only the release line
        # moves with the tuned offset.
        above = wrist.y < shoulder.y - margin
        below = wrist.y > shoulder.y + release_offset + margin
        depth = fusion.landmark_mm(wrist_i) if fusion is not None else None

        if above:
            # (Re)arm continuously while above the shoulder — the stroke's
            # baseline is the LAST above-shoulder frame, so a long wind-up
            # doesn't expire.
            armed[side] = {"t": now, "z": getattr(wrist, "z", 0.0), "depth": depth}
            continue

        state = armed.get(side)
        if state is None:
            continue

        if now - state["t"] > max_stroke_s:
            # Took too long to come down — a lower, not a throw.
            armed.pop(side, None)
            continue

        if not below:
            continue   # still crossing the margin band — keep waiting

        # Release frame: wrist crossed below the shoulder within the window.
        armed.pop(side, None)

        if not require_growth:
            context["throw_armed"] = {}
            return True

        # Optional strict mode: also demand z-approach during the stroke.
        # REAL depth delta (Gemini 335) preferred; Pose wrist z is the fallback.
        if state.get("depth") is not None and depth is not None:
            grew = (state["depth"] - depth) >= min_depth_delta
        else:
            z_now = getattr(wrist, "z", 0.0)
            grew = (state["z"] - z_now) >= min_z_delta
        if grew:
            context["throw_armed"] = {}
            return True

    return False
