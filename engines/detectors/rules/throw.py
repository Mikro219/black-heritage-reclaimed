"""
throw — overhand throwing motion: one hand winds up above the shoulder line
(arming stays active for as long as the hand is up there), then fires when the
hand snaps below the shoulder line within the stroke window.

Pose-primary: the above/below-shoulder phase tracking uses Pose wrists so the
stroke still tracks when MediaPipe Hands loses the motion-blurred hand mid-throw.
Transient wrist-visibility loss mid-stroke does NOT disarm — motion blur is
exactly when a real throw is happening.

By default the release check is just the above->below crossing — lenient, public
installation. Set `require_growth: true` to additionally demand a z-approach
toward the camera during the stroke. Preference order for that check:
REAL wrist depth delta (Gemini 335 sampler) > Hands bounding-box growth (the
same z-proxy as push_out / forward_reach) > Pose wrist z delta (hip-normalised;
more negative = closer to the camera).

Either arm can throw. Both are tracked independently; the first to complete a
valid stroke fires.

Params:
  max_stroke_ms (float): max time allowed between leaving the above-shoulder
                          zone and crossing below the shoulder line. A slow
                          lower-and-drop is not a throw. Default 600.
  shoulder_margin (float): normalised y margin around the shoulder line — the
                          wrist must be margin ABOVE it to arm and margin BELOW
                          it to release, giving the stroke hysteresis. Default 0.03.
  release_y_offset (float): shifts the RELEASE line up (negative) or down
                          (positive) in normalised screen coords, independent of
                          the wind-up line (which stays at the true shoulder).
                          Lowering it demands a deeper follow-through; raising it
                          fires earlier in the stroke. Same convention as
                          waist_y_offset in run_arms/paddle. Tuned with W/S in
                          the gesture tuner; persists in the host profile's
                          gesture_tuning section. Default 0.0.
  require_growth (bool): if True, the release must also pass the z-approach
                          check. Default False (crossing suffices).
  min_growth_pct (float): required % hand-bbox area growth between the armed
                          (above-shoulder) baseline and the release frame.
                          Only used with require_growth. Default 40.
  min_pose_z_delta (float): fallback z-approach threshold on the Pose wrist z
                          when neither depth nor hand bboxes are available.
                          Only used with require_growth. Default 0.2.
  min_depth_delta_mm (float): with the Gemini 335 depth sampler present, the
                          require_growth check uses the REAL wrist depth delta
                          between wind-up and release instead of bbox growth.
                          Default 250.
  min_visibility (float): Pose wrist visibility gate (phantom estimated wrists
                          are ignored; the Gemini depth veto stacks on top —
                          see engines/depth/fusion.py). Default 0.5.

Context keys: throw_armed (dict per side)  (reads: _pose_lm, _pose_depth)
"""

import time

from . import hand_pose
from ...depth.fusion import trusted_landmark

# (side, pose wrist index, pose shoulder index)
_ARMS = [("L", 15, 11), ("R", 16, 12)]


def _nearest_hand_area(landmarks, wx: float, wy: float, max_dist: float = 0.25):
    """Bbox area of the Hands detection whose wrist is nearest the Pose wrist,
    or None when no hand is close enough to be the same physical hand."""
    hand = hand_pose.nearest_hand(landmarks, wx, wy, max_dist)
    return hand_pose.bbox_area(hand) if hand is not None else None


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    if pose_lm is None:
        context["throw_armed"] = {}
        return False

    require_growth  = params.get("require_growth", False)
    min_growth      = params.get("min_growth_pct", 40) / 100.0
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
            # state and skip the frame; the stroke window (below) still expires
            # a stale arm on its own clock.
            continue

        # Wind-up is judged at the true shoulder line; only the release line moves
        # with the tuned offset.
        above = wrist.y < shoulder.y - margin
        below = wrist.y > shoulder.y + release_offset + margin
        area  = _nearest_hand_area(landmarks, wrist.x, wrist.y)
        depth = fusion.landmark_mm(wrist_i) if fusion is not None else None

        if above:
            # (Re)arm continuously while above the shoulder — the stroke's baseline
            # is the LAST above-shoulder frame, so a long wind-up doesn't expire.
            armed[side] = {"t": now, "area": area, "z": getattr(wrist, "z", 0.0),
                           "depth": depth}
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

        # Release frame: wrist crossed below the shoulder within the stroke window.
        armed.pop(side, None)

        if not require_growth:
            context["throw_armed"] = {}
            return True

        # Optional strict mode: also demand z-approach during the stroke.
        # Preference order: REAL depth delta (Gemini 335) > hand bbox growth >
        # pose z delta.
        if state.get("depth") is not None and depth is not None:
            grew = (state["depth"] - depth) >= min_depth_delta
        elif state["area"] and area:
            grew = (area / state["area"]) >= 1.0 + min_growth
        else:
            z_now = getattr(wrist, "z", 0.0)
            grew = (state["z"] - z_now) >= min_z_delta
        if grew:
            context["throw_armed"] = {}
            return True

    return False
