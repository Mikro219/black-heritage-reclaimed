"""
throw — overhand throwing motion: one hand starts above the shoulder line, then
within a short window snaps below it while growing sharply in apparent size
(z-approach toward the camera / "out of the screen").

Pose-primary: the above/below-shoulder phase tracking uses Pose wrists so the
stroke still tracks when MediaPipe Hands loses the motion-blurred hand mid-throw.
The z-growth check prefers the Hands bounding-box area (the same z-proxy used by
push_out / forward_reach); when no matching hand bbox is available on either end
of the stroke it falls back to the Pose wrist z delta (Pose z is hip-normalised;
more negative = closer to the camera).

Either arm can throw. Both are tracked independently; the first to complete a
valid stroke fires.

Params:
  min_growth_pct (float): required % hand-bbox area growth between the armed
                          (above-shoulder) baseline and the release (below-
                          shoulder) frame. Default 40.
  min_pose_z_delta (float): fallback z-approach threshold on the Pose wrist z
                          when hand bboxes are unavailable. Default 0.2.
  max_stroke_ms (float): max time allowed between leaving the above-shoulder
                          zone and crossing below the shoulder line. A slow
                          lower-and-drop is not a throw. Default 600.
  shoulder_margin (float): normalised y margin around the shoulder line — the
                          wrist must be margin ABOVE it to arm and margin BELOW
                          it to release, giving the stroke hysteresis. Default 0.03.
  min_visibility (float): Pose wrist visibility gate (phantom estimated wrists
                          are ignored, same rule as point_region). Default 0.5.

Future (Orbbec Gemini 335): with a real depth stream, the growth check should be
replaced by an actual depth delta on the wrist (see engines/depth/orbbec_camera.py);
the params anticipate a `min_depth_delta_mm` alternative.

Context keys: throw_armed (dict per side), throw_fired
"""

import math
import time

# (side, pose wrist index, pose shoulder index)
_ARMS = [("L", 15, 11), ("R", 16, 12)]


def _hand_bbox_area(hand) -> float:
    lm = hand.landmark
    xs = [l.x for l in lm]
    ys = [l.y for l in lm]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _nearest_hand_area(landmarks, wx: float, wy: float, max_dist: float = 0.25):
    """Bbox area of the Hands detection whose wrist is nearest the Pose wrist,
    or None when no hand is close enough to be the same physical hand."""
    best_area = None
    best_dist = max_dist
    for hand in landmarks or []:
        hw = hand.landmark[0]
        d = math.hypot(hw.x - wx, hw.y - wy)
        if d <= best_dist:
            best_dist = d
            best_area = _hand_bbox_area(hand)
    return best_area


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    if pose_lm is None:
        context["throw_armed"] = {}
        return False

    min_growth      = params.get("min_growth_pct", 40) / 100.0
    min_z_delta     = params.get("min_pose_z_delta", 0.2)
    max_stroke_s    = params.get("max_stroke_ms", 600) / 1000.0
    margin          = params.get("shoulder_margin", 0.03)
    min_visibility  = params.get("min_visibility", 0.5)

    armed: dict = context.setdefault("throw_armed", {})
    now = time.monotonic()

    for side, wrist_i, shoulder_i in _ARMS:
        try:
            wrist    = pose_lm[wrist_i]
            shoulder = pose_lm[shoulder_i]
        except (IndexError, TypeError):
            continue
        if getattr(wrist, "visibility", 1.0) < min_visibility:
            armed.pop(side, None)
            continue

        above = wrist.y < shoulder.y - margin
        below = wrist.y > shoulder.y + margin
        area  = _nearest_hand_area(landmarks, wrist.x, wrist.y)

        if above:
            # (Re)arm continuously while above the shoulder — the stroke's baseline
            # is the LAST above-shoulder frame, so a long wind-up doesn't expire.
            armed[side] = {"t": now, "area": area, "z": getattr(wrist, "z", 0.0)}
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

        # Release frame: wrist is below the shoulder within the stroke window.
        # z-growth check — hand bbox growth preferred, pose z delta as fallback.
        grew = False
        if state["area"] and area:
            grew = (area / state["area"]) >= 1.0 + min_growth
        else:
            z_now = getattr(wrist, "z", 0.0)
            grew = (state["z"] - z_now) >= min_z_delta

        armed.pop(side, None)
        if grew:
            context["throw_armed"] = {}
            return True

    return False
