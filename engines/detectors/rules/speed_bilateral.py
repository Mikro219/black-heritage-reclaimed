"""
speed_bilateral — both hands moving at or above a velocity threshold, with
enough burst events to confirm fast deliberate movement.

Used for Scene 10 `fast_gather` CG (≥3 bursts in 3s) and `forward_push` CG.

Params:
  min_bursts (int): minimum burst count within burst_window_ms. Default 3.
  burst_window_ms (int): rolling window to accumulate bursts. Default 3000.
  velocity_multiplier (float): threshold as a multiple of a nominal idle velocity. Default 2.0.
  use_pose (bool): track Pose wrists (landmarks 15/16) instead of hand wrists.
                   More robust for full-arm bilateral motion at exhibition distance.
                   Default False (uses MediaPipe Hands wrists).
  min_visibility (float): Pose wrist visibility gate (use_pose mode only).
                   Occluded/out-of-frame wrists get jittery phantom estimates
                   whose noise reads as motion bursts. Default 0.5.

Approach:
  Each frame, compute the summed bilateral wrist displacement from the previous frame.
  A "burst" is a frame where this displacement exceeds the idle threshold.
  Fire when min_bursts bursts occur within burst_window_ms.

Context keys: speed_burst_times, prev_wrist_pos
"""

import time

_IDLE_DISPLACEMENT = 0.015  # normalised units/frame ≈ resting hand wobble


def _reset(context: dict) -> None:
    """Signal lost or untrustworthy — restart motion tracking from scratch.
    Clearing burst history matters as much as prev position: bursts accumulated
    before a dropout must not count toward a fire after the signal returns."""
    context["prev_wrist_pos"] = None
    context["speed_burst_times"] = []


def detect(landmarks, params: dict, context: dict) -> bool:
    use_pose = params.get("use_pose", False)

    if use_pose:
        # Pose wrists: 15 = left_wrist, 16 = right_wrist.
        pose_lm = context.get("_pose_lm")
        if pose_lm is None:
            _reset(context)
            return False
        lw, rw = pose_lm[15], pose_lm[16]
        # Pose reports positions for occluded/out-of-frame wrists too (hands at
        # the sides, below the frame bottom). Those phantom estimates jitter
        # frame-to-frame, and the jitter registers as motion bursts — so a person
        # standing still with no visible hands could fire the gesture. Gate on
        # visibility, and drop accumulated bursts so half a gesture's worth of
        # phantom noise can't carry over into the next valid stretch.
        min_visibility = params.get("min_visibility", 0.5)
        if (getattr(lw, "visibility", 1.0) < min_visibility
                or getattr(rw, "visibility", 1.0) < min_visibility):
            _reset(context)
            return False
        current_pos = [(lw.x, lw.y), (rw.x, rw.y)]
    else:
        if not landmarks or len(landmarks) < 2:
            _reset(context)
            return False
        current_pos = [(hand.landmark[0].x, hand.landmark[0].y) for hand in landmarks]

    if "speed_burst_times" not in context:
        context["speed_burst_times"] = []
        context["prev_wrist_pos"] = None

    min_bursts = params.get("min_bursts", 3)
    burst_window_ms = params.get("burst_window_ms", 3000)
    multiplier = params.get("velocity_multiplier", 2.0)
    threshold = _IDLE_DISPLACEMENT * multiplier

    prev_pos = context.get("prev_wrist_pos")

    now = time.monotonic()
    if prev_pos and len(prev_pos) == len(current_pos):
        displacement = sum(
            ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            for (cx, cy), (px, py) in zip(current_pos, prev_pos)
        )
        if displacement >= threshold:
            context["speed_burst_times"].append(now)

    # Trim expired bursts
    window_s = burst_window_ms / 1000.0
    context["speed_burst_times"] = [
        t for t in context["speed_burst_times"] if now - t <= window_s
    ]
    context["prev_wrist_pos"] = current_pos

    if len(context["speed_burst_times"]) >= min_bursts:
        context["speed_burst_times"] = []  # reset so it can fire again after cooldown
        return True
    return False
