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

Approach:
  Each frame, compute the summed bilateral wrist displacement from the previous frame.
  A "burst" is a frame where this displacement exceeds the idle threshold.
  Fire when min_bursts bursts occur within burst_window_ms.

Context keys: speed_burst_times, prev_wrist_pos
"""

import time

_IDLE_DISPLACEMENT = 0.015  # normalised units/frame ≈ resting hand wobble


def detect(landmarks, params: dict, context: dict) -> bool:
    use_pose = params.get("use_pose", False)

    if use_pose:
        # Pose wrists: 15 = left_wrist, 16 = right_wrist.
        pose_lm = context.get("_pose_lm")
        if pose_lm is None:
            context["prev_wrist_pos"] = None
            return False
        current_pos = [(pose_lm[15].x, pose_lm[15].y),
                       (pose_lm[16].x, pose_lm[16].y)]
    else:
        if not landmarks or len(landmarks) < 2:
            context["prev_wrist_pos"] = None
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
