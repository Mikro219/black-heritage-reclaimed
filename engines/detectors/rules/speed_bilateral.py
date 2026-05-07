"""
speed_bilateral — both hands moving at ≥ velocity_multiplier × idle speed.
Used for Scene 10 fast-gather (≥3 bursts in 3s) and Scene 10 forward-push.

Params:
  min_bursts (int): number of simultaneous above-threshold bursts. Default 3.
  burst_window_ms (int): window to accumulate bursts. Default 3000.
  velocity_multiplier (float): burst threshold as multiple of idle. Default 2.0.

Context keys: speed_burst_times, prev_wrist_pos
"""


def detect(landmarks, params: dict, context: dict) -> bool:
    # TODO: compute bilateral wrist velocity, detect burst events above threshold,
    # accumulate timestamps, gate on min_bursts within burst_window_ms.
    if "speed_burst_times" not in context:
        context["speed_burst_times"] = []
        context["prev_wrist_pos"] = None
    return False
