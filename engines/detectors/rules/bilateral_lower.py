"""
bilateral_lower — both hands moving downward together.
Used for Scene 7 duck-down OI.

Params:
  min_velocity (float): minimum downward normalised velocity per hand. Default 0.05.

Context keys: prev_wrist_y
"""


def detect(landmarks, params: dict, context: dict) -> bool:
    # TODO: compute per-hand Y delta from prev_wrist_y, both must be positive
    # (downward) and above min_velocity.
    context["prev_wrist_y"] = (
        [hand.landmark[0].y for hand in landmarks] if landmarks else []
    )
    return False
