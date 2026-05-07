"""
mouth_proximity_tip — hand tip near mouth with tipping/tilting wrist motion.
Used for Scene 1 drink OI.

Params:
  hold_ms (int): ms tip must remain near mouth. Default 400.
  proximity_threshold (float): max normalised distance tip→mouth estimate. Default 0.12.

Approach: estimate mouth at approximately (0.5, 0.75) normalised frame coords;
check index or thumb tip proximity; validate tipping wrist motion from last N frames.

Context keys: tip_near_since
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    # TODO: estimate mouth position, check fingertip proximity,
    # validate tipping wrist rotation, hold-gate.
    if "tip_near_since" not in context:
        context["tip_near_since"] = None
    return False
