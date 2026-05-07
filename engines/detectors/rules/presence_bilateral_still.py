"""
presence_bilateral_still — both hands present AND total landmark velocity below
threshold for stillness_ms. Optional palm_orientation check.

Params:
  stillness_ms (int): ms both hands must stay below velocity threshold. Default 1000.
  velocity_threshold (float): max normalised units/s summed across both hands. Default 0.02.
  palm_orientation (str | None): "forward" validates palms face camera. Default None.

Context keys: bilateral_still_since, prev_landmarks
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    # TODO: compute per-hand velocity from prev_landmarks,
    # gate on threshold, then hold-time check.
    context["prev_landmarks"] = landmarks
    return False
