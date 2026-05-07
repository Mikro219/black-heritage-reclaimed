"""
bilateral_sweep — one extended arm sweeping horizontally across ≥ min_screen_fraction.
Used for Scene 3 arm sweep (L→R, ≥1/3 screen width).

Params:
  min_screen_fraction (float): minimum X travel fraction. Default 0.33.
  direction (str): "left_to_right" | "right_to_left". Default "left_to_right".

Context keys: sweep_start_x
"""


def detect(landmarks, params: dict, context: dict) -> bool:
    # TODO: track fingertip X across frames; fire when delta ≥ min_screen_fraction
    # at above-idle velocity.
    return False
