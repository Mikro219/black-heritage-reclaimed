"""
bilateral_sweep — one extended arm sweeping horizontally across ≥ min_screen_fraction.
Used for Scene 3 arm sweep (L→R, ≥1/3 screen width) CG.

Params:
  min_screen_fraction (float): minimum X travel during the sweep. Default 0.33.
  direction (str): "left_to_right" | "right_to_left" | "any". Default "left_to_right".

Approach:
  Track the wrist X across frames. Record the leftmost X seen (sweep_start_x).
  When current X exceeds sweep_start_x + min_screen_fraction, fire.
  Reset start_x whenever the hand reverses direction significantly.

Context keys: sweep_start_x, sweep_direction_sign
"""


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks:
        context["sweep_start_x"] = None
        return False

    min_frac = params.get("min_screen_fraction", 0.33)
    direction = params.get("direction", "left_to_right")
    sign = -1 if direction == "right_to_left" else 1  # +1 = L→R, -1 = R→L

    # Use any hand with extended index (fingertip above MCP)
    best_wrist_x = None
    for hand in landmarks:
        lm = hand.landmark
        tip = lm[8]
        mcp = lm[5]
        if tip.y < mcp.y:  # finger extended
            best_wrist_x = lm[0].x
            break

    if best_wrist_x is None:
        return False

    start_x = context.get("sweep_start_x")
    if start_x is None:
        context["sweep_start_x"] = best_wrist_x
        context["sweep_direction_sign"] = sign
        return False

    traveled = (best_wrist_x - start_x) * sign
    if traveled >= min_frac:
        context["sweep_start_x"] = None  # reset for next sweep
        return True

    # Reset if hand reversed by more than 10%
    if traveled < -0.10:
        context["sweep_start_x"] = best_wrist_x

    return False
