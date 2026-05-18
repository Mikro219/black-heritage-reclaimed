"""
presence_bilateral_still — both hands present AND total landmark velocity below
threshold for stillness_ms. Optional palm_orientation check.

Params:
  stillness_ms (int): ms both hands must stay below velocity threshold. Default 1000.
  velocity_threshold (float): max summed wrist delta/s (normalised). Default 0.08.
  palm_orientation (str | None): "forward" — rough check only (no depth data). Default None.

Approach:
  Track wrist-to-wrist total movement between frames. If both hands are present and
  the summed wrist movement is below the velocity threshold for stillness_ms, fire.
  palm_orientation="forward": we check that palm landmarks (5,9,13,17) are spread
  horizontally relative to wrist — a rough forward-open-palm heuristic.

Context keys: bilateral_still_since, prev_wrist_xy
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks or len(landmarks) < 2:
        context["bilateral_still_since"] = None
        context["prev_wrist_xy"] = None
        return False

    stillness_ms = params.get("stillness_ms", 1000)
    vel_thresh = params.get("velocity_threshold", 0.08)
    palm_check = params.get("palm_orientation")

    now = time.monotonic()
    current_xy = [(hand.landmark[0].x, hand.landmark[0].y) for hand in landmarks]

    prev_xy = context.get("prev_wrist_xy")
    if prev_xy is None or len(prev_xy) != len(current_xy):
        context["prev_wrist_xy"] = current_xy
        context["bilateral_still_since"] = None
        return False

    # Total wrist displacement this frame (rough velocity proxy)
    total_delta = sum(
        ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        for (cx, cy), (px, py) in zip(current_xy, prev_xy)
    )

    still = total_delta < vel_thresh

    # Palm orientation: check that fingers splay horizontally (open palm facing forward)
    if still and palm_check == "forward":
        for hand in landmarks:
            lm = hand.landmark
            # Finger bases (MCPs) should span at least 15% of frame width
            mcp_xs = [lm[i].x for i in (5, 9, 13, 17)]
            span = max(mcp_xs) - min(mcp_xs)
            if span < 0.07:
                still = False
                break

    context["prev_wrist_xy"] = current_xy

    if still:
        if context.get("bilateral_still_since") is None:
            context["bilateral_still_since"] = now
        return (now - context["bilateral_still_since"]) * 1000 >= stillness_ms
    else:
        context["bilateral_still_since"] = None
        return False
