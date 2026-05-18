"""
bilateral_alternating — alternating hand motion for ≥ min_cycles.
Used for Scene 9 Path B walk CG (slow, ≥3 cycles) and Scene 10 run arms OI (fast).

Params:
  min_cycles (int): L-R alternating strokes required. Default 3.
  cadence (str | None): "walk" (inter-stroke > 0.5s) | "run" (< 0.4s) | None (any). Default None.

Approach:
  Track each hand's wrist X position. A "stroke" is when a hand's X crosses the body
  midline (0.5) in either direction after being on the opposite side. Alternation is
  detected when the two hands take turns making strokes. Count min_cycles alternations.

Context keys: alt_cycle_count, alt_last_hand, alt_last_time, alt_prev_x
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks or len(landmarks) < 2:
        return False

    if "alt_cycle_count" not in context:
        context["alt_cycle_count"] = 0
        context["alt_last_hand"] = None
        context["alt_last_time"] = None
        context["alt_prev_x"] = {}

    min_cycles = params.get("min_cycles", 3)
    cadence = params.get("cadence")

    now = time.monotonic()
    prev_x = context["alt_prev_x"]

    for i, hand in enumerate(landmarks):
        x = hand.landmark[0].x
        p = prev_x.get(i)
        if p is None:
            prev_x[i] = x
            continue

        # Stroke = hand crossed a midpoint (0.5) between frames in either direction
        crossed = (p < 0.5 <= x) or (x < 0.5 <= p)
        if crossed and context["alt_last_hand"] != i:
            last_t = context["alt_last_time"]
            dt = (now - last_t) if last_t is not None else None

            # Cadence gating
            ok = True
            if cadence == "walk" and dt is not None and dt < 0.35:
                ok = False
            elif cadence == "run" and dt is not None and dt > 0.55:
                ok = False

            if ok:
                context["alt_cycle_count"] += 1
                context["alt_last_hand"] = i
                context["alt_last_time"] = now

                if context["alt_cycle_count"] >= min_cycles * 2:  # each hand counts
                    context["alt_cycle_count"] = 0
                    context["alt_last_hand"] = None
                    return True

        prev_x[i] = x

    context["alt_prev_x"] = prev_x
    return False
