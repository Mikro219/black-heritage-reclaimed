"""
push_out — both hands thrust forward simultaneously (Z-approach via bbox growth).

Used by:
  AL-10-009 forward_push  → high velocity: min_growth_pct 50, window_ms 250
  AL-11-012 launch_push   → moderate velocity: min_growth_pct 35, window_ms 400

Both hands must show simultaneous bounding-box area growth above min_growth_pct
within the window_ms time window. Area growth is the Z-axis depth proxy
(same principle as forward_reach, but applied bilaterally and time-windowed
rather than frame-count-windowed).

Params:
  min_growth_pct (float): required % bbox area growth over window. Default 50.
  window_ms (float): sliding window duration in milliseconds. Default 250.

Detection:
  For each hand, track (timestamp, bbox_area) pairs.
  At each frame, prune entries older than window_ms.
  Fire when BOTH hands have at least one entry AND the ratio
    current_area / oldest_area_in_window >= 1 + (min_growth_pct / 100)
  is satisfied simultaneously.

No starting-pose constraint — any position counts, as long as both hands
move forward together. This keeps it forgiving for visitors who start extended.

Context keys: push_area_history (list[list[tuple[float,float]]]), push_fired
"""

import time


def _hand_bbox_area(hand) -> float:
    lm = hand.landmark
    xs = [l.x for l in lm]
    ys = [l.y for l in lm]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks or len(landmarks) < 2:
        return False

    if context.get("push_fired"):
        return False

    min_growth = params.get("min_growth_pct", 50) / 100.0
    window_s   = params.get("window_ms", 250) / 1000.0

    histories: list = context.setdefault("push_area_history", [[], []])
    # Ensure we have exactly 2 per-hand history slots
    while len(histories) < 2:
        histories.append([])

    now = time.monotonic()

    # Assign hands to slots by index (stable within a single detection window)
    for i, hand in enumerate(landmarks[:2]):
        area = _hand_bbox_area(hand)
        histories[i].append((now, area))
        # Prune old entries
        histories[i] = [(t, a) for t, a in histories[i] if now - t <= window_s]

    # Both hands must have a history spanning the full window
    both_ready = all(len(h) >= 2 for h in histories[:2])
    if not both_ready:
        return False

    threshold = 1.0 + min_growth
    for h in histories[:2]:
        oldest_area = h[0][1]
        current_area = h[-1][1]
        if oldest_area < 1e-6:
            return False
        if current_area / oldest_area < threshold:
            return False

    context["push_fired"] = True
    return True
