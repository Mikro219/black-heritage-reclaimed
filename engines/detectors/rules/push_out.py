"""
push_out — both hands thrust forward simultaneously (Z-approach).

Used by:
  AL-10-009 forward_push  → high velocity: min_growth_pct 50, window_ms 250
  AL-11-012 launch_push   → moderate velocity: min_growth_pct 35, window_ms 400

With the Gemini 335 depth sampler present (context["_depth_mm_at"]), a push is
measured in REAL millimetres: both wrists must approach the camera by
min_depth_delta_mm within the window, and while depth is flowing it is
authoritative (the bbox proxy is suppressed). On a plain webcam, both hands
must show simultaneous bounding-box area growth above min_growth_pct within
the window — the same monocular z-proxy as forward_reach, applied bilaterally.

Params:
  min_growth_pct (float): required % bbox area growth over window. Default 50.
  window_ms (float): sliding window duration in milliseconds. Default 250.
  min_depth_delta_mm (float): required real depth decrease per wrist within the
                  window when the depth sampler is present. Default 200.

No starting-pose constraint — any position counts, as long as both hands
move forward together. This keeps it forgiving for visitors who start extended.

Context keys: push_area_history, push_depth_history, push_fired
"""

import time

from . import hand_pose


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks or len(landmarks) < 2:
        return False

    if context.get("push_fired"):
        return False

    min_growth      = params.get("min_growth_pct", 50) / 100.0
    window_s        = params.get("window_ms", 250) / 1000.0
    min_depth_delta = params.get("min_depth_delta_mm", 200)

    now = time.monotonic()
    depth_at = context.get("_depth_mm_at")

    # ── Preferred: real wrist depth (Gemini 335) — both wrists must approach
    #    the camera by min_depth_delta_mm within the window. ──────────────────
    if callable(depth_at):
        depth_hist: list = context.setdefault("push_depth_history", [[], []])
        sampled = 0
        for i, hand in enumerate(landmarks[:2]):
            hw = hand.landmark[0]
            d = depth_at(hw.x, hw.y)
            if d is not None:
                depth_hist[i].append((now, d))
                sampled += 1
            depth_hist[i] = [(t, dd) for t, dd in depth_hist[i]
                             if now - t <= window_s]
        if sampled == 2 and all(len(h) >= 2 for h in depth_hist[:2]):
            if all(max(dd for _, dd in h) - h[-1][1] >= min_depth_delta
                   for h in depth_hist[:2]):
                context["push_fired"] = True
                return True
            # Depth is authoritative when it's flowing — don't let the bbox
            # proxy fire on hands the real sensor says aren't moving forward.
            return False

    histories: list = context.setdefault("push_area_history", [[], []])
    # Ensure we have exactly 2 per-hand history slots
    while len(histories) < 2:
        histories.append([])

    # Assign hands to slots by index (stable within a single detection window)
    for i, hand in enumerate(landmarks[:2]):
        area = hand_pose.bbox_area(hand)
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
