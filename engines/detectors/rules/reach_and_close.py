"""
reach_and_close — single hand extends forward then closes (grasps).
Used for Scene 6 clasp-hand OI (AL-06-009).

Params:
  (none required — the gesture is self-contained)

Approach:
  Phase 1 (open/reach): fingers extended — fingertip Ys are above (lower than) MCP Ys.
  Phase 2 (close/grasp): fingers curled — fingertip-to-wrist distance shrinks vs. phase-1.
  We proxy Z-reach using fingertip-spread (spread increases when hand extends toward camera).
  Fire when we transition from open to closed within the window.

State machine: None → "open" → "closing" (fire)

Context keys: reach_phase, open_tip_spread
"""

import math


def _finger_curl(hand) -> float:
    """Average curl: 0=fully open, 1=fully closed. Uses tip-to-palm distance."""
    lm = hand.landmark
    wrist = lm[0]
    tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]
    avg_dist = sum(math.hypot(t.x - wrist.x, t.y - wrist.y) for t in tips) / len(tips)
    # Rough normalisation: open hand ~0.35 dist, closed fist ~0.12 dist
    curl = 1.0 - min(max((avg_dist - 0.10) / 0.25, 0.0), 1.0)
    return curl


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks:
        context["reach_phase"] = None
        return False

    # Use the first (or only) hand
    hand = landmarks[0]
    curl = _finger_curl(hand)

    phase = context.get("reach_phase")

    if phase is None:
        if curl < 0.35:  # hand is open — start watching
            context["reach_phase"] = "open"
            context["open_tip_spread"] = curl
    elif phase == "open":
        if curl > 0.65:  # hand closed after being open — fire
            context["reach_phase"] = None
            return True
        # Stay in open phase if curl is still low
        if curl > 0.50:
            # Mid-curl — transitioning
            pass

    return False
