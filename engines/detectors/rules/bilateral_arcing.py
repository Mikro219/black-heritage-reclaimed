"""
bilateral_arcing — bilateral high-to-low forward arc (paddle stroke), N times.
Used for Scene 11 `paddle` CG (AL-11-013), 4 strokes.

Params:
  strokes (int): complete stroke cycles required. Default 4.

Approach:
  A stroke is: wrist moves from high position (y < 0.45) to low position (y > 0.65)
  and back up. Track each hand's Y trajectory; count times both hands complete the
  down-then-up cycle. Both hands must be in roughly the same phase.

Context keys: arc_stroke_count, arc_phase, arc_peak_y, arc_valley_y
"""


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks or len(landmarks) < 2:
        return False

    if "arc_stroke_count" not in context:
        context["arc_stroke_count"] = 0
        context["arc_phase"] = {}   # hand_idx → "up" | "down"
        context["arc_prev_y"] = {}

    strokes = params.get("strokes", 4)
    phase = context["arc_phase"]
    prev_y = context["arc_prev_y"]

    completed = []
    for i, hand in enumerate(landmarks):
        y = hand.landmark[0].y
        p_y = prev_y.get(i, y)

        if phase.get(i) is None:
            phase[i] = "up" if y < 0.5 else "down"
        elif phase[i] == "up" and y > 0.65:
            phase[i] = "down"
        elif phase[i] == "down" and y < 0.45:
            phase[i] = "up"
            completed.append(i)

        prev_y[i] = y

    # Count stroke when both hands complete a cycle within the same update
    if len(completed) == 2:
        context["arc_stroke_count"] += 1
        if context["arc_stroke_count"] >= strokes:
            context["arc_stroke_count"] = 0
            return True

    context["arc_prev_y"] = prev_y
    return False
