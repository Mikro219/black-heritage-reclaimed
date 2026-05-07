"""
shape_match — path-recorded hand trace evaluated against a reference template.
Used for Scene 4: dipper trace (≥50%) and star trace (≥65%).

Params:
  shape (str): "dipper" | "star"
  threshold (float): override match score (defaults from config thresholds).

Recording: capture fingertip XY while hand is active; evaluate when hand
drops below shoulder for ≥500ms (pause threshold).

Context keys: shape_recording, shape_hand_down_since
"""

import time

PAUSE_THRESHOLD_MS = 500


def detect(landmarks, params: dict, context: dict) -> bool:
    # TODO:
    # 1. Append fingertip XY to context["shape_recording"] while hand active.
    # 2. When hand drops below shoulder for PAUSE_THRESHOLD_MS, evaluate path.
    # 3. Normalise, align to reference template (Procrustes or waypoint match).
    # 4. Return True if score >= threshold; reset recording.
    if "shape_recording" not in context:
        context["shape_recording"] = []
        context["shape_hand_down_since"] = None
    return False
