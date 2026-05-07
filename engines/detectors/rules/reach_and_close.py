"""
reach_and_close — single hand extends forward then closes (grasps).
Used for Scene 6 clasp-hand OI.

Params:
  hand (str): "either" | "left" | "right". Default "either".

Approach:
  Phase 1 (reach): wrist Z decreases (moves toward camera), fingers extended.
  Phase 2 (close): fingers curl — fingertip-to-palm distance drops sharply.
  Fire on phase-2 completion.

Context keys: reach_phase
"""


def detect(landmarks, params: dict, context: dict) -> bool:
    # TODO: track single-hand reach (Z decrease, open fingers) then
    # close (finger curl) phase transition.
    if "reach_phase" not in context:
        context["reach_phase"] = None
    return False
