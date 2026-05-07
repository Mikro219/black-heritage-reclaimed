"""
grlib_layer.py — Layer A gesture classifier.

Wraps GRLib's pretrained classifier and adapts it to the BHR gesture event
interface. Authoritative for the standard gesture vocabulary that resembles
GRLib's training distribution:

  - Directional points (left, right, up, down)
  - Open-palm presence (raised hand, both hands raised)
  - Sweep gestures across the screen
  - Fist closure / hand grabbing
  - Brow-shading hand position

For gestures NOT in this list, Layer B (rules/) is authoritative and this
layer is silent.

Authority is declared per-gesture in scene metadata:
  "detector_authority": "grlib"   → this layer fires the event
  "shadow_layer": "rule_based"    → rules/ logs but does not fire

Usage:
  layer = GRLibLayer(config)
  event = layer.classify(landmarks_result)
  # Returns a gesture name string or None.
"""


class GRLibLayer:
    # GRLib vocabulary this layer is authoritative for
    VOCABULARY = {
        "directional_point",
        "bilateral_sweep",
        "presence_bilateral",
        "raise_hands",
        "shade_eyes",
        "fist_close",
    }

    def __init__(self, config: dict):
        self.config = config
        self._classifier = None
        # TODO: initialise GRLib classifier
        # from grlib.pipeline import GestureRecognitionPipeline
        # self._classifier = GestureRecognitionPipeline(...)

    def classify(self, landmarks_result) -> str | None:
        """
        Feed a MediaPipe Hands result and return a gesture name or None.
        Only returns gestures in VOCABULARY; everything else returns None.
        """
        if self._classifier is None or not landmarks_result.multi_hand_landmarks:
            return None
        # TODO: pass landmarks to GRLib, map output label to BHR gesture name
        return None
