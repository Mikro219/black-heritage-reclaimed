"""
GestureEngine — MediaPipe Hands + GRLib gesture detection.

Consumes camera frames, runs hand landmark detection, and classifies
gestures via detector types defined in the interaction schema. Emits
gesture events onto the EventBus.

Detector types implemented as stubs; each raises NotImplementedError
until the MediaPipe/GRLib integration is complete.
"""

import cv2
import mediapipe as mp
from typing import Optional
import time


DETECTOR_TYPES = [
    "presence_bilateral",
    "presence_bilateral_still",
    "directional_point",
    "directional_head_or_hand",
    "bilateral_sweep",
    "bilateral_lower",
    "shape_match",
    "rhythm_bilateral",
    "speed_bilateral",
    "bilateral_alternating",
    "bilateral_arcing",
    "bilateral_rotation",
    "mouth_proximity_tip",
    "reach_and_close",
    "point_target_held",
    "trail_follow",
]


class GestureEngine:
    def __init__(self, config: dict, event_bus: "EventBus"):
        self.config = config
        self.event_bus = event_bus
        self._thresholds = config.get("detection_thresholds", {})

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=self._thresholds.get("min_detection_confidence", 0.6),
            min_tracking_confidence=self._thresholds.get("min_tracking_confidence", 0.5),
        )

        self._active_cg: Optional[dict] = None
        self._active_oi: Optional[dict] = None
        self._oi_open_time: Optional[float] = None
        self._cooldown_until: float = 0.0
        self._last_landmarks = None
        self._input_locked = False

        self.event_bus.subscribe("cg_window_open", self._on_cg_window_open)
        self.event_bus.subscribe("oi_window_open", self._on_oi_window_open)
        self.event_bus.subscribe("input_lock", self._on_input_lock)

    def process_frame(self, frame):
        if self._input_locked:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        self._last_landmarks = results.multi_hand_landmarks

        if not results.multi_hand_landmarks:
            return

        now = time.monotonic()
        if now < self._cooldown_until:
            return

        if self._active_cg:
            if self._detect(self._active_cg, results):
                self._emit_cg(self._active_cg["id"])
                self._active_cg = None

        oi_window_ms = self.config["timing_defaults"].get("oi_window_ms", 6000)
        if self._active_oi and self._oi_open_time:
            elapsed = (now - self._oi_open_time) * 1000
            if elapsed <= oi_window_ms:
                if self._detect(self._active_oi, results):
                    self._emit_oi(self._active_oi["id"])
                    self._active_oi = None
                    self._oi_open_time = None
            else:
                self._active_oi = None
                self._oi_open_time = None

    def hands_detected(self) -> bool:
        return bool(self._last_landmarks)

    def _on_cg_window_open(self, data: dict):
        self._active_cg = data.get("interaction")

    def _on_oi_window_open(self, data: dict):
        self._active_oi = data.get("interaction")
        self._oi_open_time = time.monotonic()

    def _on_input_lock(self, data: dict):
        self._input_locked = data.get("locked", False)

    def _emit_cg(self, gesture_id: str):
        cooldown = self._thresholds.get("gesture_cooldown_ms", 600) / 1000
        self._cooldown_until = time.monotonic() + cooldown
        self.event_bus.emit("cg_detected", {"gesture_id": gesture_id})

    def _emit_oi(self, gesture_id: str):
        self.event_bus.emit("oi_detected", {"gesture_id": gesture_id})

    def _detect(self, interaction: dict, results) -> bool:
        detector = interaction.get("type")
        params = interaction.get("params", {})
        landmarks = results.multi_hand_landmarks

        if detector == "presence_bilateral":
            return self._detect_presence_bilateral(landmarks, params)
        elif detector == "presence_bilateral_still":
            return self._detect_presence_bilateral_still(landmarks, params)
        elif detector == "directional_point":
            return self._detect_directional_point(landmarks, params)
        elif detector == "mouth_proximity_tip":
            return self._detect_mouth_proximity_tip(landmarks, params)
        elif detector == "point_target_held":
            return self._detect_point_target_held(landmarks, params)
        # All other detector types are stubs
        return False

    def _detect_presence_bilateral(self, landmarks, params) -> bool:
        if not landmarks or len(landmarks) < 2:
            return False
        y_min = params.get("y_min_normalized", 0.0)
        for hand in landmarks:
            wrist_y = hand.landmark[0].y
            # y increases downward; y_min_normalized means hand must be above midpoint
            if wrist_y > (1.0 - y_min):
                return False
        return True

    def _detect_presence_bilateral_still(self, landmarks, params) -> bool:
        # Stub: requires velocity tracking across frames
        return False

    def _detect_directional_point(self, landmarks, params) -> bool:
        # Stub: requires index-finger direction classification
        return False

    def _detect_mouth_proximity_tip(self, landmarks, params) -> bool:
        # Stub: requires face landmark + hand tip proximity check
        return False

    def _detect_point_target_held(self, landmarks, params) -> bool:
        # Stub: requires screen-region mapping
        return False

    def close(self):
        self._hands.close()
