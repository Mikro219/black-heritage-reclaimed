"""
GestureEngine — MediaPipe Hands + GRLib gesture detection.

Consumes camera frames, runs hand landmark detection, and dispatches to
detector modules in engines/detectors/ via a registry. Each detector
receives (landmarks, params, context) where context is a persistent dict
scoped to the active interaction — used for cross-frame state like hold
timers, velocity history, and path recordings.
"""

import cv2
import mediapipe as mp
from typing import Optional
import time

from engines.detectors import REGISTRY


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
        self._active_cg_context: dict = {}
        self._active_oi: Optional[dict] = None
        self._active_oi_context: dict = {}
        self._oi_open_time: Optional[float] = None
        self._cooldown_until: float = 0.0
        self._last_landmarks = None
        self._last_handedness = None
        self._input_locked = False
        self._last_fired: Optional[str] = None
        self._last_fired_time: float = 0.0

        self.event_bus.subscribe("cg_window_open", self._on_cg_window_open)
        self.event_bus.subscribe("oi_window_open", self._on_oi_window_open)
        self.event_bus.subscribe("input_lock", self._on_input_lock)

    def process_frame(self, frame):
        if self._input_locked:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        self._last_landmarks = results.multi_hand_landmarks
        self._last_handedness = results.multi_handedness

        if not results.multi_hand_landmarks:
            return

        now = time.monotonic()
        if now < self._cooldown_until:
            return

        if self._active_cg:
            if self._dispatch(self._active_cg, results, self._active_cg_context):
                self._emit_cg(self._active_cg["id"])
                self._active_cg = None
                self._active_cg_context = {}

        oi_window_ms = self.config["timing_defaults"].get("oi_window_ms", 6000)
        if self._active_oi and self._oi_open_time:
            elapsed = (now - self._oi_open_time) * 1000
            if elapsed <= oi_window_ms:
                if self._dispatch(self._active_oi, results, self._active_oi_context):
                    self._emit_oi(self._active_oi["id"])
                    self._active_oi = None
                    self._active_oi_context = {}
                    self._oi_open_time = None
            else:
                self._active_oi = None
                self._active_oi_context = {}
                self._oi_open_time = None

    def hands_detected(self) -> bool:
        return bool(self._last_landmarks)

    def _on_cg_window_open(self, data: dict):
        self._active_cg = data.get("interaction")
        self._active_cg_context = {}

    def _on_oi_window_open(self, data: dict):
        self._active_oi = data.get("interaction")
        self._active_oi_context = {}
        self._oi_open_time = time.monotonic()

    def _on_input_lock(self, data: dict):
        self._input_locked = data.get("locked", False)

    def _emit_cg(self, gesture_id: str):
        cooldown = self._thresholds.get("gesture_cooldown_ms", 600) / 1000
        self._cooldown_until = time.monotonic() + cooldown
        # Directional detectors store their result in context["point_direction"]
        choice = self._active_cg_context.get("point_direction")
        label = f"CG:{gesture_id}" + (f"({choice})" if choice else "")
        self._last_fired = label
        self._last_fired_time = time.monotonic()
        event = {"gesture_id": gesture_id}
        if choice:
            event["choice"] = choice
        self.event_bus.emit("cg_detected", event)

    def _emit_oi(self, gesture_id: str):
        self._last_fired = f"OI:{gesture_id}"
        self._last_fired_time = time.monotonic()
        self.event_bus.emit("oi_detected", {"gesture_id": gesture_id})

    def debug_info(self) -> dict:
        now = time.monotonic()
        last = self._last_fired if (now - self._last_fired_time) < 2.0 else None
        cg = self._active_cg
        oi = self._active_oi
        recording = self._active_cg_context.get("shape_recording")
        return {
            "active_cg": f"{cg['id']} ({cg['type']})" if cg else None,
            "active_oi": f"{oi['id']} ({oi['type']})" if oi else None,
            "last_fired": last,
            "recording_pts": len(recording) if recording else 0,
        }

    def _dispatch(self, interaction: dict, results, context: dict) -> bool:
        detector_type = interaction.get("type")
        params = interaction.get("params", {})
        landmarks = results.multi_hand_landmarks
        detector_fn = REGISTRY.get(detector_type)
        if detector_fn is None:
            return False
        return detector_fn(landmarks, params, context)

    def close(self):
        self._hands.close()
