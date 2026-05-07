"""
SceneManager — owns scene transitions and the top-level state machine.

States:
  BOOT → CALIBRATION → SCENE_PLAYBACK → FINAL_ADDRESS → ACKNOWLEDGMENTS → ATTRACT_LOOP

Does not control narration directly; receives `sequence_finished` from
NarrationEngine and decides when to transition.
"""

import json
import os
from typing import Optional


SCENES = [
    "scene_01", "scene_02", "scene_03", "scene_04", "scene_05", "scene_06",
    "scene_07", "scene_08", "scene_09", "scene_10", "scene_11",
]

STATE_BOOT = "BOOT"
STATE_CALIBRATION = "CALIBRATION"
STATE_SCENE_PLAYBACK = "SCENE_PLAYBACK"
STATE_FINAL_ADDRESS = "FINAL_ADDRESS"
STATE_ACKNOWLEDGMENTS = "ACKNOWLEDGMENTS"
STATE_ATTRACT_LOOP = "ATTRACT_LOOP"


class SceneManager:
    def __init__(self, config: dict, event_bus: "EventBus", scenes_root: str):
        self.config = config
        self.event_bus = event_bus
        self.scenes_root = scenes_root
        self._state = STATE_BOOT
        self._current_scene_id: Optional[str] = None
        self._current_metadata: Optional[dict] = None

        self.event_bus.subscribe("sequence_finished", self._on_sequence_finished)
        self.event_bus.subscribe("cg_detected", self._on_cg_detected)
        self.event_bus.subscribe("oi_detected", self._on_oi_detected)
        self.event_bus.subscribe("vi_detected", self._on_vi_detected)

    def start(self):
        self._state = STATE_CALIBRATION
        self.event_bus.emit("state_change", {"state": STATE_CALIBRATION})

    def on_calibration_passed(self):
        self._load_scene(self.config.get("start_scene", "scene_01"))

    def _load_scene(self, scene_id: str):
        metadata_path = self._find_metadata(scene_id)
        if not metadata_path:
            raise FileNotFoundError(f"No metadata.json found for {scene_id}")

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        self._current_scene_id = scene_id
        self._current_metadata = metadata
        self._state = STATE_SCENE_PLAYBACK

        self.event_bus.emit("input_lock", {"locked": False})
        self.event_bus.emit("scene_load", {
            "scene_id": scene_id,
            "metadata": metadata,
            "audio_dir": os.path.join(os.path.dirname(metadata_path), "audio"),
        })

    def _find_metadata(self, scene_id: str) -> Optional[str]:
        for artist_dir in os.listdir(self.scenes_root):
            candidate = os.path.join(self.scenes_root, artist_dir, scene_id, "metadata.json")
            if os.path.exists(candidate):
                return candidate
        return None

    def _on_sequence_finished(self, data: dict):
        if self._state == STATE_SCENE_PLAYBACK:
            next_scene = (self._current_metadata or {}).get("transitions", {}).get("next_scene")
            if next_scene and next_scene != self._current_scene_id:
                self._transition_to(next_scene)
            elif self._current_scene_id == "scene_11":
                self._enter_final_address()
        elif self._state == STATE_FINAL_ADDRESS:
            self._state = STATE_ACKNOWLEDGMENTS
            self.event_bus.emit("state_change", {"state": STATE_ACKNOWLEDGMENTS})

    def _transition_to(self, scene_id: str):
        self.event_bus.emit("input_lock", {"locked": True})
        self.event_bus.emit("scene_transition_start", {"from": self._current_scene_id, "to": scene_id})
        self._load_scene(scene_id)

    def _enter_final_address(self):
        self._state = STATE_FINAL_ADDRESS
        self.event_bus.emit("input_lock", {"locked": True})
        self.event_bus.emit("state_change", {"state": STATE_FINAL_ADDRESS})

    def _on_cg_detected(self, data: dict):
        self.event_bus.emit("narration_cg_completed", data)

    def _on_oi_detected(self, data: dict):
        self.event_bus.emit("narration_oi_detected", data)

    def _on_vi_detected(self, data: dict):
        tier = data.get("tier")
        if tier in ("cg_alternative", "cg_required"):
            self.event_bus.emit("narration_cg_completed", {"gesture_id": data.get("voice_id")})
        else:
            self.event_bus.emit("narration_vi_reaction", data)

    @property
    def state(self) -> str:
        return self._state
