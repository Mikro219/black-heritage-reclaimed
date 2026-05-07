"""
main.py — Black Heritage Reclaimed entry point.

Boot sequence:
  1. Load config.json
  2. Init Pygame + display
  3. Wire up EventBus and all engines
  4. Run calibration (Scene 1 hand-tracking liveness check)
  5. Enter the main loop: process camera frame → update engines → render
"""

import json
import os
import sys

import cv2
import pygame

from engines.event_bus import EventBus
from engines.narration_engine import NarrationEngine
from engines.gesture_engine import GestureEngine
from engines.voice_engine import VoiceEngine
from engines.scene_manager import SceneManager
from engines.render_engine import RenderEngine


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
SCENES_ROOT = os.path.join(os.path.dirname(__file__), "scenes")


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def main():
    config = load_config()

    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

    bus = EventBus()

    render = RenderEngine(config, bus)
    render.init_display()

    narration = NarrationEngine(config, bus)
    gesture = GestureEngine(config, bus)
    voice = VoiceEngine(config, bus)
    scene_mgr = SceneManager(config, bus, SCENES_ROOT)

    # Wire narration engine callbacks from bus events
    bus.subscribe("scene_load", lambda d: narration.load_scene(
        d["metadata"],
        d["audio_dir"],
        on_finished=lambda: bus.emit("sequence_finished", {})
    ))
    bus.subscribe("narration_cg_completed", lambda d: narration.on_cg_completed(d.get("gesture_id", "")))
    bus.subscribe("narration_oi_detected", lambda d: narration.on_oi_detected(d.get("gesture_id", "")))
    bus.subscribe("narration_vi_reaction", lambda d: narration.on_vi_detected(d.get("voice_id", "")))

    # Narration starts when a scene is loaded and calibration passes
    bus.subscribe("scene_load", lambda _: narration.start())

    voice.start()
    scene_mgr.start()

    cap = cv2.VideoCapture(config.get("camera_index", 0))
    clock = pygame.time.Clock()

    calibration_passed = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _shutdown(cap, gesture, voice)
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                _shutdown(cap, gesture, voice)
                return

        ret, frame = cap.read()
        if not ret:
            continue

        gesture.process_frame(frame)

        if not calibration_passed:
            if gesture.hands_detected():
                calibration_passed = True
                scene_mgr.on_calibration_passed()

        narration.update()
        render.update(landmark_data=gesture._last_landmarks)

        clock.tick(config.get("detection_thresholds", {}).get("fps_cap", 60))


def _shutdown(cap, gesture, voice):
    voice.stop()
    gesture.close()
    cap.release()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
