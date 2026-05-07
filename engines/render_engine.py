"""
RenderEngine — Pygame-based frame sequencer and display layer.

Loads frame sequences from the current scene's frames/ directory,
advances frames at the scene's fps, and responds to render_events
emitted by the NarrationEngine (e.g. quilt_illuminate, door_widens).

Debug overlay draws hand landmarks when config.debug_overlay is true.
"""

import os
import time
import pygame
from PIL import Image
from typing import Optional


class RenderEngine:
    def __init__(self, config: dict, event_bus: "EventBus"):
        self.config = config
        self.event_bus = event_bus
        self._screen: Optional[pygame.Surface] = None
        self._frames: list = []
        self._frame_index = 0
        self._fps = 24
        self._last_frame_time = 0.0
        self._debug = config.get("debug_overlay", False)
        self._pending_events: list = []
        self._landmark_data = None

        self.event_bus.subscribe("scene_load", self._on_scene_load)
        self.event_bus.subscribe("render_event", self._on_render_event)

    def init_display(self):
        pygame.display.init()
        w, h = self.config.get("resolution", [1920, 1080])
        self._screen = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
        pygame.display.set_caption("Black Heritage Reclaimed")

    def update(self, landmark_data=None):
        if not self._screen or not self._frames:
            return

        self._landmark_data = landmark_data
        now = time.monotonic()
        frame_duration = 1.0 / self._fps

        if now - self._last_frame_time >= frame_duration:
            self._frame_index = (self._frame_index + 1) % len(self._frames)
            self._last_frame_time = now

        self._screen.blit(self._frames[self._frame_index], (0, 0))

        if self._debug and landmark_data:
            self._draw_debug_overlay(landmark_data)

        pygame.display.flip()

    def _on_scene_load(self, data: dict):
        metadata = data.get("metadata", {})
        scene_id = data.get("scene_id", "")
        frames_dir = os.path.join(
            os.path.dirname(self._find_metadata_dir(scene_id)), "frames"
        )
        self._fps = metadata.get("fps", 24)
        self._frame_index = 0
        self._frames = self._load_frames(frames_dir)
        self._last_frame_time = time.monotonic()

    def _find_metadata_dir(self, scene_id: str) -> str:
        scenes_root = os.path.join(os.path.dirname(__file__), "..", "scenes")
        for artist_dir in os.listdir(scenes_root):
            candidate = os.path.join(scenes_root, artist_dir, scene_id)
            if os.path.isdir(candidate):
                return candidate
        return ""

    def _load_frames(self, frames_dir: str) -> list:
        if not os.path.isdir(frames_dir):
            return []
        w, h = self.config.get("resolution", [1920, 1080])
        frames = []
        for fname in sorted(os.listdir(frames_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(frames_dir, fname)
            img = Image.open(path).convert("RGB").resize((w, h))
            surface = pygame.image.fromstring(img.tobytes(), img.size, "RGB")
            frames.append(surface)
        return frames

    def _on_render_event(self, data: dict):
        self._pending_events.append(data.get("name"))

    def _draw_debug_overlay(self, landmark_data):
        if not landmark_data:
            return
        w, h = self.config.get("resolution", [1920, 1080])
        for hand_landmarks in landmark_data:
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                pygame.draw.circle(self._screen, (0, 255, 0), (x, y), 5)
