import json
import os
from enum import Enum, auto

import pygame
from PIL import Image


class State(Enum):
    IDLE = auto()
    TRANSITIONING = auto()
    PLAYING = auto()


class SceneManager:
    def __init__(self, config, screen, audio):
        self._config = config
        self._screen = screen
        self._audio = audio

        self._state = State.IDLE
        self._scene_id = None
        self._metadata = None
        self._frames = []
        self._frame_index = 0
        self._accepting_input = False
        self._last_frame_tick = 0
        self._ms_per_frame = 0

    def load(self, scene_id):
        self._state = State.TRANSITIONING
        self._accepting_input = False
        self._scene_id = scene_id

        path = os.path.join("scenes", scene_id)
        with open(os.path.join(path, "metadata.json")) as f:
            self._metadata = json.load(f)

        self._ms_per_frame = 1000 / self._metadata["fps"]
        self._frames = self._load_frames(os.path.join(path, "frames"))
        self._frame_index = 0
        self._last_frame_tick = pygame.time.get_ticks()

        self._audio.play_scene(self._metadata)
        self._state = State.PLAYING

    def _load_frames(self, frames_dir):
        paths = sorted(
            f for f in os.listdir(frames_dir) if f.endswith(".png")
        )
        w, h = self._config["resolution"]
        frames = []
        for p in paths:
            img = Image.open(os.path.join(frames_dir, p)).convert("RGBA")
            img = img.resize((w, h), Image.LANCZOS)
            frames.append(pygame.image.fromstring(img.tobytes(), img.size, "RGBA"))
        return frames

    def handle_gesture(self, gesture):
        if not self._accepting_input:
            return
        if gesture == "swipe_right":
            next_scene = self._metadata["transitions"].get("next_scene")
            if next_scene:
                self.load(next_scene)
        elif gesture == "swipe_left":
            pass  # TODO: back-navigation once scene order is finalised
        elif gesture == "open_palm":
            self._audio.toggle_pause()

    def update(self):
        if self._state != State.PLAYING or not self._frames:
            return

        now = pygame.time.get_ticks()
        if now - self._last_frame_tick >= self._ms_per_frame:
            self._frame_index += 1
            self._last_frame_tick = now

            threshold = self._metadata["transitions"].get("accepts_input_after_frame", 0)
            if self._frame_index >= threshold:
                self._accepting_input = True

            if self._frame_index >= len(self._frames):
                self._frame_index = 0  # loop

    def draw(self):
        if not self._frames:
            self._screen.fill((0, 0, 0))
            return
        self._screen.blit(self._frames[self._frame_index % len(self._frames)], (0, 0))
