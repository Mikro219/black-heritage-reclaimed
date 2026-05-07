"""
NarrationEngine — master clock for the dialogue-cue-driven runtime.

Owns the dialogue_sequence for the current scene. Plays each AL-XX-YYY
line on the narration audio channel, emits `dialogue_cue` events, opens
CG/OI/VI windows, holds at `wait_for_interaction` cues until the linked
CG completes, fires re-prompt lines on timeout, and fires reaction lines
when an OI or VI is detected.

Does NOT control scene transitions; signals the SceneManager when the
sequence is exhausted.
"""

import time
import pygame
import os
from typing import Callable, Optional


class NarrationEngine:
    STATE_IDLE = "idle"
    STATE_PLAYING = "playing"
    STATE_WAITING_CG = "waiting_cg"
    STATE_FINISHED = "finished"

    def __init__(self, config: dict, event_bus: "EventBus"):
        self.config = config
        self.event_bus = event_bus
        self._channel = pygame.mixer.Channel(0)
        self._sequence: list = []
        self._index = 0
        self._state = self.STATE_IDLE
        self._cg_start_time: Optional[float] = None
        self._audio_dir: Optional[str] = None
        self._timing: dict = {}
        self._on_sequence_finished: Optional[Callable] = None

    def load_scene(self, metadata: dict, audio_dir: str, on_finished: Callable):
        self._sequence = metadata.get("dialogue_sequence", [])
        self._audio_dir = audio_dir
        self._index = 0
        self._state = self.STATE_IDLE
        self._on_sequence_finished = on_finished

        profile_name = "profile_" + metadata.get("timing_profile", "standard")
        self._timing = self.config["timing_defaults"].get(
            profile_name, self.config["timing_defaults"]["profile_standard"]
        )

    def start(self):
        if self._sequence:
            self._state = self.STATE_PLAYING
            self._advance()

    def update(self):
        if self._state == self.STATE_IDLE or self._state == self.STATE_FINISHED:
            return

        if self._state == self.STATE_PLAYING:
            if not self._channel.get_busy():
                self._advance()

        elif self._state == self.STATE_WAITING_CG:
            elapsed = (time.monotonic() - self._cg_start_time) * 1000
            cue = self._current_cue()
            if elapsed >= self._timing["reprompt_second_ms"]:
                self._fire_reprompt(2)
            elif elapsed >= self._timing["reprompt_first_ms"]:
                self._fire_reprompt(1)
            if elapsed >= self._timing["auto_advance_ms"]:
                self._on_cg_timeout()

    def on_cg_completed(self, gesture_id: str):
        if self._state != self.STATE_WAITING_CG:
            return
        cue = self._current_cue()
        interaction = cue.get("interaction", {})
        if interaction.get("id") == gesture_id:
            self._cg_start_time = None
            self._state = self.STATE_PLAYING
            self._advance()

    def on_oi_detected(self, gesture_id: str):
        cue = self._current_cue()
        if not cue:
            return
        interaction = cue.get("interaction", {})
        if interaction.get("tier") == "oi" and interaction.get("id") == gesture_id:
            reaction = interaction.get("reaction_audio")
            if reaction:
                self._play_audio(reaction, channel=2)

    def on_vi_detected(self, voice_id: str):
        cue = self._current_cue()
        if not cue:
            return
        interaction = cue.get("interaction", {})
        voice_alt = interaction.get("voice_alternative", {})
        if voice_alt.get("id") == voice_id and voice_alt.get("tier") == "cg_alternative":
            self.on_cg_completed(interaction.get("id"))
        voice_req = interaction.get("voice_required", {})
        if voice_req.get("id") == voice_id:
            self.on_cg_completed(interaction.get("id"))

    def _advance(self):
        if self._index >= len(self._sequence):
            self._state = self.STATE_FINISHED
            if self._on_sequence_finished:
                self._on_sequence_finished()
            return

        cue = self._sequence[self._index]
        self._index += 1

        self.event_bus.emit("dialogue_cue", {"code": cue["cue"], "cue": cue})

        audio = cue.get("audio")
        if audio:
            self._play_audio(audio, channel=0)

        render_event = cue.get("render_event")
        if render_event:
            self.event_bus.emit("render_event", {"name": render_event})

        sfx = cue.get("sfx")
        if sfx:
            self._play_audio(sfx, channel=1)

        interaction = cue.get("interaction", {})
        tier = interaction.get("tier")

        if cue.get("wait_for_interaction") and tier == "cg":
            self.event_bus.emit("cg_window_open", {"interaction": interaction})
            self._state = self.STATE_WAITING_CG
            self._cg_start_time = time.monotonic()
        elif tier == "oi":
            self.event_bus.emit("oi_window_open", {
                "interaction": interaction,
                "window_ms": self.config["timing_defaults"]["oi_window_ms"]
            })

    def _current_cue(self) -> Optional[dict]:
        idx = self._index - 1
        if 0 <= idx < len(self._sequence):
            return self._sequence[idx]
        return None

    def _play_audio(self, filename: str, channel: int = 0):
        if not self._audio_dir:
            return
        path = os.path.join(self._audio_dir, filename)
        if not os.path.exists(path):
            return
        sound = pygame.mixer.Sound(path)
        pygame.mixer.Channel(channel).play(sound)

    def _fire_reprompt(self, level: int):
        # Placeholder: play a reprompt line if asset exists
        pass

    def _on_cg_timeout(self):
        self._state = self.STATE_PLAYING
        self._cg_start_time = None
        self._advance()
