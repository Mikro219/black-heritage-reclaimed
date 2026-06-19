"""
narration_adapter — thin bridge between ShotSequencePlayer and the audio layer.

In the shot-driven runtime, narration is a *side-effect* of shot entry, not the
clock.  ShotSequencePlayer emits "shot_audio_lines" when entering PLAY/PLAY_INTRO;
this adapter catches that event and plays each AL-XX-YYY file sequentially on
pygame mixer channel 0.

It does NOT use NarrationEngine.load_scene() or NarrationEngine.start() — those
belong to the legacy dialogue-cue-driven model (scene_manager.py + narration_engine.py).

VoiceEngine compatibility: a "dialogue_cue" event is emitted for each line so that
VoiceEngine._on_dialogue_cue can open VI windows.  Once shot metadata has vi_config
attached to chain steps, VoiceEngine responds to "vi_chain_step" directly (Phase 3
VoiceEngine change) and the dialogue_cue shim can be removed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pygame

log = logging.getLogger(__name__)


class NarrationAdapter:
    """
    Subscribes to "shot_audio_lines" and plays lines sequentially on channel 0.

    Call update() once per main-loop frame.
    """

    def __init__(self, config: dict, event_bus, shots):
        self.config     = config
        self.event_bus  = event_bus
        self._channel   = pygame.mixer.Channel(0)

        # shot_id -> Path of audio/ dir  (None when not yet on disk)
        self._audio_dirs: dict[str, Optional[Path]] = {
            s.shot: s.audio_dir for s in shots
        }

        # Pending lines: list of (resolved_path_or_None, al_code)
        self._queue: list[tuple[Optional[Path], str]] = []

        # True only while THIS adapter is driving channel 0 (playing AL lines).
        # When a shot uses whole-file audio.mp3, RenderEngine owns channel 0 and the
        # adapter must not stop it — otherwise an input_lock (every FSM transition /
        # non-interactive segment) would kill the shot's audio.mp3 mid-playback.
        self._owns_channel = False

        event_bus.subscribe("shot_load",        self._on_shot_load)
        event_bus.subscribe("shot_audio_lines", self._on_shot_audio_lines)
        event_bus.subscribe("input_lock",       self._on_input_lock)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Advance the queue when the current line finishes playing."""
        if not self._queue:
            return
        if not self._channel.get_busy():
            self._play_next()

    def stop_current(self) -> None:
        """Stop our narration playback and clear the queue (e.g. on scene transition).

        Only touches channel 0 when the adapter actually owns it — never stops a
        shot's whole-file audio.mp3 (owned by RenderEngine).
        """
        self._queue.clear()
        if self._owns_channel:
            self._channel.stop()
            self._owns_channel = False

    def debug_info(self) -> dict:
        return {
            "queue_len": len(self._queue),
            "playing":   bool(self._channel.get_busy()),
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_shot_load(self, data: dict) -> None:
        """New shot: assume whole-file audio (RenderEngine owns channel 0) until a
        shot_audio_lines event proves this shot has per-line narration to play."""
        self._owns_channel = False

    def _on_shot_audio_lines(self, data: dict) -> None:
        """Enqueue all lines for a shot and start playback immediately.

        If none of the lines resolve to a per-line audio file, the shot uses the
        whole-file ``audio.mp3`` convention instead (played by RenderEngine on the
        same channel). In that case this adapter must NOT touch the channel — doing
        so would stop the shot's audio.mp3 and leave silence.
        """
        shot_id   = data.get("shot_id", "")
        audio_dir = self._audio_dirs.get(shot_id)

        pending: list[tuple[Optional[Path], str]] = []
        for al_code in data.get("lines", []):
            resolved = None
            if audio_dir is not None:
                for ext in (".wav", ".ogg"):
                    candidate = audio_dir / f"{al_code}{ext}"
                    if candidate.exists():
                        resolved = candidate
                        break
            pending.append((resolved, al_code))

        # Emit dialogue_cue for every line regardless (VoiceEngine VI windows depend
        # on these), but only seize the audio channel if we actually have files.
        if not any(path is not None for path, _ in pending):
            for _, al_code in pending:
                self.event_bus.emit("dialogue_cue", {
                    "code": al_code,
                    "cue":  {"cue": al_code},
                })
            log.debug("NarrationAdapter: shot %s uses whole-file audio; channel left alone",
                      shot_id)
            self._queue.clear()
            self._owns_channel = False
            return

        # We have per-line narration files — take ownership of channel 0 (this
        # replaces any whole-file audio.mp3 RenderEngine may have started).
        self._queue = pending
        self._owns_channel = True
        self._channel.stop()
        self._play_next()

    def _on_input_lock(self, data: dict) -> None:
        if data.get("locked"):
            self.stop_current()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _play_next(self) -> None:
        if not self._queue:
            return

        path, al_code = self._queue.pop(0)

        # Emit dialogue_cue so VoiceEngine._on_dialogue_cue can open VI windows.
        # COUPLING NOTE: this shim is only needed while shot metadata lacks vi_config
        # on chain steps.  Once VoiceEngine responds directly to "vi_chain_step"
        # (see voice_engine.py _on_vi_chain_step), this emit can be removed.
        self.event_bus.emit("dialogue_cue", {
            "code": al_code,
            "cue":  {"cue": al_code},   # minimal cue dict; no interaction field yet
        })

        if path is None:
            log.debug("NarrationAdapter: skipping playback for %s (no file)", al_code)
            return

        try:
            sound = pygame.mixer.Sound(str(path))
            self._channel.play(sound)
            log.debug("NarrationAdapter: playing %s", path.name)
        except Exception as exc:
            log.warning("NarrationAdapter: could not play %s: %s", path, exc)
