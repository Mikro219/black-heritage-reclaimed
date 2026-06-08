"""
LEGACY — dialogue-cue-driven runtime.
In the shot-driven runtime (ShotSequencePlayer), this engine is no longer the master
clock. Audio playback is handled by NarrationAdapter (shot_audio_lines events).
This file is kept for the SceneManager-based runtime path (invoked without --shot-player).

NarrationEngine — master clock for the dialogue-cue-driven runtime.

Owns the dialogue_sequence for the current scene. Plays each AL-XX-YYY
line on the narration audio channel, emits `dialogue_cue` events, opens
CG/OI/VI windows, holds at `wait_for_interaction` cues until the linked
CG completes, fires re-prompt lines on timeout, and fires reaction lines
when an OI or VI is detected.

Freeze-frame mechanic:
  When a cue with wait_for_interaction: true also has freeze_frame_page: N,
  the engine emits a `freeze_on_page` event and stores the pending interaction.
  The render engine lets frames play naturally until reaching that page, then
  freezes and emits `freeze_activated`. Only then does the narration engine
  emit `cg_window_open` to arm the detector. This ensures detection only
  starts once the target storyboard panel is on screen.

Stroke-chain mechanic (STATE_STROKE_CHAIN):
  When interaction.type == "directional_draw_chain", the engine enters
  STATE_STROKE_CHAIN and iterates through interaction.stroke_chain. Each
  element is {freeze_page, direction, next_page}. The engine arms one
  directional_draw detector per stroke, emits page_jump on each success,
  and advances to the next stroke. After the final stroke it completes
  normally and calls _advance().

Branch mechanic (AL-02-007 point_path):
  When on_cg_completed receives a choice, it looks up interaction.branches
  and emits a page_jump to the branch's next_page. The cue's converge_at
  value is not needed here because the next cue's own freeze_frame_page
  naturally resolves the panel after the branch plays.

OI-chain mechanic (AL-01-002 drink_gesture):
  When a cue has optional_interactions: [...] instead of a single interaction,
  the engine opens each OI window in sequence. The first OI fires a page_jump
  to next_page_on_success, then opens the second OI window, and so on.
  These do not block narration.

Does NOT control scene transitions; signals the SceneManager when the
sequence is exhausted.
"""

import time
import pygame
import os
from typing import Callable, Optional


_STROKE_ID = "__stroke__"   # synthetic gesture_id used for all stroke-chain steps


class NarrationEngine:
    STATE_IDLE       = "idle"
    STATE_PLAYING    = "playing"
    STATE_WAITING_CG = "waiting_cg"
    STATE_STROKE_CHAIN = "stroke_chain"
    STATE_FINISHED   = "finished"

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

        # Stroke-chain state
        self._stroke_chain: Optional[list] = None
        self._stroke_index: int = 0
        self._stroke_parent_id: Optional[str] = None

        # OI-chain state
        self._oi_chain: list = []
        self._oi_chain_index: int = 0

        # Stores last CG choice for downstream cues (e.g. path_left / path_right)
        self._scene_choice: Optional[str] = None

        # Interactions deferred until freeze_activated fires
        self._pending_cg_interaction: Optional[dict] = None
        self._pending_oi_interaction: Optional[dict] = None
        self._pending_oi_window_ms: int = 0

        self.event_bus.subscribe("freeze_activated", self._on_freeze_activated)

    def load_scene(self, metadata: dict, audio_dir: str, on_finished: Callable):
        self._sequence = metadata.get("dialogue_sequence", [])
        self._audio_dir = audio_dir
        self._index = 0
        self._state = self.STATE_IDLE
        self._on_sequence_finished = on_finished
        self._stroke_chain = None
        self._stroke_index = 0
        self._stroke_parent_id = None
        self._oi_chain = []
        self._oi_chain_index = 0
        self._scene_choice = None
        self._pending_cg_interaction = None
        self._pending_oi_interaction = None
        self._pending_oi_window_ms = 0

        profile_name = "profile_" + metadata.get("timing_profile", "standard")
        self._timing = self.config["timing_defaults"].get(
            profile_name, self.config["timing_defaults"]["profile_standard"]
        )

    def start(self):
        if self._sequence:
            self._state = self.STATE_PLAYING
            self._advance()

    def update(self):
        if self._state in (self.STATE_IDLE, self.STATE_FINISHED):
            return

        if self._state == self.STATE_PLAYING:
            if not self._channel.get_busy():
                self._advance()

        elif self._state in (self.STATE_WAITING_CG, self.STATE_STROKE_CHAIN):
            if self._cg_start_time is None:
                return
            elapsed = (time.monotonic() - self._cg_start_time) * 1000
            if elapsed >= self._timing["reprompt_second_ms"]:
                self._fire_reprompt(2)
            elif elapsed >= self._timing["reprompt_first_ms"]:
                self._fire_reprompt(1)
            # Auto-advance intentionally omitted until frame-queue stability is confirmed

    def on_cg_completed(self, gesture_id: str, choice: str = None):
        # Stroke-chain step
        if self._state == self.STATE_STROKE_CHAIN:
            if gesture_id == _STROKE_ID:
                self._on_stroke_completed()
            return

        if self._state != self.STATE_WAITING_CG:
            return

        cue = self._current_cue()
        if not cue:
            return
        interaction = cue.get("interaction", {})
        if interaction.get("id") != gesture_id:
            return

        # Store choice for any downstream cue that needs it
        if choice:
            self._scene_choice = choice
            branches = interaction.get("branches", {})
            branch = branches.get(choice, {})
            next_page = branch.get("next_page")
            if next_page is not None:
                self.event_bus.emit("page_jump", {"page": next_page})

        self.event_bus.emit("freeze_release", {})
        self._cg_start_time = None
        self._state = self.STATE_PLAYING
        self._advance()

    def on_oi_detected(self, gesture_id: str):
        # OI-chain step
        if self._oi_chain and self._oi_chain_index < len(self._oi_chain):
            current = self._oi_chain[self._oi_chain_index]
            if current.get("id") == gesture_id:
                next_page = current.get("next_page_on_success")
                if next_page is not None:
                    self.event_bus.emit("freeze_release", {})
                    self.event_bus.emit("page_jump", {"page": next_page})
                self._oi_chain_index += 1
                self._arm_oi_chain_step()
                return

        # Standard single OI reaction
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

    # ------------------------------------------------------------------
    # Internal: advance through the dialogue sequence
    # ------------------------------------------------------------------

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
        itype = interaction.get("type")

        # OI chain — multiple optional interactions sequenced on one cue
        optional_interactions = cue.get("optional_interactions")
        if optional_interactions:
            self._oi_chain = list(optional_interactions)
            self._oi_chain_index = 0
            self._arm_oi_chain_step()
            # OI chains don't block; fall through so narration continues

        if cue.get("wait_for_interaction") and tier == "cg":
            if itype == "directional_draw_chain":
                self._start_stroke_chain(cue)
            else:
                freeze_page = cue.get("freeze_frame_page")
                if freeze_page is not None:
                    # Let frames play through to the target page, then arm detector
                    self.event_bus.emit("freeze_on_page", {"page": freeze_page})
                    self._pending_cg_interaction = interaction
                else:
                    self.event_bus.emit("cg_window_open", {"interaction": interaction})
                self._state = self.STATE_WAITING_CG
                self._cg_start_time = time.monotonic()
        elif tier == "oi" and not optional_interactions:
            self.event_bus.emit("oi_window_open", {
                "interaction": interaction,
                "window_ms": self.config["timing_defaults"]["oi_window_ms"]
            })

    # ------------------------------------------------------------------
    # Stroke-chain helpers
    # ------------------------------------------------------------------

    def _start_stroke_chain(self, cue: dict):
        interaction = cue.get("interaction", {})
        strokes = interaction.get("stroke_chain", [])
        if not strokes:
            self._advance()
            return
        self._stroke_chain = strokes
        self._stroke_index = 0
        self._stroke_parent_id = interaction.get("id")
        self._state = self.STATE_STROKE_CHAIN
        self._cg_start_time = time.monotonic()
        self._arm_stroke(0)

    def _arm_stroke(self, idx: int):
        """Schedule play-through freeze and defer cg_window_open until freeze_activated."""
        stroke = self._stroke_chain[idx]
        synthetic = {
            "tier": "cg",
            "id": _STROKE_ID,
            "type": "directional_draw",
            "params": {
                "direction": stroke.get("direction", "right"),
                "window_ms": stroke.get("window_ms", 400),
            },
        }
        freeze_page = stroke.get("freeze_page")
        if freeze_page is not None:
            self.event_bus.emit("freeze_on_page", {"page": freeze_page})
            self._pending_cg_interaction = synthetic
        else:
            self.event_bus.emit("cg_window_open", {"interaction": synthetic})

    def _on_stroke_completed(self):
        stroke = self._stroke_chain[self._stroke_index]
        next_page = stroke.get("next_page")
        if next_page is not None:
            self.event_bus.emit("page_jump", {"page": next_page})

        self._stroke_index += 1

        if self._stroke_index >= len(self._stroke_chain):
            # Chain complete
            self.event_bus.emit("freeze_release", {})
            self._stroke_chain = None
            self._stroke_index = 0
            self._stroke_parent_id = None
            self._cg_start_time = None
            self._state = self.STATE_PLAYING
            self._advance()
        else:
            # Arm the next stroke (context is reset because gesture_engine creates
            # a new empty context dict each time cg_window_open fires)
            self._arm_stroke(self._stroke_index)

    # ------------------------------------------------------------------
    # OI-chain helpers
    # ------------------------------------------------------------------

    def _arm_oi_chain_step(self):
        if self._oi_chain_index >= len(self._oi_chain):
            self._oi_chain = []
            return
        oi = self._oi_chain[self._oi_chain_index]
        synthetic_interaction = {
            "tier": "oi",
            "id": oi["id"],
            "type": oi.get("type") or oi.get("detector"),
            "params": oi.get("params", {}),
        }
        window_ms = oi.get("window_ms", self.config["timing_defaults"]["oi_window_ms"])
        freeze_page = oi.get("freeze_frame_page")
        if freeze_page is not None:
            self.event_bus.emit("freeze_on_page", {"page": freeze_page})
            self._pending_oi_interaction = synthetic_interaction
            self._pending_oi_window_ms = window_ms
        else:
            self.event_bus.emit("oi_window_open", {
                "interaction": synthetic_interaction,
                "window_ms": window_ms
            })

    # ------------------------------------------------------------------
    # Freeze-activated callback — arms detector once frames reach target page
    # ------------------------------------------------------------------

    def _on_freeze_activated(self, data: dict):
        if self._pending_cg_interaction is not None:
            self.event_bus.emit("cg_window_open", {"interaction": self._pending_cg_interaction})
            self._pending_cg_interaction = None
        elif self._pending_oi_interaction is not None:
            self.event_bus.emit("oi_window_open", {
                "interaction": self._pending_oi_interaction,
                "window_ms": self._pending_oi_window_ms
            })
            self._pending_oi_interaction = None
            self._pending_oi_window_ms = 0

    # ------------------------------------------------------------------
    # Audio / utility
    # ------------------------------------------------------------------

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

    def debug_info(self) -> dict:
        cue = self._current_cue()
        waiting_id = None
        stroke_info = None

        if self._state == self.STATE_WAITING_CG and cue:
            waiting_id = cue.get("interaction", {}).get("id")
        elif self._state == self.STATE_STROKE_CHAIN:
            waiting_id = self._stroke_parent_id
            if self._stroke_chain and self._stroke_index < len(self._stroke_chain):
                s = self._stroke_chain[self._stroke_index]
                stroke_info = f"{self._stroke_index + 1}/{len(self._stroke_chain)} {s.get('direction', '?')}"

        return {
            "state": self._state,
            "cue_code": cue["cue"] if cue else None,
            "waiting_id": waiting_id,
            "stroke_info": stroke_info,
            "scene_choice": self._scene_choice,
        }

    def _fire_reprompt(self, level: int):
        pass

    def _on_cg_timeout(self):
        self.event_bus.emit("freeze_release", {})
        self._state = self.STATE_PLAYING
        self._cg_start_time = None
        self._advance()
