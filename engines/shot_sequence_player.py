"""
shot_sequence_player — shot-driven runtime player (Phase 2).

Implements the per-shot state machine from CLAUDE.md:

    playback:     PLAY  ──────────────────────────────────────► ADVANCE

    interactive:  PLAY_INTRO ─► HOLD(idle_loop, detectors armed) ─► PLAY_RES ─► ADVANCE
                                        │
                                        └─ timeout ─► FALLBACK (auto_advance | auto_complete)

The master clock is the shot sequence, not the narration engine.  Narration
(AL-XX-YYY audio) is a side-effect of shot entry, triggered via the
"shot_audio_lines" event; the actual audio plumbing is wired in Phase 3.

Detectors are armed on entering HOLD (via "cg_window_open" / "oi_window_open"
bus events) and are effectively disarmed on exiting HOLD because the player
stops responding to "cg_detected" events outside the HOLD state.

COUPLING FLAGS (to be resolved in Phase 3):
  1. VI windows: VoiceEngine subscribes to "dialogue_cue" to open VI windows.
     That event no longer exists in the shot-driven model. The player emits
     "vi_chain_step" when a voice chain step is armed; Phase 3 will adapt
     VoiceEngine to subscribe to this event and open a window.
  2. Narration audio: NarrationEngine.load_scene() expects a dialogue_sequence
     dict.  Phase 3 will add a thin NarrationAdapter that translates a shot's
     audio_lines list into individual play calls without using the old sequence.
  3. Render engine: the player emits "shot_load" and "shot_state_change".
     RenderEngine currently only responds to "scene_load"/"dev_frames_load".
     Phase 3 will add a "segment_playback_done" callback from the render engine
     so the player knows when non-looping segments finish.  Until then, the
     player advances play segments immediately when assets_pending=True, or
     after an estimated duration when frames are present.
"""

from __future__ import annotations

import time
from typing import Optional

from .sequence_loader import Shot


# ---------------------------------------------------------------------------
# Per-shot states
# ---------------------------------------------------------------------------

STATE_PRELOADING = "PRELOADING"   # waiting for frame preloader before starting shot
STATE_PLAY       = "PLAY"         # playback shot: playing frames start-to-end
STATE_PLAY_INTRO = "PLAY_INTRO"   # interactive: playing intro segment
STATE_HOLD       = "HOLD"         # interactive: idle_loop; detectors armed
STATE_PLAY_RES   = "PLAY_RES"     # interactive: playing resolution segment post-success
STATE_TRANSITION = "TRANSITION"   # brief input-locked gap between shots (unused, Phase 3)

# Top-level player states
PLAYER_IDLE     = "IDLE"
PLAYER_RUNNING  = "RUNNING"
PLAYER_FINAL    = "FINAL_ADDRESS"
PLAYER_FINISHED = "FINISHED"


# ---------------------------------------------------------------------------
# ShotSequencePlayer
# ---------------------------------------------------------------------------

class ShotSequencePlayer:
    """
    Shot-sequence-driven runtime player.

    Call update() every frame from the main loop.

    Events emitted:
        shot_load           {"shot": Shot, "index": int}
        shot_state_change   {"shot_id": str, "state": str, "segment": str}
        shot_audio_lines    {"shot_id": str, "lines": list, "trigger": str}
        shot_choice         {"shot_id": str, "choice": str}
        cg_reprompt         {"shot_id": str, "level": int, "interaction_id": str}
        vi_chain_step       {"shot_id": str, "step_index": int, "step": dict}
        cg_window_open      {"interaction": dict}   -> GestureEngine
        oi_window_open      {"interaction": dict, "window_ms": int}  -> GestureEngine
        input_lock          {"locked": bool}
        state_change        {"state": str}   (for FINAL_ADDRESS)

    Events consumed:
        cg_detected         {"gesture_id": str, "choice": str|None}
        oi_detected         {"gesture_id": str}
        vi_detected         {"voice_id": str}   [checked but not acted on until Phase 3]
    """

    def __init__(self, shots: list[Shot], config: dict, event_bus):
        self.shots     = shots
        self.config    = config
        self.event_bus = event_bus

        self._index        = 0
        self._player_state = PLAYER_IDLE
        self._shot_state: Optional[str] = None

        # HOLD tracking
        self._hold_start:    float      = 0.0
        self._chain:         list[dict] = []     # effective CG chain for current HOLD
        self._chain_index:   int        = 0
        self._reprompt_fired: set       = set()
        self._oi_expiry:     Optional[float] = None  # for OI-only HOLDs

        # PLAY / PLAY_INTRO / PLAY_RES tracking
        self._segment_start:    float = 0.0
        self._segment_duration: float = 0.0   # estimated when entering segment
        self._segment_done:     bool  = False  # set True by Phase 3 segment_playback_done

        self._fsm: Optional["ShotFSM"] = None

        event_bus.subscribe("cg_detected",          self._on_cg_detected)
        event_bus.subscribe("oi_detected",           self._on_oi_detected)
        event_bus.subscribe("vi_detected",           self._on_vi_detected)
        event_bus.subscribe("shot_frames_ready",     self._on_shot_frames_ready)
        event_bus.subscribe("frame_window_enter",    self._on_frame_window_enter)
        event_bus.subscribe("segment_playback_done", self._on_segment_playback_done)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def current_shot(self) -> Optional[Shot]:
        if 0 <= self._index < len(self.shots):
            return self.shots[self._index]
        return None

    def start(self, start_index: int = 0) -> None:
        if not self.shots:
            self._player_state = PLAYER_FINISHED
            self.event_bus.emit("sequence_finished", {})
            return
        self._player_state = PLAYER_RUNNING
        self._enter_shot(max(0, min(start_index, len(self.shots) - 1)))

    def update(self) -> None:
        """Drive the state machine. Call once per main-loop frame."""
        if self._player_state != PLAYER_RUNNING:
            return
        shot = self.current_shot
        if shot is None:
            return

        if self._shot_state == STATE_PRELOADING:
            pass  # waiting for shot_frames_ready from render engine
        elif self._shot_state in (STATE_PLAY, STATE_PLAY_INTRO, STATE_PLAY_RES):
            self._update_play(shot)
        elif self._shot_state == STATE_HOLD:
            self._update_hold(shot)

    def debug_info(self) -> dict:
        """
        Return a dict compatible with RenderEngine's narration_debug parameter.

        Existing overlay keys (WAIT:, CUE:, REC:) are preserved so the render
        engine works without modification.  The new SHOT: line is driven by
        the shot_id / shot_state / shot_elapsed_s keys added here.
        """
        shot = self.current_shot

        waiting_id  = None
        stroke_info = None
        if self._shot_state == STATE_HOLD and self._chain:
            if self._chain_index < len(self._chain):
                step = self._chain[self._chain_index]
                waiting_id = step.get("id")
                # For directional_draw_chain steps, expose stroke progress (Phase 3)
                if step.get("type") == "directional_draw_chain":
                    strokes = step.get("params", {}).get("strokes", [])
                    if strokes:
                        stroke_info = f"{self._chain_index + 1}/{len(strokes)}"

        elapsed = None
        if self._shot_state == STATE_HOLD:
            elapsed = round(time.monotonic() - self._hold_start, 1)

        return {
            # Keys consumed by the existing RenderEngine overlay
            "state":        self._shot_state or self._player_state,
            "cue_code":     shot.audio_lines[0] if (shot and shot.audio_lines) else None,
            "waiting_id":   waiting_id,
            "stroke_info":  stroke_info,
            "scene_choice": None,
            # New keys for SHOT: overlay (render engine reads these if present)
            "shot_id":       shot.shot if shot else None,
            "shot_state":    self._shot_state,
            "shot_act":      shot.act  if shot else None,
            "shot_elapsed_s": elapsed,
        }

    # ------------------------------------------------------------------
    # Shot entry
    # ------------------------------------------------------------------

    def _enter_shot(self, index: int) -> None:
        self._index = index
        shot = self.shots[index]

        # Set PRELOADING before emitting shot_load so _on_shot_frames_ready transitions
        # correctly when the prefetch was already done (synchronous event chain).
        if not shot.assets_pending:
            self._shot_state = STATE_PRELOADING

        self.event_bus.emit("shot_load", {"shot": shot, "index": index})
        print(f"[ShotPlayer] >> shot {shot.shot}  kind={shot.kind}  act={shot.act}  "
              f"{'PENDING' if shot.assets_pending else 'frames_ready'}")

        if shot.assets_pending:
            # No frames on disk yet — skip straight to playback (placeholder/TODO shot)
            if shot.kind == "playback":
                self._enter_segment(STATE_PLAY, shot)
            else:
                self._enter_segment(STATE_PLAY_INTRO, shot)
        else:
            # STATE_PRELOADING already set above; emit change event and prefetch next
            self.event_bus.emit("shot_state_change", {
                "shot_id": shot.shot,
                "state":   STATE_PRELOADING,
                "segment": "preloading",
            })
            self._prefetch_next(index)

    def _enter_segment(self, state: str, shot: Shot) -> None:
        self._shot_state      = state
        self._segment_start   = time.monotonic()
        self._segment_done    = False
        self._segment_duration = _estimate_duration(shot, state)

        label_map = {
            STATE_PLAY:       "playback",
            STATE_PLAY_INTRO: "intro",
            STATE_PLAY_RES:   "resolution",
        }
        self.event_bus.emit("shot_state_change", {
            "shot_id": shot.shot,
            "state":   state,
            "segment": label_map.get(state, state),
        })

        # Arm OI window for playback shots that declare an OI interaction
        if state == STATE_PLAY and shot.interaction:
            tier = shot.interaction.get("tier", "").upper()
            if tier == "OI":
                oi_frame_window = shot.interaction.get("oi_frame_window")
                if oi_frame_window and len(oi_frame_window) == 2:
                    # Frame-gated: tell the render engine which frame range to watch;
                    # it emits frame_window_enter when playback reaches that frame.
                    self.event_bus.emit("set_frame_window", {
                        "start": oi_frame_window[0],
                        "end":   oi_frame_window[1],
                    })
                else:
                    # No frame gate — arm immediately for the full segment duration
                    window_ms = max(int(self._segment_duration * 1000), 5000)
                    oi = _single_as_oi(shot.shot, shot.interaction)
                    self.event_bus.emit("oi_window_open", {
                        "interaction": oi,
                        "window_ms":   window_ms,
                    })

        # Announce audio lines so Phase 3 can trigger narration
        if state in (STATE_PLAY, STATE_PLAY_INTRO) and shot.audio_lines:
            self.event_bus.emit("shot_audio_lines", {
                "shot_id": shot.shot,
                "lines":   shot.audio_lines,
                "trigger": "intro",
            })

    def _enter_hold(self, shot: Shot) -> None:
        print(f"[ShotPlayer][DBG] _enter_hold shot {shot.shot} "
              f"(prev fsm={'set' if self._fsm else 'none'})")
        self._shot_state      = STATE_HOLD
        self._hold_start      = time.monotonic()
        self._chain_index     = 0
        self._reprompt_fired  = set()
        self._segment_done    = False
        self._segment_duration = 0.0
        self._fsm             = None

        self.event_bus.emit("shot_state_change", {
            "shot_id": shot.shot,
            "state":   STATE_HOLD,
            "segment": "idle_loop",
        })

        # FSM path: interaction_fsm key inside the interaction dict
        fsm_spec = (shot.interaction or {}).get("interaction_fsm")
        if fsm_spec:
            self._fsm = ShotFSM(fsm_spec, shot.segments)
            print(f"[ShotPlayer] shot {shot.shot} FSM init: "
                  f"states={list(self._fsm.states)} initial={self._fsm.current}")
            self._fsm_enter_state()
            return

        self._chain = _build_chain(shot.shot, shot.interaction)

        if not self._chain:
            tier = (shot.interaction or {}).get("tier", "").upper()
            if tier == "OI" and shot.interaction:
                # OI-only shot — arm window, advance when it expires
                oi_ms = self.config["timing_defaults"].get("oi_window_ms", 6000) / 1000.0
                self._oi_expiry = time.monotonic() + oi_ms
                self._arm_oi_window(shot)
            else:
                # interaction is None (TODO) or unrecognised — auto-advance immediately.
                # Normal path while shots are being wired; prevents a silent 6s stall
                # per unwired interactive shot in the dry-run.
                print(f"[ShotPlayer] shot {shot.shot} HOLD: no chain (interaction TODO) -> advance")
                self._advance()
            return

        self._oi_expiry = None
        self._arm_chain_step(shot)

    def _enter_resolution(self, shot: Shot) -> None:
        self._shot_state    = STATE_PLAY_RES
        self._segment_start = time.monotonic()
        self._segment_done  = False

        self.event_bus.emit("shot_state_change", {
            "shot_id": shot.shot,
            "state":   STATE_PLAY_RES,
            "segment": "resolution",
        })

    def _advance(self) -> None:
        shot = self.current_shot
        if shot:
            print(f"[ShotPlayer] ADVANCE  shot {shot.shot}  was {self._shot_state}")

        self.event_bus.emit("input_lock", {"locked": True})
        next_index = self._index + 1

        if next_index >= len(self.shots):
            self._player_state = PLAYER_FINAL
            self.event_bus.emit("state_change", {"state": PLAYER_FINAL})
            self.event_bus.emit("input_lock", {"locked": False})
            print(f"[ShotPlayer] sequence complete -> FINAL_ADDRESS")
            return

        self.event_bus.emit("input_lock", {"locked": False})
        self._prefetch_next(next_index)
        self._enter_shot(next_index)

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def _update_play(self, shot: Shot) -> None:
        """Advance PLAY / PLAY_INTRO / PLAY_RES when the segment is done."""
        if self._segment_done:
            # Phase 3: segment_playback_done event set this
            self._on_segment_done(shot)
            return

        if shot.assets_pending:
            # No real frames — skip segment immediately
            self._on_segment_done(shot)
            return

        if time.monotonic() - self._segment_start >= self._segment_duration:
            self._on_segment_done(shot)

    def _on_segment_done(self, shot: Shot) -> None:
        if self._shot_state == STATE_PLAY:
            self._advance()
        elif self._shot_state == STATE_PLAY_INTRO:
            self._enter_hold(shot)
        elif self._shot_state == STATE_PLAY_RES:
            self._advance()

    def _update_hold(self, shot: Shot) -> None:
        """Drive HOLD reprompts, OI expiry, and final timeout."""
        elapsed = time.monotonic() - self._hold_start
        timing  = _effective_timing(shot, self.config)

        # OI-only HOLD: advance when the OI window expires
        if self._oi_expiry is not None:
            if time.monotonic() >= self._oi_expiry:
                print(f"[ShotPlayer] shot {shot.shot} OI window expired -> advance")
                self._advance()
            return

        # CG HOLD: reprompts then timeout
        reprompt_s = timing["reprompt_s"]
        timeout_s  = timing["timeout_s"]

        for level, threshold in enumerate(reprompt_s, start=1):
            if elapsed >= threshold and level not in self._reprompt_fired:
                self._reprompt_fired.add(level)
                self._fire_reprompt(shot, level)

        if elapsed >= timeout_s:
            on_timeout = timing["on_timeout"]
            print(f"[ShotPlayer] shot {shot.shot} HOLD timeout ({timeout_s}s) -> {on_timeout}")
            if on_timeout == "auto_complete":
                self._enter_resolution(shot)
            else:
                self._advance()

    def _fire_reprompt(self, shot: Shot, level: int) -> None:
        interaction_id = "--"
        if self._chain and self._chain_index < len(self._chain):
            interaction_id = self._chain[self._chain_index].get("id", "--")
        print(f"[ShotPlayer] shot {shot.shot} reprompt {level}  waiting: {interaction_id}")
        self.event_bus.emit("cg_reprompt", {
            "shot_id":        shot.shot,
            "level":          level,
            "interaction_id": interaction_id,
        })

    # ------------------------------------------------------------------
    # Chain / detector arming
    # ------------------------------------------------------------------

    def _arm_chain_step(self, shot: Shot) -> None:
        if self._chain_index >= len(self._chain):
            # All CG steps done
            if _has_resolution(shot):
                self._enter_resolution(shot)
            else:
                self._advance()
            return

        step      = self._chain[self._chain_index]
        step_type = step.get("type", "")

        if step_type == "voice":
            # --- COUPLING FLAG (Phase 3) ---
            # VI chain steps need a VoiceEngine window.  VoiceEngine currently
            # subscribes to "dialogue_cue" (old model); that event no longer
            # exists.  Phase 3 will adapt VoiceEngine to listen to "vi_chain_step"
            # and open a window.  Until then, emit the event and let HOLD timeout
            # handle the auto-advance path.
            print(f"[ShotPlayer] shot {shot.shot} step {self._chain_index}: "
                  f"VI '{step.get('keyword', '?')}' -- VI chain not wired until Phase 3")
            self.event_bus.emit("vi_chain_step", {
                "shot_id":    shot.shot,
                "step_index": self._chain_index,
                "step":       step,
            })
            return   # wait for vi_detected or timeout

        # Build the interaction dict GestureEngine expects
        tier = (shot.interaction or {}).get("tier", "CG")
        interaction = _step_to_interaction(shot.shot, self._chain_index, step, tier)

        print(f"[ShotPlayer] shot {shot.shot} arming CG step {self._chain_index}: "
              f"type={interaction['type']!r}  id={interaction['id']!r}")
        self.event_bus.emit("cg_window_open", {"interaction": interaction})

    def _arm_oi_window(self, shot: Shot) -> None:
        if not shot.interaction:
            return
        oi_ms       = self.config["timing_defaults"].get("oi_window_ms", 6000)
        interaction = _single_as_oi(shot.shot, shot.interaction)
        if interaction.get("type"):
            print(f"[ShotPlayer] shot {shot.shot} arming OI: {interaction['id']}")
            self.event_bus.emit("oi_window_open", {
                "interaction": interaction,
                "window_ms":   oi_ms,
            })

    # ------------------------------------------------------------------
    # FSM helpers
    # ------------------------------------------------------------------

    def _fsm_enter_state(self) -> None:
        """Enter the FSM's current state: switch frame range, arm detectors, open voice window."""
        if not self._fsm:
            return
        shot = self.current_shot
        if not shot:
            return
        state_id = self._fsm.current
        print(f"[ShotPlayer] FSM → {state_id}  loop={self._fsm.is_loop()}")

        # Switch render engine to this state's frame range
        seg = self._fsm.segment_range()
        if seg:
            self.event_bus.emit("play_segment", {
                "start": seg[0],
                "end":   seg[1],
                "loop":  self._fsm.is_loop(),
            })

        # SFX on state entry
        sfx = self._fsm.on_enter_sfx()
        if sfx:
            sfx_path = _resolve_sfx(shot, sfx)
            if sfx_path:
                self.event_bus.emit("play_sfx", {"path": sfx_path})
            else:
                print(f"[ShotPlayer] FSM SFX not found: {sfx!r}")

        # Arm gesture detector for any outgoing transitions.
        # gesture_type in the FSM spec selects the detector family:
        #   "point"  (default) — directional_point: hold finger in direction
        #   "draw"             — directional_draw:  swipe motion in direction
        directions = self._fsm.outgoing_point_directions()
        keyword    = self._fsm.voice_keyword()
        hold_ms    = (shot.interaction or {}).get("hold_ms", 500)
        fsm_spec   = (shot.interaction or {}).get("interaction_fsm", {})
        gesture_type = fsm_spec.get("gesture_type", "point")

        # States with no gesture or voice transitions (transition animations, confirm
        # segments) lock all input so stale detections can't fire during playback.
        # States with active transitions unlock so the player can interact.
        accepts_input = bool(directions or keyword)
        self.event_bus.emit("input_lock", {"locked": not accepts_input})
        print(f"[ShotPlayer] FSM input_lock={not accepts_input}  state={state_id}  gesture_type={gesture_type}")

        if directions and gesture_type == "draw":
            # directional_draw: one direction per state, motion-based swipe.
            # Use the first declared direction (states should only list one for draw).
            # Gesture id is "draw_<direction>" so _on_cg_detected fires FSM event directly.
            direction = directions[0]
            self.event_bus.emit("cg_window_open", {
                "interaction": {
                    "id":     f"draw_{direction}",
                    "type":   "directional_draw",
                    "params": {"direction": direction},
                    "tier":   "cg",
                }
            })
        elif directions:
            params = {"directions": directions, "hold_ms": hold_ms}
            # Position/target mode: if the shot declares on-screen path targets,
            # pass them so the detector classifies by where the fingertip is (which
            # path it's over) rather than by the finger-vector angle.
            interaction = shot.interaction or {}
            point_targets = interaction.get("point_targets")
            if point_targets:
                params["targets"] = point_targets
                params["proximity_threshold"] = interaction.get("point_proximity", 0.22)
            self.event_bus.emit("cg_window_open", {
                "interaction": {
                    "id":     f"{shot.shot}_fsm_{state_id}",
                    "type":   "directional_point",
                    "params": params,
                    "tier":   "cg",
                }
            })
        else:
            self.event_bus.emit("cg_window_open", {"interaction": None})

        if keyword:
            self.event_bus.emit("vi_window_open", {
                "vi_config": {
                    "id":       f"voice_{keyword}",
                    "keywords": [keyword],
                    "mode":     "keyword",
                    "tier":     "cg_required",
                    "window_ms": 10000,
                }
            })

    def _fsm_fire(self, event: str) -> None:
        """Fire an FSM event from the current state, enter the resulting state or advance."""
        if not self._fsm:
            print(f"[ShotPlayer][DBG] _fsm_fire({event!r}) but no FSM active")
            return
        shot = self.current_shot
        if not shot:
            return
        from_state = self._fsm.current
        next_state = self._fsm.fire(event)
        print(f"[ShotPlayer][DBG] _fsm_fire: from={from_state!r} event={event!r} "
              f"-> {next_state!r}")
        if next_state is None:
            return   # no matching transition — silent
        print(f"[ShotPlayer] FSM event={event!r} → {next_state}")
        if next_state == "__advance__":
            self._fsm = None
            self.event_bus.emit("input_lock", {"locked": False})
            self._advance()
        else:
            self._fsm_enter_state()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_shot_frames_ready(self, data: dict) -> None:
        """Render engine finished preloading — start playback now."""
        if self._shot_state != STATE_PRELOADING:
            return
        shot = self.current_shot
        if shot is None:
            return
        if shot.kind == "playback":
            self._enter_segment(STATE_PLAY, shot)
        else:
            self._enter_segment(STATE_PLAY_INTRO, shot)

    def _prefetch_next(self, current_index: int) -> None:
        """Tell the render engine to start preloading the next shot in background."""
        next_index = current_index + 1
        if next_index < len(self.shots):
            next_shot = self.shots[next_index]
            if not next_shot.assets_pending:
                self.event_bus.emit("prefetch_shot", {"shot": next_shot})

    def _on_segment_playback_done(self, data: dict) -> None:
        """Non-looping FSM segment finished — fire segment_end transition."""
        if self._shot_state == STATE_HOLD and self._fsm:
            self._fsm_fire("segment_end")

    def _on_cg_detected(self, data: dict) -> None:
        print(f"[ShotPlayer][DBG] cg_detected raw={data}  state={self._shot_state}  "
              f"fsm_current={self._fsm.current if self._fsm else None}")
        if self._shot_state != STATE_HOLD:
            return
        shot = self.current_shot
        if shot is None:
            return

        # FSM path: convert choice → point_<direction> event (normalize diagonals)
        if self._fsm:
            choice = data.get("choice", "")
            if choice:
                canonical = self._fsm.normalize_direction(choice)
                self._fsm_fire(f"point_{canonical}")
            else:
                self._fsm_fire(data.get("gesture_id", ""))
            return

        # Existing chain path
        if not self._chain or self._chain_index >= len(self._chain):
            return
        gesture_id = data.get("gesture_id", "")
        expected   = self._chain[self._chain_index].get("id")
        if gesture_id != expected:
            return   # not the step we're waiting for

        choice = data.get("choice")
        print(f"[ShotPlayer] shot {shot.shot} CG step {self._chain_index} DONE: "
              f"{gesture_id}" + (f"  choice={choice}" if choice else ""))

        if choice:
            self.event_bus.emit("shot_choice", {
                "shot_id": shot.shot,
                "choice":  choice,
            })

        self._chain_index += 1
        self._arm_chain_step(shot)

    def _on_oi_detected(self, data: dict) -> None:
        shot = self.current_shot
        if not shot:
            return
        print(f"[ShotPlayer] shot {shot.shot} OI: {data.get('gesture_id')}")
        interaction = shot.interaction or {}
        sfx = interaction.get("sfx")
        feedback = interaction.get("feedback")
        if sfx:
            sfx_path = _resolve_sfx(shot, sfx)
            if sfx_path:
                self.event_bus.emit("play_sfx", {"path": sfx_path})
            else:
                print(f"[ShotPlayer] SFX not found: {sfx!r} (checked audio/ subdir and shot root)")
        if feedback == "green_flash":
            self.event_bus.emit("oi_flash", {"color": (0, 255, 80), "duration_ms": 800})

    def _on_vi_detected(self, data: dict) -> None:
        shot = self.current_shot
        if not shot:
            return
        voice_id = data.get("voice_id", "")
        print(f"[ShotPlayer] shot {shot.shot} VI detected: {voice_id!r}  tier={data.get('tier')}"
              f"  state={self._shot_state}  fsm_current={self._fsm.current if self._fsm else None}")

        # FSM path: voice_id is used directly as the transition event (e.g. "voice_go")
        if self._fsm and self._shot_state == STATE_HOLD:
            self._fsm_fire(voice_id)
            return

        # For OI playback shots with a voice_alternative, treat VI exactly like OI detected
        if self._shot_state == STATE_PLAY and shot.interaction:
            tier = shot.interaction.get("tier", "").upper()
            voice_alt = shot.interaction.get("voice_alternative", {})
            if tier == "OI" and voice_id == voice_alt.get("id"):
                sfx = shot.interaction.get("sfx")
                feedback = shot.interaction.get("feedback")
                if sfx:
                    sfx_path = _resolve_sfx(shot, sfx)
                    if sfx_path:
                        self.event_bus.emit("play_sfx", {"path": sfx_path})
                if feedback == "green_flash":
                    self.event_bus.emit("oi_flash", {"color": (0, 255, 80), "duration_ms": 800})
                return

        # Phase 3: CG chain voice steps
        print(f"[ShotPlayer] shot {shot.shot} VI {voice_id!r} not acted on in current state "
              f"({self._shot_state})")

    def _on_frame_window_enter(self, data: dict) -> None:
        """Render engine crossed the oi_frame_window start — arm gesture OI and voice window."""
        shot = self.current_shot
        if not shot or not shot.interaction:
            return
        oi_frame_window = shot.interaction.get("oi_frame_window", [])
        if len(oi_frame_window) == 2:
            frame_count = oi_frame_window[1] - oi_frame_window[0]
            window_ms = max(int(frame_count / max(1, shot.fps) * 1000), 2000)
        else:
            window_ms = 5000
        oi = _single_as_oi(shot.shot, shot.interaction)
        print(f"[ShotPlayer] shot {shot.shot} frame_window_enter -> arming OI {oi['id']} "
              f"for {window_ms}ms")
        self.event_bus.emit("oi_window_open", {"interaction": oi, "window_ms": window_ms})

        # Also open a voice window if the interaction declares a voice_alternative
        voice_alt = shot.interaction.get("voice_alternative")
        if voice_alt:
            vi_config = dict(voice_alt)
            vi_config.setdefault("window_ms", window_ms)
            print(f"[ShotPlayer] shot {shot.shot} frame_window_enter -> arming VI "
                  f"{vi_config.get('id')!r} keywords={vi_config.get('keywords')}")
            self.event_bus.emit("vi_window_open", {"vi_config": vi_config})


# ---------------------------------------------------------------------------
# ShotFSM — mini state machine for branching interactive shots
# ---------------------------------------------------------------------------

class ShotFSM:
    """
    Minimal FSM for shots that have multiple selectable paths (point_path,
    point_fork, choose_path).

    Spec lives under interaction["interaction_fsm"] in metadata.json:

        states      — {state_id: {segment, loop, on_enter_sfx, voice}}
        transitions — [{from, on, to}, ...]  (to "__advance__" exits the shot)
        initial     — starting state id

    Gesture events are named "point_<direction>" (left/right/up/down/…).
    Voice events use the vi_config id, typically "voice_<keyword>".
    The render engine emits "segment_playback_done" → "segment_end" event.
    """

    # Maps FSM event name → directions accepted by the directional_point detector.
    # Cardinal events (point_left/right/up/down) expand to include adjacent diagonals
    # so real-world pointing at ~45° off-axis still registers.
    _DIR_MAP = {
        "point_left":       ["left",  "up_left",   "down_left"  ],
        "point_right":      ["right", "up_right",  "down_right" ],
        "point_up":         ["up",    "up_left",   "up_right"   ],
        "point_down":       ["down",  "down_left", "down_right" ],
        "point_up_left":    ["up_left"   ],
        "point_up_right":   ["up_right"  ],
        "point_down_left":  ["down_left" ],
        "point_down_right": ["down_right"],
    }

    # Maps a raw detected direction back to the canonical FSM event direction
    # (diagonals collapse to the cardinal they belong to under _DIR_MAP expansion).
    _DIR_NORMALIZE = {
        "left":       "left",  "up_left":   "left",  "down_left":  "left",
        "right":      "right", "up_right":  "right", "down_right": "right",
        "up":         "up",    "down":      "down",
    }

    def __init__(self, spec: dict, segments: Optional[dict]):
        self.states      = spec.get("states", {})
        self.transitions = spec.get("transitions", [])
        self.fallback    = spec.get("fallback", {})
        self._segments   = segments or {}
        self.current     = spec.get("initial", "waiting")
        self.done        = False

    def segment_range(self) -> Optional[tuple]:
        seg_name = self.states.get(self.current, {}).get("segment")
        if not seg_name:
            return None
        seg = self._segments.get(seg_name)
        if isinstance(seg, (list, tuple)) and len(seg) == 2:
            return (int(seg[0]), int(seg[1]))
        return None

    def is_loop(self) -> bool:
        return self.states.get(self.current, {}).get("loop", False)

    def on_enter_sfx(self) -> Optional[str]:
        return self.states.get(self.current, {}).get("on_enter_sfx")

    def voice_keyword(self) -> Optional[str]:
        return self.states.get(self.current, {}).get("voice")

    def outgoing_point_directions(self) -> list:
        """Accepted directions for the current state's outgoing point_* transitions.

        If the current state declares a "directions" list in the FSM metadata, that
        list is used exactly (no expansion). Otherwise cardinal events expand to include
        adjacent diagonals so off-axis pointing registers.
        """
        # Per-state override: metadata can restrict or custom-set accepted directions.
        # Use this to keep "waiting" strict (cardinal only) while allowing diagonal
        # leniency in the selected/switch states.
        override = self.states.get(self.current, {}).get("directions")
        if override is not None:
            return list(override)

        result = []
        for t in self.transitions:
            if t.get("from") == self.current:
                dirs = self._DIR_MAP.get(t.get("on", ""), [])
                for d in dirs:
                    if d not in result:
                        result.append(d)
        return result

    def normalize_direction(self, raw: str) -> str:
        """Collapse a detected direction to the canonical FSM event direction."""
        return self._DIR_NORMALIZE.get(raw, raw)

    def fire(self, event: str) -> Optional[str]:
        """
        Fire event from current state.
        Returns next state id, "__advance__", or None (no matching transition).
        Updates self.current on success.
        """
        for t in self.transitions:
            if t.get("from") == self.current and t.get("on") == event:
                next_state = t["to"]
                if next_state == "__advance__":
                    self.done = True
                    return "__advance__"
                self.current = next_state
                return next_state
        return None


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no side-effects)
# ---------------------------------------------------------------------------

def _build_chain(shot_id: str, interaction: Optional[dict]) -> list[dict]:
    """
    Return an ordered list of CG chain steps with guaranteed 'id' fields.

    OI interactions return an empty list (non-blocking; handled separately).
    Single-step interactions are normalised to a one-element chain.
    """
    if not interaction:
        return []
    tier = interaction.get("tier", "").upper()
    if tier == "OI":
        return []

    if "chain" in interaction:
        steps = []
        for i, raw_step in enumerate(interaction["chain"]):
            step = dict(raw_step)
            if "id" not in step:
                step["id"] = f"{shot_id}_step{i}"
            steps.append(step)
        return steps

    # Single-step (no chain array)
    step = dict(interaction)
    if "id" not in step:
        step["id"] = f"{shot_id}_cg"
    return [step]


def _step_to_interaction(shot_id: str, step_index: int, step: dict,
                          tier: str = "CG") -> dict:
    """
    Convert a chain step dict to the format GestureEngine's cg_window_open expects:
        id, type, params, tier
    """
    return {
        "id":     step.get("id", f"{shot_id}_step{step_index}"),
        "type":   step.get("type"),
        "params": step.get("params", {}),
        "tier":   tier.lower(),
    }


def _single_as_oi(shot_id: str, interaction: dict) -> dict:
    """Build a minimal OI interaction dict from a shot's full interaction spec."""
    return {
        "id":     interaction.get("id", f"{shot_id}_oi"),
        "type":   interaction.get("type"),
        "params": interaction.get("params", {}),
        "tier":   "oi",
    }


def _has_resolution(shot: Shot) -> bool:
    """True if the shot has a non-TODO resolution frame range."""
    if not shot.segments or shot.segments_todo:
        return False
    res = shot.segments.get("resolution")
    return isinstance(res, (list, tuple)) and len(res) == 2


def _get_segment_range(shot: Shot, state: str) -> Optional[tuple[int, int]]:
    """Return (a, b) frame range for the given segment state, or None."""
    if not shot.segments or shot.segments_todo:
        return None
    key = {STATE_PLAY_INTRO: "intro", STATE_PLAY_RES: "resolution"}.get(state)
    if key is None:
        return None
    val = shot.segments.get(key)
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return (int(val[0]), int(val[1]))
    return None


def _estimate_duration(shot: Shot, state: str) -> float:
    """
    Return the expected play duration in seconds for a PLAY/PLAY_INTRO/PLAY_RES segment.

    Priority:
      1. Explicit segment frame range in shot.segments → exact frame count / fps.
      2. No segment range but frames_dir exists → count files in directory (one read per shot).
      3. assets_pending or no frames_dir → 0.0 (advance immediately).
    """
    seg_range = _get_segment_range(shot, state)
    if seg_range is not None:
        a, b = seg_range
        return max(1, b - a) / max(1, shot.fps) + 0.5

    if shot.assets_pending or shot.frames_dir is None:
        return 0.0

    from pathlib import Path
    d = Path(shot.frames_dir)
    image_exts = {".png", ".jpg", ".jpeg"}
    try:
        num_frames = sum(1 for f in d.iterdir() if f.suffix.lower() in image_exts)
    except OSError:
        return 0.0
    return num_frames / max(1, shot.fps)


def _resolve_sfx(shot: Shot, filename: str) -> Optional[str]:
    """
    Resolve an SFX filename to an absolute path.
    Checks audio/ subdirectory first, then the shot root directory.
    """
    from pathlib import Path
    if shot.audio_dir:
        p = Path(shot.audio_dir) / filename
        if p.exists():
            return str(p)
    if shot.frames_dir:
        p = Path(shot.frames_dir).parent / filename
        if p.exists():
            return str(p)
    return None


def _effective_timing(shot: Shot, config: dict) -> dict:
    """
    Merge timing from config timing profiles with per-shot fallback overrides.

    Resolution order (most specific wins):
        global profile_standard / profile_urgent  ← shot.fallback
    """
    profile_key = "profile_" + shot.timing_profile
    profile     = config["timing_defaults"].get(
        profile_key, config["timing_defaults"]["profile_standard"]
    )
    base = {
        "timeout_s":  profile.get("auto_advance_ms",   30000) / 1000.0,
        "reprompt_s": [
            profile.get("reprompt_first_ms",   8000)  / 1000.0,
            profile.get("reprompt_second_ms", 16000)  / 1000.0,
        ],
        "on_timeout": "auto_advance",
    }
    base.update(shot.fallback)   # per-shot overrides win
    return base
