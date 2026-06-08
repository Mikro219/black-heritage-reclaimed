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

        event_bus.subscribe("cg_detected", self._on_cg_detected)
        event_bus.subscribe("oi_detected", self._on_oi_detected)
        event_bus.subscribe("vi_detected", self._on_vi_detected)
        # Phase 3: event_bus.subscribe("segment_playback_done", self._on_segment_playback_done)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def current_shot(self) -> Optional[Shot]:
        if 0 <= self._index < len(self.shots):
            return self.shots[self._index]
        return None

    def start(self) -> None:
        if not self.shots:
            self._player_state = PLAYER_FINISHED
            self.event_bus.emit("sequence_finished", {})
            return
        self._player_state = PLAYER_RUNNING
        self._enter_shot(0)

    def update(self) -> None:
        """Drive the state machine. Call once per main-loop frame."""
        if self._player_state != PLAYER_RUNNING:
            return
        shot = self.current_shot
        if shot is None:
            return

        if self._shot_state in (STATE_PLAY, STATE_PLAY_INTRO, STATE_PLAY_RES):
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

        self.event_bus.emit("shot_load", {"shot": shot, "index": index})
        print(f"[ShotPlayer] >> shot {shot.shot}  kind={shot.kind}  act={shot.act}  "
              f"{'PENDING' if shot.assets_pending else 'frames_ready'}")

        if shot.kind == "playback":
            self._enter_segment(STATE_PLAY, shot)
        else:
            self._enter_segment(STATE_PLAY_INTRO, shot)

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

        # Announce audio lines so Phase 3 can trigger narration
        if state in (STATE_PLAY, STATE_PLAY_INTRO) and shot.audio_lines:
            self.event_bus.emit("shot_audio_lines", {
                "shot_id": shot.shot,
                "lines":   shot.audio_lines,
                "trigger": "intro",
            })

    def _enter_hold(self, shot: Shot) -> None:
        self._shot_state      = STATE_HOLD
        self._hold_start      = time.monotonic()
        self._chain_index     = 0
        self._reprompt_fired  = set()
        self._segment_done    = False
        self._segment_duration = 0.0

        self.event_bus.emit("shot_state_change", {
            "shot_id": shot.shot,
            "state":   STATE_HOLD,
            "segment": "idle_loop",
        })

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
    # Event handlers
    # ------------------------------------------------------------------

    def _on_cg_detected(self, data: dict) -> None:
        if self._shot_state != STATE_HOLD:
            return
        shot = self.current_shot
        if shot is None or not self._chain:
            return
        if self._chain_index >= len(self._chain):
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
        if shot:
            print(f"[ShotPlayer] shot {shot.shot} OI: {data.get('gesture_id')}")

    def _on_vi_detected(self, data: dict) -> None:
        # Phase 3: if a VI chain step is armed and voice_id matches, advance the chain
        shot = self.current_shot
        if shot:
            print(f"[ShotPlayer] shot {shot.shot} VI: {data.get('voice_id')} "
                  f"[not acted on until Phase 3]")


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
