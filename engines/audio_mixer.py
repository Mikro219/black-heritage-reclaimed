"""
audio_mixer — layered stem playback for the shot-driven runtime (July 2026).

Plays each shot's ``audio_events`` (music/ambience beds + frame-anchored SFX
one-shots), replacing the old per-shot baked ``audio.mp3`` slice. Events are
authored by the CapCut importer (scripts/capcut_audio.py) or the Experience
Builder and resolved onto Shot objects by sequence_loader.

Semantics (the behaviour contract — see tests/test_audio_events.py):

* Every event is frame-anchored: it arms when the render engine's playback
  frame crosses ``at_s × fps``, and fires at most once per shot entry. Events
  parked after an idle_loop simply never fire if the hold times out past them.
* ``sustain`` events (music/ambience beds) loop indefinitely once started and
  keep playing through HOLD states and shot transitions.
* On the next shot becoming ready: an active bed whose file matches one of the
  new shot's ``continues`` events is handed over silently (no restart; the new
  event's gain is adopted). Unclaimed beds fade out over their fade_out_ms.
* Non-sustain events ("sfx") play once on a pooled channel and are left to run
  out — they are short by construction (the importer pre-trims clips).
* ``input_lock`` does NOT stop anything: bed continuity across transitions is
  the point of this layer.

Channel plan (pygame.mixer, NUM_CHANNELS total):
  0     VO / narration (NarrationAdapter + RenderEngine audio.mp3 — untouched)
  1     stroke / on_enter SFX          (RenderEngine play_sfx)
  2     detect.mp3 reaction SFX        (RenderEngine play_sfx)
  3–5   beds (music/ambience; spare channel leaves room for crossfades)
  6–15  one-shot pool (round-robin)

RenderEngine stamps master volume only on channels 0–2 and broadcasts a
"master_volume" event; this mixer owns 3+ and sets per-channel volume to
event gain × master.

Runtime never slices audio: the importer pre-renders any trimmed/tempo'd
clip to its own file, so every event's ``path`` is played whole.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Callable, Optional

import pygame

log = logging.getLogger(__name__)

NUM_CHANNELS      = 16
BED_CHANNELS      = (3, 4, 5)
ONESHOT_CHANNELS  = tuple(range(6, 16))
_DEFAULT_FADE_OUT_MS = 400
_SOUND_CACHE_MAX  = 32


class ShotAudioMixer:
    """Subscribes to shot lifecycle events and schedules audio_events.

    Call ``update()`` once per main-loop frame (same cadence as
    NarrationAdapter.update()).

    ``frame_provider`` returns the render engine's current playback frame, or
    None while a shot is loading/paused (RenderEngine.playback_frame).
    """

    def __init__(self, config: dict, event_bus,
                 frame_provider: Optional[Callable[[], Optional[int]]] = None):
        self.config    = config
        self.event_bus = event_bus
        self._frame_provider = frame_provider or (lambda: None)

        self._master = float(config.get("audio", {}).get("master_volume", 1.0))

        # Active shot state
        self._events: list = []          # current shot's audio_events
        self._fps: int = 24
        self._fired: set = set()         # event indices already started this shot
        self._warned: set = set()        # file names already warned unresolvable

        # Pending shot (stashed at shot_load, activated at shot_frames_ready)
        self._pending: Optional[tuple[list, int]] = None

        # channel index -> {"event": dict, "channel": Channel, "since": float}
        self._beds: dict[int, dict] = {}
        self._oneshot_rr = 0             # round-robin cursor into ONESHOT_CHANNELS

        # path(str) -> Sound, LRU-capped (decoded beds are RAM-heavy)
        self._sounds: OrderedDict[str, pygame.mixer.Sound] = OrderedDict()

        if pygame.mixer.get_init():
            pygame.mixer.set_num_channels(max(NUM_CHANNELS,
                                              pygame.mixer.get_num_channels()))

        event_bus.subscribe("shot_load",         self._on_shot_load)
        event_bus.subscribe("shot_frames_ready", self._on_shot_frames_ready)
        event_bus.subscribe("master_volume",     self._on_master_volume)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Fire any event whose anchor frame has been crossed."""
        if not self._events:
            return
        frame = self._frame_provider()
        if frame is None:
            return
        for i, ev in enumerate(self._events):
            if i in self._fired:
                continue
            if frame >= self._at_frame(ev):
                self._fired.add(i)
                self._start_event(ev)

    def stop(self) -> None:
        """Stop everything this mixer owns (shutdown)."""
        if not pygame.mixer.get_init():
            return
        for ch in BED_CHANNELS + ONESHOT_CHANNELS:
            if ch < pygame.mixer.get_num_channels():
                pygame.mixer.Channel(ch).stop()
        self._beds.clear()
        self._events = []
        self._pending = None

    def debug_info(self) -> dict:
        return {
            "beds":    [b["event"]["file"] for b in self._beds.values()],
            "fired":   len(self._fired),
            "events":  len(self._events),
            "pending": self._pending is not None,
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_shot_load(self, data: dict) -> None:
        shot = data.get("shot")
        if shot is None:
            return
        events = list(getattr(shot, "audio_events", None) or [])
        fps    = int(getattr(shot, "fps", 24) or 24)
        self._pending = (events, fps)

    def _on_shot_frames_ready(self, data: dict) -> None:
        """The new shot starts playing now: hand over / fade beds, arm events."""
        if self._pending is None:
            return
        events, fps = self._pending
        self._pending = None
        self._fps = fps

        claimed: set = set()
        for ch_idx, bed in list(self._beds.items()):
            match = None
            for i, ev in enumerate(events):
                if i in claimed or not ev.get("continues"):
                    continue
                if ev.get("file") == bed["event"].get("file"):
                    match = i
                    break
            if match is not None:
                # Silent handover: keep the channel playing, adopt the new gain.
                claimed.add(match)
                bed["event"] = events[match]
                bed["channel"].set_volume(self._gain(events[match]))
            else:
                fade = int(bed["event"].get("fade_out_ms") or _DEFAULT_FADE_OUT_MS)
                bed["channel"].fadeout(max(1, fade))
                del self._beds[ch_idx]

        self._events = events
        self._fired  = set(claimed)   # handed-over beds are already "started"

    def _on_master_volume(self, data: dict) -> None:
        self._master = max(0.0, min(1.0, float(data.get("volume", 1.0))))
        for bed in self._beds.values():
            bed["channel"].set_volume(self._gain(bed["event"]))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _at_frame(self, ev: dict) -> int:
        return int(float(ev.get("at_s", 0.0)) * self._fps)

    def _gain(self, ev: dict) -> float:
        return max(0.0, min(1.0, float(ev.get("gain", 1.0)) * self._master))

    def _load_sound(self, ev: dict) -> Optional[pygame.mixer.Sound]:
        path = ev.get("path")
        if path is None:
            if ev.get("file") not in self._warned:
                self._warned.add(ev.get("file"))
                log.warning("ShotAudioMixer: %s not found (shot audio/ or _audio pool)",
                            ev.get("file"))
            return None
        key = str(path)
        sound = self._sounds.get(key)
        if sound is not None:
            self._sounds.move_to_end(key)
            return sound
        try:
            sound = pygame.mixer.Sound(key)
        except Exception as exc:
            if ev.get("file") not in self._warned:
                self._warned.add(ev.get("file"))
                log.warning("ShotAudioMixer: could not load %s: %s", key, exc)
            return None
        self._sounds[key] = sound
        while len(self._sounds) > _SOUND_CACHE_MAX:
            self._sounds.popitem(last=False)
        return sound

    def _start_event(self, ev: dict) -> None:
        sound = self._load_sound(ev)
        if sound is None:
            return
        fade_in = max(0, int(ev.get("fade_in_ms") or 0))
        if ev.get("sustain"):
            ch_idx = self._free_bed_channel()
            if ch_idx is None:
                log.warning("ShotAudioMixer: no free bed channel for %s", ev.get("file"))
                return
            channel = pygame.mixer.Channel(ch_idx)
            channel.set_volume(self._gain(ev))
            channel.play(sound, loops=-1, fade_ms=fade_in)
            self._beds[ch_idx] = {"event": ev, "channel": channel,
                                  "since": time.monotonic()}
        else:
            ch_idx = ONESHOT_CHANNELS[self._oneshot_rr % len(ONESHOT_CHANNELS)]
            self._oneshot_rr += 1
            channel = pygame.mixer.Channel(ch_idx)
            channel.set_volume(self._gain(ev))
            channel.play(sound, fade_ms=fade_in)

    def _free_bed_channel(self) -> Optional[int]:
        for ch in BED_CHANNELS:
            if ch not in self._beds:
                return ch
        # All busy: evict the oldest bed (fade it) — newest event wins.
        oldest = min(self._beds, key=lambda c: self._beds[c]["since"])
        bed = self._beds.pop(oldest)
        fade = int(bed["event"].get("fade_out_ms") or _DEFAULT_FADE_OUT_MS)
        bed["channel"].fadeout(max(1, fade))
        return oldest
