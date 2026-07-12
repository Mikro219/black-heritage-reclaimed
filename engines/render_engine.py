"""
RenderEngine — Pygame-based frame sequencer and display layer.

Serves the shot-driven runtime: loads each shot's frame pack through the
look-ahead FrameCacheManager, advances frames on the shot clock (whole-shot
time-based playback or FSM segment ranges via play_segment), runs the
frame-gated OI windows, and draws the player-facing layer (hand-icon cursors,
interaction indicators, green success flash, pause menu, tutorial cards) plus
the debug overlay (D) and skeleton mini-panel (K).
"""

import math
import os
import time
import pygame
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from .frame_cache import FrameCacheManager

# Max converted Surfaces kept in RAM per shot (LRU). 240 frames at 1080p ≈ 1.9 GB.
SURFACE_LRU_CAP = 240


class FrameView:
    """Lazy, RAM-bounded view over a shot's frames, indexable like a list of
    pygame Surfaces.

    Frames are converted from the cache's mmapped raw bytes on demand and only an
    LRU window of Surfaces is kept resident, so a 5000-frame shot never holds all
    its Surfaces in memory at once. Supports len(), [i] and [a:b] so it drops into
    the existing render code wherever a frame list was used.
    """

    def __init__(self, cache, frames_dir, count, convert_fn, lru_cap=SURFACE_LRU_CAP):
        self._cache = cache
        self._dir = frames_dir
        self._count = count
        self._convert = convert_fn
        self._cap = lru_cap
        self._surf: "OrderedDict[int, pygame.Surface]" = OrderedDict()
        self._blank: Optional[pygame.Surface] = None

    def __len__(self):
        return self._count

    def __bool__(self):
        return self._count > 0

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self._count))]
        if i < 0:
            i += self._count
        i = max(0, min(i, self._count - 1))

        surf = self._surf.get(i)
        if surf is not None:
            self._surf.move_to_end(i)
            return surf

        got = self._cache.get_frame_bytes(self._dir, i)
        if got is None:
            if self._blank is None:
                self._blank = pygame.Surface(self._cache.resolution())
            return self._blank
        data, size = got
        surf = self._convert(data, size)
        self._surf[i] = surf
        if len(self._surf) > self._cap:
            self._surf.popitem(last=False)   # evict least-recently-used
        return surf


class RenderEngine:
    def __init__(self, config: dict, event_bus: "EventBus"):
        self.config = config
        self.event_bus = event_bus
        self._screen: Optional[pygame.Surface] = None
        self._fullscreen: bool = False
        self._display_size: tuple = (1920, 1080)
        self._frames: list = []
        self._frame_index = 0
        self._fps = 24
        self._last_frame_time = 0.0
        # Debug overlay starts OFF regardless of profile/config; toggle it at
        # runtime with the D key (render.toggle_debug()).
        self._debug = False
        self._landmark_data = None
        self._font: Optional[pygame.font.Font] = None
        self._small_font: Optional[pygame.font.Font] = None
        self._playback_start_time: float = 0.0
        self._pending_audio: Optional[str] = None
        self._current_frames_dir: Optional[Path] = None

        # Look-ahead frame cache (continuous background preload of all shots with art)
        self._cache: Optional[FrameCacheManager] = None
        self._loading_dir: Optional[Path] = None   # incoming shot being converted
        self._loading_kind: str = "playback"       # kind of the incoming shot (debug only)

        # OI flash overlay (full-screen surface pre-allocated once in init_display)
        self._flash_color: tuple = (0, 255, 80)
        self._flash_start: float = 0.0
        self._flash_until: float = 0.0
        self._flash_alpha: int   = 80   # peak alpha (0-255)
        self._flash_overlay: Optional[pygame.Surface] = None

        # Frame-gated OI window (1-based frame numbers, matching filenames)
        self._oi_frame_start: Optional[int] = None
        self._oi_frame_end:   Optional[int] = None
        self._oi_window_open: bool = False

        # FSM segment-constrained playback (set by play_segment event)
        self._seg_start:  Optional[int] = None  # inclusive start frame index
        self._seg_end:    Optional[int] = None  # inclusive end frame index
        self._seg_loop:   bool = True
        self._seg_anchor: float = 0.0           # time.monotonic() when segment started
        self._seg_done:   bool = False

        # Pause/resume — freezes frame advance; on resume every monotonic time
        # anchor is shifted forward by the paused duration so playback, segment
        # loops and the OI flash all continue exactly where they left off.
        self._paused:        bool  = False
        self._pause_started: float = 0.0

        # Master volume (0.0-1.0), applied to every mixer channel. Adjustable from
        # the pause menu (Up/Down or dragging the slider).
        self._volume: float = float(config.get("audio", {}).get("master_volume", 1.0))
        self._volume_slider_rect: Optional[pygame.Rect] = None  # set when drawn

        # Skeleton-in-corner display option (pause menu, K key) — shows the
        # bottom-right skeleton mini-panel WITHOUT the rest of the debug overlay.
        self._show_skeleton: bool = False

        # Hand-icon cursors (July 2026): illustrated hands from assets/hand_icons/
        # replace the old crosshair+label cursors. Keyed "open"/"fist"/"point"/
        # "knock" + "_l"/"_r". Per-side last-seen state persists through Hands
        # dropouts so the icon doesn't flicker.
        self._hand_icons: dict = {}
        self._hand_cursor_state: dict = {
            "L": {"pos": None, "shape": "open", "t": 0.0},
            "R": {"pos": None, "shape": "open", "t": 0.0},
        }

        # Play-through segment overshoot carry: when a non-loop FSM segment ends,
        # playback has typically run a fraction of a frame past the boundary by the
        # time the tick notices. The next play_segment subtracts this so chained
        # segments stay in lockstep with the shot's continuous audio track instead
        # of drifting later with every transition (Scene 5/8/11 lag).
        self._seg_overshoot: float = 0.0

        self.event_bus.subscribe("shot_load",     self._on_shot_load)
        self.event_bus.subscribe("prefetch_shot", self._on_prefetch_shot)
        self.event_bus.subscribe("oi_flash",          self._on_oi_flash)
        self.event_bus.subscribe("play_sfx",          self._on_play_sfx)
        self.event_bus.subscribe("set_frame_window",  self._on_set_frame_window)
        self.event_bus.subscribe("play_segment",      self._on_play_segment)

    def init_display(self):
        pygame.display.init()
        pygame.font.init()
        display_cfg = self.config.get("_profile", {}).get("display", {})
        w, h = display_cfg.get("resolution") or self.config.get("resolution", [1920, 1080])
        self._display_size = (w, h)
        self._fullscreen = bool(display_cfg.get("fullscreen", False))
        flags = pygame.FULLSCREEN if self._fullscreen else 0
        self._screen = pygame.display.set_mode((w, h), flags)
        pygame.display.set_caption("Black Heritage Reclaimed")
        self._font = pygame.font.SysFont("monospace", 20, bold=True)
        self._small_font = pygame.font.SysFont("monospace", 14, bold=True)
        # Cached fonts for the scene panel (avoids constructing SysFont per frame).
        # Pre-allocate the full-screen OI flash overlay once (reused each flash frame).
        self._flash_overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        self._load_hand_icons()

    def _load_hand_icons(self) -> None:
        """Load the illustrated hand cursors produced by scripts/prepare_hand_icons.py.
        Missing files degrade gracefully to the dot cursors."""
        icons_dir = Path(__file__).resolve().parent.parent / "assets" / "hand_icons"
        target_h = max(60, int(self._display_size[1] * 0.09))
        for shape in ("open", "fist", "point", "knock"):
            for side in ("l", "r"):
                p = icons_dir / f"{shape}_{side}.png"
                if not p.exists():
                    continue
                try:
                    img = pygame.image.load(str(p)).convert_alpha()
                except pygame.error:
                    continue
                scale = target_h / max(1, img.get_height())
                img = pygame.transform.smoothscale(
                    img, (max(1, int(img.get_width() * scale)), target_h))
                self._hand_icons[f"{shape}_{side}"] = img
        if self._hand_icons:
            print(f"[RenderEngine] hand icons loaded: {len(self._hand_icons)}")
        else:
            print("[RenderEngine] no hand icons found — falling back to dot cursors "
                  "(run: py -3.12 scripts/prepare_hand_icons.py)")

    def toggle_fullscreen(self) -> None:
        """Switch between fullscreen and windowed at the same resolution.

        Recreates the display surface (more reliable than display.toggle_fullscreen
        on Windows). Existing cached frame Surfaces stay blittable; new conversions
        pick up the new display format on demand via the FrameView LRU.
        """
        if self._screen is None:
            return
        self._fullscreen = not self._fullscreen
        flags = pygame.FULLSCREEN if self._fullscreen else 0
        self._screen = pygame.display.set_mode(self._display_size, flags)
        print(f"[RenderEngine] fullscreen={'on' if self._fullscreen else 'off'}")

    def toggle_debug(self) -> None:
        """Show/hide the debug overlay (HANDS/POSE/CG/OI/SHOT panels, OI target flag,
        scene panel). Does not affect playback."""
        self._debug = not self._debug
        print(f"[RenderEngine] debug_overlay={'on' if self._debug else 'off'}")

    def toggle_skeleton(self) -> None:
        """Show/hide the bottom-right skeleton mini-panel on its own (pause-menu
        display option, K key) — independent of the full debug overlay."""
        self._show_skeleton = not self._show_skeleton
        print(f"[RenderEngine] skeleton_panel={'on' if self._show_skeleton else 'off'}")

    # ------------------------------------------------------------------
    # Master volume
    # ------------------------------------------------------------------

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def playback_frame(self):
        """Current frame index while a shot is actively playing, else None (loading
        freeze / no frames / paused). ShotAudioMixer's frame clock for
        frame-anchored audio_events."""
        if self._paused or not self._frames or self._loading_dir is not None:
            return None
        return self._frame_index

    # Channels the render engine owns volume-wise: 0 VO/narration, 1 stroke sfx,
    # 2 detect sfx. Channels 3+ belong to ShotAudioMixer, which mixes per-event
    # gain × master volume itself — stamping them here would clobber those gains.
    _OWNED_CHANNELS = (0, 1, 2)

    def _apply_volume(self) -> None:
        """Push the master volume onto the channels this engine owns (channel volume
        persists across plays, so this covers current and future narration/sfx
        playback) and broadcast it so the stem mixer can rescale its channels."""
        if pygame.mixer.get_init():
            n = pygame.mixer.get_num_channels()
            for i in self._OWNED_CHANNELS:
                if i < n:
                    pygame.mixer.Channel(i).set_volume(self._volume)
        self.event_bus.emit("master_volume", {"volume": self._volume})

    def set_volume(self, v: float) -> None:
        self._volume = max(0.0, min(1.0, v))
        self._apply_volume()

    def adjust_volume(self, delta: float) -> None:
        self.set_volume(self._volume + delta)

    def handle_volume_click(self, pos) -> bool:
        """Set volume from a mouse position over the pause-menu slider. Returns True
        if the position hit the slider (so the caller can treat it as consumed)."""
        rect = self._volume_slider_rect
        if rect is None:
            return False
        hit = rect.inflate(20, 24)   # generous grab area around the thin bar
        if not hit.collidepoint(pos):
            return False
        frac = (pos[0] - rect.x) / max(1, rect.w)
        self.set_volume(frac)
        return True

    # ------------------------------------------------------------------
    # Pause / resume
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Freeze frame advance. Idempotent."""
        if self._paused:
            return
        self._paused = True
        self._pause_started = time.monotonic()

    def resume(self) -> None:
        """Resume playback, shifting every time anchor forward by the paused
        duration so the current frame, segment loop and flash continue seamlessly."""
        if not self._paused:
            return
        now = time.monotonic()
        delta = now - self._pause_started
        self._playback_start_time += delta
        self._seg_anchor          += delta
        # Flashes fired and consumed while paused (tutorial success flashes render
        # in real time on the paused branch) must not be shifted into the future,
        # or they'd replay after resume. Only shift a flash that is still live.
        if self._flash_until > now:
            self._flash_start += delta
            self._flash_until += delta
        self._last_frame_time     += delta
        self._paused = False

    def _draw_paused_overlay(self) -> None:
        """Re-blit the held frame and a dimmed PAUSED banner over it."""
        if self._frames and 0 <= self._frame_index < len(self._frames):
            self._screen.blit(self._frames[self._frame_index], (0, 0))
        else:
            self._screen.fill((0, 0, 0))

        sw, sh = self._screen.get_size()
        veil = pygame.Surface((sw, sh), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 120))
        self._screen.blit(veil, (0, 0))

        if self._font:
            title = self._font.render("|| PAUSED", True, (255, 255, 255))
            self._screen.blit(title, ((sw - title.get_width()) // 2, sh // 2 - 70))
        if self._small_font:
            hint = self._small_font.render(
                "Space: resume    S: skip prologue    K: skeleton panel    "
                "D: debug    F: fullscreen", True, (200, 200, 200))
            self._screen.blit(hint, ((sw - hint.get_width()) // 2, sh // 2 - 34))

        self._draw_volume_slider(sw, sh)

    def _draw_volume_slider(self, sw: int, sh: int) -> None:
        """Volume slider for the pause menu. Up/Down adjust; the bar is drag-clickable."""
        bar_w, bar_h = 360, 8
        bx = (sw - bar_w) // 2
        by = sh // 2 + 20
        rect = pygame.Rect(bx, by, bar_w, bar_h)
        self._volume_slider_rect = rect

        if self._small_font:
            label = self._small_font.render(
                f"Volume  {int(round(self._volume * 100))}%   (Up / Down)", True, (220, 220, 220))
            self._screen.blit(label, ((sw - label.get_width()) // 2, by - 26))

        # Track
        pygame.draw.rect(self._screen, (90, 90, 90), rect, border_radius=4)
        # Fill
        fill_w = int(bar_w * self._volume)
        if fill_w > 0:
            pygame.draw.rect(self._screen, (90, 200, 120),
                             pygame.Rect(bx, by, fill_w, bar_h), border_radius=4)
        # Knob
        kx = bx + fill_w
        pygame.draw.circle(self._screen, (240, 240, 240), (kx, by + bar_h // 2), 9)
        pygame.draw.circle(self._screen, (60, 60, 60), (kx, by + bar_h // 2), 9, 2)

    def update(self, landmark_data=None, handedness_data=None,
               pose_data=None,
               gesture_debug: dict | None = None,
               voice_debug: dict | None = None,
               narration_debug: dict | None = None,
               tutorial_card: dict | None = None):
        if not self._screen:
            return

        # Tutorial runs while the shot player (and its time anchors) stay paused:
        # draw the code-rendered card, the live hand cursors and the success flash
        # in real time, and nothing else advances.
        if self._paused and tutorial_card is not None:
            self._draw_tutorial_card(tutorial_card)
            now = time.monotonic()
            if now < self._flash_until and self._flash_overlay is not None:
                duration = self._flash_until - self._flash_start
                progress = (now - self._flash_start) / duration if duration > 0 else 1.0
                alpha = int(self._flash_alpha * math.sin(progress * math.pi))
                self._flash_overlay.fill((*self._flash_color, max(0, min(255, alpha))))
                self._screen.blit(self._flash_overlay, (0, 0))
            self._draw_hand_cursors(landmark_data, handedness_data, pose_data, gesture_debug)
            if self._debug or self._show_skeleton:
                if landmark_data or pose_data:
                    self._draw_hand_mini_panel(landmark_data, handedness_data, pose_data)
            pygame.display.flip()
            return

        # While paused, hold the current frame and overlay a PAUSED banner.
        # No time-based frame advance, no events emitted.
        if self._paused:
            self._draw_paused_overlay()
            pygame.display.flip()
            return

        self._landmark_data = landmark_data
        self._pose_data = pose_data
        now = time.monotonic()

        # ── Swap in the incoming shot once the disk-backed cache can serve it. The
        #    FrameView converts frames to Surfaces on demand (LRU-bounded), so RAM
        #    stays flat regardless of shot length. Until ready, the previous shot's
        #    last frame stays frozen on screen.
        self._service_loading()

        playing = bool(self._frames) and self._loading_dir is None

        if playing:
            if self._seg_start is not None:
                # FSM segment-constrained playback: lock to a named frame range.
                # Frames are guaranteed fully loaded here (a shot only begins playing
                # after its whole frame set is cached), so no partial-load handling.
                seg_len  = max(1, self._seg_end - self._seg_start + 1)
                elapsed  = now - self._seg_anchor
                raw_local = int(elapsed * self._fps)
                seg_start_before = self._seg_start
                if self._seg_loop:
                    local = raw_local % seg_len
                else:
                    local = min(raw_local, seg_len - 1)
                    if not self._seg_done and raw_local >= seg_len:
                        self._seg_done = True
                        # How far past the boundary this tick landed. The next
                        # play_segment consumes it so chained play-through
                        # segments don't accumulate lag against the shot audio.
                        self._seg_overshoot = min(max(0.0, elapsed - seg_len / self._fps), 0.5)
                        print(f"[RenderEngine] segment_playback_done "
                              f"[{self._seg_start}–{self._seg_end}] "
                              f"overshoot={self._seg_overshoot*1000:.0f}ms")
                        self.event_bus.emit("segment_playback_done", {})
                # The emit above can synchronously start a NEW segment (FSM advance)
                # or clear playback (shot advance). In that case `local` (and seg
                # bounds) are stale — applying _seg_start + local would land on a
                # frame from elsewhere in the shot and flash for one tick. Detect the
                # change and position at the new segment's first frame instead.
                if self._seg_start is None or not self._frames:
                    pass  # shot advanced / frames cleared — nothing to index
                elif self._seg_start != seg_start_before:
                    self._frame_index = min(self._seg_start, len(self._frames) - 1)
                else:
                    self._frame_index = min(self._seg_start + local,
                                            len(self._frames) - 1)

            else:
                # Standard time-based frame index: stays in sync with audio.
                elapsed = now - self._playback_start_time
                target = max(0, int(elapsed * self._fps))
                self._frame_index = min(target, len(self._frames) - 1)

                # Frame-gated OI window: emit events when _frame_index crosses thresholds
                if self._oi_frame_start is not None and not self._oi_window_open:
                    if self._frame_index >= self._oi_frame_start:
                        self._oi_window_open = True
                        print(f"[RenderEngine] frame_window_enter at frame {self._frame_index} "
                              f"(gate={self._oi_frame_start}-{self._oi_frame_end})")
                        self.event_bus.emit("frame_window_enter", {})
                if self._oi_frame_end is not None and self._oi_window_open:
                    if self._frame_index >= self._oi_frame_end:
                        self._oi_window_open = False
                        self._oi_frame_start = None   # clear so we don't re-trigger on clamped last frame
                        self._oi_frame_end   = None
                        print(f"[RenderEngine] frame_window_exit at frame {self._frame_index}")
                        self.event_bus.emit("frame_window_exit", {})

            if self._frames and 0 <= self._frame_index < len(self._frames):
                self._screen.blit(self._frames[self._frame_index], (0, 0))
            else:
                self._screen.fill((0, 0, 0))

        elif self._frames:
            # Loading the next shot: freeze on the current (last shown) frame and
            # wait until it's fully preloaded — do not advance, do not show stand-ins.
            self._frame_index = max(0, min(self._frame_index, len(self._frames) - 1))
            self._screen.blit(self._frames[self._frame_index], (0, 0))
            if narration_debug and self._loading_dir is not None:
                self._draw_loading_indicator()

        else:
            # No previous frame to hold (cold start) — black until first shot loads.
            self._screen.fill((0, 0, 0))
            if self._loading_dir is not None:
                self._draw_loading_indicator()
            elif narration_debug and narration_debug.get("waiting_id"):
                self._draw_wait_for_cg_placeholder(narration_debug.get("waiting_id"))

        # OI flash overlay — fade in then fade out using a sine curve.
        # Reuses the pre-allocated full-screen surface (no per-frame alloc).
        if now < self._flash_until and self._flash_overlay is not None:
            duration = self._flash_until - self._flash_start
            progress = (now - self._flash_start) / duration if duration > 0 else 1.0
            import math as _math
            alpha = int(self._flash_alpha * _math.sin(progress * _math.pi))
            alpha = max(0, min(255, alpha))
            self._flash_overlay.fill((*self._flash_color, alpha))
            self._screen.blit(self._flash_overlay, (0, 0))

        # Player-facing interaction indicator (target ring / draw arrow) and the
        # illustrated hand cursors — always on, not debug-gated.
        self._draw_interaction_indicator(gesture_debug)
        self._draw_hand_cursors(landmark_data, handedness_data, pose_data, gesture_debug)

        if self._debug:
            self._draw_oi_target_flag(gesture_debug)
            self._draw_debug_panel(landmark_data, gesture_debug, voice_debug, narration_debug)
        if self._debug or self._show_skeleton:
            # Skeleton preview: debug overlay OR the standalone pause-menu option.
            if landmark_data or pose_data:
                self._draw_hand_mini_panel(landmark_data, handedness_data, pose_data)

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Shot / frame loading
    # ------------------------------------------------------------------

    def attach_cache(self, shots) -> None:
        """Start the continuous look-ahead frame cache over every shot with art.

        Call once after the sequence is loaded. The cache decodes all shots that
        have frames on disk in the background, prioritising forward from whatever
        shot becomes current, so later shots are fully preloaded before we reach
        them. assets_pending (placeholder) shots are skipped — they have no art.
        """
        dirs = []
        for s in shots:
            fd = getattr(s, "frames_dir", None)
            if fd and not getattr(s, "assets_pending", True):
                dirs.append(Path(fd))
        keep_ahead = self.config.get("frame_cache_keep_ahead", 2)
        self._cache = FrameCacheManager(self._get_resolution(), keep_ahead=keep_ahead)
        self._cache.start(dirs)

    def _convert_surface(self, data, size) -> pygame.Surface:
        """Raw RGB bytes → display-format Surface (.convert() makes blits a memcpy)."""
        return pygame.image.fromstring(data, size, "RGB").convert()

    def _service_loading(self) -> None:
        """Swap in the incoming shot once the cache can serve its frames.

        No upfront conversion: as soon as the cache is ready (mmap pack live, or the
        decode fallback's file list available), we point _frames at a FrameView that
        converts frames to Surfaces on demand and keeps only an LRU window in RAM.
        Until then the previous shot's last frame stays frozen on screen.
        """
        if self._loading_dir is None or self._cache is None:
            return
        if not self._cache.is_ready(self._loading_dir):
            return   # pack still building / paths not listed — hold previous frame

        count = self._cache.frame_count(self._loading_dir) or 0
        if count <= 0:
            self._loading_dir = None
            return

        self._frames = FrameView(self._cache, self._loading_dir, count,
                                  self._convert_surface)
        self._frame_index         = 0
        self._playback_start_time = time.monotonic()
        self._loading_dir         = None
        self._begin_audio()
        self.event_bus.emit("shot_frames_ready", {})
        print(f"[RenderEngine] shot ready: {count} frames "
              f"({'pack' if self._cache.pack_ready(self._current_frames_dir) else 'decode fallback'})")

    def _begin_audio(self) -> None:
        if self._pending_audio:
            try:
                sound = pygame.mixer.Sound(self._pending_audio)
                ch = pygame.mixer.Channel(0)
                ch.set_volume(self._volume)
                ch.play(sound)
            except Exception as exc:
                print(f"[RenderEngine] audio load failed: {exc}")
            self._pending_audio = None

    def _on_shot_load(self, data: dict):
        shot = data.get("shot")
        if shot is None:
            return
        self._fps = getattr(shot, "fps", 24)
        self._loading_kind = getattr(shot, "kind", "playback")
        self._last_frame_time = time.monotonic()
        audio_file = getattr(shot, "audio_file", None)
        self._pending_audio = str(audio_file) if audio_file else None

        # Always reset interaction state regardless of whether frames exist.
        # If a synchronous shot transition fires mid-frame (e.g. confirm segment
        # ends and _advance() emits shot_load before update() returns), these must
        # be cleared so the render loop doesn't continue executing a stale segment.
        self._oi_frame_start    = None
        self._oi_frame_end      = None
        self._oi_window_open    = False
        self._seg_start         = None
        self._seg_end           = None
        self._seg_done          = False
        self._seg_overshoot     = 0.0

        frames_dir = getattr(shot, "frames_dir", None)
        if frames_dir is None or getattr(shot, "assets_pending", True):
            # Placeholder shot with no art — clear the screen, nothing to load.
            self._frames = []
            self._frame_index = 0
            self._current_frames_dir = None
            self._loading_dir = None
            return

        # Keep the previous shot's frames on screen (frozen on the last frame) while
        # the incoming shot preloads — _service_loading swaps it in once complete.
        self._current_frames_dir = Path(frames_dir)
        self._loading_dir = Path(frames_dir)
        if self._cache is not None:
            self._cache.prioritize(self._loading_dir)

    def _on_prefetch_shot(self, data: dict):
        """No-op: the look-ahead cache already preloads every shot forward of the
        current one. Retained because ShotSequencePlayer still emits prefetch_shot."""
        return

    def _get_resolution(self) -> tuple[int, int]:
        display_cfg = self.config.get("_profile", {}).get("display", {})
        w, h = display_cfg.get("resolution") or self.config.get("resolution", [1920, 1080])
        return (w, h)

    def _on_oi_flash(self, data: dict):
        self._flash_color = data.get("color", (0, 255, 80))
        duration_ms = data.get("duration_ms", 800)
        self._flash_start = time.monotonic()
        self._flash_until = self._flash_start + duration_ms / 1000.0

    def _on_play_sfx(self, data: dict):
        path = data.get("path")
        if not path or not os.path.exists(path):
            return
        # channel 1: stroke / on_enter_sfx (can be long — carries tail audio).
        # channel 2: OI reaction sfx (detect.mp3) — overlays without cutting ch.1.
        channel = data.get("channel", 1)
        try:
            sound = pygame.mixer.Sound(path)
            ch = pygame.mixer.Channel(channel)
            ch.set_volume(self._volume)
            ch.play(sound)
        except Exception as exc:
            print(f"[RenderEngine] SFX load failed: {exc}")

    def _on_set_frame_window(self, data: dict):
        self._oi_frame_start = data.get("start")
        self._oi_frame_end   = data.get("end")
        self._oi_window_open = False

    def _on_play_segment(self, data: dict):
        self._seg_start  = data.get("start")
        self._seg_end    = data.get("end")
        self._seg_loop   = data.get("loop", True)
        # Consume the previous segment's boundary overshoot so back-to-back
        # play-through segments track the continuous audio instead of drifting
        # one tick later per transition. Loop segments hold indefinitely, so
        # exact phase doesn't matter there — start them clean.
        carry = 0.0 if self._seg_loop else self._seg_overshoot
        self._seg_overshoot = 0.0
        self._seg_anchor = time.monotonic() - carry
        self._seg_done   = False
        print(f"[RenderEngine] play_segment [{self._seg_start}-{self._seg_end}]  "
              f"loop={self._seg_loop}" + (f"  carry={carry*1000:.0f}ms" if carry else ""))

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_oi_target_flag(self, gesture_debug: dict | None) -> None:
        """Draw a flag-style overlay for the active OI detection zone (debug mode only).

        For region_rect: a labelled cyan rectangle with a flag tab at the top-left.
        For directions:  compass arrows radiating from the bottom-centre of the screen.
        Rect x-coordinates are mirrored to match display orientation (same transform
        as the fingertip cursor: display_x = 1 - raw_x).
        """
        if not gesture_debug:
            return
        params = gesture_debug.get("active_oi_params")
        if not params:
            return

        sw, sh = self._screen.get_size()
        COLOR   = (80, 220, 255)    # cyan — distinct from CG yellow
        FILL_A  = 35                # fill alpha
        BORDER  = 2

        rect = params.get("region_rect")
        if rect:
            # Mirror x so the rect aligns with the fingertip cursor on screen.
            rx0 = int((1.0 - rect["x"] - rect["w"]) * sw)
            rx1 = int((1.0 - rect["x"]) * sw)
            ry0 = int(rect["y"] * sh)
            ry1 = int((rect["y"] + rect["h"]) * sh)
            rw  = max(1, rx1 - rx0)
            rh  = max(1, ry1 - ry0)

            # Semi-transparent fill
            fill = pygame.Surface((rw, rh), pygame.SRCALPHA)
            fill.fill((*COLOR, FILL_A))
            self._screen.blit(fill, (rx0, ry0))

            # Border
            pygame.draw.rect(self._screen, COLOR, (rx0, ry0, rw, rh), BORDER)

            # Flag tab — small filled rectangle with label, pinned to top-left corner
            label_text = gesture_debug.get("active_oi", "OI target")
            if self._small_font:
                lbl = self._small_font.render(label_text, True, (0, 0, 0))
                tab_w = lbl.get_width() + 10
                tab_h = lbl.get_height() + 6
                tab_surf = pygame.Surface((tab_w, tab_h), pygame.SRCALPHA)
                tab_surf.fill((*COLOR, 220))
                tab_surf.blit(lbl, (5, 3))
                self._screen.blit(tab_surf, (rx0, ry0 - tab_h))

        directions = params.get("directions")
        if directions:
            # Draw directional arrows from a fixed anchor point at the bottom-centre.
            _DIR_VEC = {
                "up":         ( 0, -1), "down":       ( 0,  1),
                "left":       (-1,  0), "right":      ( 1,  0),
                "up_left":    (-1, -1), "up_right":   ( 1, -1),
                "down_left":  (-1,  1), "down_right": ( 1,  1),
            }
            import math as _math
            ax = sw // 2
            ay = sh - 80          # near bottom-centre
            arrow_len = 50

            for d in directions:
                vec = _DIR_VEC.get(d)
                if not vec:
                    continue
                mag = _math.hypot(*vec)
                dx = int(vec[0] / mag * arrow_len)
                dy = int(vec[1] / mag * arrow_len)
                ex, ey = ax + dx, ay + dy
                pygame.draw.line(self._screen, COLOR, (ax, ay), (ex, ey), 3)
                # Arrowhead: two short lines at ~135° from the direction
                head = 12
                ang  = _math.atan2(dy, dx)
                for side in (0.6, -0.6):
                    hx = int(ex - head * _math.cos(ang + side * _math.pi))
                    hy = int(ey - head * _math.sin(ang + side * _math.pi))
                    pygame.draw.line(self._screen, COLOR, (ex, ey), (hx, hy), 2)

            if self._small_font and directions:
                label = "OI: " + "/".join(directions)
                lbl = self._small_font.render(label, True, COLOR)
                self._screen.blit(lbl, (ax - lbl.get_width() // 2, ay + arrow_len + 6))

    # ------------------------------------------------------------------
    # Hand cursors + player-facing interaction indicators (July 2026)
    # ------------------------------------------------------------------

    # Interaction type -> cursor treatment
    _POINT_TYPES = {"directional_point", "point_target_held", "forward_point",
                    "point_region"}
    _KNOCK_TYPES = {"rhythm_bilateral"}
    # Draw strokes show the big direction arrow instead of any tracked cursor —
    # the tracking visuals were confusing players mid-trace (Scene 4/5 punch list).
    _NO_CURSOR_TYPES = {"directional_draw"}

    _DOT_COLORS = {"L": (60, 220, 90), "R": (70, 160, 255)}   # green / blue

    # Pose landmark indices per side: wrist + the pose "hand point" (index).
    _POSE_SIDE_POINTS = {"L": (15, 19), "R": (16, 20)}

    def _draw_hand_cursors(self, landmark_data, handedness_data=None,
                           pose_data=None, gesture_debug: dict | None = None):
        """Illustrated hand cursors (assets/hand_icons/), POSE-DRIVEN since the
        July 2026 pose-only rework: position comes from the Pose index landmark
        (19/20) when visible, else the wrist (15/16); side labels are inherent
        to the skeleton. Icon picked by the active interaction — knock fists
        during knock windows, pointing hand during point windows, open hand
        during grab windows, plain green (L) / blue (R) dots when no window is
        open. Last-seen position persists through pose dropouts. (landmark_data
        / handedness_data are the legacy Hands slots — always None now.)"""
        if not self._screen:
            return
        gd = gesture_debug or {}
        active_type = gd.get("active_type")
        if active_type in self._NO_CURSOR_TYPES:
            return

        mode = "dots"
        if active_type and not gd.get("input_locked"):
            if active_type in self._POINT_TYPES:
                mode = "point"
            elif active_type in self._KNOCK_TYPES:
                mode = "knock"
            else:
                mode = "grab"

        w, h = self._screen.get_size()
        now = time.monotonic()

        if pose_data:
            for side, (wrist_i, index_i) in self._POSE_SIDE_POINTS.items():
                lm = None
                # Prefer the pose index (hand point); fall back to the wrist.
                for idx in (index_i, wrist_i):
                    if idx < len(pose_data):
                        cand = pose_data[idx]
                        if getattr(cand, "visibility", 1.0) >= 0.5:
                            lm = cand
                            break
                if lm is None:
                    continue
                state = self._hand_cursor_state[side]
                state["pos"] = (int((1 - lm.x) * w), int(lm.y * h))
                state["t"] = now

        for side, state in self._hand_cursor_state.items():
            if state["pos"] is None:
                continue
            age = now - state["t"]
            if age > 4.0:
                continue   # long gone — drop the ghost cursor
            x, y = state["pos"]

            if mode == "dots":
                color = self._DOT_COLORS.get(side, (220, 220, 220))
                pygame.draw.circle(self._screen, (0, 0, 0), (x, y), 12, 0)
                pygame.draw.circle(self._screen, color, (x, y), 9, 0)
                pygame.draw.circle(self._screen, (255, 255, 255), (x, y), 12, 2)
                continue

            if mode == "knock":
                icon_shape = "knock"
            elif mode == "point":
                icon_shape = "point"
            else:
                icon_shape = state["shape"]   # "open" | "fist"
            icon = self._hand_icons.get(f"{icon_shape}_{side.lower()}")
            if icon is None:
                color = self._DOT_COLORS.get(side, (220, 220, 220))
                pygame.draw.circle(self._screen, color, (x, y), 10)
                continue
            if age > 0.5:
                icon = icon.copy()
                icon.set_alpha(120)   # stale — hand not currently tracked
            self._screen.blit(icon, (x - icon.get_width() // 2,
                                     y - icon.get_height() // 2))

    # forward_point's named regions, in PLAYER/screen space (already mirrored).
    _NAMED_REGIONS = {
        "top_left_quadrant":  {"x": 0.0,  "y": 0.0,  "w": 0.5, "h": 0.5},
        "top_right_quadrant": {"x": 0.5,  "y": 0.0,  "w": 0.5, "h": 0.5},
        "lower_third":        {"x": 0.0,  "y": 0.67, "w": 1.0, "h": 0.33},
        "center":             {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5},
    }

    def _draw_interaction_indicator(self, gesture_debug: dict | None) -> None:
        """Player-facing hint for the active window (always on, unlike the debug
        target flag): a pulsing ring around point targets, a big pulsing arrow
        for draw strokes. Replaces the confusing tracked-hand visuals as the
        'simple on-screen indication' from the playtest punch list."""
        if not self._screen:
            return
        gd = gesture_debug or {}
        active_type = gd.get("active_type")
        if not active_type or gd.get("input_locked"):
            return
        params = gd.get("active_params") or {}
        sw, sh = self._screen.get_size()
        now = time.monotonic()
        pulse = 0.5 + 0.5 * math.sin(now * 4.0)   # 0..1

        if active_type == "directional_draw":
            direction = params.get("direction", "right")
            _DIR_VEC = {
                "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
                "up_left": (-1, -1), "up_right": (1, -1),
                "down_left": (-1, 1), "down_right": (1, 1),
            }
            vec = _DIR_VEC.get(direction, (1, 0))
            mag = math.hypot(*vec)
            ind = params.get("indicator_xy") or (0.5, 0.42)
            cx, cy = int(ind[0] * sw), int(ind[1] * sh)
            length = int(sh * (0.10 + 0.02 * pulse))
            dx, dy = vec[0] / mag, vec[1] / mag
            sx, sy = int(cx - dx * length), int(cy - dy * length)
            ex, ey = int(cx + dx * length), int(cy + dy * length)
            color = (255, 210, 90)
            # Start dot ("begin your stroke here") + shaft + arrowhead
            pygame.draw.circle(self._screen, (0, 0, 0), (sx, sy), 14)
            pygame.draw.circle(self._screen, color, (sx, sy), 10 + int(3 * pulse))
            pygame.draw.line(self._screen, (0, 0, 0), (sx, sy), (ex, ey), 12)
            pygame.draw.line(self._screen, color, (sx, sy), (ex, ey), 8)
            head = 26
            ang = math.atan2(ey - sy, ex - sx)
            for side_a in (0.65, -0.65):
                hx = int(ex - head * math.cos(ang + side_a * math.pi))
                hy = int(ey - head * math.sin(ang + side_a * math.pi))
                pygame.draw.line(self._screen, (0, 0, 0), (ex, ey), (hx, hy), 12)
                pygame.draw.line(self._screen, color, (ex, ey), (hx, hy), 8)
            return

        if active_type in ("point_target_held", "forward_point"):
            # Resolve the target rect. point_target_held rects are RAW camera
            # space (mirror to screen); forward_point named regions are already
            # player/screen space.
            rect = None
            if active_type == "point_target_held":
                raw = params.get("region_rect")
                if raw:
                    rect = {"x": 1.0 - raw["x"] - raw["w"], "y": raw["y"],
                            "w": raw["w"], "h": raw["h"]}
            else:
                rect = self._NAMED_REGIONS.get(params.get("target_region", ""))
            if not rect:
                return
            cx = int((rect["x"] + rect["w"] / 2) * sw)
            cy = int((rect["y"] + rect["h"] / 2) * sh)
            rx = max(30, int(rect["w"] * sw / 2))
            ry = max(30, int(rect["h"] * sh / 2))
            # Soft pulsing ellipse ring around the target area
            ring = pygame.Surface((rx * 2 + 20, ry * 2 + 20), pygame.SRCALPHA)
            alpha = int(90 + 100 * pulse)
            pygame.draw.ellipse(ring, (255, 210, 90, alpha),
                                pygame.Rect(4, 4, rx * 2 + 12, ry * 2 + 12),
                                width=5 + int(3 * pulse))
            self._screen.blit(ring, (cx - rx - 10, cy - ry - 10))

    def _draw_tutorial_card(self, card: dict) -> None:
        """Full-screen code-rendered tutorial/calibration card. The card dict
        comes from TutorialEngine.card_info()."""
        sw, sh = self._screen.get_size()
        self._screen.fill((12, 14, 24))

        title_font = getattr(self, "_tut_title_font", None)
        if title_font is None:
            self._tut_title_font  = pygame.font.SysFont("arial", 52, bold=True)
            self._tut_prompt_font = pygame.font.SysFont("arial", 30)
            title_font = self._tut_title_font

        title = card.get("title", "")
        if title:
            surf = title_font.render(title, True, (255, 235, 200))
            self._screen.blit(surf, ((sw - surf.get_width()) // 2, int(sh * 0.14)))

        prompt = card.get("prompt", "")
        if prompt:
            for li, line in enumerate(self._wrap_text(prompt, self._tut_prompt_font,
                                                      int(sw * 0.7))):
                surf = self._tut_prompt_font.render(line, True, (230, 230, 230))
                self._screen.blit(surf, ((sw - surf.get_width()) // 2,
                                         int(sh * 0.26) + li * 40))

        # Optional on-card target box (player-space rect) for the pointing steps.
        rect = card.get("target_rect")
        if rect:
            pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 4.0)
            rx0 = int(rect["x"] * sw)
            ry0 = int(rect["y"] * sh)
            rw  = int(rect["w"] * sw)
            rh  = int(rect["h"] * sh)
            fill = pygame.Surface((rw, rh), pygame.SRCALPHA)
            fill.fill((255, 210, 90, 30 + int(30 * pulse)))
            self._screen.blit(fill, (rx0, ry0))
            pygame.draw.rect(self._screen, (255, 210, 90),
                             (rx0, ry0, rw, rh), 4 + int(2 * pulse))

        # Optional big demo icon (e.g. the open-hand illustration).
        icon_key = card.get("icon")
        if icon_key:
            icon = self._hand_icons.get(icon_key)
            if icon:
                big = pygame.transform.smoothscale(
                    icon, (icon.get_width() * 2, icon.get_height() * 2))
                self._screen.blit(big, ((sw - big.get_width()) // 2,
                                        int(sh * 0.44)))

        if self._small_font:
            step = card.get("step")
            total = card.get("total")
            if step is not None:
                surf = self._small_font.render(f"Step {step} of {total}", True,
                                               (170, 170, 170))
                self._screen.blit(surf, ((sw - surf.get_width()) // 2, int(sh * 0.86)))
            hint = self._small_font.render("S: skip tutorial", True, (120, 120, 120))
            self._screen.blit(hint, ((sw - hint.get_width()) // 2, int(sh * 0.90)))

    # MediaPipe Pose: upper-body skeleton for the debug panel.
    _POSE_CONNECTIONS = [
        (11, 12),                    # shoulders
        (11, 13), (13, 15),          # left arm
        (12, 14), (14, 16),          # right arm
        (11, 23), (12, 24), (23, 24),# torso
        (0, 11), (0, 12),            # neck-ish (nose to shoulders)
    ]
    _POSE_KEY_POINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]  # nose, ears, etc.

    def _draw_hand_mini_panel(self, landmark_data, handedness_data=None, pose_data=None):
        """Pose-skeleton preview in bottom-right corner — keeps the main screen
        clean. (landmark_data / handedness_data are the legacy Hands slots —
        always None since the pose-only rework.)"""
        if not self._screen:
            return
        sw, sh = self._screen.get_size()
        panel_w, panel_h = 240, 180
        pad = 10
        px = sw - panel_w - 10
        py = sh - panel_h - 10

        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 210))
        self._screen.blit(bg, (px, py))

        inner_w = panel_w - pad * 2
        inner_h = panel_h - pad * 2

        if pose_data:
            ppts = [
                (px + pad + int((1 - lm.x) * inner_w),
                 py + pad + int(lm.y * inner_h))
                for lm in pose_data
            ]
            def _vis(idx):
                return 0 <= idx < len(ppts)
            for a, b in self._POSE_CONNECTIONS:
                if _vis(a) and _vis(b):
                    pygame.draw.line(self._screen, (255, 170, 40), ppts[a], ppts[b], 2)
            for idx in self._POSE_KEY_POINTS:
                if _vis(idx):
                    pygame.draw.circle(self._screen, (255, 210, 90), ppts[idx], 3)
            # Pose hand points (pinky/index/thumb), green L / blue R — the
            # closest thing to a hand readout the pose-only runtime has.
            for color, idxs in (((0, 255, 80), (17, 19, 21)),
                                ((60, 200, 255), (18, 20, 22))):
                for idx in idxs:
                    if _vis(idx) and getattr(pose_data[idx], "visibility", 1.0) >= 0.5:
                        pygame.draw.circle(self._screen, color, ppts[idx], 3)
            if self._small_font:
                tag = self._small_font.render("POSE", True, (255, 190, 70))
                self._screen.blit(tag, (px + 4, py + 2))

    def _draw_loading_indicator(self) -> None:
        """Small corner readout while the incoming shot's frame pack is building."""
        if not self._screen or not self._small_font or self._cache is None:
            return
        name  = self._loading_dir.parent.name if self._loading_dir else "?"
        total = self._cache.frame_count(self._loading_dir) if self._loading_dir else None
        if total:
            txt = f"building frame pack: {name}  ({total} frames)..."
        else:
            txt = f"preparing {name}..."
        surf = self._small_font.render(txt, True, (180, 180, 180))
        self._screen.blit(surf, (20, self._screen.get_height() - 30))

    def _draw_wait_for_cg_placeholder(self, waiting_id: str):
        """Placeholder card when WAIT_FOR_CG is active but no storyboard frame exists."""
        if not self._screen or not self._font:
            return
        sw, sh = self._screen.get_size()
        card_w, card_h = 640, 220
        cx = (sw - card_w) // 2
        cy = (sh - card_h) // 2

        card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card.fill((20, 10, 0, 230))
        pygame.draw.rect(card, (180, 120, 40), (0, 0, card_w, card_h), 2)
        self._screen.blit(card, (cx, cy))

        title = self._font.render("[ AWAITING GESTURE ]", True, (255, 200, 80))
        id_surf = self._font.render(waiting_id, True, (200, 160, 255))
        mark = self._small_font.render("PLACEHOLDER — NO STORYBOARD PANEL", True, (100, 80, 60))
        self._screen.blit(title, (cx + (card_w - title.get_width()) // 2, cy + 20))
        self._screen.blit(id_surf, (cx + (card_w - id_surf.get_width()) // 2, cy + 80))
        self._screen.blit(mark, (cx + (card_w - mark.get_width()) // 2, cy + 180))

    def _draw_debug_panel(self, landmark_data, gesture_debug: dict | None,
                          voice_debug: dict | None = None,
                          narration_debug: dict | None = None):
        if not self._font:
            return
        w, h = self._screen.get_size()

        lines = []
        if gesture_debug:
            pose_status = gesture_debug.get("pose_status", "NONE")
            pose_color = ((255, 170, 40) if pose_status == "OK"
                          else (180, 120, 40) if pose_status.startswith("STALE")
                          else (120, 90, 50))
            lines.append((f"POSE: {pose_status}", pose_color))

            cg = gesture_debug.get("active_cg")
            oi = gesture_debug.get("active_oi")
            lines.append((f"CG: {cg or '--'}", (255, 220, 0) if cg else (140, 140, 140)))
            lines.append((f"OI: {oi or '--'}", (100, 200, 255) if oi else (140, 140, 140)))
            point_dir = gesture_debug.get("point_dir")
            if point_dir:
                lines.append((f"POINT: {point_dir}", (255, 255, 120)))
            last = gesture_debug.get("last_fired")
            if last:
                lines.append((f"FIRED: {last}", (255, 140, 0)))

        if narration_debug is not None:
            lines.append(("", (0, 0, 0)))
            # SHOT: overlay — populated by ShotSequencePlayer.debug_info()
            shot_id      = narration_debug.get("shot_id")
            shot_state   = narration_debug.get("shot_state")
            shot_elapsed = narration_debug.get("shot_elapsed_s")
            if shot_id is not None:
                elapsed_str = f" ({shot_elapsed}s)" if shot_elapsed is not None else ""
                lines.append((f"SHOT: {shot_id} [{shot_state or '?'}]{elapsed_str}",
                               (160, 255, 200)))
            nstate = narration_debug.get("state", "?")
            cue = narration_debug.get("cue_code") or "--"
            waiting_id = narration_debug.get("waiting_id")
            stroke_info = narration_debug.get("stroke_info")
            if waiting_id:
                lines.append((f"WAIT: {waiting_id}", (255, 80, 255)))
                lines.append((f"CUE: {cue}", (200, 160, 255)))
                if stroke_info:
                    lines.append((f"STROKE: {stroke_info}", (255, 180, 0)))
            else:
                lines.append((f"NAR: {nstate.upper()}", (140, 140, 140)))
            rec_pts = (gesture_debug or {}).get("recording_pts", 0)
            if rec_pts > 0:
                lines.append((f"REC: {rec_pts} pts", (255, 200, 0)))

        if voice_debug is not None:
            lines.append(("", (0, 0, 0)))
            mic_on = voice_debug.get("mic_active", False)
            lines.append(("MIC: ON" if mic_on else "MIC: OFF",
                          (0, 255, 80) if mic_on else (255, 60, 60)))
            vi_window = voice_debug.get("active_window")
            lines.append((f"VI: {vi_window or '--'}",
                          (255, 220, 0) if vi_window else (140, 140, 140)))
            heard = voice_debug.get("last_heard")
            if heard:
                lines.append((f"HEARD: {heard}", (0, 220, 255)))

        line_h = 26
        pad = 8
        panel_w = 320
        panel_h = len(lines) * line_h + pad * 2
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        self._screen.blit(panel, (10, 10))

        for i, (text, color) in enumerate(lines):
            surf = self._font.render(text, True, color)
            self._screen.blit(surf, (10 + pad, 10 + pad + i * line_h))

    @staticmethod
    def _wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
        words = text.split()
        lines = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            if font.size(test)[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]
