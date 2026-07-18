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
import random
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
        # Debug overlay starts OFF regardless of profile/config; toggle it at
        # runtime with the D key (render.toggle_debug()).
        self._debug = False
        self._font: Optional[pygame.font.Font] = None
        self._small_font: Optional[pygame.font.Font] = None
        self._playback_start_time: float = 0.0
        self._pending_audio: Optional[str] = None
        self._current_frames_dir: Optional[Path] = None

        # Baked whole-file shot audio (audio.mp3) seek support: a play_segment
        # that lands far from the audio's current position (start-at-shot /
        # skip-prologue) restarts it at the matching offset via
        # pygame.mixer.music (Sound objects can't seek; the music stream can).
        self._audio_path:  Optional[str] = None   # the playing shot audio
        self._audio_pos0:  float = 0.0            # offset it started at (s)
        self._audio_epoch: float = 0.0            # monotonic time it started
        self._music_active: bool = False          # audio runs on mixer.music

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
        self._sound_cache: OrderedDict = OrderedDict()   # path -> decoded Sound (LRU)
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

        # Cursor visibility fade (July 2026): pose cursors are shown ONLY while
        # an interaction window is armed — alpha ramps in when the window opens
        # and out when it closes (the last window's icon persists through the
        # fade-out so it doesn't flip mid-animation).
        self._cursor_fade_alpha: float = 0.0
        self._cursor_fade_mode: str = "grab"
        self._cursor_fade_t: float = time.monotonic()

        # Star-trail particle layer (July 2026): tiny 4-point stars trail the
        # visitor's hands while a directional_draw window is armed. Sprites are
        # pre-rendered in init_display; particles are dicts with position,
        # velocity, birth time, lifetime and sprite index.
        self._star_sprites: list = []            # [size][rotation] -> Surface
        self._hand_glow: Optional[pygame.Surface] = None   # per-hand "pen" dot
        self._trail_particles: list = []
        self._trail_last_pos: dict = {"L": None, "R": None}
        self._fx_overlay: Optional[pygame.Surface] = None

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
        self._fx_overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        self._build_star_sprites()
        self._load_hand_icons()

    _STAR_COLOR = (255, 210, 90)   # the player-layer amber

    def _build_star_sprites(self) -> None:
        """Pre-render the 4-point star sprites used by the comet indicator and
        the hand star-trail: three sizes x two rotations, amber with a white-hot
        core. Rendered once; per-blit alpha is set with Surface.set_alpha."""
        self._star_sprites = []
        for size in (10, 15, 22):
            rots = []
            for rot_deg in (0.0, 22.5):
                pad = 2
                surf = pygame.Surface((size + pad * 2, size + pad * 2),
                                      pygame.SRCALPHA)
                cx = cy = size / 2 + pad
                r_out = size / 2
                r_in = r_out * 0.38
                pts = []
                for i in range(8):
                    ang = math.radians(rot_deg) + i * math.pi / 4 - math.pi / 2
                    r = r_out if i % 2 == 0 else r_in
                    pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
                pygame.draw.polygon(surf, self._STAR_COLOR, pts)
                pygame.draw.circle(surf, (255, 250, 235), (int(cx), int(cy)),
                                   max(1, int(r_in * 0.7)))
                rots.append(surf)
            self._star_sprites.append(rots)
        # The "pen" dot pinned to each hand while a draw window is armed —
        # same layered-glow language as the comet head.
        g = pygame.Surface((36, 36), pygame.SRCALPHA)
        pygame.draw.circle(g, (*self._STAR_COLOR, 45), (18, 18), 16)
        pygame.draw.circle(g, (*self._STAR_COLOR, 110), (18, 18), 10)
        pygame.draw.circle(g, (255, 250, 235, 230), (18, 18), 5)
        self._hand_glow = g

    def _load_hand_icons(self) -> None:
        """Load the illustrated hand cursors produced by scripts/prepare_hand_icons.py.
        Missing files degrade gracefully to the dot cursors."""
        # Frozen (PyInstaller) builds: __file__ lives inside _internal/, so
        # resolve assets/ next to BHR.exe instead (same pattern as main.py).
        import sys
        if getattr(sys, "frozen", False):
            app_root = Path(sys.executable).parent
        else:
            app_root = Path(__file__).resolve().parent.parent
        icons_dir = app_root / "assets" / "hand_icons"
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
            if self._music_active:
                pygame.mixer.music.set_volume(self._volume)
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
        # pygame.mixer.pause() (main's pause path) does not touch the music
        # stream — pause a seek-restarted shot audio explicitly.
        if self._music_active:
            try:
                pygame.mixer.music.pause()
            except Exception:
                pass

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
        self._audio_epoch += delta   # keep the audio-position estimate honest
        if self._music_active:
            try:
                pygame.mixer.music.unpause()
            except Exception:
                pass
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

    def update(self, pose_data=None,
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
            self._draw_hand_cursors(pose_data, gesture_debug)
            if (self._debug or self._show_skeleton) and pose_data:
                self._draw_hand_mini_panel(pose_data)
            pygame.display.flip()
            return

        # While paused, hold the current frame and overlay a PAUSED banner.
        # No time-based frame advance, no events emitted.
        if self._paused:
            self._draw_paused_overlay()
            pygame.display.flip()
            return

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

        # Player-facing layer — always on, not debug-gated: star trail under the
        # comet indicator, hand cursors on top.
        self._draw_star_trail(pose_data, gesture_debug)
        self._draw_interaction_indicator(gesture_debug)
        self._draw_hand_cursors(pose_data, gesture_debug)

        if self._debug:
            self._draw_oi_target_flag(gesture_debug)
            self._draw_debug_panel(gesture_debug, voice_debug, narration_debug)
        if (self._debug or self._show_skeleton) and pose_data:
            # Skeleton preview: debug overlay OR the standalone pause-menu option.
            self._draw_hand_mini_panel(pose_data)

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

    # Decoded Sound cache: detect.mp3 fires on every OI detection and shots
    # replay across the end-loop — decoding from disk on every play is waste.
    _SOUND_CACHE_MAX = 16

    def _load_sound(self, path) -> "pygame.mixer.Sound | None":
        key = str(path)
        sound = self._sound_cache.get(key)
        if sound is not None:
            self._sound_cache.move_to_end(key)
            return sound
        try:
            sound = pygame.mixer.Sound(key)
        except Exception as exc:
            print(f"[RenderEngine] audio load failed: {key}: {exc}")
            return None
        self._sound_cache[key] = sound
        while len(self._sound_cache) > self._SOUND_CACHE_MAX:
            self._sound_cache.popitem(last=False)
        return sound

    def _begin_audio(self) -> None:
        if self._pending_audio:
            sound = self._load_sound(self._pending_audio)
            if sound is not None:
                ch = pygame.mixer.Channel(0)
                ch.set_volume(self._volume)
                ch.play(sound)
                self._audio_path  = self._pending_audio
                self._audio_pos0  = 0.0
                self._audio_epoch = time.monotonic()
            self._pending_audio = None

    def _stop_music(self) -> None:
        if self._music_active:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            self._music_active = False

    def _seek_shot_audio(self, pos_s: float) -> None:
        """Restart the shot's baked audio.mp3 at pos_s seconds. Channel 0's
        Sound copy is stopped (it is our own audio in a master_audio shot) and
        the file continues on the mixer.music stream, which can start at an
        offset."""
        if not self._audio_path:
            return
        try:
            pygame.mixer.Channel(0).stop()
            pygame.mixer.music.load(self._audio_path)
            pygame.mixer.music.set_volume(self._volume)
            pygame.mixer.music.play(start=max(0.0, pos_s))
            self._music_active = True
            self._audio_pos0   = pos_s
            self._audio_epoch  = time.monotonic()
            print(f"[RenderEngine] shot audio re-synced to {pos_s:.1f}s")
        except Exception as exc:
            print(f"[RenderEngine] shot audio seek failed: {exc}")

    def _on_shot_load(self, data: dict):
        shot = data.get("shot")
        if shot is None:
            return
        self._fps = getattr(shot, "fps", 24)
        self._loading_kind = getattr(shot, "kind", "playback")
        audio_file = getattr(shot, "audio_file", None)
        self._pending_audio = str(audio_file) if audio_file else None
        self._stop_music()          # a seeked shot's audio must not outlive it
        self._audio_path = None

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
        sound = self._load_sound(path)
        if sound is None:
            return
        try:
            ch = pygame.mixer.Channel(channel)
            ch.set_volume(self._volume)
            ch.play(sound)
        except Exception as exc:
            print(f"[RenderEngine] SFX play failed: {exc}")

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

        # Baked-audio re-sync: if this segment start is far from where the
        # shot's audio.mp3 currently is (a seek — start-at-shot / skip-prologue),
        # restart the audio at the matching offset. Contiguous play-through
        # transitions land within the tolerance and never restart.
        if self._audio_path and self._seg_start and not self._seg_loop:
            expected = max(0.0, (self._seg_start - 1) / max(1, self._fps))
            playing  = self._audio_pos0 + (time.monotonic() - self._audio_epoch)
            if abs(expected - playing) > 2.0:
                self._seek_shot_audio(expected)

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
                head = 12
                ang  = _math.atan2(dy, dx)
                for hx, hy in self._arrow_wings(ex, ey, ang, head):
                    pygame.draw.line(self._screen, COLOR, (ex, ey),
                                     (int(hx), int(hy)), 2)

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

    # Window-scoped cursor fade (seconds full-off -> full-on and back).
    _CURSOR_FADE_IN_S = 0.4
    _CURSOR_FADE_OUT_S = 0.5

    def _draw_hand_cursors(self, pose_data=None, gesture_debug: dict | None = None):
        """Illustrated hand cursors (assets/hand_icons/), POSE-DRIVEN since the
        July 2026 pose-only rework: position comes from the Pose index landmark
        (19/20) when visible, else the wrist (15/16); side labels are inherent
        to the skeleton. Icon picked by the active interaction — knock fists
        during knock windows, pointing hand during point windows, open hand
        during grab windows. Cursors exist ONLY while an interaction window is
        armed: they fade in when the window opens and fade out when it closes
        (no tracked visuals between windows — playtest feedback). Last-seen
        position persists through pose dropouts."""
        if not self._screen:
            return
        gd = gesture_debug or {}
        active_type = gd.get("active_type")
        # A window can force its icon via params {"cursor": ...}: an icon name,
        # "dots" (plain tracking dots) or "hidden" (no cursor at all).
        override = (gd.get("active_params") or {}).get("cursor")
        window_open = (bool(active_type)
                       and not gd.get("input_locked")
                       and active_type not in self._NO_CURSOR_TYPES
                       and override != "hidden")

        now = time.monotonic()
        dt = max(0.0, min(0.1, now - self._cursor_fade_t))  # clamp pause stalls
        self._cursor_fade_t = now
        if window_open:
            self._cursor_fade_alpha = min(
                1.0, self._cursor_fade_alpha + dt / self._CURSOR_FADE_IN_S)
        else:
            self._cursor_fade_alpha = max(
                0.0, self._cursor_fade_alpha - dt / self._CURSOR_FADE_OUT_S)
        if self._cursor_fade_alpha <= 0.0:
            return

        if window_open:
            # e.g. reach_star is detected as point_target_held but the player
            # should reach with an open hand, not point.
            if override in ("open", "fist", "point", "knock", "dots"):
                mode = override
            elif active_type in self._POINT_TYPES:
                mode = "point"
            elif active_type in self._KNOCK_TYPES:
                mode = "knock"
            else:
                mode = "grab"
            self._cursor_fade_mode = mode
        else:
            # Window just closed — keep its icon while the fade-out plays.
            mode = self._cursor_fade_mode

        w, h = self._screen.get_size()

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

        fade = self._cursor_fade_alpha
        for side, state in self._hand_cursor_state.items():
            if state["pos"] is None:
                continue
            age = now - state["t"]
            if age > 4.0:
                continue   # long gone — drop the ghost cursor
            x, y = state["pos"]

            if mode in ("knock", "point", "open", "fist"):
                icon_shape = mode             # explicit (incl. cursor override)
            else:
                icon_shape = state["shape"]   # grab: "open" | "fist"
            # Mirror chirality (playtest-verified per art set): the open/fist
            # art is drawn anatomically, so on the mirrored display pose side L
            # needs the *_r art (a mirror flips handedness). The point/knock
            # art was authored already-mirrored — same-side art is correct.
            if icon_shape in ("open", "fist"):
                art_side = "r" if side == "L" else "l"
            else:
                art_side = side.lower()
            # "dots" mode draws the plain tracking dots instead of an icon.
            icon = (None if mode == "dots"
                    else self._hand_icons.get(f"{icon_shape}_{art_side}"))
            # Stale (pose dropout) dims the cursor; the window fade multiplies.
            alpha = int(255 * fade * (0.47 if age > 0.5 else 1.0))
            if icon is None:
                color = self._DOT_COLORS.get(side, (220, 220, 220))
                r = 10
                dot = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(dot, (*color, alpha), (r + 1, r + 1), r)
                self._screen.blit(dot, (x - r - 1, y - r - 1))
                continue
            if alpha < 255:
                icon = icon.copy()
                icon.set_alpha(alpha)
            self._screen.blit(icon, (x - icon.get_width() // 2,
                                     y - icon.get_height() // 2))

    # 8-way direction unit-ish vectors (screen space) shared by the draw
    # indicator and the debug arrows.
    _DIR_VEC_8 = {
        "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
        "up_left": (-1, -1), "up_right": (1, -1),
        "down_left": (-1, 1), "down_right": (1, 1),
    }

    @staticmethod
    def _arrow_wings(ex, ey, ang, head):
        """The two wing endpoints of an arrowhead whose tip is (ex, ey) and
        heading is `ang`: wings land BEHIND the tip (the V opens backward) so
        the arrow points along the heading, away from the line."""
        spread = 0.5   # rad half-opening (~29°)
        return [(ex - head * math.cos(ang + s), ey - head * math.sin(ang + s))
                for s in (spread, -spread)]

    @classmethod
    def _indicator_span(cls, params, sw, sh):
        """(sx, sy, ex, ey) of the draw-stroke indicator. Placement: the
        window's authored `indicator_rect` (SCREEN-space {x,y,w,h} drawn in
        the Experience Builder — centre of the rect, stroke length fitted to
        its extent along the direction), else `indicator_xy` (normalised
        centre), else the default mid-screen anchor."""
        vec = cls._DIR_VEC_8.get(params.get("direction", "right"), (1, 0))
        mag = math.hypot(*vec)
        dx, dy = vec[0] / mag, vec[1] / mag
        rect = params.get("indicator_rect")
        if rect:
            cx = (rect["x"] + rect["w"] / 2.0) * sw
            cy = (rect["y"] + rect["h"] / 2.0) * sh
            # Half-length of the chord through the rect centre along the
            # direction (first boundary hit), slightly inset.
            hw, hh = rect["w"] * sw / 2.0, rect["h"] * sh / 2.0
            tx = hw / abs(dx) if abs(dx) > 1e-6 else float("inf")
            ty = hh / abs(dy) if abs(dy) > 1e-6 else float("inf")
            length = max(30.0, min(tx, ty) * 0.9)
        else:
            ind = params.get("indicator_xy") or (0.5, 0.42)
            cx, cy = ind[0] * sw, ind[1] * sh
            length = sh * 0.11
        return (cx - dx * length, cy - dy * length,
                cx + dx * length, cy + dy * length)

    # forward_point's named regions, in PLAYER/screen space (already mirrored).
    _NAMED_REGIONS = {
        "top_left_quadrant":  {"x": 0.0,  "y": 0.0,  "w": 0.5, "h": 0.5},
        "top_right_quadrant": {"x": 0.5,  "y": 0.0,  "w": 0.5, "h": 0.5},
        "lower_third":        {"x": 0.0,  "y": 0.67, "w": 1.0, "h": 0.33},
        "center":             {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5},
    }

    _TRAIL_MAX_PARTICLES = 200
    _TRAIL_MIN_MOVE_PX = 4.0

    def _draw_star_trail(self, pose_data=None,
                         gesture_debug: dict | None = None) -> None:
        """Hand feedback while a directional_draw window is armed (the tracked
        cursor is suppressed there): a glowing "pen" dot pinned to each hand
        with star-trail particles emanating from it. Spawn rate scales with
        hand movement — a still hand keeps its dot but leaves no build-up;
        live particles keep fading after the window closes."""
        if not self._screen or not self._star_sprites:
            return
        gd = gesture_debug or {}
        active = (gd.get("active_type") == "directional_draw"
                  and not gd.get("input_locked"))
        now = time.monotonic()
        hand_pts = []

        if active and pose_data:
            w, h = self._screen.get_size()
            for side, (wrist_i, index_i) in self._POSE_SIDE_POINTS.items():
                lm = None
                for idx in (index_i, wrist_i):
                    if idx < len(pose_data):
                        cand = pose_data[idx]
                        if getattr(cand, "visibility", 1.0) >= 0.5:
                            lm = cand
                            break
                if lm is None:
                    self._trail_last_pos[side] = None
                    continue
                pos = ((1.0 - lm.x) * w, lm.y * h)
                hand_pts.append(pos)
                last = self._trail_last_pos[side]
                self._trail_last_pos[side] = pos
                if last is None:
                    continue
                dist = math.hypot(pos[0] - last[0], pos[1] - last[1])
                if dist < self._TRAIL_MIN_MOVE_PX:
                    continue
                n = min(4, 1 + int(dist / 30))
                for _ in range(n):
                    if len(self._trail_particles) >= self._TRAIL_MAX_PARTICLES:
                        self._trail_particles.pop(0)
                    k = random.random()   # sample along last->pos
                    self._trail_particles.append({
                        "x": last[0] + (pos[0] - last[0]) * k + random.uniform(-6, 6),
                        "y": last[1] + (pos[1] - last[1]) * k + random.uniform(-6, 6),
                        "vx": random.uniform(-14, 14),
                        "vy": random.uniform(-26, -4),   # gentle upward drift
                        "born": now,
                        "life": random.uniform(0.5, 0.9),
                        "size_i": random.randrange(len(self._star_sprites)),
                        "rot_i": random.randrange(2),
                    })
        elif not active:
            self._trail_last_pos = {"L": None, "R": None}

        alive = []
        for p in self._trail_particles:
            age = now - p["born"]
            if age >= p["life"]:
                continue
            k = 1.0 - age / p["life"]
            sprite = self._star_sprites[p["size_i"]][p["rot_i"]]
            sprite.set_alpha(int(250 * k ** 1.5))   # ease-out fade
            x = p["x"] + p["vx"] * age
            y = p["y"] + p["vy"] * age
            self._screen.blit(sprite, (int(x - sprite.get_width() / 2),
                                       int(y - sprite.get_height() / 2)))
            alive.append(p)
        self._trail_particles = alive

        # The pen dot rides on top of its trail.
        if self._hand_glow is not None:
            half = self._hand_glow.get_width() // 2
            for hx, hy in hand_pts:
                self._screen.blit(self._hand_glow,
                                  (int(hx - half), int(hy - half)))

    def _draw_interaction_indicator(self, gesture_debug: dict | None) -> None:
        """Player-facing hint for the active window (always on, unlike the debug
        target flag): a pulsing ring around point targets, an animated comet
        sweep for draw strokes. Replaces the confusing tracked-hand visuals as
        the 'simple on-screen indication' from the playtest punch list."""
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
            sx, sy, ex, ey = self._indicator_span(params, sw, sh)
            color = self._STAR_COLOR

            fx = self._fx_overlay
            if fx is None:
                return
            fx.fill((0, 0, 0, 0))

            # Faint guide line (where the stroke goes) with a dark underlay.
            pygame.draw.line(fx, (0, 0, 0, 90), (sx, sy), (ex, ey), 7)
            pygame.draw.line(fx, (*color, 70), (sx, sy), (ex, ey), 3)

            # Start ring ("begin here") — soft pulse.
            pygame.draw.circle(fx, (0, 0, 0, 150), (int(sx), int(sy)), 13, 5)
            pygame.draw.circle(fx, (*color, 220), (int(sx), int(sy)),
                               10 + int(2 * pulse), 3)

            # Arrowhead at the end (outlined, always visible), pointing along
            # the stroke direction.
            head = 24
            ang = math.atan2(ey - sy, ex - sx)
            for hx, hy in self._arrow_wings(ex, ey, ang, head):
                pygame.draw.line(fx, (0, 0, 0, 150), (ex, ey), (hx, hy), 9)
                pygame.draw.line(fx, (*color, 220), (ex, ey), (hx, hy), 5)

            # Comet sweep: a bright dot travels start -> end (~1.4s loop) with a
            # trail of small stars, fading in/out at the loop seam so the
            # restart doesn't pop.
            period = 1.4
            t = (now % period) / period
            prog = t ** 0.85                      # slight ease-out
            seam = min(1.0, t / 0.10, (1.0 - t) / 0.15)   # fade in/out window
            px = sx + (ex - sx) * prog
            py = sy + (ey - sy) * prog

            if self._star_sprites:
                for k in range(1, 6):             # trail behind the comet
                    tp = prog - k * 0.055
                    if tp <= 0:
                        break
                    tx = sx + (ex - sx) * tp
                    ty = sy + (ey - sy) * tp
                    sprite = self._star_sprites[min(1, len(self._star_sprites) - 1)][k % 2]
                    sprite.set_alpha(int(190 * seam * (1.0 - k / 6.0)))
                    fx.blit(sprite, (int(tx - sprite.get_width() / 2),
                                     int(ty - sprite.get_height() / 2)))

            # Comet head: layered glow + white-hot core.
            a = seam
            pygame.draw.circle(fx, (*color, int(45 * a)), (int(px), int(py)), 16)
            pygame.draw.circle(fx, (*color, int(110 * a)), (int(px), int(py)), 10)
            pygame.draw.circle(fx, (255, 250, 235, int(230 * a)),
                               (int(px), int(py)), 5)

            self._screen.blit(fx, (0, 0))
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

    # Full-body extension for the camera-setup screen (legs matter there).
    _POSE_CONNECTIONS_FULL = _POSE_CONNECTIONS + [
        (23, 25), (25, 27),          # left leg
        (24, 26), (26, 28),          # right leg
    ]

    def draw_camera_setup(self, frame_bgr, pose_lm) -> bool:
        """Camera-setup screen (config.json "camera_setup"): the live camera
        view, mirrored like the player-facing cursors, with the detected
        skeleton drawn on top and a prompt to adjust the camera until the whole
        body is in frame. Confirmed by ENTER/Space or the voice command
        "ready" (handled in main.py). Returns the body-in-frame verdict.

        Owns the whole frame: fills, draws and flips — the caller skips the
        normal render update while this screen is active."""
        if not self._screen:
            return False
        import numpy as np

        sw, sh = self._screen.get_size()
        self._screen.fill((8, 10, 14))

        disp = None   # (x, y, w, h) of the displayed camera rect
        if frame_bgr is not None:
            # BGR -> RGB and mirror in one slice so the view behaves like a mirror.
            rgb = np.ascontiguousarray(frame_bgr[:, ::-1, ::-1])
            fh, fw = rgb.shape[:2]
            surf = pygame.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB")
            scale = min(sw / fw, (sh * 0.82) / fh)   # leave room for the prompt
            dw, dh = int(fw * scale), int(fh * scale)
            dx, dy = (sw - dw) // 2, int(sh * 0.10)
            self._screen.blit(pygame.transform.scale(surf, (dw, dh)), (dx, dy))
            pygame.draw.rect(self._screen, (70, 80, 95), (dx, dy, dw, dh), 2)
            disp = (dx, dy, dw, dh)

        # Body-in-frame verdict: head and both ankles confidently inside view.
        def _in_frame(idx):
            if pose_lm is None or idx >= len(pose_lm):
                return False
            lm = pose_lm[idx]
            return (getattr(lm, "visibility", 0.0) >= 0.5
                    and 0.02 <= lm.x <= 0.98 and 0.02 <= lm.y <= 0.98)
        body_ok = _in_frame(0) and _in_frame(27) and _in_frame(28)

        # Skeleton over the displayed rect (nearest/most prominent person —
        # the one MediaPipe Pose tracks).
        if pose_lm is not None and disp is not None:
            dx, dy, dw, dh = disp
            pts = [(dx + int((1 - lm.x) * dw), dy + int(lm.y * dh))
                   for lm in pose_lm]
            def _vis(idx):
                return (idx < len(pose_lm)
                        and getattr(pose_lm[idx], "visibility", 0.0) >= 0.5)
            color = (60, 220, 90) if body_ok else (255, 170, 40)
            for a, b in self._POSE_CONNECTIONS_FULL:
                if _vis(a) and _vis(b):
                    pygame.draw.line(self._screen, color, pts[a], pts[b], 3)
            for idx, lm in enumerate(pose_lm):
                if _vis(idx):
                    pygame.draw.circle(self._screen, color, pts[idx], 4)

        # Prompt block
        if self._font:
            title = self._font.render("CAMERA SETUP", True, (200, 210, 225))
            self._screen.blit(title, ((sw - title.get_width()) // 2, int(sh * 0.03)))
            if pose_lm is None:
                msg, col = "No one detected - step into the camera's view", (255, 170, 40)
            elif not body_ok:
                msg, col = "Adjust the camera until your WHOLE body is in the frame", (255, 170, 40)
            else:
                msg, col = "Body in frame - press ENTER or say \"READY\" to continue", (60, 220, 90)
            line = self._font.render(msg, True, col)
            self._screen.blit(line, ((sw - line.get_width()) // 2, int(sh * 0.935)))
        if self._small_font:
            hint = self._small_font.render(
                "ENTER / Space or say \"ready\" to confirm", True, (120, 130, 145))
            self._screen.blit(hint, ((sw - hint.get_width()) // 2, int(sh * 0.972)))

        pygame.display.flip()
        return body_ok

    def _draw_hand_mini_panel(self, pose_data=None):
        """Pose-skeleton preview in bottom-right corner — keeps the main screen
        clean."""
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

    def _draw_debug_panel(self, gesture_debug: dict | None,
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
