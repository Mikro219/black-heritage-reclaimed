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
import threading
import time
import pygame
from collections import OrderedDict, deque
from pathlib import Path
from typing import Optional

from .frame_cache import FrameCacheManager
# Player-facing colours live in one place. Keep this module's imports cheap:
# engines.detectors.rules.* pulls in every rule module AND the Orbbec SDK
# (~3s + stdout noise), which the render layer and tests/test_render.py must
# not pay for — any pose math needed here is done inline against landmark
# indices, mirroring engines/detectors/rules/pose_helpers.py.
from .palette import PALETTE as P

# Max converted Surfaces kept in RAM per shot (LRU). 240 frames at 1080p ≈ 1.9 GB.
SURFACE_LRU_CAP = 240

# Main-thread stall reporting. Anything on the render/player thread that blocks
# for this long shows as a frozen picture while the audio runs on — the class of
# bug behind "the frames freeze when the OI window opens". Costs one
# perf_counter pair; only prints when something is actually slow.
SLOW_MS = 60.0


def warn_slow(label: str, t0: float) -> float:
    ms = (time.perf_counter() - t0) * 1000.0
    if ms >= SLOW_MS:
        print(f"[perf] {label} blocked the main thread for {ms:.0f}ms")
    return ms


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

        surf = None
        # Pack live: wrap the mmap slice and convert ONCE — the bytes path
        # below copies the frame three times (tobytes -> fromstring parse ->
        # convert). The frombuffer surface aliases the mmap only until
        # .convert() returns its own pixels, and the priority shot's pack is
        # never evicted while it plays, so that window is safe.
        getbuf = getattr(self._cache, "get_frame_buffer", None)
        if getbuf is not None:
            view = getbuf(self._dir, i)
            if view is not None:
                try:
                    surf = pygame.image.frombuffer(
                        view[0], view[1], "RGB").convert()
                except Exception:
                    surf = None   # odd buffer/mock — the bytes path handles it
        if surf is None:
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
        # Baked master audio held back by master_audio_offset_ms: monotonic
        # time it should start, or None. Serviced from update().
        self._pending_audio_at: Optional[float] = None
        self._pending_audio_pos: float = 0.0      # where in the file to start
        self._shot_id: Optional[str] = None       # selects the per-shot offset
        # Scheduled one-shot clips (choice pick/switch audio): each entry is
        # {at, path, source_offset_s, duration_s, gain, channel}. Serviced from
        # update(), shifted by pause/resume, dropped on shot_load.
        self._pending_clips: list = []
        self._clip_slices: OrderedDict = OrderedDict()   # (path,off,dur) -> Sound
        self._last_update_t: float = 0.0                 # frame-gap watchdog
        self._gap_mute_until: float = 0.0                # watchdog print rate limit
        self._pace_times: deque = deque(maxlen=240)      # update() timestamps, ~2s
        self._pace_report_t: float = 0.0                 # pacing print rate limit
        self._last_audio_resync: float = 0.0             # re-sync cooldown

        # Look-ahead frame cache (continuous background preload of all shots with art)
        self._cache: Optional[FrameCacheManager] = None
        self._loading_dir: Optional[Path] = None   # incoming shot being converted
        self._loading_kind: str = "playback"       # kind of the incoming shot (debug only)

        # OI flash overlay (full-screen surface pre-allocated once in init_display)
        self._flash_color: tuple = P.SUCCESS
        self._flash_start: float = 0.0
        self._flash_until: float = 0.0
        self._flash_alpha: int   = P.SUCCESS_ALPHA   # peak alpha (0-255)
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
        # Guards _sound_cache AND _clip_slices: the prefetch worker inserts
        # from its own thread while the live path reads on the main thread
        # (same pattern as audio_mixer._sounds_lock). A cached entry is never
        # replaced — first decode wins.
        self._sound_lock = threading.Lock()
        self._volume_slider_rect: Optional[pygame.Rect] = None  # set when drawn

        # Skeleton-in-corner display option (pause menu, K key) — shows the
        # bottom-right skeleton mini-panel WITHOUT the rest of the debug overlay.
        self._show_skeleton: bool = False

        # Captions (Experience Builder authored / script-generated). Stashed per
        # shot from Shot.captions; drawn against the shot playhead. The font is
        # built eagerly in init_display (a SysFont preference-chain lookup costs
        # 20-60ms — it used to land on the first captioned frame), with a lazy
        # fallback kept for surfaces created without init_display. The cache
        # memoises finished caption CARDS keyed on (text, rect w, rect h) —
        # re-wrapping + re-rendering every visible frame cost 1.5-4ms/frame.
        self._captions: list = []
        self._caption_font = None
        self._caption_render_cache: OrderedDict = OrderedDict()

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
        # Per-side aim, latched alongside the mode. The gesture engine clears
        # its active window BEFORE emitting the detection, so on the firing
        # frame debug_info() reports active_params={} — without this the icon
        # would snap upright for the whole fade-out (and the green flash).
        # Two hands carry two independent radial angles, hence per-side.
        self._cursor_fade_angle: dict = {"L": None, "R": None}

        # Torso centre (screen space) for radial point cursors, latched through
        # pose dropouts — see _torso_center_screen.
        self._torso_screen: Optional[tuple] = None
        self._torso_t: float = 0.0

        # Per-frame draw caches (steady-state costs measured Aug 2026):
        # pulsing target rings per (size, pulse bucket), rotated/faded cursor
        # icons per (icon, angle bucket, alpha bucket).
        self._ring_cache: dict = {}
        self._rot_icon_cache: dict = {}

        # Star-trail particle layer (July 2026): tiny 4-point stars trail the
        # visitor's hands while a directional_draw window is armed. Sprites are
        # pre-rendered in init_display; particles are dicts with position,
        # velocity, birth time, lifetime and sprite index.
        self._star_sprites: list = []            # [size][rotation] -> Surface
        self._hand_glow: Optional[pygame.Surface] = None   # per-hand "pen" dot
        self._trail_particles: list = []
        self._trail_last_pos: dict = {"L": None, "R": None}
        self._fx_overlay: Optional[pygame.Surface] = None
        # Dirty-rect latch for the draw-window FX composite: the rect that
        # currently holds ink on _fx_overlay (last armed frame's draw bounds).
        # Persists through window close — it is only swept (erased + replaced)
        # on the next armed frame, so stale ink can never leak into a new
        # window's blit region.
        self._fx_dirty: Optional[pygame.Rect] = None

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
        self.event_bus.subscribe("play_clip",         self._on_play_clip)
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
        # Caption font up front too: resolving its preference chain costs
        # 20-60ms, which used to stall the first captioned frame (the lazy
        # build in _draw_captions remains as a fallback).
        self._caption_font = self._build_caption_font(h)
        # Cached fonts for the scene panel (avoids constructing SysFont per frame).
        # Pre-allocate the full-screen OI flash overlay once (reused each flash frame).
        self._flash_overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        self._fx_overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        self._build_star_sprites()
        self._load_hand_icons()

    _STAR_COLOR = P.LANTERN   # the player-layer amber (alias — see engines/palette.py)

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
                pygame.draw.circle(surf, P.NORTH_STAR, (int(cx), int(cy)),
                                   max(1, int(r_in * 0.7)))
                rots.append(surf)
            self._star_sprites.append(rots)
        # The "pen" dot pinned to each hand while a draw window is armed —
        # same layered-glow language as the comet head.
        g = pygame.Surface((36, 36), pygame.SRCALPHA)
        pygame.draw.circle(g, (*self._STAR_COLOR, 45), (18, 18), 16)
        pygame.draw.circle(g, (*self._STAR_COLOR, 110), (18, 18), 10)
        pygame.draw.circle(g, (*P.NORTH_STAR, 230), (18, 18), 5)
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

    def toggle_captions(self) -> None:
        """Flip subtitle display (pause-menu option, C key). Session-scoped:
        flips the in-memory `captions_enabled` config flag that gates
        `_draw_captions`; the config.json default is untouched on disk."""
        on = not self.config.get("captions_enabled", True)
        self.config["captions_enabled"] = on
        print(f"[RenderEngine] captions={'on' if on else 'off'}")

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
        if self._pending_audio_at is not None:
            self._pending_audio_at += delta   # a held-back track waits out the pause
        for clip in self._pending_clips:
            clip["at"] += delta               # ... and so do delayed pick sounds
        if self._music_active:
            try:
                pygame.mixer.music.unpause()
            except Exception:
                pass
        self._paused = False

    def _draw_paused_overlay(self) -> None:
        """Re-blit the held frame and a dimmed PAUSED banner over it.

        The whole screen is static while paused (the kiosk idles here between
        visitors), yet it was re-composited from scratch every frame — a fresh
        full-screen SRCALPHA veil plus per-glyph text, ~6-10ms/frame of pure
        waste. Composite once into a cached surface, invalidated by anything
        that can change the picture (frame, volume, captions flag, size)."""
        sw, sh = self._screen.get_size()
        key = (self._frame_index if self._frames else None, self._volume,
               bool(self.config.get("captions_enabled", True)), sw, sh)
        cached = getattr(self, "_pause_cache", None)
        if cached is None or cached[0] != key:
            comp = pygame.Surface((sw, sh))
            if self._frames and 0 <= self._frame_index < len(self._frames):
                comp.blit(self._frames[self._frame_index], (0, 0))
            else:
                comp.fill((0, 0, 0))
            veil = pygame.Surface((sw, sh), pygame.SRCALPHA)
            veil.fill(P.VEIL_RGBA)
            comp.blit(veil, (0, 0))

            screen, self._screen = self._screen, comp   # draw helpers -> comp
            try:
                # Serif display title — the camera/tutorial redesign's type
                # ladder applied to the pause screen.
                title = self._serif_font(max(30, int(sh * 0.062))).render(
                    "Paused", True, P.NORTH_STAR)
                comp.blit(title, ((sw - title.get_width()) // 2, sh // 2 - 130))
                self._draw_volume_slider(sw, sh)
                self._draw_pause_keys(sw, sh, sh // 2 + 30)
            finally:
                self._screen = screen
            cached = (key, comp)
            self._pause_cache = cached
        self._screen.blit(cached[1], (0, 0))

    def _draw_pause_keys(self, sw: int, sh: int, top: int) -> None:
        """Operator key reference, laid out as two labelled columns.

        It used to be one run-on line of six items, which is unreadable at a
        glance on a projected wall. Keys are amber (the interactive colour),
        actions are dim: the eye finds the key first, then reads across."""
        font = self._small_font
        if not font:
            return
        cap_state = "ON" if self.config.get("captions_enabled", True) else "OFF"
        columns = (
            ("PLAYBACK", (("Space", "resume"),
                          ("S", "skip prologue / epilogue"),
                          ("R", "restart from the beginning"),
                          ("Esc", "quit"))),
            ("DISPLAY", (("C", f"captions  {cap_state}"),
                         ("K", "skeleton panel"),
                         ("D", "debug overlay"),
                         ("F", "fullscreen"))),
        )

        line_h = font.get_linesize() + 4
        gutter = 12          # key column -> action column
        col_gap = 54         # between the two columns
        # Size each column to its own content so the two blocks sit evenly.
        metrics = []
        for header, rows in columns:
            kw = max(font.size(k)[0] for k, _ in rows)
            aw = max(font.size(a)[0] for _, a in rows)
            hw = font.size(header)[0] + 3 * max(0, len(header) - 1)  # tracked
            metrics.append((max(kw + gutter + aw, hw), kw))
        total_w = sum(w for w, _ in metrics) + col_gap
        x = (sw - total_w) // 2

        for (header, rows), (col_w, key_w) in zip(columns, metrics):
            head = self._tracked_label(font, [(header, P.LANTERN_DIM)], 3)
            self._screen.blit(head, (x, top))
            y = top + line_h + 2
            for key, action in rows:
                ks = font.render(key, True, P.LANTERN)
                # Right-align the key against the gutter so the actions line up.
                self._screen.blit(ks, (x + key_w - ks.get_width(), y))
                self._screen.blit(font.render(action, True, P.LINEN_DIM),
                                  (x + key_w + gutter, y))
                y += line_h
            x += col_w + col_gap

    def _draw_volume_slider(self, sw: int, sh: int) -> None:
        """Volume slider for the pause menu. Up/Down adjust; the bar is drag-clickable."""
        bar_w, bar_h = 360, 8
        bx = (sw - bar_w) // 2
        by = sh // 2 - 30
        rect = pygame.Rect(bx, by, bar_w, bar_h)
        self._volume_slider_rect = rect

        if self._small_font:
            label = self._small_font.render(
                f"Volume  {int(round(self._volume * 100))}%   (Up / Down)", True, P.LINEN_DIM)
            self._screen.blit(label, ((sw - label.get_width()) // 2, by - 26))

        # Track
        pygame.draw.rect(self._screen, P.TRACK, rect, border_radius=4)
        # Fill — amber, like every other live interactive value. It used to be a
        # green inches away from the success-flash green, on the one screen an
        # operator stares at; green now means exactly one thing.
        fill_w = int(bar_w * self._volume)
        if fill_w > 0:
            pygame.draw.rect(self._screen, P.LANTERN,
                             pygame.Rect(bx, by, fill_w, bar_h), border_radius=4)
        # Knob
        kx = bx + fill_w
        pygame.draw.circle(self._screen, P.LINEN, (kx, by + bar_h // 2), 9)
        pygame.draw.circle(self._screen, P.NIGHT_DEEP, (kx, by + bar_h // 2), 9, 2)

    def update(self, pose_data=None,
               gesture_debug: dict | None = None,
               voice_debug: dict | None = None,
               narration_debug: dict | None = None,
               tutorial_card: dict | None = None):
        if not self._screen:
            return

        # Frame-gap watchdog: reports a stall wherever it happened in the main
        # loop (player, bus handlers, frame decode), which the per-operation
        # timers above then attribute. A frozen picture while the audio runs on
        # always shows up here first. Threshold is 80ms (2.5 dropped frames at
        # 30fps): the first-gesture freeze produced NO logs at the old 150ms —
        # a sequence of sub-150ms slow iterations reads as a freeze on screen
        # but is invisible to a single-gap check. Rate-limited so the print
        # itself can't feed a stall loop.
        t_now = time.perf_counter()
        if self._last_update_t:
            gap_ms = (t_now - self._last_update_t) * 1000.0
            if gap_ms >= 500.0:
                # Mode switch (camera setup, boot screen) — not a pacing
                # problem. Stale history must not fake a degradation report.
                self._pace_times.clear()
            if gap_ms >= 80.0 and not self._paused and t_now >= self._gap_mute_until:
                self._gap_mute_until = t_now + 0.5
                print(f"[perf] frame gap {gap_ms:.0f}ms "
                      f"(shot {self._shot_id}, frame {self._frame_index})",
                      flush=True)
        self._last_update_t = t_now

        # Rolling pacing monitor: catches SUSTAINED degradation no single-gap
        # check can see — e.g. a second of 60-80ms iterations halves the frame
        # rate (a visible stutter) without one gap crossing the watchdog. One
        # line at most every 2s, only while a shot is actually on screen.
        self._pace_times.append(t_now)
        if not self._paused and self._frames:
            n = sum(1 for t in self._pace_times if t_now - t <= 1.0)
            if 2 <= n <= 20 and t_now - self._pace_report_t >= 2.0 \
                    and t_now - self._pace_times[0] >= 1.0:
                self._pace_report_t = t_now
                print(f"[perf] pacing: {n} updates in the last second "
                      f"(target ~30, shot {self._shot_id}, "
                      f"frame {self._frame_index})", flush=True)

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
        self._service_pending_audio()
        self._service_pending_clips()

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
            alpha = int(self._flash_alpha * math.sin(progress * math.pi))
            alpha = max(0, min(255, alpha))
            self._flash_overlay.fill((*self._flash_color, alpha))
            self._screen.blit(self._flash_overlay, (0, 0))

        # Player-facing layer — always on, not debug-gated: star trail under the
        # comet indicator, hand cursors on top.
        self._draw_star_trail(pose_data, gesture_debug)
        self._draw_interaction_indicator(gesture_debug)
        self._draw_hand_cursors(pose_data, gesture_debug)
        self._draw_captions()

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
        with self._sound_lock:
            sound = self._sound_cache.get(key)
            if sound is not None:
                self._sound_cache.move_to_end(key)
                return sound
        t0 = time.perf_counter()
        try:
            sound = pygame.mixer.Sound(key)
        except Exception as exc:
            print(f"[RenderEngine] audio load failed: {key}: {exc}")
            return None
        # Decoding a whole shot's audio.mp3 is hundreds of ms of PCM — never do
        # it on a frame the visitor is watching. (The prefetch worker takes
        # exactly that hit off-thread, where it is fine — don't report it.)
        if threading.current_thread() is threading.main_thread():
            warn_slow(f"decode {os.path.basename(key)}", t0)
        with self._sound_lock:
            # The prefetch worker and the live path can race on the same file:
            # first decode wins — a cached Sound may already be playing on a
            # channel, so it is never replaced.
            hit = self._sound_cache.get(key)
            if hit is not None:
                self._sound_cache.move_to_end(key)
                return hit
            self._sound_cache[key] = sound
            while len(self._sound_cache) > self._SOUND_CACHE_MAX:
                self._sound_cache.popitem(last=False)
        return sound

    def master_audio_offset_ms(self, shot_id=None) -> int:
        """Lip-sync trim for this shot's BAKED master audio, in ms.

        Only baked whole-file audio (`audio.mp3`) is affected: shots built from
        layered `audio_events` are frame-anchored to the picture and can't
        drift, so trimming them would only break something that works.

        POSITIVE holds the audio back (the fix when audio runs ahead of the
        lips); NEGATIVE starts the file that far in, so it runs earlier.
        `master_audio_offset_ms_by_shot["<id>"]` overrides the global
        `master_audio_offset_ms`; null/absent falls back to the global. Read at
        shot start, so tuning is an edit + relaunch — no re-export."""
        per_shot = self.config.get("master_audio_offset_ms_by_shot") or {}
        val = per_shot.get(str(shot_id)) if shot_id is not None else None
        if val is None:
            val = self.config.get("master_audio_offset_ms", 0)
        try:
            return int(val)
        except (TypeError, ValueError):
            print(f"[RenderEngine] bad master audio offset {val!r} — using 0")
            return 0

    def _begin_audio(self) -> None:
        """Start (or schedule) the shot's baked audio.mp3 at the top of the shot."""
        self._sync_shot_audio_to_picture(0.0)

    def _sync_shot_audio_to_picture(self, picture_s: float) -> None:
        """Place the shot's baked audio for picture position `picture_s`,
        honouring the master offset.

        ONE positioner for every entry point — shot start AND every seek
        (start-at-shot, skip-prologue, skip-epilogue). The offset used to live
        only in the shot-start path, so skipping the prologue re-started the
        audio through _seek_shot_audio and silently threw the trim away.

        A positive offset means the audio should LAG the picture, so the audio
        target is `picture_s - offset`. When that is still negative (near the
        top of the shot) there is nothing to play yet: hold the start until the
        offset has elapsed, then begin at 0."""
        path = self._pending_audio or self._audio_path
        if not path:
            return
        offset_s = self.master_audio_offset_ms(self._shot_id) / 1000.0
        target = picture_s - offset_s
        self._pending_audio = path
        if target < 0.0:
            # Not due yet — silence anything already running so the old
            # position can't leak through the wait.
            self._silence_shot_audio()
            self._pending_audio_at = time.monotonic() - target
            self._pending_audio_pos = 0.0
            print(f"[RenderEngine] shot {self._shot_id} master audio held "
                  f"{-target:.3f}s (offset {offset_s * 1000:.0f}ms)")
            return
        self._pending_audio_at = None
        self._pending_audio_pos = target
        self._start_pending_audio()

    def _silence_shot_audio(self) -> None:
        """Stop whatever is carrying the shot's baked audio, leaving the
        layered audio_events channels (3+) alone."""
        self._stop_music()
        try:
            pygame.mixer.Channel(0).stop()
        except Exception:
            pass

    def _start_pending_audio(self) -> None:
        """Play the pending baked audio at `_pending_audio_pos` seconds in,
        streamed on mixer.music (see below — never a decoded Sound)."""
        path = self._pending_audio
        pos = max(0.0, self._pending_audio_pos or 0.0)
        self._pending_audio = None
        self._pending_audio_at = None
        self._pending_audio_pos = 0.0
        if not path:
            return
        # Always stream via mixer.music. The old pos-0 path decoded the WHOLE
        # mp3 into a Channel-0 Sound — 689ms measured on the main thread for
        # shot 01's 5-minute master track, seen as a picture freeze at every
        # baked shot start, with the picture clock running through the stall
        # so the audio then lagged by the same ~700ms for the rest of the
        # shot (under the 2s drift net — never self-corrected). The music
        # stream decodes incrementally (~1ms to start) and can seek, so one
        # path serves shot starts and skips alike.
        self._audio_path = path
        self._seek_shot_audio(pos)

    def _service_pending_audio(self) -> None:
        """Start a delayed master audio track once its offset has elapsed."""
        if self._pending_audio_at is None or self._paused:
            return
        if time.monotonic() >= self._pending_audio_at:
            self._start_pending_audio()

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
        t0 = time.perf_counter()
        try:
            pygame.mixer.Channel(0).stop()
            pygame.mixer.music.load(self._audio_path)
            pygame.mixer.music.set_volume(self._volume)
            # The stop+load above can cost ~100ms on a long mp3 while the
            # picture clock keeps running. Starting at the position computed
            # BEFORE the load bakes that latency in as permanent extra audio
            # lag — it sits far below the 2s drift net, so it never
            # self-corrects, and after a prologue skip it landed on top of
            # master_audio_offset_ms (a 50ms trim played as ~150ms). Fold the
            # elapsed time into the start position so the stream begins where
            # the picture is NOW.
            start = max(0.0, pos_s + (time.perf_counter() - t0))
            pygame.mixer.music.play(start=start)
            warn_slow(f"music.load+seek {os.path.basename(str(self._audio_path))}", t0)
            self._music_active = True
            self._audio_pos0   = start
            self._audio_epoch  = time.monotonic()
            print(f"[RenderEngine] shot audio re-synced to {start:.1f}s "
                  f"(+{(start - pos_s) * 1000:.0f}ms load latency)")
        except Exception as exc:
            print(f"[RenderEngine] shot audio seek failed: {exc}")

    def _on_shot_load(self, data: dict):
        shot = data.get("shot")
        if shot is None:
            return
        # prefetch_shot only ever carries the NEXT shot, so the FIRST shot of
        # a run (and any seek target) would still decode its entry sounds on
        # the main thread — warm the loading shot's own sounds too.
        self._on_prefetch_shot({"shot": shot})
        self._fps = getattr(shot, "fps", 24)
        self._captions = getattr(shot, "captions", []) or []
        self._loading_kind = getattr(shot, "kind", "playback")
        self._shot_id = getattr(shot, "shot", None)
        audio_file = getattr(shot, "audio_file", None)
        self._pending_audio = str(audio_file) if audio_file else None
        self._pending_audio_at = None   # drop any unfired delay from the last shot
        self._pending_clips.clear()     # ... and any pick sound still counting down
        self._last_audio_resync = 0.0   # a fresh shot may re-sync immediately
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
        """Pre-decode the NEXT shot's state-entry sounds while the current
        shot is still playing (frames are already covered by the look-ahead
        cache).

        pygame.mixer.Sound() decodes the WHOLE file synchronously, so the
        first on_enter_sfx / OI sfx / pick clip of a shot used to decode on
        the main thread at the exact moment the FSM entered its state — a
        picture hitch on the frame a prompt appears. Walks the interaction
        metadata for every sound the FSM can fire at state entry and warms
        _load_sound / _slice_sound on a throwaway daemon thread (never block
        the bus — same pattern as audio_mixer's prefetch preload). Missing
        files are skipped silently here; the live path already warns."""
        shot = data.get("shot")
        if shot is None:
            return
        jobs = self._collect_entry_sounds(shot)
        if not jobs:
            return
        threading.Thread(target=self._prefetch_sounds_worker, args=(jobs,),
                         daemon=True, name="RenderSfxPrefetch").start()

    def _collect_entry_sounds(self, shot) -> list:
        """(path, source_offset_s, duration_s) for every sound the shot's FSM
        wiring can fire at state entry: on_enter_sfx and oi "sfx" names, the
        shot-level interaction "sfx", and on_enter_audio pick/switch clips. A
        plain sfx is (path, 0, 0) — _slice_sound hands that straight through
        to _load_sound, so one job shape covers both."""
        interaction = getattr(shot, "interaction", None) or {}
        names, clips = [], []
        if interaction.get("sfx"):
            names.append(interaction["sfx"])
        fsm = interaction.get("interaction_fsm") or {}
        for state in (fsm.get("states") or {}).values():
            if state.get("on_enter_sfx"):
                names.append(state["on_enter_sfx"])
            oi = state.get("oi") or {}
            if oi.get("sfx"):
                names.append(oi["sfx"])
            clip = state.get("on_enter_audio")
            if isinstance(clip, dict) and clip.get("file"):
                clips.append(clip)

        jobs = []
        for name in names:
            path = self._resolve_shot_sound(shot, name)
            if path:
                jobs.append((path, 0.0, 0.0))
        for clip in clips:
            path = self._resolve_shot_sound(shot, clip["file"])
            if path:
                # Same coercion as _on_play_clip, so the slice key warmed
                # here is byte-identical to the one the live path computes.
                jobs.append((path,
                             float(clip.get("source_offset_s", 0.0) or 0.0),
                             float(clip.get("duration_s", 0.0) or 0.0)))
        return jobs

    @staticmethod
    def _resolve_shot_sound(shot, filename) -> Optional[str]:
        """Mirror of the player's _resolve_sfx (audio/ subdir first, then the
        shot root) — kept in lockstep so the path warmed at prefetch is the
        path play_sfx / play_clip will carry at fire time."""
        audio_dir = getattr(shot, "audio_dir", None)
        if audio_dir:
            p = Path(audio_dir) / filename
            if p.exists():
                return str(p)
        frames_dir = getattr(shot, "frames_dir", None)
        if frames_dir:
            p = Path(frames_dir).parent / filename
            if p.exists():
                return str(p)
        return None

    def _prefetch_sounds_worker(self, jobs: list) -> None:
        """Daemon-thread body: warm the LRUs. A failure here costs nothing —
        the live path just decodes (and reports) as before."""
        for path, offset, dur in jobs:
            try:
                self._slice_sound(path, offset, dur)
            except Exception:
                pass

    def _get_resolution(self) -> tuple[int, int]:
        display_cfg = self.config.get("_profile", {}).get("display", {})
        w, h = display_cfg.get("resolution") or self.config.get("resolution", [1920, 1080])
        return (w, h)

    def _on_oi_flash(self, data: dict):
        self._flash_color = data.get("color", P.SUCCESS)
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

    _CLIP_SLICE_MAX = 12

    def _slice_sound(self, path, source_offset_s: float, duration_s: float):
        """A Sound covering [source_offset_s, +duration_s) of `path`.

        Cut from the decoded PCM (Sounds can't seek, and pre-rendering trimmed
        files would mean a re-export every time someone nudges a number), so
        offset/duration stay tunable in the Builder alone. duration_s <= 0
        means "to the end of the file". Slices are LRU-cached — a choice pick
        can fire many times per visit."""
        offset = max(0.0, float(source_offset_s or 0.0))
        dur = max(0.0, float(duration_s or 0.0))
        if offset <= 0.0 and dur <= 0.0:
            return self._load_sound(path)
        key = (str(path), round(offset, 3), round(dur, 3))
        with self._sound_lock:
            hit = self._clip_slices.get(key)
            if hit is not None:
                self._clip_slices.move_to_end(key)
                return hit
        base = self._load_sound(path)
        if base is None:
            return None
        init = pygame.mixer.get_init()
        if not init:
            return base
        # get_raw() on a long file copies the whole decoded buffer — cheap for a
        # UI blip, very much not for a 5-minute bed.
        t0 = time.perf_counter()
        freq, size, channels = init
        frame_bytes = max(1, channels * (abs(size) // 8))
        raw = base.get_raw()
        start = int(offset * freq) * frame_bytes
        if start >= len(raw):
            print(f"[RenderEngine] clip offset {offset:.2f}s is past the end of "
                  f"{os.path.basename(str(path))} — playing from the start")
            start = 0
        end = len(raw) if dur <= 0.0 else min(len(raw),
                                              start + int(dur * freq) * frame_bytes)
        try:
            sound = pygame.mixer.Sound(buffer=raw[start:end])
        except Exception as exc:
            print(f"[RenderEngine] clip slice failed: {exc}")
            return base
        if threading.current_thread() is threading.main_thread():
            warn_slow(f"slice {os.path.basename(str(path))}", t0)
        with self._sound_lock:
            # Same never-replace rule as _load_sound (prefetch worker race).
            hit = self._clip_slices.get(key)
            if hit is not None:
                self._clip_slices.move_to_end(key)
                return hit
            self._clip_slices[key] = sound
            while len(self._clip_slices) > self._CLIP_SLICE_MAX:
                self._clip_slices.popitem(last=False)
        return sound

    def _on_play_clip(self, data: dict):
        """Play a one-shot clip with optional delay / source offset / duration.

        Used by the choice blocks' pick & switch audio. `delay_s` waits before
        starting (the visitor's gesture lands, then the sound answers it)."""
        path = data.get("path")
        if not path or not os.path.exists(str(path)):
            if path:
                print(f"[RenderEngine] clip not found: {path}")
            return
        entry = {
            "path": str(path),
            "source_offset_s": float(data.get("source_offset_s", 0.0) or 0.0),
            "duration_s": float(data.get("duration_s", 0.0) or 0.0),
            "gain": float(data.get("gain", 1.0) or 1.0),
            "channel": int(data.get("channel", 1)),
        }
        delay = max(0.0, float(data.get("delay_s", 0.0) or 0.0))
        if delay > 0.0:
            entry["at"] = time.monotonic() + delay
            self._pending_clips.append(entry)
        else:
            self._play_clip_now(entry)

    def _play_clip_now(self, entry: dict) -> None:
        sound = self._slice_sound(entry["path"], entry["source_offset_s"],
                                  entry["duration_s"])
        if sound is None:
            return
        try:
            ch = pygame.mixer.Channel(entry["channel"])
            ch.set_volume(max(0.0, min(1.0, entry["gain"])) * self._volume)
            ch.play(sound)
        except Exception as exc:
            print(f"[RenderEngine] clip play failed: {exc}")

    def _service_pending_clips(self) -> None:
        if not self._pending_clips or self._paused:
            return
        now = time.monotonic()
        due = [c for c in self._pending_clips if c["at"] <= now]
        if due:
            self._pending_clips = [c for c in self._pending_clips if c["at"] > now]
            for entry in due:
                self._play_clip_now(entry)

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
        # re-place the audio for the new picture position. Contiguous
        # play-through transitions land within the tolerance and never restart.
        # Routed through _sync_shot_audio_to_picture so the master offset
        # survives a skip (it used to be applied only at shot start).
        if self._seg_start and not self._seg_loop:
            t0_sync = time.perf_counter()
            expected = max(0.0, (self._seg_start - 1) / max(1, self._fps))
            if self._pending_audio:
                # Still waiting to start: the queued position is now wrong.
                self._sync_shot_audio_to_picture(expected)
            elif self._audio_path:
                playing = self._audio_pos0 + (time.monotonic() - self._audio_epoch)
                drift = abs(expected - playing)
                # Rate-limited: a re-sync reloads the WHOLE master mp3 into the
                # music stream, which blocks the main thread long enough to be
                # seen as a freeze. If real drift ever sits above the threshold
                # this would otherwise fire at every segment boundary — each
                # stall adding drift, so it never recovers. Seeks are one-off,
                # so a cooldown costs them nothing.
                now = time.monotonic()
                if drift > 2.0 and now - self._last_audio_resync >= 10.0:
                    self._last_audio_resync = now
                    print(f"[RenderEngine] audio re-sync: picture {expected:.1f}s "
                          f"vs audio {playing:.1f}s (drift {drift:.1f}s)")
                    self._sync_shot_audio_to_picture(expected)
                elif drift > 2.0:
                    print(f"[RenderEngine] audio drift {drift:.1f}s — re-sync "
                          f"suppressed (cooling down)")
            # An audio (re)placement above blocks the main thread — the mp3
            # seek-scan inside music.play(start=) alone measured ~100ms — and
            # the picture anchor was set BEFORE the stall, so the picture
            # would jump the stall forward while the audio starts at the
            # pre-stall position: ~100ms of extra audio lag on every seek,
            # invisible to the 2s drift net. Shift the anchor by the measured
            # stall so picture and audio both start "now", aligned.
            blocked = time.perf_counter() - t0_sync
            if blocked >= 0.005:
                self._seg_anchor += blocked

        # Warm the incoming segment's frames so its first reads (and a loop's
        # wrap-around) come out of the pack instead of page-faulting / decode-
        # falling-back on a frame the visitor is watching. Index space matches
        # the fetch path exactly: update() reads _frames[_seg_start + local]
        # with local in [0, seg_len), clamped to the frame count — i.e. the
        # FrameView indices _seg_start.._seg_end. warm_segment is advisory
        # (non-blocking, never raises), and older caches may not have it.
        if (self._seg_start is not None and self._seg_end is not None
                and self._current_frames_dir is not None
                and self._cache is not None
                and hasattr(self._cache, "warm_segment")):
            start_idx, end_idx = self._seg_start, self._seg_end
            if self._frames:
                last = len(self._frames) - 1
                start_idx = min(start_idx, last)
                end_idx = min(end_idx, last)
            try:
                self._cache.warm_segment(self._current_frames_dir,
                                         start_idx, end_idx)
            except Exception:
                pass   # a warm hint must never break playback

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
            ax = sw // 2
            ay = sh - 80          # near bottom-centre
            arrow_len = 50

            for d in directions:
                vec = _DIR_VEC.get(d)
                if not vec:
                    continue
                mag = math.hypot(*vec)
                dx = int(vec[0] / mag * arrow_len)
                dy = int(vec[1] / mag * arrow_len)
                ex, ey = ax + dx, ay + dy
                pygame.draw.line(self._screen, COLOR, (ax, ay), (ex, ey), 3)
                head = 12
                ang  = math.atan2(dy, dx)
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

    _DOT_COLORS = {"L": P.HAND_L, "R": P.HAND_R}   # green / blue — matches the tinted PNGs

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
        active_params = gd.get("active_params") or {}
        # A window can force its icon via params {"cursor": ...}: an icon name,
        # "dots" (plain tracking dots) or "hidden" (no cursor at all).
        override = active_params.get("cursor")
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
            # Fully hidden — the next window starts from a clean aim. This is
            # the only reset point: clearing on window-open would blank a still
            # valid angle on a frame where the pose happens to be stale.
            self._cursor_fade_angle["L"] = None
            self._cursor_fade_angle["R"] = None
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
        # Computed every frame (four attribute reads) so the latch stays warm
        # between windows and a radial window opens already aimed.
        torso = self._torso_center_screen(pose_data, w, h)

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
            # All four icon families are colour-tinted per side (green *_l for
            # the green-dot L hand, blue *_r for the blue-dot R hand), so art
            # side must follow the tracked side — same-side for EVERY shape.
            # (open/fist used to flip L<->r "for mirror chirality", which put
            # the blue fist on the green hand and vice versa — the playtest
            # "swapped hands on the fist" report.)
            art_side = side.lower()
            # "dots" mode draws the plain tracking dots instead of an icon.
            icon = (None if mode == "dots"
                    else self._hand_icons.get(f"{icon_shape}_{art_side}"))
            # Stale (pose dropout) dims the cursor; the window fade multiplies.
            alpha = int(255 * fade * (0.47 if age > 0.5 else 1.0))
            if icon is None:
                color = self._DOT_COLORS.get(side, P.LINEN)
                r = 10
                dot = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(dot, (*color, alpha), (r + 1, r + 1), r)
                self._screen.blit(dot, (x - r - 1, y - r - 1))
                continue
            # The point hand is drawn finger-up; rotate it to aim at the target
            # (region box), radially out from the torso on a left/right choice,
            # or along a declared direction — so it reads as "point THERE".
            if mode == "point":
                if window_open:
                    ang = self._point_icon_angle(active_params, x, y, w, h,
                                                 torso=torso)
                    self._cursor_fade_angle[side] = ang
                else:
                    # Hold the last aim through the fade-out.
                    ang = self._cursor_fade_angle.get(side)
                # `is not None`, not truthiness: 0.0 is a legitimate aim (hand
                # straight above the torso centre, or direction "up").
                if ang is not None:
                    icon = self._rotated_icon(icon, ang, alpha)
                    alpha = 255   # baked into the cached copy
            if alpha < 255:
                icon = self._rotated_icon(icon, 0.0, alpha)
            self._screen.blit(icon, (x - icon.get_width() // 2,
                                     y - icon.get_height() // 2))

    def _rotated_icon(self, icon, ang: float, alpha: int) -> pygame.Surface:
        """Rotated + faded cursor icon from a small cache. transform.rotate +
        copy()/set_alpha ran per hand per frame (~0.3-0.8ms); 5-degree angle
        buckets and 16 alpha steps are invisible at cursor size."""
        ang_b = int(round(ang / 5.0)) * 5 % 360
        alpha_b = min(255, (alpha // 16) * 16 + 15)
        key = (id(icon), ang_b, alpha_b)
        out = self._rot_icon_cache.get(key)
        if out is None:
            out = pygame.transform.rotate(icon, ang_b) if ang_b else icon.copy()
            if alpha_b < 255:
                out.set_alpha(alpha_b)
            if len(self._rot_icon_cache) >= 128:
                self._rot_icon_cache.pop(next(iter(self._rot_icon_cache)))
            self._rot_icon_cache[key] = out
        return out

    def _draw_captions(self) -> None:
        """Subtitle overlay: draw the caption(s) active at the shot playhead.

        Each caption {at_s, duration_s, text, rect?} is authored in the
        Experience Builder (rect = screen-space placement) or generated from
        the script; without a rect it lands in a default bottom band. Gated by
        config `captions_enabled` (default on)."""
        if not self._captions or not self._screen:
            return
        if not self.config.get("captions_enabled", True):
            return
        frame = getattr(self, "_frame_index", None)
        if frame is None:
            return
        t = frame / max(1, self._fps)
        active = [c for c in self._captions
                  if c["at_s"] <= t < c["at_s"] + max(0.1, c["duration_s"])]
        if not active:
            return
        w, h = self._screen.get_size()
        if self._caption_font is None:
            self._caption_font = self._build_caption_font(h)
        for cap in active:
            rect = cap.get("rect")
            if rect:
                rx, ry = int(rect["x"] * w), int(rect["y"] * h)
                rw, rh = int(rect["w"] * w), int(rect["h"] * h)
            else:
                rw, rh = int(w * 0.80), int(h * 0.16)
                rx, ry = (w - rw) // 2, int(h * 0.80)
            self._blit_caption(cap["text"], rx, ry, rw, rh)

    @staticmethod
    def _build_caption_font(h: int):
        if not pygame.font.get_init():
            pygame.font.init()
        # Preference chain resolves to the first installed face; on the
        # Windows kiosk that's Segoe UI Semibold (clean, wide counters —
        # reads well against moving art on a projected wall).
        return pygame.font.SysFont(
            "segoeuisemibold,segoeui,trebuchetms,tahoma,arial",
            max(18, int(h * 0.040)))

    # Finished caption cards kept per (text, rect w, rect h) — a caption is on
    # screen for seconds, and shots rarely carry more than a couple at once.
    _CAPTION_CARD_MAX = 8

    def _blit_caption(self, text: str, rx: int, ry: int, rw: int, rh: int) -> None:
        """One caption card inside the placement rect: word-wrapped text on a
        rounded, semi-opaque panel sized to the text (not the whole rect),
        centred in the rect, with a soft drop shadow under the glyphs.

        The finished card is memoised on (text, rect w, rect h): wrapping,
        two font.render calls per line and a fresh SRCALPHA surface every
        visible frame measured 1.5-4ms/frame. Only the blit position math
        stays live."""
        key = (str(text), rw, rh)
        card = self._caption_render_cache.get(key)
        if card is None:
            card = self._render_caption_card(str(text), rw)
            self._caption_render_cache[key] = card
            while len(self._caption_render_cache) > self._CAPTION_CARD_MAX:
                self._caption_render_cache.popitem(last=False)   # evict oldest
        else:
            self._caption_render_cache.move_to_end(key)

        # Centre the card in the placement rect, clamped on-screen.
        card_w, card_h = card.get_width(), card.get_height()
        sw, sh = self._screen.get_size()
        cx = rx + (rw - card_w) // 2
        cy = ry + max(0, (rh - card_h) // 2)
        cx = max(0, min(cx, sw - card_w))
        cy = max(0, min(cy, sh - card_h))
        self._screen.blit(card, (cx, cy))

    def _render_caption_card(self, text: str, rw: int) -> pygame.Surface:
        """Build one caption card surface for _blit_caption's memo."""
        font = self._caption_font
        pad_x = max(14, int(rw * 0.028))
        pad_y = max(8, pad_x // 2)
        maxw = max(1, rw - 2 * pad_x)
        lines, cur = [], ""
        for word in text.split():
            trial = (cur + " " + word).strip()
            if not cur or font.size(trial)[0] <= maxw:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        lh = font.get_linesize()
        text_w = max((font.size(ln)[0] for ln in lines), default=1)
        card_w = min(rw, text_w + 2 * pad_x)
        card_h = lh * max(1, len(lines)) + 2 * pad_y

        card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        radius = max(3, min(8, card_h // 8))   # gently rounded, not pill-shaped
        # Borderless (Aug 2026, Mike's call): just the semi-opaque panel — the
        # drop shadow under the glyphs carries the separation from the frame.
        pygame.draw.rect(card, P.CAPTION_BG_RGBA,
                         pygame.Rect(0, 0, card_w, card_h), border_radius=radius)

        y = pad_y
        for ln in lines:
            surf = font.render(ln, True, P.LINEN)
            shadow = font.render(ln, True, (0, 0, 0))
            shadow.set_alpha(P.SHADOW_ALPHA)
            lx = (card_w - surf.get_width()) // 2
            card.blit(shadow, (lx + 2, y + 2))
            card.blit(surf, (lx, y))
            y += lh
        return card

    # 8-way direction unit-ish vectors (screen space) shared by the draw
    # indicator and the debug arrows.
    _DIR_VEC_8 = {
        "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
        "up_left": (-1, -1), "up_right": (1, -1),
        "down_left": (-1, 1), "down_right": (1, 1),
    }

    # Torso reference frame for radial point cursors. Same landmark set as
    # engines/detectors/rules/pose_helpers.geometry (shoulders + hips), read
    # inline here to keep this module's imports cheap (see the import block).
    _TORSO_LM = (11, 12, 23, 24)
    _TORSO_TTL_S = 4.0    # matches the cursor last-seen ghost window

    @staticmethod
    def _icon_angle_from_vec(dx, dy):
        """Rotation (deg, pygame CCW) that aims the finger-up icon along
        (dx, dy), or None for the zero vector.

        pygame.transform.rotate is CCW while screen-y is down, and the icon
        points up (screen angle -90deg), so R = -90 - atan2(dy, dx)."""
        if dx == 0 and dy == 0:
            return None
        return -90.0 - math.degrees(math.atan2(dy, dx))

    @classmethod
    def _wants_radial_point(cls, params) -> bool:
        """True for a two-way LEFT/RIGHT choice window with no region_rect —
        the decision blocks (shots 09/37/50), armed by the fork FSM as
        point_region/directional_point with directions ["left","right"].

        Those cursors aim radially OUT from the visitor's torso so each hand
        reads as "push this way". Single-direction windows (the tutorial's
        point LEFT/RIGHT/DOWN steps) deliberately stay upright."""
        if params.get("region_rect"):
            return False
        dirs = params.get("directions")
        return bool(dirs) and set(dirs) == {"left", "right"}

    def _torso_center_screen(self, pose_data, sw, sh):
        """Screen-space torso centre, or None when the pose can't supply one.

        Definition mirrors engines/detectors/rules/pose_helpers.geometry: x
        from the shoulder midpoint (steadier than hips under an occluding
        arm), y from the mean of shoulder-y and hip-y — roughly sternum to
        navel. x is mirrored like every other player-layer draw.

        Stricter than geometry() in one way: all four landmarks must pass the
        visibility gate. geometry() checks presence only, and a phantom hip
        would swing every icon on screen. Latched for _TORSO_TTL_S so it
        survives pose dropouts exactly like the cursor position does."""
        now = time.monotonic()
        pts = []
        if pose_data:
            for idx in self._TORSO_LM:
                if idx >= len(pose_data):
                    pts = []
                    break
                lm = pose_data[idx]
                if getattr(lm, "visibility", 1.0) < 0.5:
                    pts = []
                    break
                pts.append(lm)
        if len(pts) == 4:
            sh_l, sh_r, hip_l, hip_r = pts
            mid_x = (sh_l.x + sh_r.x) / 2.0
            mid_y = ((sh_l.y + sh_r.y) / 2.0 + (hip_l.y + hip_r.y) / 2.0) / 2.0
            self._torso_screen = (int((1 - mid_x) * sw), int(mid_y * sh))
            self._torso_t = now
            return self._torso_screen
        if (self._torso_screen is not None
                and now - self._torso_t <= self._TORSO_TTL_S):
            return self._torso_screen
        return None

    def _point_icon_angle(self, params, x, y, sw, sh, torso=None):
        """Rotation (deg, pygame CCW) to aim the finger-up point icon at its
        target: the mirrored centre of a `region_rect`, else radially outward
        from the torso centre on a two-way left/right choice, else the
        declared `direction`. Returns None (draw upright) when there's nothing
        to aim at — including a hand sitting exactly on the torso centre."""
        rect = params.get("region_rect")
        direction = params.get("direction")
        if rect:
            # Same mirror the indicator ring / debug rect use (display_x = 1-raw_x).
            tx = (1.0 - rect.get("x", 0.0) - rect.get("w", 0.0) / 2.0) * sw
            ty = (rect.get("y", 0.0) + rect.get("h", 0.0) / 2.0) * sh
            dx, dy = tx - x, ty - y
        elif torso is not None and self._wants_radial_point(params):
            # Radial: the ray from the body's centre through this hand, so the
            # icon tilts left/right with the reach and up/down with its height.
            dx, dy = x - torso[0], y - torso[1]
        elif direction in self._DIR_VEC_8:
            dx, dy = self._DIR_VEC_8[direction]
        else:
            return None
        return self._icon_angle_from_vec(dx, dy)

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

    # Dirty-rect padding around the draw-indicator span: every piece of ink
    # (guide line, start ring + pulse, arrowhead wings, comet glow, trail
    # stars) hangs off the span line by well under this — generous on purpose
    # so a tweak to a radius or stroke width can't clip.
    _FX_PAD = 48

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

        # Pre-alpha'd sprite variants + one batched blits() call: set_alpha on
        # the shared sprite plus 200 individual blits cost ~0.5-1ms/frame.
        alive = []
        blits = []
        for p in self._trail_particles:
            age = now - p["born"]
            if age >= p["life"]:
                continue
            k = 1.0 - age / p["life"]
            a_b = min(7, int((k ** 1.5) * 8))       # 8 fade steps (ease-out)
            sprite = self._star_alpha_sprite(p["size_i"], p["rot_i"], a_b)
            x = p["x"] + p["vx"] * age
            y = p["y"] + p["vy"] * age
            blits.append((sprite, (int(x - sprite.get_width() / 2),
                                   int(y - sprite.get_height() / 2))))
            alive.append(p)
        if blits:
            self._screen.blits(blits)
        self._trail_particles = alive

    def _star_alpha_sprite(self, size_i: int, rot_i: int, a_b: int):
        """Star sprite pre-faded to one of 8 alpha buckets (48 tiny surfaces,
        built lazily, so the draw loop never mutates the shared sprites)."""
        cache = getattr(self, "_star_alpha_cache", None)
        if cache is None:
            cache = self._star_alpha_cache = {}
        key = (size_i, rot_i, a_b)
        sprite = cache.get(key)
        if sprite is None:
            sprite = self._star_sprites[size_i][rot_i].copy()
            sprite.set_alpha(int(250 * (a_b + 0.5) / 8))
            cache[key] = sprite
        return sprite

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
            # Dirty-rect composite: clear + blit only the region actually
            # inked, not the whole overlay — a full-screen SRCALPHA fill+blit
            # cost 8-18ms per frame at 1080p, every frame a draw window was
            # armed. The work rect is THIS frame's draw bounds unioned with
            # the PREVIOUS frame's (_fx_dirty latch), so ink left by a moving
            # comet / re-authored span is erased cleanly before redrawing.
            pad = self._FX_PAD
            draw_rect = pygame.Rect(int(min(sx, ex) - pad),
                                    int(min(sy, ey) - pad),
                                    int(abs(ex - sx)) + 2 * pad,
                                    int(abs(ey - sy)) + 2 * pad)
            work = (draw_rect.union(self._fx_dirty)
                    if self._fx_dirty is not None else draw_rect)
            work = work.clip(fx.get_rect())
            if work.w > 0 and work.h > 0:
                fx.fill((0, 0, 0, 0), work)

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
            pygame.draw.circle(fx, (*P.NORTH_STAR, int(230 * a)),
                               (int(px), int(py)), 5)

            if work.w > 0 and work.h > 0:
                self._screen.blit(fx, work.topleft, work)
            clipped = draw_rect.clip(fx.get_rect())
            self._fx_dirty = clipped if clipped.w > 0 and clipped.h > 0 else None
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
            # Soft pulsing ellipse ring around the target area. Authored rects
            # reach ~900x530 px: allocating + filling that SRCALPHA surface
            # every frame cost 2-4ms for the life of the window, so the ring
            # is cached per (size, pulse bucket) — 8 buckets read as a smooth
            # pulse at 30fps.
            bucket = min(7, int(pulse * 8))
            key = (rx, ry, bucket)
            ring = self._ring_cache.get(key)
            if ring is None:
                b = bucket / 7.0
                ring = pygame.Surface((rx * 2 + 20, ry * 2 + 20),
                                      pygame.SRCALPHA)
                pygame.draw.ellipse(ring, (*P.LANTERN, int(90 + 100 * b)),
                                    pygame.Rect(4, 4, rx * 2 + 12, ry * 2 + 12),
                                    width=5 + int(3 * b))
                if len(self._ring_cache) >= 32:   # a few window sizes at most
                    self._ring_cache.pop(next(iter(self._ring_cache)))
                self._ring_cache[key] = ring
            self._screen.blit(ring, (cx - rx - 10, cy - ry - 10))

    # ------------------------------------------------------------------
    # Tutorial step figures (code-drawn vector art)
    # ------------------------------------------------------------------
    #
    # A stylised person performing the step's gesture, drawn in a UNIT box
    # (u, v in 0..1) and scaled into the card's corner panel — same spirit as
    # _POSE_CONNECTIONS below, but hand-authored rather than tracked. Only the
    # arms and the accent change per pose, so six poses cost six dict entries
    # instead of six drawing functions.
    #
    # MIRROR NOTE: this is a DIAGRAM IN SCREEN SPACE, not an anatomical figure.
    # "point_left" reaches toward the LEFT OF THE BOX because the prompt says
    # "point to the LEFT side of the screen" and the visitor sees themselves
    # mirrored. Do not "correct" it to anatomical left.
    _FIG_HEAD = (0.50, 0.17, 0.075)          # cu, cv, radius (in box height)
    _FIG_SPINE = ((0.50, 0.255), (0.50, 0.58))
    _FIG_LEGS = (((0.50, 0.58), (0.39, 0.86)), ((0.50, 0.58), (0.61, 0.86)))
    _FIG_SHOULDER = {"L": (0.395, 0.33), "R": (0.605, 0.33)}
    # Arms hanging at rest — the default for any side a pose doesn't pose.
    _FIG_REST = {"L": ((0.34, 0.45), (0.32, 0.58)),
                 "R": ((0.66, 0.45), (0.68, 0.58))}

    # figure key -> arms {side: (elbow_uv, hand_uv)}, which sides are "doing"
    # the gesture, the accent mark, and the caption under the figure.
    _FIGURES = {
        "raise_both": {
            "arms": {"L": ((0.33, 0.24), (0.30, 0.06)),
                     "R": ((0.67, 0.24), (0.70, 0.06))},
            "active": ("L", "R"), "accent": "spark_hands",
            "caption": "Both hands up",
        },
        "point_target": {
            "arms": {"R": ((0.68, 0.34), (0.76, 0.30))},
            "active": ("R",), "accent": "target_box",
            "caption": "Point and hold",
        },
        "point_left": {
            "arms": {"L": ((0.30, 0.33), (0.20, 0.29))},
            "active": ("L",), "accent": "arrow_left",
            "caption": "Point left",
        },
        "point_right": {
            "arms": {"R": ((0.70, 0.33), (0.80, 0.29))},
            "active": ("R",), "accent": "arrow_right",
            "caption": "Point right",
        },
        "point_down": {
            "arms": {"R": ((0.70, 0.42), (0.76, 0.58))},
            "active": ("R",), "accent": "arrow_down",
            "caption": "Point down",
        },
        # "reach_in" removed with the tutorial's Reach IN step (August 2026);
        # the "ripple" accent drawer remains available for future figures.
    }

    @staticmethod
    def _fig_pt(box, u: float, v: float):
        """Unit-box (u, v) -> pixel point inside `box`."""
        return (int(box.x + u * box.w), int(box.y + v * box.h))

    def _tutorial_figure_box(self, sw: int, sh: int):
        """Panel rect for the step figure — CENTRED under the prompt (Quilt
        Card design, 2a). The centred panel never collides with the K-key
        skeleton mini-panel in the bottom-right, so the old corner-swap rule
        is gone. Always fully on screen."""
        fw = max(200, int(sw * 0.20))
        fh = max(180, int(sh * 0.34))
        bx = (sw - fw) // 2
        by = int(sh * 0.50)
        bx = max(0, min(bx, max(0, sw - fw)))
        by = max(0, min(by, max(0, sh - fh)))
        box = pygame.Rect(bx, by, fw, fh)
        # On a tiny debug window the FIXED-size mini panel can still reach the
        # centre — nudge left just enough to clear it.
        if self._debug or self._show_skeleton:
            pw, ph = self._MINI_PANEL_SIZE
            pm = self._MINI_PANEL_MARGIN
            panel = pygame.Rect(sw - pw - pm, sh - ph - pm, pw, ph)
            if box.colliderect(panel):
                box.x = max(0, panel.left - box.w - 8)
        return box

    def _draw_step_figure(self, figure, box) -> None:
        """Panel + stick figure + accent + caption for one tutorial step.

        STATIC — no animation. The figure is a reference diagram the visitor
        reads once; a moving one competes with the live hand cursors and reads
        as noise on a projected wall.

        Everything is drawn into an inset ART rect, never the panel rect, so
        no limb or arrowhead can crowd the border, and the caption gets a
        reserved band of its own. No-op for a missing or unknown figure key."""
        spec = self._FIGURES.get(figure) if figure else None
        if spec is None or not self._screen:
            return

        panel = pygame.Surface((box.w, box.h), pygame.SRCALPHA)
        radius = max(4, int(box.h * 0.06))
        pygame.draw.rect(panel, (*P.CLOTH, 235), panel.get_rect(),
                         border_radius=radius)
        pygame.draw.rect(panel, P.EDGE_RGBA, panel.get_rect(), width=1,
                         border_radius=radius)
        self._screen.blit(panel, box.topleft)

        # Inset drawing area: breathing room on every side, plus a caption band.
        pad = max(8, int(min(box.w, box.h) * 0.10))
        font = getattr(self, "_tut_label_font", None) or self._small_font
        cap_h = (font.get_linesize() + pad // 2) if font else pad
        art = pygame.Rect(box.x + pad, box.y + pad,
                          box.w - 2 * pad, box.h - 2 * pad - cap_h)
        if art.w <= 0 or art.h <= 0:
            return

        stroke = max(2, int(art.h * 0.028))
        active = spec.get("active", ())
        arms = {**self._FIG_REST, **spec["arms"]}

        def line(a, b, color, w):
            pygame.draw.line(self._screen, color, self._fig_pt(art, *a),
                             self._fig_pt(art, *b), w)

        # Body — dim, so the acting limb reads first.
        cu, cv, r = self._FIG_HEAD
        pygame.draw.circle(self._screen, P.LANTERN_DIM, self._fig_pt(art, cu, cv),
                           max(3, int(r * art.h)), stroke)
        line(*self._FIG_SPINE, P.LANTERN_DIM, stroke)
        for a, b in self._FIG_LEGS:
            line(a, b, P.LANTERN_DIM, stroke)

        for side, (elbow, hand) in arms.items():
            color = P.LANTERN if side in active else P.LANTERN_DIM
            w = stroke + 1 if side in active else stroke
            line(self._FIG_SHOULDER[side], elbow, color, w)
            line(elbow, hand, color, w)
            if side in active:
                pygame.draw.circle(self._screen, color,
                                   self._fig_pt(art, *hand),
                                   max(2, int(art.h * 0.022)))

        self._draw_figure_accent(spec, arms, art, stroke)

        caption = spec.get("caption")
        if caption and font:
            surf = font.render(caption, True, P.LINEN_DIM)
            self._screen.blit(surf, (box.x + (box.w - surf.get_width()) // 2,
                                     box.bottom - pad // 2 - cap_h))

    # Accent -> the direction its arrow points (screen space).
    _ACCENT_DIR = {"arrow_left": (-1.0, 0.0), "arrow_right": (1.0, 0.0),
                   "arrow_down": (0.0, 1.0)}

    def _draw_figure_accent(self, spec, arms, art, stroke: int) -> None:
        """The mark that says what the gesture DOES: a directional arrow off
        the hand, the box being pointed at, sparks over raised hands, or
        rings travelling toward the viewer. Static, and sized to stay inside
        the art rect."""
        accent = spec.get("accent")
        active = spec.get("active", ())
        hands = [arms[s][1] for s in active if s in arms]
        if not hands:
            return

        if accent in self._ACCENT_DIR:
            dx, dy = self._ACCENT_DIR[accent]
            hx, hy = hands[0]
            sx, sy = self._fig_pt(art, hx + dx * 0.04, hy + dy * 0.04)
            ex, ey = self._fig_pt(art, hx + dx * 0.15, hy + dy * 0.15)
            pygame.draw.line(self._screen, P.LANTERN, (sx, sy), (ex, ey),
                             stroke + 1)
            ang = math.atan2(ey - sy, ex - sx)
            head = max(6, int(art.h * 0.055))
            # Shared arrowhead helper — wings land behind the tip.
            for wing in self._arrow_wings(ex, ey, ang, head):
                pygame.draw.line(self._screen, P.LANTERN, (ex, ey), wing,
                                 stroke + 1)

        elif accent == "target_box":
            # Oval, matching the card's glowing target and the runtime's
            # pulsing target rings.
            hx, hy = hands[0]
            bw, bh = int(art.w * 0.15), int(art.h * 0.15)
            bx, by = self._fig_pt(art, hx + 0.05, hy - 0.07)
            tgt = pygame.Surface((bw, bh), pygame.SRCALPHA)
            pygame.draw.ellipse(tgt, (*P.LANTERN, 60), tgt.get_rect())
            self._screen.blit(tgt, (bx, by))
            pygame.draw.ellipse(self._screen, P.LANTERN, (bx, by, bw, bh),
                                max(2, stroke - 1))

        elif accent == "spark_hands":
            for hx, hy in hands:
                pygame.draw.circle(self._screen, P.LANTERN,
                                   self._fig_pt(art, hx, hy),
                                   max(3, int(art.h * 0.030)))

        elif accent == "ripple":
            hx, hy = hands[0]
            cx, cy = self._fig_pt(art, hx, hy)
            # Concentric rings, brightest at the hand — motion toward the viewer.
            for frac, alpha in ((0.055, 255), (0.105, 170), (0.155, 95)):
                rad = int(art.h * frac)
                if rad <= 0:
                    continue
                ring = pygame.Surface((rad * 2 + 4, rad * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(ring, (*P.LANTERN, alpha),
                                   (rad + 2, rad + 2), rad, max(2, stroke - 1))
                self._screen.blit(ring, (cx - rad - 2, cy - rad - 2))

    # Serif stack for the piece's display type (Camera & Tutorial redesign,
    # Aug 2026 — the design canvas sets Cormorant Garamond / Lora; the kiosk
    # ships no webfonts, so this is the closest installed-serif ladder).
    _SERIF_STACK = "garamond,georgia,palatinolinotype,book antiqua,timesnewroman"

    def _serif_font(self, size: int, italic: bool = False,
                    bold: bool = False) -> pygame.font.Font:
        key = (size, italic, bold)
        cache = getattr(self, "_serif_cache", None)
        if cache is None:
            cache = self._serif_cache = {}
        f = cache.get(key)
        if f is None:
            f = pygame.font.SysFont(self._SERIF_STACK, size,
                                    bold=bold, italic=italic)
            cache[key] = f
        return f

    def _tracked_label(self, font, segments, tracking: int) -> pygame.Surface:
        """Letter-spaced label (pygame has no tracking). segments is
        [(text, color), ...] so one label can mix colours (the amber READY
        inside a faint hint line)."""
        glyphs = []
        for text, color in segments:
            for ch in text:
                glyphs.append(font.render(ch, True, color))
        w = sum(g.get_width() for g in glyphs) + tracking * max(0, len(glyphs) - 1)
        surf = pygame.Surface((max(1, w), font.get_linesize()), pygame.SRCALPHA)
        x = 0
        for g in glyphs:
            surf.blit(g, (x, 0))
            x += g.get_width() + tracking
        return surf

    def _tutorial_fonts(self, sh: int):
        """Screen-relative tutorial fonts, rebuilt when the display height
        changes. Serif display type per the Quilt Card design (2a); the label
        font stays sans for the small tracked captions."""
        if getattr(self, "_tut_font_h", None) != sh:
            self._tut_title_font = self._serif_font(max(30, int(sh * 0.098)))
            self._tut_prompt_font = self._serif_font(max(18, int(sh * 0.031)))
            self._tut_label_font = pygame.font.SysFont(
                "arial", max(12, int(sh * 0.020)), bold=True)
            self._tut_font_h = sh
        return self._tut_title_font, self._tut_prompt_font, self._tut_label_font

    def _draw_tutorial_card(self, card: dict) -> None:
        """Full-screen code-rendered tutorial/calibration card. The card dict
        comes from TutorialEngine.card_info()."""
        sw, sh = self._screen.get_size()
        self._screen.fill(P.NIGHT)
        title_font, prompt_font, _ = self._tutorial_fonts(sh)
        # One pulse for the whole card so the target box and the figure's
        # arrow breathe in phase.
        pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 4.0)

        # Quilt-block progress row (2a): one diamond per step, the current one
        # a filled lantern patch, the rest outlined seams.
        step, total = card.get("step"), card.get("total")
        if step and total:
            side = max(7, int(sh * 0.020))
            # A rotated square spans 2*side — centres need that plus daylight.
            gap = side * 2 + max(10, int(sh * 0.020))
            cx0 = sw // 2 - gap * (total - 1) // 2
            cy = int(sh * 0.115)
            for i in range(total):
                cx = cx0 + i * gap
                pts = [(cx, cy - side), (cx + side, cy),
                       (cx, cy + side), (cx - side, cy)]
                if i + 1 == step:
                    pygame.draw.polygon(self._screen, P.LANTERN, pts)
                else:
                    pygame.draw.polygon(self._screen, P.LANTERN_DIM, pts, 1)

        title = card.get("title", "")
        if title:
            surf = title_font.render(title, True, P.NORTH_STAR)
            self._screen.blit(surf, ((sw - surf.get_width()) // 2, int(sh * 0.17)))

        prompt = card.get("prompt", "")
        if prompt:
            lh = int(prompt_font.get_linesize() * 1.35)   # airy leading (2a)
            for li, line in enumerate(self._wrap_text(prompt, prompt_font,
                                                      int(sw * 0.58))):
                surf = prompt_font.render(line, True, P.LINEN_DIM)
                self._screen.blit(surf, ((sw - surf.get_width()) // 2,
                                         int(sh * 0.33) + li * lh))

        # Optional on-card target box (player-space rect) for the pointing steps.
        rect = card.get("target_rect")
        if rect:
            # Drawn as a glowing OVAL inscribed in the detection rect — the
            # detector still tests the rect, but the visitor-facing target
            # speaks the same circular language as the runtime's pulsing
            # target rings.
            rx0 = int(rect["x"] * sw)
            ry0 = int(rect["y"] * sh)
            rw  = int(rect["w"] * sw)
            rh  = int(rect["h"] * sh)
            fill = pygame.Surface((rw, rh), pygame.SRCALPHA)
            pygame.draw.ellipse(fill, (*P.LANTERN, 30 + int(30 * pulse)),
                                fill.get_rect())
            self._screen.blit(fill, (rx0, ry0))
            pygame.draw.ellipse(self._screen, P.LANTERN,
                                (rx0, ry0, rw, rh), 4 + int(2 * pulse))

        # (The old big background hand icon is gone — the centred step figure
        # is the demonstration now, and the icon drew right behind it.)

        # Centred panel: a static vector figure performing this step's gesture,
        # for visitors who won't read the prompt.
        self._draw_step_figure(card.get("figure"),
                               self._tutorial_figure_box(sw, sh))

        # Skip hint, tracked small caps (the diamonds carry the step count).
        if self._small_font:
            hint = self._tracked_label(
                self._small_font,
                [('S TO SKIP  ·  OR SAY "SKIP"', P.LINEN_FAINT)], 3)
            self._screen.blit(hint, ((sw - hint.get_width()) // 2, int(sh * 0.93)))

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

    def _camera_setup_overlay(self, sw: int, sh: int) -> pygame.Surface:
        """Static dressing for the Lantern Vignette camera screen (1a), baked
        once per display size: an inset vignette pulling the frame into
        darkness, the four lantern corner brackets, and the bottom gradient
        band the prompt sits on."""
        cached = getattr(self, "_cam_overlay", None)
        if cached is not None and cached.get_size() == (sw, sh):
            return cached
        import numpy as np

        fade = max(60, int(min(sw, sh) * 0.26))
        yy = np.minimum(np.arange(sh), sh - 1 - np.arange(sh))
        xx = np.minimum(np.arange(sw), sw - 1 - np.arange(sw))
        edge = np.minimum.outer(yy, xx).astype(np.float32) / fade
        alpha = np.clip(1.0 - edge, 0.0, 1.0) ** 1.6 * 225
        # Bottom band: the prompt's gradient (to top, ~0.95 -> transparent).
        band_h = max(1, int(sh * 0.24))
        band = np.zeros(sh, np.float32)
        band[sh - band_h:] = np.linspace(0.0, 1.0, band_h) * 242
        alpha = np.maximum(alpha, band[:, None])

        rgba = np.empty((sh, sw, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = P.NIGHT_DEEP
        rgba[..., 3] = alpha.astype(np.uint8)
        overlay = pygame.image.frombuffer(rgba.tobytes(), (sw, sh),
                                          "RGBA").convert_alpha()

        m, arm = max(16, int(sh * 0.053)), max(18, int(sh * 0.058))
        bracket = (*P.LANTERN, 153)
        for cx, sx in ((m, 1), (sw - m, -1)):
            for cy, sy in ((m, 1), (sh - m, -1)):
                pygame.draw.line(overlay, bracket, (cx, cy),
                                 (cx + sx * arm, cy))
                pygame.draw.line(overlay, bracket, (cx, cy),
                                 (cx, cy + sy * arm))
        self._cam_overlay = overlay
        return overlay

    def draw_camera_setup_1a(self, frame_bgr, pose_lm) -> bool:
        """Lantern Vignette camera-setup design (1a) — KEPT but NOT CALLED.

        The live screen is draw_camera_setup below (1c North Star Arch,
        Mike's pick Aug 2026); to revert to 1a, swap which method main.py's
        caller reaches (rename this back over draw_camera_setup). Same
        contract: mirrored live view, skeleton, body-in-frame verdict."""
        if not self._screen:
            return False
        import numpy as np

        sw, sh = self._screen.get_size()
        self._screen.fill(P.NIGHT_DEEP)

        disp = None   # (x, y, w, h) of the displayed camera rect
        if frame_bgr is not None:
            # BGR -> RGB and mirror in one slice so the view behaves like a mirror.
            rgb = np.ascontiguousarray(frame_bgr[:, ::-1, ::-1])
            fh, fw = rgb.shape[:2]
            surf = pygame.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB")
            scale = min(sw / fw, sh / fh)   # full-bleed feed; vignette frames it
            dw, dh = int(fw * scale), int(fh * scale)
            dx, dy = (sw - dw) // 2, (sh - dh) // 2
            self._screen.blit(pygame.transform.scale(surf, (dw, dh)), (dx, dy))
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
            # Lantern skeleton; success green stays the one learned "you did
            # it" signal, so a framed body still reads green.
            color = P.SUCCESS if body_ok else P.LANTERN
            for a, b in self._POSE_CONNECTIONS_FULL:
                if _vis(a) and _vis(b):
                    pygame.draw.line(self._screen, color, pts[a], pts[b], 2)
            for idx, lm in enumerate(pose_lm):
                if _vis(idx):
                    pygame.draw.circle(self._screen, color, pts[idx], 4)

        # Vignette + brackets + bottom gradient over the feed and skeleton.
        self._screen.blit(self._camera_setup_overlay(sw, sh), (0, 0))

        # Header: tracked small caps flanked by hairline rules.
        if self._small_font:
            label = self._tracked_label(self._small_font,
                                        [("CAMERA SETUP", P.LANTERN_DIM)], 5)
            lx = (sw - label.get_width()) // 2
            ly = int(sh * 0.055)
            cy = ly + label.get_height() // 2
            rule_w, gap = max(40, int(sw * 0.075)), 16
            for x0 in (lx - gap - rule_w, lx + label.get_width() + gap):
                seam = pygame.Surface((rule_w, 1), pygame.SRCALPHA)
                seam.fill(P.EDGE_RGBA)
                self._screen.blit(seam, (x0, cy))
            self._screen.blit(label, (lx, ly))

        # Prompt block on the gradient band: serif italic line + tracked hint.
        if pose_lm is None:
            msg = "Step into the lantern's view."
        elif not body_ok:
            msg = "Step back until your whole body rests in the frame."
        else:
            msg = "The frame holds all of you — continue when ready."
        prompt_font = self._serif_font(max(20, int(sh * 0.052)), italic=True)
        line = prompt_font.render(msg, True,
                                  P.SUCCESS if body_ok else P.LINEN)
        self._screen.blit(line, ((sw - line.get_width()) // 2, int(sh * 0.845)))
        if self._small_font:
            hint = self._tracked_label(
                self._small_font,
                [("ENTER · SPACE · OR SAY ", P.LINEN_FAINT),
                 ('"READY"', P.LANTERN)], 3)
            self._screen.blit(hint, ((sw - hint.get_width()) // 2, int(sh * 0.925)))

        pygame.display.flip()
        return body_ok

    def _arch_mask(self, w: int, h: int) -> pygame.Surface:
        """Arch-shaped alpha mask (rounded-semicircle top, near-square feet)
        for the North Star Arch viewport. Cached per size."""
        cached = getattr(self, "_arch_mask_cache", None)
        if cached is not None and cached.get_size() == (w, h):
            return cached
        mask = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(),
                         border_top_left_radius=w // 2,
                         border_top_right_radius=w // 2,
                         border_bottom_left_radius=max(3, h // 72),
                         border_bottom_right_radius=max(3, h // 72))
        self._arch_mask_cache = mask
        return mask

    def draw_camera_setup(self, frame_bgr, pose_lm) -> bool:
        """Camera-setup screen (config.json "camera_setup"), North Star Arch
        design (1c): the mirrored live view held inside an arched window
        under a glowing North Star on the night ground, with the skeleton
        drawn inside the arch. Confirmed by ENTER/Space or the voice command
        "ready" (handled in main.py). Returns the body-in-frame verdict.
        (1a Lantern Vignette is kept above as draw_camera_setup_1a.)

        Owns the whole frame: fills, draws and flips — the caller skips the
        normal render update while this screen is active."""
        if not self._screen:
            return False
        import numpy as np

        sw, sh = self._screen.get_size()
        self._screen.fill(P.NIGHT)

        # Arch geometry (design: 220x290 in a 450-high canvas, centred).
        ah = int(sh * 0.62)
        aw = int(ah * (220 / 290))
        ax = (sw - aw) // 2
        ay = int(sh * 0.155)

        # Body-in-frame verdict: head and both ankles confidently inside view.
        def _in_frame(idx):
            if pose_lm is None or idx >= len(pose_lm):
                return False
            lm = pose_lm[idx]
            return (getattr(lm, "visibility", 0.0) >= 0.5
                    and 0.02 <= lm.x <= 0.98 and 0.02 <= lm.y <= 0.98)
        body_ok = _in_frame(0) and _in_frame(27) and _in_frame(28)

        # Compose feed + skeleton on an arch-sized layer, then mask to shape.
        comp = pygame.Surface((aw, ah), pygame.SRCALPHA)
        comp.fill((*P.NIGHT_DEEP, 255))
        ox = oy = 0
        dw, dh = aw, ah
        if frame_bgr is not None:
            # BGR -> RGB and mirror in one slice so the view behaves like a
            # mirror; scaled to COVER the arch (crop, don't letterbox).
            rgb = np.ascontiguousarray(frame_bgr[:, ::-1, ::-1])
            fh, fw = rgb.shape[:2]
            surf = pygame.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB")
            scale = max(aw / fw, ah / fh)
            dw, dh = int(fw * scale), int(fh * scale)
            ox, oy = (aw - dw) // 2, (ah - dh) // 2
            comp.blit(pygame.transform.scale(surf, (dw, dh)), (ox, oy))
        if pose_lm is not None:
            pts = [(ox + int((1 - lm.x) * dw), oy + int(lm.y * dh))
                   for lm in pose_lm]
            def _vis(idx):
                return (idx < len(pose_lm)
                        and getattr(pose_lm[idx], "visibility", 0.0) >= 0.5)
            # Lantern skeleton; success green stays the learned "you did it".
            color = P.SUCCESS if body_ok else P.LANTERN
            for a, b in self._POSE_CONNECTIONS_FULL:
                if _vis(a) and _vis(b):
                    pygame.draw.line(comp, color, pts[a], pts[b], 2)
            for idx in range(len(pose_lm)):
                if _vis(idx):
                    pygame.draw.circle(comp, color, pts[idx], 3)
        comp.blit(self._arch_mask(aw, ah), (0, 0),
                  special_flags=pygame.BLEND_RGBA_MULT)
        self._screen.blit(comp, (ax, ay))
        # Arch outline — a thin lantern seam.
        pygame.draw.rect(self._screen, P.LANTERN_DIM,
                         (ax, ay, aw, ah), 1,
                         border_top_left_radius=aw // 2,
                         border_top_right_radius=aw // 2,
                         border_bottom_left_radius=max(3, ah // 72),
                         border_bottom_right_radius=max(3, ah // 72))

        # The North Star above the arch: soft lantern glow + 4-point star.
        cx, cy = sw // 2, int(ay - sh * 0.055)
        r = max(6, int(sh * 0.020))
        glow = pygame.Surface((r * 8, r * 8), pygame.SRCALPHA)
        for i in range(6):   # many faint rings read as one soft halo
            gr = int(r * (3.4 - i * 0.45))
            pygame.draw.circle(glow, (*P.LANTERN, 8 + i * 7),
                               (r * 4, r * 4), gr)
        self._screen.blit(glow, (cx - r * 4, cy - r * 4))
        ri = max(2, int(r * 0.30))
        pygame.draw.polygon(self._screen, P.NORTH_STAR,
                            [(cx, cy - r), (cx + ri, cy - ri), (cx + r, cy),
                             (cx + ri, cy + ri), (cx, cy + r),
                             (cx - ri, cy + ri), (cx - r, cy),
                             (cx - ri, cy - ri)])

        # Prompt: serif italic line + tracked hint under the arch.
        if pose_lm is None:
            msg = "Step into the lantern's view."
        elif not body_ok:
            msg = "Let the frame hold all of you."
        else:
            msg = "The frame holds all of you — continue when ready."
        prompt_font = self._serif_font(max(18, int(sh * 0.048)), italic=True)
        line = prompt_font.render(msg, True,
                                  P.SUCCESS if body_ok else P.LINEN)
        self._screen.blit(line, ((sw - line.get_width()) // 2, int(sh * 0.835)))
        if self._small_font:
            hint = self._tracked_label(
                self._small_font,
                [("ENTER · SPACE · SAY ", P.LINEN_FAINT),
                 ('"READY"', P.LANTERN)], 3)
            self._screen.blit(hint, ((sw - hint.get_width()) // 2, int(sh * 0.915)))

        pygame.display.flip()
        return body_ok

    # Panel geometry — the tutorial figure box reads this to avoid the corner.
    _MINI_PANEL_SIZE = (240, 180)
    _MINI_PANEL_MARGIN = 10

    def _draw_hand_mini_panel(self, pose_data=None):
        """Pose-skeleton preview in bottom-right corner — keeps the main screen
        clean.

        DELIBERATELY UNTHEMED: this and the other diagnostic surfaces (debug
        overlay, camera setup, the tuners) are instruments, not part of the
        piece. They keep their high-contrast diagnostic colours — don't
        "finish the job" by running engines/palette.py through them."""
        if not self._screen:
            return
        sw, sh = self._screen.get_size()
        panel_w, panel_h = self._MINI_PANEL_SIZE
        pad = 10
        px = sw - panel_w - self._MINI_PANEL_MARGIN
        py = sh - panel_h - self._MINI_PANEL_MARGIN

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
