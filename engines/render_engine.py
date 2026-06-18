"""
RenderEngine — Pygame-based frame sequencer and display layer.

Loads frame sequences from the current scene's frames/ directory,
advances frames at the scene's fps, and responds to render_events
emitted by the NarrationEngine (e.g. quilt_illuminate, door_widens).

Freeze-frame mechanic:
  When the narration engine enters a wait_for_interaction cue that has a
  freeze_frame_page, it emits a `freeze_frame` event with {"page": N} where
  N is the PDF-header page number (e.g. 58). The render engine looks up that
  page in its page-to-index map (built from the storyboard filename convention
  page_NNN.jpg = PDF page NNN-1), jumps to that frame, and holds it.
  `freeze_release` unfreezes. `page_jump` is a one-shot jump without freezing
  (used for branch transitions and stroke-lock-in pages).

Debug overlay draws hand landmarks when config.debug_overlay is true.
"""

import os
import time
import pygame
from PIL import Image
from pathlib import Path
from typing import Optional

from .frame_cache import FrameCacheManager

PLAY_THRESHOLD = 100  # raw frames buffered before playback starts (drip mode kicks in after)


class RenderEngine:
    def __init__(self, config: dict, event_bus: "EventBus"):
        self.config = config
        self.event_bus = event_bus
        self._screen: Optional[pygame.Surface] = None
        self._frames: list = []
        self._frame_index = 0
        self._fps = 24
        self._last_frame_time = 0.0
        profile_perf = config.get("_profile", {}).get("performance", {})
        self._debug = profile_perf.get("debug_overlay", config.get("debug_overlay", False))
        self._pending_events: list = []
        self._landmark_data = None
        self._font: Optional[pygame.font.Font] = None
        self._small_font: Optional[pygame.font.Font] = None
        self._pending_load_paths: list = []
        self._pending_load_fps: float = 0.5
        self._playback_start_time: float = 0.0
        self._pending_audio: Optional[str] = None
        self._current_frames_dir: Optional[Path] = None

        # Look-ahead frame cache (continuous background preload of all shots with art)
        self._cache: Optional[FrameCacheManager] = None
        self._loading_dir: Optional[Path] = None   # incoming shot being converted
        self._loading_frames: list = []            # Surfaces built for the incoming shot
        self._raw_cursor: int = 0                  # raw frames converted so far
        self._convert_chunk = config.get("frame_convert_chunk", 120)  # surfaces/ tick during swap
        self._drip_active: bool = False            # True after early-start swap; drip-feeds remaining frames

        # OI flash overlay
        self._flash_color: tuple = (0, 255, 80)
        self._flash_start: float = 0.0
        self._flash_until: float = 0.0
        self._flash_alpha: int   = 80   # peak alpha (0-255)

        # Frame-gated OI window (1-based frame numbers, matching filenames)
        self._oi_frame_start: Optional[int] = None
        self._oi_frame_end:   Optional[int] = None
        self._oi_window_open: bool = False

        # Freeze-frame state
        self._page_to_frame_index: dict = {}   # PDF page number → frame list index
        self._freeze_frame_index: Optional[int] = None
        self._freeze_active: bool = False
        self._current_freeze_page: Optional[int] = None  # for debug display

        # Play-through freeze: frames advance naturally until reaching this index
        self._freeze_on_page_index: Optional[int] = None
        self._freeze_on_page_target: Optional[int] = None  # PDF page for debug

        # FSM segment-constrained playback (set by play_segment event)
        self._seg_start:  Optional[int] = None  # inclusive start frame index
        self._seg_end:    Optional[int] = None  # inclusive end frame index
        self._seg_loop:   bool = True
        self._seg_anchor: float = 0.0           # time.monotonic() when segment started
        self._seg_done:   bool = False

        self.event_bus.subscribe("shot_load",     self._on_shot_load)
        self.event_bus.subscribe("prefetch_shot", self._on_prefetch_shot)
        self.event_bus.subscribe("oi_flash",          self._on_oi_flash)
        self.event_bus.subscribe("play_sfx",          self._on_play_sfx)
        self.event_bus.subscribe("set_frame_window",  self._on_set_frame_window)
        self.event_bus.subscribe("play_segment",      self._on_play_segment)
        self.event_bus.subscribe("scene_load",    self._on_scene_load)
        self.event_bus.subscribe("dev_frames_load", self._on_dev_frames_load)
        self.event_bus.subscribe("render_event",  self._on_render_event)
        self.event_bus.subscribe("freeze_frame",  self._on_freeze_frame)
        self.event_bus.subscribe("freeze_on_page", self._on_freeze_on_page)
        self.event_bus.subscribe("freeze_release", self._on_freeze_release)
        self.event_bus.subscribe("page_jump",     self._on_page_jump)

    def init_display(self):
        pygame.display.init()
        pygame.font.init()
        display_cfg = self.config.get("_profile", {}).get("display", {})
        w, h = display_cfg.get("resolution") or self.config.get("resolution", [1920, 1080])
        flags = pygame.FULLSCREEN if display_cfg.get("fullscreen", False) else 0
        self._screen = pygame.display.set_mode((w, h), flags)
        pygame.display.set_caption("Black Heritage Reclaimed")
        self._font = pygame.font.SysFont("monospace", 20, bold=True)
        self._small_font = pygame.font.SysFont("monospace", 14, bold=True)

    def update(self, landmark_data=None, handedness_data=None,
               gesture_debug: dict | None = None,
               scene_debug: dict | None = None, voice_debug: dict | None = None,
               narration_debug: dict | None = None):
        if not self._screen:
            return

        self._landmark_data = landmark_data
        now = time.monotonic()

        # ── Look-ahead cache: convert the incoming shot's frames to Surfaces only
        #    once the cache has it FULLY decoded, then swap it in atomically. Until
        #    then the previous shot's last frame stays frozen on screen — we never
        #    play partially-loaded frames or fall back to whatever happens to be
        #    decoded. Conversion is chunked so the swap doesn't hitch the loop.
        self._service_loading()

        playing = bool(self._frames) and (self._loading_dir is None or self._drip_active)

        if playing:
            if self._freeze_active and self._freeze_frame_index is not None:
                # Hold the freeze frame — only jump if that index is already loaded
                if self._freeze_frame_index < len(self._frames):
                    self._frame_index = self._freeze_frame_index

            elif self._seg_start is not None:
                # FSM segment-constrained playback: lock to a named frame range.
                # Frames are guaranteed fully loaded here (a shot only begins playing
                # after its whole frame set is cached), so no partial-load handling.
                seg_len  = max(1, self._seg_end - self._seg_start + 1)
                elapsed  = now - self._seg_anchor
                raw_local = int(elapsed * self._fps)
                if self._seg_loop:
                    local = raw_local % seg_len
                else:
                    local = min(raw_local, seg_len - 1)
                    if not self._seg_done and raw_local >= seg_len:
                        self._seg_done = True
                        print(f"[RenderEngine] segment_playback_done "
                              f"[{self._seg_start}–{self._seg_end}]")
                        self.event_bus.emit("segment_playback_done", {})
                # Guard: the emit above can synchronously advance to the next shot,
                # clearing _frames/_seg_start via _on_shot_load.
                if self._seg_start is not None and self._frames:
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

                # Play-through freeze: activate when time-based playback reaches target
                if (self._freeze_on_page_index is not None and
                        self._freeze_on_page_index < len(self._frames) and
                        self._frame_index >= self._freeze_on_page_index):
                    self._frame_index = self._freeze_on_page_index
                    self._freeze_frame_index = self._freeze_on_page_index
                    self._freeze_active = True
                    self._current_freeze_page = self._freeze_on_page_target
                    self._freeze_on_page_index = None
                    self._freeze_on_page_target = None
                    self.event_bus.emit("freeze_activated", {})

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

        # OI flash overlay — fade in then fade out using a sine curve
        if now < self._flash_until:
            duration = self._flash_until - self._flash_start
            progress = (now - self._flash_start) / duration if duration > 0 else 1.0
            import math as _math
            alpha = int(self._flash_alpha * _math.sin(progress * _math.pi))
            alpha = max(0, min(255, alpha))
            sw, sh = self._screen.get_size()
            overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
            overlay.fill((*self._flash_color, alpha))
            self._screen.blit(overlay, (0, 0))

        if landmark_data:
            self._draw_index_cursors(landmark_data, handedness_data)

        if self._debug:
            self._draw_oi_target_flag(gesture_debug)
            self._draw_debug_panel(landmark_data, gesture_debug, voice_debug, narration_debug)
            if scene_debug:
                self._draw_scene_panel(scene_debug)

        if landmark_data:
            self._draw_hand_mini_panel(landmark_data, handedness_data)

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Freeze / page-jump event handlers
    # ------------------------------------------------------------------

    def _on_freeze_frame(self, data: dict):
        page = data.get("page")
        self._current_freeze_page = page
        if page is not None:
            idx = self._page_to_frame_index.get(page)
            if idx is not None:
                self._freeze_frame_index = idx
                if idx < len(self._frames):
                    self._frame_index = idx
            else:
                # Page not yet in map (still loading) — freeze at current position
                self._freeze_frame_index = self._frame_index
        else:
            self._freeze_frame_index = self._frame_index
        self._freeze_active = True

    def _on_freeze_on_page(self, data: dict):
        """Schedule a play-through freeze: frames advance naturally until reaching this page."""
        page = data.get("page")
        self._freeze_on_page_target = page
        if page is not None:
            idx = self._page_to_frame_index.get(page)
            if idx is not None:
                self._freeze_on_page_index = idx
            else:
                # Page not in map — freeze immediately at current position as fallback
                self._freeze_frame_index = self._frame_index
                self._freeze_active = True
                self._current_freeze_page = page
                self.event_bus.emit("freeze_activated", {})

    def _on_freeze_release(self, data: dict):
        self._freeze_active = False
        self._freeze_frame_index = None
        self._current_freeze_page = None
        self._freeze_on_page_index = None
        self._freeze_on_page_target = None

    def _on_page_jump(self, data: dict):
        page = data.get("page")
        if page is not None:
            idx = self._page_to_frame_index.get(page)
            if idx is not None and idx < len(self._frames):
                self._frame_index = idx
                self._last_frame_time = time.monotonic()
        # One-shot: does not activate freeze

    # ------------------------------------------------------------------
    # Scene / frame loading
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
        self._cache = FrameCacheManager(self._get_resolution())
        self._cache.start(dirs)

    def _service_loading(self) -> None:
        """Convert the incoming shot's cached raw frames to Surfaces, with early-start drip.

        Once PLAY_THRESHOLD raw frames are buffered by the cache worker, conversion
        begins immediately. As soon as PLAY_THRESHOLD surfaces are ready, playback
        starts (audio fires, shot_frames_ready emits) even if the rest of the shot
        hasn't decoded yet. Remaining frames are drip-fed into the live _frames list
        while the shot plays — _frames and _loading_frames are the same list object
        during drip mode, so appends are visible to the render loop instantly.

        For shots shorter than PLAY_THRESHOLD the old behaviour is preserved: wait
        for is_complete, then convert all and swap.
        """
        if self._loading_dir is None or self._cache is None:
            return

        total     = self._cache.total_count(self._loading_dir) or 0
        available = self._cache.loaded_count(self._loading_dir)

        # Gate: wait until enough raw frames are buffered.
        if total > 0 and total <= PLAY_THRESHOLD:
            # Short shot — wait for full decode before converting.
            if not self._cache.is_complete(self._loading_dir):
                return
        elif available < PLAY_THRESHOLD:
            return  # not enough buffered yet for early start

        # Convert a chunk of raw frames to Surfaces.
        end = min(available, self._raw_cursor + self._convert_chunk)
        if end > self._raw_cursor:
            for data, size in self._cache.get_raw_slice(
                    self._loading_dir, self._raw_cursor, end - self._raw_cursor):
                surf = pygame.image.fromstring(data, size, "RGB")
                if self._drip_active:
                    self._frames.append(surf)       # live list — render loop sees it immediately
                else:
                    self._loading_frames.append(surf)
            self._raw_cursor = end

        # Early-start swap: enough Surfaces converted to begin playing.
        swap_threshold = min(PLAY_THRESHOLD, total) if total > 0 else PLAY_THRESHOLD
        if not self._drip_active and len(self._loading_frames) >= swap_threshold:
            self._frames              = self._loading_frames   # same list object — drip appends work
            self._frame_index         = 0
            self._playback_start_time = time.monotonic()
            self._begin_audio()
            self.event_bus.emit("shot_frames_ready", {})
            self._drip_active = True
            print(f"[RenderEngine] early start: {len(self._frames)} frames buffered, "
                  f"{total - self._raw_cursor} remaining")

        # Fully converted: tear down loading state.
        is_done = (self._cache.is_complete(self._loading_dir) and
                   self._raw_cursor >= self._cache.loaded_count(self._loading_dir))
        if is_done:
            if not self._drip_active:
                # Short shot — swap in now (waited for full load above).
                self._frames              = self._loading_frames
                self._frame_index         = 0
                self._playback_start_time = time.monotonic()
                self._begin_audio()
                self.event_bus.emit("shot_frames_ready", {})
                print(f"[RenderEngine] shot ready: {len(self._frames)} frames")
            else:
                print(f"[RenderEngine] drip complete: {len(self._frames)} frames total")
            self._loading_frames = []
            self._loading_dir    = None
            self._drip_active    = False

    def _begin_audio(self) -> None:
        if self._pending_audio:
            try:
                sound = pygame.mixer.Sound(self._pending_audio)
                pygame.mixer.Channel(0).play(sound)
            except Exception as exc:
                print(f"[RenderEngine] audio load failed: {exc}")
            self._pending_audio = None

    def _on_shot_load(self, data: dict):
        shot = data.get("shot")
        if shot is None:
            return
        self._fps = getattr(shot, "fps", 24)
        self._pending_load_paths = []
        self._freeze_active = False
        self._freeze_frame_index = None
        self._current_freeze_page = None
        self._freeze_on_page_index = None
        self._freeze_on_page_target = None
        self._page_to_frame_index = {}
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
        # Reset incoming-shot conversion state.
        self._loading_frames = []
        self._raw_cursor     = 0
        self._drip_active    = False

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

    def _on_scene_load(self, data: dict):
        metadata = data.get("metadata", {})
        scene_id = data.get("scene_id", "")
        frames_dir = os.path.join(
            os.path.dirname(self._find_metadata_dir(scene_id)), "frames"
        )
        self._fps = metadata.get("fps", 24)
        self._frame_index = 0
        self._frames = self._load_frames(frames_dir)
        self._loading_dir = None   # legacy synchronous load path; not cache-driven
        self._last_frame_time = time.monotonic()
        self._freeze_active = False
        self._freeze_frame_index = None
        self._current_freeze_page = None
        self._freeze_on_page_index = None
        self._freeze_on_page_target = None
        self._page_to_frame_index = {}

    def _find_metadata_dir(self, scene_id: str) -> str:
        scenes_root = os.path.join(os.path.dirname(__file__), "..", "scenes")
        for artist_dir in os.listdir(scenes_root):
            candidate = os.path.join(scenes_root, artist_dir, scene_id)
            if os.path.isdir(candidate):
                return candidate
        return ""

    def _load_frames(self, frames_dir: str) -> list:
        if not os.path.isdir(frames_dir):
            return []
        w, h = self.config.get("resolution", [1920, 1080])
        frames = []
        for fname in sorted(os.listdir(frames_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(frames_dir, fname)
            img = Image.open(path).convert("RGB").resize((w, h))
            surface = pygame.image.fromstring(img.tobytes(), img.size, "RGB")
            frames.append(surface)
        return frames

    def _on_dev_frames_load(self, data: dict):
        paths = data.get("frames_paths", [])
        self._fps = data.get("fps", 0.5)
        self._frame_index = 0
        self._frames = []
        self._loading_dir = None   # legacy synchronous load path; not cache-driven
        self._pending_load_paths = list(paths)
        self._pending_load_fps = self._fps
        self._freeze_active = False
        self._freeze_frame_index = None
        self._current_freeze_page = None
        self._freeze_on_page_index = None
        self._freeze_on_page_target = None
        self._last_frame_time = time.monotonic()

        # Pre-build PDF-page → frame-index map from sorted filenames.
        # Storyboard convention: file page_NNN.jpg = PDF header page NNN-1.
        self._page_to_frame_index = {}
        for idx, path in enumerate(paths):
            basename = os.path.basename(path)
            if basename.startswith("page_") and basename.split(".")[-1].lower() in ("jpg", "png", "jpeg"):
                try:
                    file_num = int(basename[5:].split(".")[0])
                    pdf_page = file_num - 1
                    self._page_to_frame_index[pdf_page] = idx
                except ValueError:
                    pass

        print(f"[RenderEngine] Dev frames queued: {len(paths)} panels at {self._fps} fps (lazy loading)")

    def _load_frames_from_paths(self, paths: list) -> list:
        display_cfg = self.config.get("_profile", {}).get("display", {})
        w, h = display_cfg.get("resolution") or self.config.get("resolution", [1920, 1080])
        frames = []
        for path in paths:
            if not os.path.isfile(path):
                continue
            try:
                img = Image.open(path).convert("RGB").resize((w, h))
                surface = pygame.image.fromstring(img.tobytes(), img.size, "RGB")
                frames.append(surface)
            except Exception as exc:
                print(f"[RenderEngine] Failed to load frame {path}: {exc}")
        return frames

    def _on_oi_flash(self, data: dict):
        self._flash_color = data.get("color", (0, 255, 80))
        duration_ms = data.get("duration_ms", 800)
        self._flash_start = time.monotonic()
        self._flash_until = self._flash_start + duration_ms / 1000.0

    def _on_play_sfx(self, data: dict):
        path = data.get("path")
        if not path or not os.path.exists(path):
            return
        try:
            sound = pygame.mixer.Sound(path)
            pygame.mixer.Channel(1).play(sound)
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
        self._seg_anchor = time.monotonic()
        self._seg_done   = False
        print(f"[RenderEngine] play_segment [{self._seg_start}–{self._seg_end}]  "
              f"loop={self._seg_loop}")

    def _on_render_event(self, data: dict):
        self._pending_events.append(data.get("name"))

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

    def _draw_index_cursors(self, landmark_data, handedness_data=None):
        """Draw a crosshair at each hand's index fingertip with L/R label."""
        w, h = self._screen.get_size()
        for i, hand_landmarks in enumerate(landmark_data):
            lm = hand_landmarks.landmark[8]  # index fingertip
            x, y = int((1 - lm.x) * w), int(lm.y * h)
            r = 14
            pygame.draw.circle(self._screen, (255, 255, 255), (x, y), r, 2)
            pygame.draw.circle(self._screen, (255, 80, 0), (x, y), 6)
            pygame.draw.line(self._screen, (255, 255, 255), (x - r - 4, y), (x + r + 4, y), 1)
            pygame.draw.line(self._screen, (255, 255, 255), (x, y - r - 4), (x, y + r + 4), 1)
            if self._small_font:
                label = "?"
                if handedness_data and i < len(handedness_data):
                    label = handedness_data[i].classification[0].label[0]
                surf = self._small_font.render(label, True, (255, 255, 255))
                self._screen.blit(self._small_font.render(label, True, (0, 0, 0)), (x + r + 4, y - r - 2))
                self._screen.blit(surf, (x + r + 3, y - r - 3))

    _HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17),
    ]

    def _draw_hand_mini_panel(self, landmark_data, handedness_data=None):
        """Skeleton preview in bottom-right corner — keeps the main screen clean."""
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

        for i, hand_lm in enumerate(landmark_data):
            label = "?"
            if handedness_data and i < len(handedness_data):
                label = handedness_data[i].classification[0].label[0]
            color_bone = (0, 200, 60) if label != "R" else (0, 160, 220)
            color_dot = (0, 255, 80) if label != "R" else (60, 200, 255)

            pts = [
                (px + pad + int((1 - lm.x) * inner_w),
                 py + pad + int(lm.y * inner_h))
                for lm in hand_lm.landmark
            ]
            for a, b in self._HAND_CONNECTIONS:
                pygame.draw.line(self._screen, color_bone, pts[a], pts[b], 1)
            for pt in pts:
                pygame.draw.circle(self._screen, color_dot, pt, 2)

            if self._small_font:
                wx, wy = pts[0]
                surf = self._small_font.render(label, True, color_dot)
                self._screen.blit(surf, (wx + 4, wy - 4))

    def _draw_loading_indicator(self) -> None:
        """Small corner readout while the incoming shot finishes preloading."""
        if not self._screen or not self._small_font or self._cache is None:
            return
        loaded = self._cache.loaded_count(self._loading_dir) if self._loading_dir else 0
        total  = self._cache.total_count(self._loading_dir) if self._loading_dir else None
        name   = self._loading_dir.parent.name if self._loading_dir else "?"
        done, shots_total = self._cache.progress()
        if total:
            txt = f"preloading {name}  {loaded}/{total}  ({done}/{shots_total} shots)"
        else:
            txt = f"preloading {name}…  ({done}/{shots_total} shots)"
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
        hand_count = len(landmark_data) if landmark_data else 0

        lines = []
        if hand_count:
            lines.append((f"HANDS: {hand_count} detected", (0, 255, 80)))
        else:
            lines.append(("HANDS: none", (255, 60, 60)))

        if gesture_debug:
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
                if self._freeze_active and self._current_freeze_page is not None:
                    lines.append((f"FREEZE p{self._current_freeze_page}", (255, 220, 80)))
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

    def _draw_scene_panel(self, scene_debug: dict):
        if not self._font:
            return
        sw, sh = self._screen.get_size()
        pad = 10
        panel_w = 440
        line_h = 22
        small_font = pygame.font.SysFont("monospace", 16)
        title_font = pygame.font.SysFont("monospace", 22, bold=True)

        idx = scene_debug["scene_idx"]
        total = scene_debug["scene_total"]
        title = scene_debug["title"]
        desc = scene_debug["description"]
        gestures = scene_debug["gestures"]

        content: list[tuple[str, tuple, object]] = []
        content.append((f"SCENE {idx + 1:02d} / {total:02d}", (200, 160, 255), title_font))
        content.append((title, (255, 240, 200), title_font))
        content.append(("", (0, 0, 0), small_font))

        for line in self._wrap_text(desc, small_font, panel_w - pad * 2):
            content.append((line, (200, 200, 200), small_font))
        content.append(("", (0, 0, 0), small_font))

        content.append(("GESTURES:", (160, 220, 255), small_font))
        for g in gestures:
            color = (255, 220, 80) if g.startswith("CG") else \
                    (100, 200, 255) if g.startswith("OI") else \
                    (180, 255, 140)
            for line in self._wrap_text(g, small_font, panel_w - pad * 2 - 8):
                content.append(("  " + line, color, small_font))
        content.append(("", (0, 0, 0), small_font))
        content.append(("  Raise hands to advance", (160, 160, 160), small_font))

        panel_h = len(content) * line_h + pad * 2
        panel_x = sw - panel_w - 10
        panel_y = 10

        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surf.fill((0, 0, 0, 170))
        self._screen.blit(panel_surf, (panel_x, panel_y))

        y = panel_y + pad
        for text, color, font in content:
            if text:
                surf = font.render(text, True, color)
                self._screen.blit(surf, (panel_x + pad, y))
            y += line_h

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
