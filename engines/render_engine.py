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
from typing import Optional


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

        # Freeze-frame state
        self._page_to_frame_index: dict = {}   # PDF page number → frame list index
        self._freeze_frame_index: Optional[int] = None
        self._freeze_active: bool = False
        self._current_freeze_page: Optional[int] = None  # for debug display

        # Play-through freeze: frames advance naturally until reaching this index
        self._freeze_on_page_index: Optional[int] = None
        self._freeze_on_page_target: Optional[int] = None  # PDF page for debug

        self.event_bus.subscribe("scene_load", self._on_scene_load)
        self.event_bus.subscribe("dev_frames_load", self._on_dev_frames_load)
        self.event_bus.subscribe("render_event", self._on_render_event)
        self.event_bus.subscribe("freeze_frame", self._on_freeze_frame)
        self.event_bus.subscribe("freeze_on_page", self._on_freeze_on_page)
        self.event_bus.subscribe("freeze_release", self._on_freeze_release)
        self.event_bus.subscribe("page_jump", self._on_page_jump)

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

        # Drain one pending frame per tick so the main thread never blocks on image load
        if self._pending_load_paths:
            path = self._pending_load_paths.pop(0)
            display_cfg = self.config.get("_profile", {}).get("display", {})
            w, h = display_cfg.get("resolution") or self.config.get("resolution", [1920, 1080])
            try:
                img = Image.open(path).convert("RGB").resize((w, h))
                surface = pygame.image.fromstring(img.tobytes(), img.size, "RGB")
                self._frames.append(surface)
            except Exception as exc:
                print(f"[RenderEngine] Failed to load frame {path}: {exc}")

        if self._frames:
            if self._freeze_active and self._freeze_frame_index is not None:
                # Hold the freeze frame — only jump if that index is already loaded
                if self._freeze_frame_index < len(self._frames):
                    self._frame_index = self._freeze_frame_index
            else:
                frame_duration = 1.0 / self._fps
                if now - self._last_frame_time >= frame_duration:
                    self._frame_index = (self._frame_index + 1) % len(self._frames)
                    self._last_frame_time = now

                    # Play-through freeze: activate when natural playback reaches target
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

            self._screen.blit(self._frames[self._frame_index], (0, 0))
        else:
            self._screen.fill((0, 0, 0))
            if narration_debug and narration_debug.get("waiting_id"):
                self._draw_wait_for_cg_placeholder(narration_debug.get("waiting_id"))

        if landmark_data:
            self._draw_index_cursors(landmark_data, handedness_data)

        if self._debug:
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

    def _on_scene_load(self, data: dict):
        metadata = data.get("metadata", {})
        scene_id = data.get("scene_id", "")
        frames_dir = os.path.join(
            os.path.dirname(self._find_metadata_dir(scene_id)), "frames"
        )
        self._fps = metadata.get("fps", 24)
        self._frame_index = 0
        self._frames = self._load_frames(frames_dir)
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

    def _on_render_event(self, data: dict):
        self._pending_events.append(data.get("name"))

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

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
            last = gesture_debug.get("last_fired")
            if last:
                lines.append((f"FIRED: {last}", (255, 140, 0)))

        if narration_debug is not None:
            lines.append(("", (0, 0, 0)))
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
