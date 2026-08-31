"""RenderEngine headless smoke: hand cursors, interaction indicators, tutorial
card, pause/flash bookkeeping, segment overshoot carry, skeleton toggle."""

import os
import time
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from engines.render_engine import RenderEngine
from tests.mocks import Bus, LM


class TestRenderEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.bus = Bus()
        cls.r = RenderEngine({"resolution": [640, 360],
                              "timing_defaults": {"oi_window_ms": 6000}}, cls.bus)
        cls.r.init_display()
        cls.pose = [LM(0.5, 0.5, visibility=0.9) for _ in range(33)]

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_hand_icons_loaded(self):
        self.assertEqual(len(self.r._hand_icons), 8)

    def test_captions_draw_only_when_active(self):
        r = self.r
        r._fps = 30
        r.config["captions_enabled"] = True
        r._captions = [{"at_s": 1.0, "duration_s": 2.0, "text": "hello there"}]
        # playhead before the window -> font not built, nothing drawn
        r._caption_font = None
        r._frame_index = 0
        r._draw_captions()
        self.assertIsNone(r._caption_font)
        # playhead inside the window -> font built (something drawn)
        r._frame_index = 45          # 1.5s at 30fps, inside [1,3]
        r._draw_captions()
        self.assertIsNotNone(r._caption_font)

    def test_captions_respect_disable_flag(self):
        r = self.r
        r._fps = 30
        r._caption_font = None
        r._captions = [{"at_s": 0.0, "duration_s": 5.0, "text": "x"}]
        r._frame_index = 30
        r.config["captions_enabled"] = False
        r._draw_captions()
        self.assertIsNone(r._caption_font)     # disabled -> never rendered
        r.config["captions_enabled"] = True

    def test_toggle_captions_flips_config_gate(self):
        r = self.r
        r.config["captions_enabled"] = True
        r.toggle_captions()
        self.assertFalse(r.config["captions_enabled"])
        r.toggle_captions()
        self.assertTrue(r.config["captions_enabled"])

    def test_point_icon_angle_aims_at_region_and_direction(self):
        r = self.r
        # region box on screen-right (raw x 0.1 -> mirrored screen-right) from a
        # hand at screen-left should rotate the up-icon clockwise (negative deg).
        ang = r._point_icon_angle(
            {"region_rect": {"x": 0.1, "y": 0.45, "w": 0.1, "h": 0.1}},
            50, 180, 640, 360)
        self.assertIsNotNone(ang)
        # direction 'up' needs no rotation; None-worthy inputs return None
        self.assertEqual(r._point_icon_angle({"direction": "up"}, 0, 0, 640, 360),
                         -90.0 - -90.0)   # atan2(-1,0) = -90 -> R = 0
        self.assertIsNone(r._point_icon_angle({}, 0, 0, 640, 360))

    def test_cursor_fades_in_during_window_and_out_after(self):
        """Cursors are window-scoped: alpha ramps up while a window is armed,
        ramps back to zero (hidden) once it closes, and never shows outside."""
        r = self.r
        r._cursor_fade_alpha = 0.0
        # no window -> stays hidden
        r._cursor_fade_t = time.monotonic() - 0.2
        r._draw_hand_cursors(pose_data=self.pose,
                             gesture_debug={"active_type": None})
        self.assertEqual(r._cursor_fade_alpha, 0.0)

        # window opens -> alpha climbs (0.1s clamped step over 0.4s fade-in)
        armed = {"active_type": "presence_bilateral", "active_params": {}}
        r._cursor_fade_t = time.monotonic() - 0.2
        r._draw_hand_cursors(pose_data=self.pose, gesture_debug=armed)
        self.assertGreater(r._cursor_fade_alpha, 0.0)
        for _ in range(8):
            r._cursor_fade_t = time.monotonic() - 0.2
            r._draw_hand_cursors(pose_data=self.pose, gesture_debug=armed)
        self.assertEqual(r._cursor_fade_alpha, 1.0)
        self.assertEqual(r._cursor_fade_mode, "grab")

        # window closes -> fades out, keeping the last mode, then fully hides
        r._cursor_fade_t = time.monotonic() - 0.2
        r._draw_hand_cursors(pose_data=self.pose,
                             gesture_debug={"active_type": None})
        self.assertLess(r._cursor_fade_alpha, 1.0)
        self.assertGreater(r._cursor_fade_alpha, 0.0)
        self.assertEqual(r._cursor_fade_mode, "grab")
        for _ in range(8):
            r._cursor_fade_t = time.monotonic() - 0.2
            r._draw_hand_cursors(pose_data=self.pose,
                                 gesture_debug={"active_type": None})
        self.assertEqual(r._cursor_fade_alpha, 0.0)

    def test_camera_setup_screen(self):
        import numpy as np
        # no frame, no pose -> draws the prompt, body not in frame
        self.assertFalse(self.r.draw_camera_setup(None, None))
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        # pose without visible ankles -> still not ok
        pose = [LM(0.5, 0.5, visibility=0.9) for _ in range(33)]
        pose[27] = LM(0.5, 0.9, visibility=0.1)
        pose[28] = LM(0.5, 0.9, visibility=0.1)
        self.assertFalse(self.r.draw_camera_setup(frame, pose))
        # head + both ankles confidently inside the frame -> ok
        pose[0]  = LM(0.5, 0.10, visibility=0.9)
        pose[27] = LM(0.45, 0.92, visibility=0.9)
        pose[28] = LM(0.55, 0.92, visibility=0.9)
        self.assertTrue(self.r.draw_camera_setup(frame, pose))
        # a landmark hugging the border does not count as in frame
        pose[0] = LM(0.5, 0.005, visibility=0.9)
        self.assertFalse(self.r.draw_camera_setup(frame, pose))

    def test_input_locked_window_counts_as_closed(self):
        r = self.r
        r._cursor_fade_alpha = 1.0
        r._cursor_fade_t = time.monotonic() - 0.2
        r._draw_hand_cursors(pose_data=self.pose,
                             gesture_debug={"active_type": "presence_bilateral",
                                            "input_locked": True})
        self.assertLess(r._cursor_fade_alpha, 1.0)

    def test_cursor_and_indicator_modes_draw(self):
        cases = [
            {"active_type": None},                                    # dots
            {"active_type": "point_target_held",
             "active_params": {"region_rect": {"x": 0.4, "y": 0.3,
                                               "w": 0.2, "h": 0.2}}}, # ring
            {"active_type": "rhythm_bilateral", "active_params": {}}, # knock
            {"active_type": "directional_draw",
             "active_params": {"direction": "up_left"}},              # arrow
            {"active_type": "forward_point",
             "active_params": {"target_region": "center"}},           # named region
        ]
        for gd in cases:
            with self.subTest(gd=gd):
                self.r.update(pose_data=self.pose, gesture_debug=gd)

    def test_tutorial_card_while_paused(self):
        self.r.pause()
        try:
            self.r.update(pose_data=self.pose,
                          gesture_debug={"active_type": "presence_bilateral",
                                         "active_params": {}},
                          tutorial_card={"title": "Welcome!", "prompt": "Raise hands",
                                         "icon": "open_r", "step": 1, "total": 6,
                                         "target_rect": {"x": 0.6, "y": 0.3,
                                                         "w": 0.2, "h": 0.3}})
        finally:
            self.r.resume()

    def test_expired_flash_not_replayed_after_resume(self):
        self.r.pause()
        self.bus.emit("oi_flash", {"duration_ms": 50})
        time.sleep(0.1)   # flash expires while paused
        self.r.resume()
        self.assertLessEqual(self.r._flash_until, time.monotonic())

    def test_skeleton_toggle_independent_of_debug(self):
        self.r.toggle_skeleton()
        try:
            self.assertTrue(self.r._show_skeleton)
            self.assertFalse(self.r._debug)
            self.r.update(pose_data=self.pose, gesture_debug={"active_type": None})
        finally:
            self.r.toggle_skeleton()

    def test_segment_overshoot_carry(self):
        r = self.r
        r._fps = 24
        r._frames = [pygame.Surface((640, 360)) for _ in range(30)]
        r._loading_dir = None

        self.bus.emit("play_segment", {"start": 0, "end": 9, "loop": False})
        r._seg_anchor -= (10 / 24 + 0.120)   # overran the boundary by ~120ms
        r.update()
        self.assertTrue(0.100 <= r._seg_overshoot <= 0.150)

        self.bus.emit("play_segment", {"start": 10, "end": 19, "loop": False})
        lag = time.monotonic() - r._seg_anchor
        self.assertTrue(0.09 <= lag <= 0.16)      # carried into the next anchor
        self.assertEqual(r._seg_overshoot, 0.0)   # and consumed

        r._seg_overshoot = 0.2
        self.bus.emit("play_segment", {"start": 0, "end": 9, "loop": True})
        self.assertLess(time.monotonic() - r._seg_anchor, 0.05)  # loops start clean
        self.assertEqual(r._seg_overshoot, 0.0)

    def test_cursor_param_override_beats_detector_type(self):
        """A window can force its icon via params {"cursor": ...} — reach_star
        is detected as point_target_held but should show the open hand."""
        r = self.r
        r._cursor_fade_alpha = 1.0
        r._cursor_fade_t = time.monotonic() - 0.2
        r._draw_hand_cursors(pose_data=self.pose, gesture_debug={
            "active_type": "point_target_held",
            "active_params": {"cursor": "open", "hold_ms": 400}})
        self.assertEqual(r._cursor_fade_mode, "open")
        r._cursor_fade_t = time.monotonic() - 0.2
        r._draw_hand_cursors(pose_data=self.pose, gesture_debug={
            "active_type": "point_target_held", "active_params": {}})
        self.assertEqual(r._cursor_fade_mode, "point")

    def test_star_trail_spawns_on_draw_movement_and_expires(self):
        r = self.r
        r._trail_particles = []
        r._trail_last_pos = {"L": None, "R": None}
        gd = {"active_type": "directional_draw",
              "active_params": {"direction": "left"}}
        still = [LM(0.5, 0.5, visibility=0.9) for _ in range(33)]
        moved = [LM(0.4, 0.45, visibility=0.9) for _ in range(33)]
        r._draw_star_trail(pose_data=still, gesture_debug=gd)  # primes last pos
        self.assertEqual(len(r._trail_particles), 0)
        r._draw_star_trail(pose_data=moved, gesture_debug=gd)  # big move spawns
        self.assertGreater(len(r._trail_particles), 0)
        n = len(r._trail_particles)
        r._draw_star_trail(pose_data=moved, gesture_debug=gd)  # still: no growth
        self.assertEqual(len(r._trail_particles), n)
        for p in r._trail_particles:                            # age out
            p["born"] -= 5.0
        r._draw_star_trail(pose_data=moved, gesture_debug=gd)
        self.assertEqual(len(r._trail_particles), 0)
        # outside a draw window nothing spawns and the anchor resets
        r._draw_star_trail(pose_data=still, gesture_debug={"active_type": None})
        self.assertEqual(r._trail_last_pos, {"L": None, "R": None})

    def test_draw_indicator_comet_smoke(self):
        for direction in ("right", "up_left", "down"):
            self.r._draw_interaction_indicator({
                "active_type": "directional_draw",
                "active_params": {"direction": direction}})

    def test_cursor_dots_and_hidden_modes(self):
        """params.cursor "dots" shows plain tracking dots; "hidden" suppresses
        the cursor entirely (fades out like a closed window)."""
        r = self.r
        r._cursor_fade_alpha = 1.0
        r._cursor_fade_t = time.monotonic() - 0.2
        r._draw_hand_cursors(pose_data=self.pose, gesture_debug={
            "active_type": "point_target_held",
            "active_params": {"cursor": "dots"}})
        self.assertEqual(r._cursor_fade_mode, "dots")
        self.assertEqual(r._cursor_fade_alpha, 1.0)
        r._cursor_fade_t = time.monotonic() - 0.2
        r._draw_hand_cursors(pose_data=self.pose, gesture_debug={
            "active_type": "point_target_held",
            "active_params": {"cursor": "hidden"}})
        self.assertLess(r._cursor_fade_alpha, 1.0)

    def test_arrow_wings_point_backward(self):
        """Arrowhead wings must land BEHIND the tip along the heading — the
        arrow points away from the line (the old math put them beyond it)."""
        import math
        for ang in (0.0, math.pi / 2, -math.pi / 4, 2.5):
            ex, ey = 100.0, 100.0
            dx, dy = math.cos(ang), math.sin(ang)
            for hx, hy in RenderEngine._arrow_wings(ex, ey, ang, 24):
                proj = (hx - ex) * dx + (hy - ey) * dy
                self.assertLess(proj, 0.0,
                                f"wing ahead of the tip at ang={ang}")

    def test_indicator_span_from_rect(self):
        """An authored indicator_rect anchors the stroke at its centre with
        the length fitted to its extent along the direction."""
        params = {"direction": "right",
                  "indicator_rect": {"x": 0.2, "y": 0.4, "w": 0.4, "h": 0.2}}
        sx, sy, ex, ey = RenderEngine._indicator_span(params, 1000, 1000)
        self.assertAlmostEqual((sx + ex) / 2, 400.0)   # rect centre x
        self.assertAlmostEqual(sy, ey)
        self.assertAlmostEqual(sy, 500.0)              # rect centre y
        self.assertAlmostEqual(ex - sx, 2 * 200.0 * 0.9)   # fitted, inset
        # no rect -> legacy fixed anchor
        sx, sy, ex, ey = RenderEngine._indicator_span(
            {"direction": "right"}, 1000, 1000)
        self.assertAlmostEqual((sx + ex) / 2, 500.0)
        self.assertAlmostEqual(sy, 420.0)

    def test_star_trail_hand_glow_on_still_hand(self):
        """The 'pen' dot marks each tracked hand even when it is not moving
        (the trail only spawns on movement)."""
        r = self.r
        self.assertIsNotNone(r._hand_glow)
        r._trail_particles = []
        r._trail_last_pos = {"L": None, "R": None}
        gd = {"active_type": "directional_draw",
              "active_params": {"direction": "left"}}
        still = [LM(0.5, 0.5, visibility=0.9) for _ in range(33)]
        r._draw_star_trail(pose_data=still, gesture_debug=gd)
        r._draw_star_trail(pose_data=still, gesture_debug=gd)  # smoke: dot path
        self.assertEqual(len(r._trail_particles), 0)           # no build-up

    # -- torso-radial point cursors (decision blocks) --------------------

    RADIAL = {"directions": ["left", "right"]}

    def test_point_icon_angle_radial_from_torso(self):
        """A two-way left/right choice aims each hand radially OUT from the
        torso centre: the body is the reference frame, so the icon tilts with
        both the reach and the height of the hand."""
        r = self.r
        torso = (320, 180)
        for name, (x, y), want in (
                ("far left",  (100, 180),  90.0),   # icon points screen-left
                ("far right", (540, 180), 270.0),   # ... screen-right
                ("overhead",  (320,  60),   0.0),   # ... straight up
                ("up-left",   (220,  80),  45.0)):
            with self.subTest(hand=name):
                ang = r._point_icon_angle(self.RADIAL, x, y, 640, 360, torso=torso)
                self.assertIsNotNone(ang)
                self.assertAlmostEqual(ang % 360, want, places=4)
        # a hand sitting exactly on the centre has no direction -> upright
        self.assertIsNone(
            r._point_icon_angle(self.RADIAL, 320, 180, 640, 360, torso=torso))
        # and without a torso there is nothing to be radial about
        self.assertIsNone(
            r._point_icon_angle(self.RADIAL, 100, 180, 640, 360, torso=None))

    def test_region_rect_beats_torso_radial(self):
        """An authored target box still wins: the visitor should be aimed at
        the box, not away from their own body."""
        r = self.r
        rect = {"x": 0.65, "y": 0.25, "w": 0.3, "h": 0.5}
        both = r._point_icon_angle({"region_rect": rect, **self.RADIAL},
                                   320, 180, 640, 360, torso=(320, 300))
        alone = r._point_icon_angle({"region_rect": rect}, 320, 180, 640, 360)
        self.assertIsNotNone(alone)
        self.assertEqual(both, alone)

    def test_radial_requires_two_way_lateral_choice(self):
        """Only a genuine left/right fork goes radial. Single-direction
        windows (the tutorial's point LEFT / RIGHT / DOWN steps) stay upright
        — a radial aim there could contradict the prompt."""
        r = self.r
        torso = (320, 180)
        for params in ({"directions": ["left"]},
                       {"directions": ["right"]},
                       {"directions": ["down", "down_left", "down_right"]},
                       {"directions": []},
                       {}):
            with self.subTest(params=params):
                self.assertIsNone(r._point_icon_angle(params, 100, 250,
                                                      640, 360, torso=torso))

    def test_torso_center_screen_latches_and_expires(self):
        from tests.mocks import pose33
        r = self.r
        r._torso_screen, r._torso_t = None, 0.0
        # pose33: shoulders x .6/.4 y .40, hips y .75 -> mirrored x .5, y .575
        got = r._torso_center_screen(pose33(), 640, 360)
        self.assertEqual(got[0], 320)
        self.assertAlmostEqual(got[1], 207, delta=1)   # pixel rounding is not the contract
        # a dropout inside the TTL keeps serving the last known centre
        self.assertEqual(r._torso_center_screen(None, 640, 360), got)
        r._torso_t -= (RenderEngine._TORSO_TTL_S + 1.0)
        self.assertIsNone(r._torso_center_screen(None, 640, 360))
        # a low-visibility hip is a phantom: better upright than swinging
        r._torso_screen, r._torso_t = None, 0.0
        self.assertIsNone(r._torso_center_screen(
            pose33({23: LM(0.58, 0.75, visibility=0.1)}), 640, 360))

    def test_point_rotation_latches_through_detection_frame(self):
        """The gesture engine clears its window BEFORE emitting the detection,
        so the firing frame reports active_params={}. The aim must hold through
        the fade-out instead of snapping the icon upright."""
        from tests.mocks import pose33
        r = self.r
        r._cursor_fade_alpha = 1.0
        r._cursor_fade_angle = {"L": None, "R": None}
        r._torso_screen, r._torso_t = None, 0.0
        # hands out to either side, well clear of the torso centre
        pose = pose33({15: LM(0.90, 0.45, visibility=0.9),
                       16: LM(0.10, 0.45, visibility=0.9),
                       19: LM(0.92, 0.45, visibility=0.9),
                       20: LM(0.08, 0.45, visibility=0.9)})
        armed = {"active_type": "point_region", "active_params": self.RADIAL}
        r._cursor_fade_t = time.monotonic() - 0.2
        r._draw_hand_cursors(pose_data=pose, gesture_debug=armed)
        aimed = dict(r._cursor_fade_angle)
        self.assertIsNotNone(aimed["L"])
        self.assertIsNotNone(aimed["R"])
        self.assertNotEqual(aimed["L"], aimed["R"])   # genuinely per-side

        # the detection frame: window gone, params empty
        r._cursor_fade_t = time.monotonic() - 0.2
        r._draw_hand_cursors(pose_data=pose,
                             gesture_debug={"active_type": None,
                                            "active_params": {}})
        self.assertLess(r._cursor_fade_alpha, 1.0)          # fading out
        self.assertEqual(dict(r._cursor_fade_angle), aimed)  # aim preserved

        # fully hidden -> the next window starts from a clean aim
        for _ in range(8):
            r._cursor_fade_t = time.monotonic() - 0.2
            r._draw_hand_cursors(pose_data=pose,
                                 gesture_debug={"active_type": None})
        self.assertEqual(r._cursor_fade_alpha, 0.0)
        self.assertEqual(r._cursor_fade_angle, {"L": None, "R": None})

    # -- tutorial step figures ------------------------------------------

    def test_tutorial_figure_box_stays_on_screen(self):
        """The figure panel is always fully on screen, and yields the
        bottom-right corner to the K-key skeleton mini-panel."""
        r = self.r
        pw, ph = RenderEngine._MINI_PANEL_SIZE
        pm = RenderEngine._MINI_PANEL_MARGIN
        for sw, sh in ((640, 360), (1280, 720), (1920, 1080)):
            for skeleton in (False, True):
                with self.subTest(size=(sw, sh), skeleton=skeleton):
                    r._show_skeleton = skeleton
                    try:
                        box = r._tutorial_figure_box(sw, sh)
                    finally:
                        r._show_skeleton = False
                    self.assertGreaterEqual(box.left, 0)
                    self.assertGreaterEqual(box.top, 0)
                    self.assertLessEqual(box.right, sw)
                    self.assertLessEqual(box.bottom, sh)
                    if skeleton:
                        panel = pygame.Rect(sw - pw - pm, sh - ph - pm, pw, ph)
                        self.assertFalse(box.colliderect(panel))

    def test_every_tutorial_step_has_a_drawable_figure(self):
        from engines.tutorial_engine import _steps
        keys = [s.get("figure") for s in _steps()]
        self.assertTrue(all(keys), "every tutorial step needs a figure key")
        self.assertLessEqual(set(keys), set(RenderEngine._FIGURES))
        box = self.r._tutorial_figure_box(640, 360)
        for key in RenderEngine._FIGURES:
            with self.subTest(figure=key):
                self.r._draw_step_figure(key, box)

    def test_figure_art_never_touches_the_panel_border(self):
        """The figure is drawn into an inset art rect, so no limb, arrowhead
        or caption may reach the panel edge — checked by rendering each figure
        onto a blank surface and looking for ink in the border ring."""
        r = self.r
        box = pygame.Rect(0, 0, 220, 200)
        inset = 4                      # ring of pixels that must stay clear
        for key in RenderEngine._FIGURES:
            with self.subTest(figure=key):
                surf = pygame.Surface((box.w, box.h))
                surf.fill((0, 0, 0))
                real, r._screen = r._screen, surf
                try:
                    r._draw_step_figure(key, box)
                finally:
                    r._screen = real
                # Ink = anything brighter than the panel fill; the rounded
                # panel itself is drawn to the box, so sample the corners of
                # the border ring where the panel is transparent instead.
                for x in range(box.w):
                    for y in list(range(inset)) + list(range(box.h - inset, box.h)):
                        px = surf.get_at((x, y))[:3]
                        self.assertLess(max(px), 90,
                                        f"{key}: ink at top/bottom edge {(x, y)}")
                for y in range(box.h):
                    for x in list(range(inset)) + list(range(box.w - inset, box.w)):
                        px = surf.get_at((x, y))[:3]
                        self.assertLess(max(px), 90,
                                        f"{key}: ink at left/right edge {(x, y)}")

    def test_tutorial_card_without_figure_draws(self):
        """The figure is optional — an older/partial card dict must still
        render rather than raising."""
        box = self.r._tutorial_figure_box(640, 360)
        self.r._draw_step_figure(None, box)
        self.r._draw_step_figure("no_such_figure", box)
        self.r._draw_tutorial_card({"title": "T", "prompt": "p",
                                    "step": 1, "total": 6})


class TestMasterAudioOffset(unittest.TestCase):
    """Lip-sync trim for baked master audio (audio.mp3), tunable per shot
    without a re-export."""

    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.bus = Bus()
        cls.r = RenderEngine({"resolution": [320, 180],
                              "timing_defaults": {"oi_window_ms": 6000}}, cls.bus)
        cls.r.init_display()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        self.r.config.pop("master_audio_offset_ms", None)
        self.r.config.pop("master_audio_offset_ms_by_shot", None)
        self.r._pending_audio = None
        self.r._pending_audio_at = None

    def test_per_shot_overrides_global_and_falls_back(self):
        r = self.r
        r.config["master_audio_offset_ms"] = 40
        r.config["master_audio_offset_ms_by_shot"] = {"08": 120, "05": None}
        self.assertEqual(r.master_audio_offset_ms("08"), 120)   # per-shot wins
        self.assertEqual(r.master_audio_offset_ms("05"), 40)    # null -> global
        self.assertEqual(r.master_audio_offset_ms("99"), 40)    # absent -> global
        self.assertEqual(r.master_audio_offset_ms(None), 40)
        # no config at all -> no trim
        del r.config["master_audio_offset_ms"]
        del r.config["master_audio_offset_ms_by_shot"]
        self.assertEqual(r.master_audio_offset_ms("08"), 0)

    def test_bad_value_falls_back_to_zero(self):
        self.r.config["master_audio_offset_ms"] = "later please"
        self.assertEqual(self.r.master_audio_offset_ms(None), 0)

    def test_positive_offset_defers_the_start(self):
        r = self.r
        r.config["master_audio_offset_ms"] = 150
        r._shot_id = "08"
        r._pending_audio = "nonexistent.mp3"   # never actually decoded
        r._begin_audio()
        # scheduled, not started
        self.assertIsNotNone(r._pending_audio_at)
        self.assertEqual(r._pending_audio, "nonexistent.mp3")
        self.assertAlmostEqual(r._pending_audio_at - time.monotonic(), 0.15,
                               delta=0.05)
        # not due yet -> still pending
        r._service_pending_audio()
        self.assertIsNotNone(r._pending_audio_at)
        # due -> consumed (the load fails harmlessly; the contract is that the
        # deferred start fires exactly once)
        r._pending_audio_at = time.monotonic() - 0.01
        r._service_pending_audio()
        self.assertIsNone(r._pending_audio_at)
        self.assertIsNone(r._pending_audio)

    def test_zero_offset_starts_immediately(self):
        r = self.r
        r.config["master_audio_offset_ms"] = 0
        r._shot_id = "01"
        r._pending_audio = "nonexistent.mp3"
        r._begin_audio()
        self.assertIsNone(r._pending_audio_at)   # never scheduled
        self.assertIsNone(r._pending_audio)      # consumed at once

    def test_pause_holds_a_delayed_track(self):
        """A held-back track must wait out a pause, not fire the moment the
        operator resumes early."""
        r = self.r
        r.config["master_audio_offset_ms"] = 300
        r._shot_id = "08"
        r._pending_audio = "nonexistent.mp3"
        r._begin_audio()
        due_before = r._pending_audio_at
        r.pause()
        r._service_pending_audio()               # paused: never starts
        self.assertIsNotNone(r._pending_audio_at)
        r._pause_started = time.monotonic() - 0.5   # half a second paused
        r.resume()
        self.assertGreater(r._pending_audio_at, due_before)  # deadline shifted

    def test_shot_load_drops_an_unfired_delay(self):
        """A delay that never fired must not leak into the next shot."""
        r = self.r
        r.config["master_audio_offset_ms"] = 5000
        r._shot_id = "08"
        r._pending_audio = "nonexistent.mp3"
        r._begin_audio()
        self.assertIsNotNone(r._pending_audio_at)

        class _Shot:
            shot, fps, kind = "09", 30, "playback"
            captions, audio_file, frames_dir = [], None, None
            assets_pending = True
        r._on_shot_load({"shot": _Shot()})
        self.assertIsNone(r._pending_audio_at)
        self.assertEqual(r._shot_id, "09")

    def _spy_seek(self):
        """Record the position _seek_shot_audio is asked for instead of touching
        the mixer."""
        seen = []
        self.r._seek_shot_audio = lambda pos: seen.append(pos)
        self.addCleanup(lambda: self.r.__dict__.pop("_seek_shot_audio", None))
        return seen

    def test_seek_keeps_the_offset(self):
        """Skipping the prologue re-places the audio; the trim must survive it.
        It used to be applied only at shot start, so a skip threw it away."""
        r = self.r
        r.config["master_audio_offset_ms"] = 200
        r._shot_id = "01"
        r._audio_path = "nonexistent.mp3"
        r._pending_audio = None
        seen = self._spy_seek()
        r._sync_shot_audio_to_picture(100.0)
        self.assertEqual(len(seen), 1)
        self.assertAlmostEqual(seen[0], 99.8, places=3)   # audio lags by 200ms

    def test_negative_offset_on_a_seek_runs_the_audio_earlier(self):
        r = self.r
        r.config["master_audio_offset_ms"] = -300
        r._shot_id = "01"
        r._audio_path = "nonexistent.mp3"
        seen = self._spy_seek()
        r._sync_shot_audio_to_picture(50.0)
        self.assertAlmostEqual(seen[0], 50.3, places=3)

    def test_seek_inside_the_offset_window_holds_instead_of_seeking(self):
        """A seek to 0.05s with a 500ms trim has no audio to play yet — hold,
        then start at 0 rather than seeking to a negative position."""
        r = self.r
        r.config["master_audio_offset_ms"] = 500
        r._shot_id = "01"
        r._audio_path = "nonexistent.mp3"
        seen = self._spy_seek()
        r._sync_shot_audio_to_picture(0.05)
        self.assertEqual(seen, [])
        self.assertIsNotNone(r._pending_audio_at)
        self.assertEqual(r._pending_audio_pos, 0.0)
        self.assertAlmostEqual(r._pending_audio_at - time.monotonic(), 0.45,
                               delta=0.05)

    def test_seek_while_still_held_repositions_instead_of_starting_at_zero(self):
        """Skip fired during the initial hold: the queued start position is
        stale and must follow the seek, not play the top of the file."""
        r = self.r
        r.config["master_audio_offset_ms"] = 400
        r._shot_id = "01"
        r._pending_audio = "nonexistent.mp3"
        r._audio_path = None
        r._begin_audio()
        self.assertIsNotNone(r._pending_audio_at)      # held at the shot top
        seen = self._spy_seek()
        r._fps, r._seg_start, r._seg_loop = 30, 3001, False
        r._on_play_segment({"start": 3001, "end": 3200, "loop": False})
        self.assertAlmostEqual(seen[0], 100.0 - 0.4, places=3)
        self.assertIsNone(r._pending_audio_at)         # no stale countdown left

    def test_shot_start_streams_instead_of_decoding(self):
        """A baked shot's audio must start on the mixer.music STREAM, never as
        a decoded Sound: Sound decodes the whole file up front — 689ms of
        main-thread stall measured for shot 01's 5-minute master track, felt
        as a picture freeze at every baked shot start, after which the audio
        lagged the picture by the stall for the rest of the shot."""
        r = self.r
        r._pending_audio = "x.mp3"
        r._pending_audio_pos = 0.0
        seen = self._spy_seek()
        r._start_pending_audio()
        self.assertEqual(seen, [0.0])          # streamed, position 0
        self.assertEqual(r._audio_path, "x.mp3")
        self.assertIsNone(r._pending_audio)    # consumed
        r._audio_path = None

    def test_segment_anchor_absorbs_audio_resync_stall(self):
        """The audio (re)placement at a seek blocks the main thread (~100ms
        mp3 seek-scan). The picture anchor is set before the stall, so
        without compensation the picture jumps the stall forward while the
        audio starts at the pre-stall position — permanent extra lag. The
        anchor must absorb the measured stall."""
        r = self.r
        r._pending_audio = "x.mp3"
        r._fps = 30
        r.__dict__["_sync_shot_audio_to_picture"] = \
            lambda pos: time.sleep(0.05)
        self.addCleanup(
            lambda: r.__dict__.pop("_sync_shot_audio_to_picture", None))
        t_before = time.monotonic()
        r._on_play_segment({"start": 3001, "end": 3200, "loop": False})
        # anchor would be ~t_before without compensation; the 50ms stall
        # must have been folded in
        self.assertGreaterEqual(r._seg_anchor, t_before + 0.04)
        r._pending_audio = None

    def test_seek_compensates_load_latency(self):
        """The mp3 stop+load inside _seek_shot_audio costs real time (~100ms
        measured on a skip) while the picture clock keeps running. The stream
        must start where the picture is at play() time, not where it was when
        the seek was requested — otherwise the load cost becomes permanent
        extra audio lag (it sits below the 2s drift net and never corrects,
        and after a prologue skip it landed on top of the authored
        master_audio_offset_ms)."""
        from unittest import mock
        r = self.r
        r._audio_path = "whatever.mp3"
        try:
            with mock.patch.object(pygame.mixer.music, "load",
                                   side_effect=lambda p: time.sleep(0.08)), \
                 mock.patch.object(pygame.mixer.music, "set_volume"), \
                 mock.patch.object(pygame.mixer.music, "play") as play:
                r._seek_shot_audio(100.0)
            start = play.call_args.kwargs["start"]
            self.assertGreaterEqual(start, 100.08)   # load time folded in
            self.assertLess(start, 100.5)            # compensation, not runaway
            # the drift model must describe the compensated stream
            self.assertAlmostEqual(r._audio_pos0, start, places=6)
        finally:
            r._music_active = False
            r._audio_path = None

    def test_resync_is_rate_limited(self):
        """A re-sync reloads the whole master mp3 (a main-thread stall). If
        drift sits above the threshold it must NOT fire at every segment
        boundary — each stall would add drift, so it could never recover."""
        r = self.r
        r.config["master_audio_offset_ms"] = 0
        r._shot_id = "01"
        r._pending_audio = None
        r._audio_path = "nonexistent.mp3"
        r._fps = 30
        r._last_audio_resync = 0.0
        r._audio_pos0, r._audio_epoch = 0.0, time.monotonic()   # audio at ~0s
        seen = self._spy_seek()
        # picture at 100s vs audio at 0s -> way past the 2s threshold
        r._on_play_segment({"start": 3001, "end": 3200, "loop": False})
        self.assertEqual(len(seen), 1)
        # a second boundary moments later must be suppressed
        r._on_play_segment({"start": 3201, "end": 3400, "loop": False})
        self.assertEqual(len(seen), 1)
        # ... but a new shot may re-sync straight away
        class _Shot:
            shot, fps, kind = "05", 30, "playback"
            captions, audio_file, frames_dir = [], None, None
            assets_pending = True
        r._on_shot_load({"shot": _Shot()})
        self.assertEqual(r._last_audio_resync, 0.0)

    def test_only_baked_audio_is_affected(self):
        """audio_events shots carry no audio_file, so there is nothing to
        delay — the trim can never touch frame-anchored audio."""
        r = self.r
        r.config["master_audio_offset_ms"] = 500

        class _EventsShot:
            shot, fps, kind = "02", 30, "interactive"
            captions, audio_file, frames_dir = [], None, None
            assets_pending = True
        r._on_shot_load({"shot": _EventsShot()})
        self.assertIsNone(r._pending_audio)
        r._begin_audio()
        self.assertIsNone(r._pending_audio_at)


class TestChoiceClipPlayback(unittest.TestCase):
    """play_clip: the choice blocks' pick/switch audio — delay, source offset
    and duration applied at runtime so they stay tunable in the Builder."""

    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.bus = Bus()
        cls.r = RenderEngine({"resolution": [320, 180],
                              "timing_defaults": {"oi_window_ms": 6000}}, cls.bus)
        cls.r.init_display()
        # a real decodable file already shipped in the repo
        cls.snd = None
        for p in Path("assets/audio/stems").glob("*.mp3"):
            cls.snd = str(p)
            break

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        self.r._pending_clips.clear()
        self.r._clip_slices.clear()

    def test_missing_file_is_ignored(self):
        self.r._on_play_clip({"path": "no/such/file.mp3"})
        self.assertEqual(self.r._pending_clips, [])
        self.r._on_play_clip({})
        self.assertEqual(self.r._pending_clips, [])

    def test_delay_defers_and_fires_once(self):
        if not self.snd:
            self.skipTest("no stem mp3 available")
        r = self.r
        r._on_play_clip({"path": self.snd, "delay_s": 0.25})
        self.assertEqual(len(r._pending_clips), 1)
        r._service_pending_clips()                 # not due yet
        self.assertEqual(len(r._pending_clips), 1)
        r._pending_clips[0]["at"] = time.monotonic() - 0.01
        r._service_pending_clips()
        self.assertEqual(r._pending_clips, [])     # consumed exactly once

    def test_no_delay_plays_immediately(self):
        if not self.snd:
            self.skipTest("no stem mp3 available")
        self.r._on_play_clip({"path": self.snd})
        self.assertEqual(self.r._pending_clips, [])

    def test_offset_and_duration_slice_the_sound(self):
        if not self.snd:
            self.skipTest("no stem mp3 available")
        r = self.r
        full = r._load_sound(self.snd)
        self.assertIsNotNone(full)
        if full.get_length() < 1.5:
            self.skipTest("stem too short to slice")
        cut = r._slice_sound(self.snd, 0.2, 0.5)
        self.assertIsNotNone(cut)
        self.assertAlmostEqual(cut.get_length(), 0.5, delta=0.05)
        # duration 0 == to the end of the file
        rest = r._slice_sound(self.snd, 0.2, 0)
        self.assertAlmostEqual(rest.get_length(), full.get_length() - 0.2, delta=0.05)
        # no offset and no duration hands back the whole sound, uncut
        self.assertIs(r._slice_sound(self.snd, 0, 0), full)

    def test_slices_are_cached(self):
        if not self.snd:
            self.skipTest("no stem mp3 available")
        r = self.r
        a = r._slice_sound(self.snd, 0.1, 0.3)
        b = r._slice_sound(self.snd, 0.1, 0.3)
        self.assertIs(a, b)                        # a pick can fire many times

    def test_offset_past_the_end_falls_back_to_the_start(self):
        if not self.snd:
            self.skipTest("no stem mp3 available")
        cut = self.r._slice_sound(self.snd, 99999.0, 0.2)
        self.assertIsNotNone(cut)                  # warns, does not crash

    def test_pause_holds_a_delayed_clip_and_shot_load_drops_it(self):
        if not self.snd:
            self.skipTest("no stem mp3 available")
        r = self.r
        r._on_play_clip({"path": self.snd, "delay_s": 5.0})
        due_before = r._pending_clips[0]["at"]
        r.pause()
        r._service_pending_clips()                 # paused: nothing fires
        self.assertEqual(len(r._pending_clips), 1)
        r._pause_started = time.monotonic() - 0.4
        r.resume()
        self.assertGreater(r._pending_clips[0]["at"], due_before)

        class _Shot:
            shot, fps, kind = "03", 30, "playback"
            captions, audio_file, frames_dir = [], None, None
            assets_pending = True
        r._on_shot_load({"shot": _Shot()})
        self.assertEqual(r._pending_clips, [])     # never leaks into the next shot


class TestPalette(unittest.TestCase):
    """The player-facing palette is a contract, not decoration."""

    @staticmethod
    def _contrast(a, b):
        def lum(c):
            def ch(v):
                v /= 255.0
                return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
            return 0.2126 * ch(c[0]) + 0.7152 * ch(c[1]) + 0.0722 * ch(c[2])
        la, lb = sorted((lum(a), lum(b)), reverse=True)
        return (la + 0.05) / (lb + 0.05)

    def test_render_engine_uses_the_palette(self):
        from engines.palette import PALETTE as P
        self.assertIs(RenderEngine._STAR_COLOR, P.LANTERN)
        self.assertEqual(RenderEngine._DOT_COLORS, {"L": P.HAND_L, "R": P.HAND_R})

    def test_hand_dot_colours_match_the_tinted_art(self):
        """The hand icon PNGs are pre-tinted green (*_l) / blue (*_r) on disk.
        These fallbacks must keep matching them or the L/R read breaks —
        pinned as literals so a palette pass can't quietly drift them."""
        from engines.palette import PALETTE as P
        self.assertEqual(P.HAND_L, (60, 220, 90))
        self.assertEqual(P.HAND_R, (70, 160, 255))

    def test_success_flash_is_still_unmistakably_green(self):
        from engines.palette import PALETTE as P
        r, g, b = P.SUCCESS
        self.assertGreater(g, 200)
        self.assertLess(r, 60)
        self.assertLess(b, 140)

    def test_text_colours_clear_the_contrast_floor(self):
        """Projected in a lit public venue: body text >= 7:1 on NIGHT, the
        operator-only hint >= 4.5:1. Keeps the look from eroding one tweak
        at a time."""
        from engines.palette import PALETTE as P
        for name in ("NORTH_STAR", "LINEN", "LANTERN", "LINEN_DIM"):
            with self.subTest(colour=name):
                self.assertGreaterEqual(
                    self._contrast(getattr(P, name), P.NIGHT), 7.0)
        self.assertGreaterEqual(self._contrast(P.LINEN_FAINT, P.NIGHT), 4.5)


if __name__ == "__main__":
    unittest.main()
