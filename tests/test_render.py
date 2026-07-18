"""RenderEngine headless smoke: hand cursors, interaction indicators, tutorial
card, pause/flash bookkeeping, segment overshoot carry, skeleton toggle."""

import os
import time
import unittest

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


if __name__ == "__main__":
    unittest.main()
