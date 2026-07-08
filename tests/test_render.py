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


class Hand:
    def __init__(self):
        self.landmark = [LM(0.5, 0.5) for _ in range(21)]


class _Cls:
    def __init__(self, label):
        self.label = label


class Handed:
    def __init__(self, label):
        self.classification = [_Cls(label)]


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
                self.r.update(landmark_data=[Hand()],
                              handedness_data=[Handed("Left")],
                              pose_data=self.pose, gesture_debug=gd)

    def test_tutorial_card_while_paused(self):
        self.r.pause()
        try:
            self.r.update(landmark_data=[Hand()], handedness_data=[Handed("L")],
                          pose_data=self.pose,
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
            self.r.update(landmark_data=[Hand()], handedness_data=[Handed("L")],
                          pose_data=self.pose, gesture_debug={"active_type": None})
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


if __name__ == "__main__":
    unittest.main()
