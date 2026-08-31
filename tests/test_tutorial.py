"""TutorialEngine flow: begin/arm/success/flash/advance/timeout/skip/restore."""

import time
import unittest

import engines.tutorial_engine as te
from engines.tutorial_engine import TutorialEngine
from tests.mocks import Bus


class FakeGesture:
    _input_locked = True


class TestDetectSound(unittest.TestCase):
    def test_success_plays_detect_sfx_when_wired(self):
        bus = Bus()
        tut = TutorialEngine({"tutorial_enabled": True}, bus, FakeGesture(),
                             detect_sfx="/scenes/scene_01/audio/detect.mp3")
        tut.begin()
        bus.emit("cg_detected", {"gesture_id": "tutorial_0"})
        sfx = [d for n, d in bus.log if n == "play_sfx"]
        self.assertEqual(len(sfx), 1)
        self.assertTrue(sfx[0]["path"].endswith("detect.mp3"))

    def test_no_sfx_event_without_a_path(self):
        bus = Bus()
        tut = TutorialEngine({"tutorial_enabled": True}, bus, FakeGesture())
        tut.begin()
        bus.emit("cg_detected", {"gesture_id": "tutorial_0"})
        self.assertFalse([d for n, d in bus.log if n == "play_sfx"])


class TestTutorialFlow(unittest.TestCase):
    def setUp(self):
        self._linger, self._timeout = te.SUCCESS_LINGER_S, te.STEP_TIMEOUT_S
        te.SUCCESS_LINGER_S = 0.05
        te.STEP_TIMEOUT_S = 0.3
        self.bus = Bus()
        self.tut = TutorialEngine({"tutorial_enabled": True}, self.bus, FakeGesture())

    def tearDown(self):
        te.SUCCESS_LINGER_S, te.STEP_TIMEOUT_S = self._linger, self._timeout

    def _armed_ids(self):
        return [d["interaction"]["id"] for n, d in self.bus.log
                if n == "cg_window_open" and d.get("interaction")]

    def test_full_flow(self):
        self.assertTrue(self.tut.begin())
        self.assertTrue(self.tut.active)
        self.assertEqual(self._armed_ids()[-1], "tutorial_0")
        self.assertIn(("input_lock", {"locked": False}), self.bus.log)
        card = self.tut.card_info()
        # 5 steps since August 2026 (the Reach IN step was removed).
        self.assertEqual((card["step"], card["total"]), (1, 5))

        # success on step 1 -> green flash, linger, advance
        self.bus.emit("cg_detected", {"gesture_id": "tutorial_0"})
        self.assertTrue(any(n == "oi_flash" for n, _ in self.bus.log))
        self.assertEqual(self.tut.card_info()["prompt"], "Nice!")
        time.sleep(0.07)
        self.tut.update()
        self.assertEqual(self.tut.card_info()["step"], 2)
        self.assertEqual(self._armed_ids()[-1], "tutorial_1")

        # stale detection from an earlier step is ignored
        self.bus.emit("cg_detected", {"gesture_id": "tutorial_0"})
        self.assertNotEqual(self.tut.card_info()["prompt"], "Nice!")

        # timeout auto-advance keeps the installation un-weddgeable
        time.sleep(0.35)
        self.tut.update()
        self.assertEqual(self.tut.card_info()["step"], 3)

        # skip finishes, disarms and restores the input lock
        self.tut.skip()
        self.assertFalse(self.tut.active)
        self.assertTrue(self.tut.done)
        self.assertIn(("cg_window_open", {"interaction": None}), self.bus.log)
        self.assertIn(("input_lock", {"locked": True}), self.bus.log)
        self.assertFalse(self.tut.begin())   # one-shot per session

    def test_card_carries_a_per_step_figure(self):
        """`figure` keys the code-drawn vector figure in the card's corner box
        and is the step's stable id — it must change with the step."""
        self.assertTrue(self.tut.begin())
        first = self.tut.card_info()["figure"]
        self.assertTrue(first)
        self.bus.emit("cg_detected", {"gesture_id": "tutorial_0"})
        time.sleep(0.07)
        self.tut.update()
        self.assertNotEqual(self.tut.card_info()["figure"], first)

    def test_disabled_never_starts(self):
        tut = TutorialEngine({"tutorial_enabled": False}, Bus(), FakeGesture())
        self.assertFalse(tut.begin())


if __name__ == "__main__":
    unittest.main()
