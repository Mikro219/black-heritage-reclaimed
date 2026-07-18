"""Meta-flow contract (July 2026): voice "skip" plumbing for tutorial /
prologue / epilogue, the end-of-experience loop restart, and the camera-setup
screen's confirm path.

Player tests run the real ShotSequencePlayer over the real scenes/ tree with a
mock bus — entering shots only emits events, so this is hardware-free.
"""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from engines.sequence_loader import load_sequence
from engines.shot_sequence_player import (ShotSequencePlayer, PLAYER_RUNNING,
                                          PLAYER_FINAL)
from tests.mocks import Bus

ROOT = Path(__file__).resolve().parent.parent


def make_player():
    shots = load_sequence(ROOT / "scenes", {"fps": 30})
    return ShotSequencePlayer(shots, {"timing_defaults": {}}, Bus()), shots


class TestSkipEpilogue(unittest.TestCase):
    def test_no_op_outside_epilogue(self):
        player, shots = make_player()
        player.start(0)
        self.assertEqual(player.current_shot.act, "00")
        self.assertFalse(player.skip_epilogue())
        self.assertEqual(player.player_state, PLAYER_RUNNING)

    def test_skips_to_final_from_epilogue_act(self):
        player, shots = make_player()
        target = next(i for i, s in enumerate(shots)
                      if s.act in ShotSequencePlayer.EPILOGUE_ACTS)
        player.start(target)
        self.assertTrue(player.skip_epilogue())
        self.assertEqual(player.player_state, PLAYER_FINAL)

    def test_skip_prologue_then_epilogue_guard(self):
        player, shots = make_player()
        player.start(0)
        self.assertTrue(player.skip_prologue())
        self.assertNotEqual(player.current_shot.act, "00")
        # mid-story: neither skip may fire
        self.assertFalse(player.skip_prologue())
        self.assertFalse(player.skip_epilogue())


class TestLoopRestart(unittest.TestCase):
    def test_start_resets_final_state_and_fork_choices(self):
        player, shots = make_player()
        target = next(i for i, s in enumerate(shots)
                      if s.act in ShotSequencePlayer.EPILOGUE_ACTS)
        player.start(target)
        player._fork_choices["09"] = "left"
        player.skip_epilogue()
        self.assertEqual(player.player_state, PLAYER_FINAL)

        player.start(0)   # the end-of-experience loop restart
        self.assertEqual(player.player_state, PLAYER_RUNNING)
        self.assertEqual(player.current_shot.shot, shots[0].shot)
        self.assertEqual(player._fork_choices, {},
                         "restart must not inherit the previous visitor's forks")


class TestCrossroadsSwitch(unittest.TestCase):
    """Shot 09 fork: switching a picked path plays the switch animation ONCE
    and then holds silently on its end frame — it must NOT re-enter the
    *_selected state (which would replay pick.mp3 + the full pick animation:
    the 'two audios and two animations' bug)."""

    def _fsm(self):
        import json
        from engines.shot_sequence_player import ShotFSM
        meta_path = (ROOT / "scenes" / "act_02_crossroads" / "shot_09" /
                     "metadata.json")
        with open(meta_path, encoding="utf-8-sig") as f:
            meta = json.load(f)
        return ShotFSM(meta["interaction"]["interaction_fsm"], meta["segments"])

    def test_switch_lands_in_silent_hold_not_reselect(self):
        fsm = self._fsm()
        self.assertEqual(fsm.fire("point_left"), "left_selected")
        self.assertEqual(fsm.on_enter_sfx(), "pick.mp3")      # first pick: audible
        self.assertEqual(fsm.fire("point_right"), "left_to_right")
        self.assertEqual(fsm.on_enter_sfx(), "switch.mp3")    # switch: audible
        hold = fsm.fire("segment_end")
        self.assertNotIn("selected", hold, "switch must not replay a pick state")
        self.assertIsNone(fsm.on_enter_sfx(), "hold state is silent")
        self.assertTrue(fsm.is_loop(), "hold freezes (1-frame loop)")
        a, b = fsm.segment_range()
        self.assertEqual(a, b, "hold is a single frame")
        # the hold frame is the switch animation's end frame — no visual jump
        self.assertEqual(a, meta_seg_end(fsm, "left_to_right"))

    def test_hold_still_confirms_and_switches_back(self):
        fsm = self._fsm()
        fsm.fire("point_left"); fsm.fire("point_right"); fsm.fire("segment_end")
        self.assertEqual(fsm.voice_keyword(), "go")
        self.assertEqual(fsm.fire("voice_go"), "confirm_right")

        fsm = self._fsm()
        fsm.fire("point_left"); fsm.fire("point_right"); fsm.fire("segment_end")
        self.assertEqual(fsm.fire("point_left"), "right_to_left")
        self.assertEqual(fsm.fire("segment_end"), "left_switch_hold")
        self.assertEqual(fsm.fire("voice_go"), "confirm_left")
        self.assertEqual(fsm.fire("segment_end"), "__advance__")


def meta_seg_end(fsm, segment_name):
    return int(fsm._segments[segment_name][1])


class FakeVoice:
    """Stand-in for VoiceEngine: open/close/window_open + subscriber fan-out."""

    def __init__(self):
        self.open_ids = {}
        self.n = 0
        self.cbs = []

    def subscribe(self, cb):
        self.cbs.append(cb)

    def open_window(self, vi_config):
        self.n += 1
        wid = f"w{self.n}"
        self.open_ids[wid] = vi_config
        return wid

    def close_window(self, wid):
        self.open_ids.pop(wid, None)

    def window_open(self, wid):
        return wid in self.open_ids

    def fire(self, vi_id):
        for cb in self.cbs:
            cb(SimpleNamespace(vi_id=vi_id, window_id="x", tier="reaction",
                               trigger="keyword", matched="skip"))


class TestMetaVoice(unittest.TestCase):
    def _mv(self):
        import main
        voice = FakeVoice()
        return main.MetaVoice(voice), voice

    def test_ensure_opens_once_and_reopens_after_clear(self):
        mv, voice = self._mv()
        mv.ensure(("meta_skip", ["skip"]))
        self.assertEqual(voice.n, 1)
        mv.ensure(("meta_skip", ["skip"]))          # unchanged — no reopen
        self.assertEqual(voice.n, 1)
        voice.open_ids.clear()                      # engine cleared on input_lock
        mv.ensure(("meta_skip", ["skip"]))          # vanished — reopen
        self.assertEqual(voice.n, 2)
        cfg = list(voice.open_ids.values())[0]
        self.assertEqual(cfg["keywords"], ["skip"])

    def test_ensure_none_closes(self):
        mv, voice = self._mv()
        mv.ensure(("meta_ready", ["ready"]))
        mv.ensure(None)
        self.assertEqual(voice.open_ids, {})

    def test_switching_desc_swaps_window(self):
        mv, voice = self._mv()
        mv.ensure(("meta_ready", ["ready"]))
        mv.ensure(("meta_skip", ["skip"]))
        self.assertEqual(len(voice.open_ids), 1)
        cfg = list(voice.open_ids.values())[0]
        self.assertEqual(cfg["id"], "meta_skip")

    def test_only_meta_events_queue_and_drain(self):
        mv, voice = self._mv()
        voice.fire("meta_skip")
        voice.fire("freedom")                       # scene VI — not ours
        voice.fire("meta_ready")
        self.assertEqual(list(mv.drain()), ["meta_skip", "meta_ready"])
        self.assertEqual(list(mv.drain()), [])


class TestVoiceWindowOpen(unittest.TestCase):
    def test_window_open_tracks_lifecycle(self):
        from engines.voice_engine import VoiceEngine
        engine = VoiceEngine({"voice": {}, "timing_defaults": {}})
        wid = engine.open_window({"id": "meta_skip", "keywords": ["skip"],
                                  "mode": "keyword", "tier": "reaction",
                                  "window_ms": 60000})
        self.assertTrue(engine.window_open(wid))
        self.assertFalse(engine.window_open("nope"))
        self.assertFalse(engine.window_open(None))
        engine._on_input_lock({"locked": True})     # lock clears every window
        self.assertFalse(engine.window_open(wid))


if __name__ == "__main__":
    unittest.main()
