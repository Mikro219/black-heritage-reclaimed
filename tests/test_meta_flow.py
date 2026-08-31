"""Meta-flow contract (July 2026): the MetaVoice window plumbing, the
VoiceEngine window-open lifecycle, and the player's seek contract
(start-at-frame / skip-prologue), all over synthetic shots — hardware-free.

The tests that ran the player over the hand-authored scenes/ tree were
retired with that tree (Aug 2026 — the .bhrx export is the single source
of truth).
"""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from engines.shot_sequence_player import ShotSequencePlayer
from tests.mocks import Bus

ROOT = Path(__file__).resolve().parent.parent


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


class TestStartSeek(unittest.TestCase):
    """start(index, start_frame) / set_prologue_end seek contract (July 2026):
    exported trees merge many original shots into one stretch shot, so the
    player must be able to enter a shot at an arbitrary local frame — landing
    in the intro, inside an FSM state's segment, or mid plain-playback."""

    FSM = {
        "initial": "oi_1",
        "states": {
            "oi_1":    {"segment": "oi_1", "loop": False},
            "between": {"segment": "between", "loop": False},
            "oi_2":    {"segment": "oi_2", "loop": False},
            "outro":   {"segment": "outro", "loop": False},
        },
        "transitions": [
            {"from": "oi_1", "on": "segment_end", "to": "between"},
            {"from": "between", "on": "segment_end", "to": "oi_2"},
            {"from": "oi_2", "on": "segment_end", "to": "outro"},
            {"from": "outro", "on": "segment_end", "to": "__advance__"},
        ],
    }
    SEGMENTS = {"intro": [1, 100], "oi_1": [101, 200], "between": [201, 260],
                "oi_2": [261, 320], "outro": [321, 400]}

    def _shot(self, **over):
        from engines.sequence_loader import Shot
        base = dict(shot="01", act="01", kind="interactive", audio_lines=[],
                    tracker_type="", tracker_notes="", reuse_of=None,
                    segments=dict(self.SEGMENTS),
                    interaction={"tier": "OI",
                                 "interaction_fsm": dict(self.FSM)},
                    fallback={"timeout_s": 300, "reprompt_s": [],
                              "on_timeout": "auto_advance"},
                    play_if=None, fps=30, timing_profile="standard",
                    frames_dir=Path("frames"), audio_dir=None, audio_file=None,
                    audio_events=[], captions=[], assets_pending=False,
                    segments_todo=False,
                    interaction_todo=False, reuse_self=False)
        base.update(over)
        return Shot(**base)

    def _segments_emitted(self, bus):
        return [(d["start"], d["end"], d["loop"])
                for n, d in bus.log if n == "play_segment"]

    def test_seek_lands_in_intro(self):
        bus = Bus()
        player = ShotSequencePlayer([self._shot()], {"timing_defaults": {}}, bus)
        player.start(0, start_frame=50)
        bus.emit("shot_frames_ready", {})
        self.assertEqual(self._segments_emitted(bus)[-1], (50, 100, False))
        self.assertEqual(player.debug_info()["shot_state"], "PLAY_INTRO")

    def test_seek_fast_forwards_fsm_to_containing_state(self):
        bus = Bus()
        player = ShotSequencePlayer([self._shot()], {"timing_defaults": {}}, bus)
        player.start(0, start_frame=230)
        bus.emit("shot_frames_ready", {})
        self.assertEqual(player._fsm.current, "between")
        self.assertEqual(self._segments_emitted(bus)[-1], (230, 260, False))

    def test_seek_plain_playback_shot(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            for i in range(6):
                (Path(td) / f"f{i:03d}.png").write_bytes(b"")
            bus = Bus()
            shot = self._shot(kind="playback", segments=None, interaction=None,
                              frames_dir=Path(td))
            player = ShotSequencePlayer([shot], {"timing_defaults": {}}, bus)
            player.start(0, start_frame=4)
            bus.emit("shot_frames_ready", {})
            self.assertEqual(self._segments_emitted(bus)[-1], (4, 6, False))

    def test_skip_prologue_frame_based(self):
        bus = Bus()
        player = ShotSequencePlayer([self._shot()], {"timing_defaults": {}}, bus)
        player.set_prologue_end(0, 60)
        player.start(0)
        bus.emit("shot_frames_ready", {})
        self.assertTrue(player.in_prologue())
        self.assertTrue(player.skip_prologue())
        self.assertEqual(self._segments_emitted(bus)[-1], (60, 100, False))
        # past the prologue point now: both report done, no double jump
        self.assertFalse(player.in_prologue())
        self.assertFalse(player.skip_prologue())

    def test_skip_prologue_frame_based_across_shots(self):
        bus = Bus()
        shots = [self._shot(), self._shot(shot="02")]
        player = ShotSequencePlayer(shots, {"timing_defaults": {}}, bus)
        player.set_prologue_end(1, 50)
        player.start(0)
        bus.emit("shot_frames_ready", {})
        self.assertTrue(player.skip_prologue())
        self.assertEqual(player.current_shot.shot, "02")
        bus.emit("shot_frames_ready", {})
        self.assertEqual(self._segments_emitted(bus)[-1], (50, 100, False))


if __name__ == "__main__":
    unittest.main()
