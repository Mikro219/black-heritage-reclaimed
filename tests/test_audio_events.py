"""Behaviour contract for the layered stem audio (July 2026).

Covers the loader's audio_events parsing (file resolution against the shot's
audio/ dir and the shared _audio pool; suppression of the baked audio.mp3)
and the ShotAudioMixer semantics: frame-anchored one-shots fire exactly once,
sustain beds loop through holds, bed continuity hands over across shots
without a restart, unclaimed beds fade out, and master volume rescales bed
gains without clobbering them. VO events (July 2026) play once on the
dedicated channel 0 and a `continues` VO piece never restarts the line.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from engines.sequence_loader import load_sequence
from engines import audio_mixer as am
from tests.mocks import Bus


# ---------------------------------------------------------------------------
# Fake pygame.mixer
# ---------------------------------------------------------------------------

class FakeSound:
    def __init__(self, path):
        self.path = path


class FakeChannel:
    def __init__(self, idx):
        self.idx = idx
        self.volume = 1.0
        self.plays = []       # (sound, loops, fade_ms)
        self.fadeouts = []    # fade_ms
        self.stopped = False
        self.busy = False     # tests flip this to simulate a finished sound

    def set_volume(self, v):
        self.volume = v

    def play(self, sound, loops=0, fade_ms=0):
        self.plays.append((sound, loops, fade_ms))
        self.stopped = False
        self.busy = True

    def fadeout(self, ms):
        self.fadeouts.append(ms)
        self.busy = False

    def stop(self):
        self.stopped = True
        self.busy = False

    def get_busy(self):
        return self.busy


class FakeMixer:
    def __init__(self):
        self.channels = {}
        self.num = 8

    def get_init(self):
        return True

    def get_num_channels(self):
        return self.num

    def set_num_channels(self, n):
        self.num = n

    def Channel(self, idx):
        if idx not in self.channels:
            self.channels[idx] = FakeChannel(idx)
        return self.channels[idx]

    def Sound(self, path):
        return FakeSound(path)


def make_mixer(events=None, fps=30, master=1.0):
    """ShotAudioMixer against a FakeMixer, already inside a ready shot."""
    fake = FakeMixer()
    bus = Bus()
    frame = {"v": 0}
    with mock.patch.object(am, "pygame", SimpleNamespace(mixer=fake)):
        mixer = am.ShotAudioMixer({"audio": {"master_volume": master}}, bus,
                                  frame_provider=lambda: frame["v"])
    # keep the patched pygame alive for the mixer's lifetime
    mixer._test_patch = mock.patch.object(am, "pygame", SimpleNamespace(mixer=fake))
    mixer._test_patch.start()
    if events is not None:
        bus.emit("shot_load", {"shot": SimpleNamespace(audio_events=events, fps=fps)})
        bus.emit("shot_frames_ready", {})
    return mixer, fake, bus, frame


def ev(file="bed.mp3", role="ambience", at_s=0.0, gain=1.0, sustain=None,
       continues=False, fade_in_ms=0, fade_out_ms=500, path="X"):
    return {"file": file, "role": role, "at_s": at_s, "duration_s": None,
            "source_offset_s": 0.0, "gain": gain, "fade_in_ms": fade_in_ms,
            "fade_out_ms": fade_out_ms,
            "sustain": (role in ("music", "ambience")) if sustain is None
                       else sustain,
            "continues": continues,
            "path": (file if path == "X" else path)}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class TestLoaderAudioEvents(unittest.TestCase):
    def _tree(self, tmp, audio_events, with_pool_file=True, with_audio_mp3=True):
        root = Path(tmp)
        act = root / "act_01_test" / "shot_01"
        (act / "frames").mkdir(parents=True)
        (act / "frames" / "0001.png").write_bytes(b"x")
        if with_audio_mp3:
            (act / "audio.mp3").write_bytes(b"x")
        if with_pool_file:
            (root / "_audio").mkdir()
            (root / "_audio" / "bed.mp3").write_bytes(b"x")
        meta = {"shot": "01", "kind": "playback", "audio": "audio.mp3",
                "audio_events": audio_events}
        with open(act / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        with open(root / "sequence.json", "w", encoding="utf-8") as f:
            json.dump({"fps": 30, "shots": [{"shot": "01", "act": "01"}]}, f)
        return root

    def test_events_parsed_and_pool_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp, [{"file": "bed.mp3", "role": "ambience",
                                     "at_s": 1.5, "gain": 0.4}])
            shot = load_sequence(root, {})[0]
            self.assertEqual(len(shot.audio_events), 1)
            e = shot.audio_events[0]
            self.assertEqual(e["role"], "ambience")
            self.assertTrue(e["sustain"])           # bed roles sustain by default
            self.assertFalse(e["continues"])
            self.assertEqual(e["path"], root / "_audio" / "bed.mp3")

    def test_audio_file_suppressed_when_events_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp, [{"file": "bed.mp3"}])
            shot = load_sequence(root, {})[0]
            self.assertIsNone(shot.audio_file)      # stems replace the baked mix

    def test_audio_file_kept_without_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp, [])
            shot = load_sequence(root, {})[0]
            self.assertIsNotNone(shot.audio_file)
            self.assertEqual(shot.audio_events, [])

    def test_shot_audio_dir_wins_over_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp, [{"file": "bed.mp3"}])
            local = root / "act_01_test" / "shot_01" / "audio"
            local.mkdir()
            (local / "bed.mp3").write_bytes(b"y")
            shot = load_sequence(root, {})[0]
            self.assertEqual(shot.audio_events[0]["path"], local / "bed.mp3")

    def test_unresolvable_file_keeps_none_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp, [{"file": "ghost.mp3"}], with_pool_file=False)
            shot = load_sequence(root, {})[0]
            self.assertIsNone(shot.audio_events[0]["path"])

    def test_bad_role_defaults_to_sfx(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp, [{"file": "bed.mp3", "role": "narration"}])
            shot = load_sequence(root, {})[0]
            self.assertEqual(shot.audio_events[0]["role"], "sfx")

    def test_vo_role_accepted_and_never_sustains_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp, [{"file": "bed.mp3", "role": "vo",
                                     "gain": 0.55}])
            shot = load_sequence(root, {})[0]
            e = shot.audio_events[0]
            self.assertEqual(e["role"], "vo")
            self.assertFalse(e["sustain"])   # a voice line must never loop


# ---------------------------------------------------------------------------
# Mixer
# ---------------------------------------------------------------------------

class TestOneShots(unittest.TestCase):
    def test_fires_once_when_frame_crossed(self):
        e = ev("crinkle.mp3", role="sfx", at_s=1.0)
        mixer, fake, bus, frame = make_mixer([e], fps=30)
        mixer.update()                       # frame 0 — before the anchor
        self.assertFalse(any(c.plays for c in fake.channels.values()
                             if c.idx in am.ONESHOT_CHANNELS))
        frame["v"] = 30
        mixer.update()
        mixer.update()                       # crossing again must not re-fire
        frame["v"] = 45
        mixer.update()
        plays = [p for c in fake.channels.values()
                 if c.idx in am.ONESHOT_CHANNELS for p in c.plays]
        self.assertEqual(len(plays), 1)
        self.assertEqual(plays[0][1], 0)     # loops=0: one-shot

    def test_no_fire_while_loading(self):
        e = ev("crinkle.mp3", role="sfx", at_s=0.0)
        mixer, fake, bus, frame = make_mixer([e])
        frame["v"] = None                    # render engine still loading
        mixer.update()
        self.assertFalse(any(c.plays for c in fake.channels.values()))

    def test_unresolved_path_skipped(self):
        e = ev("ghost.mp3", role="sfx", at_s=0.0, path=None)
        mixer, fake, bus, frame = make_mixer([e])
        mixer.update()                       # must not raise
        self.assertFalse(any(c.plays for c in fake.channels.values()))


class TestBeds(unittest.TestCase):
    def test_bed_loops_indefinitely(self):
        e = ev("night.mp3", at_s=0.0, gain=0.3, fade_in_ms=200)
        mixer, fake, bus, frame = make_mixer([e], master=1.0)
        mixer.update()
        ch = fake.channels[am.BED_CHANNELS[0]]
        self.assertEqual(len(ch.plays), 1)
        _, loops, fade = ch.plays[0]
        self.assertEqual(loops, -1)          # sustains through any HOLD length
        self.assertEqual(fade, 200)
        self.assertAlmostEqual(ch.volume, 0.3)

    def test_handover_does_not_restart(self):
        a = ev("night.mp3", at_s=0.0, gain=0.3)
        mixer, fake, bus, frame = make_mixer([a])
        mixer.update()
        ch = fake.channels[am.BED_CHANNELS[0]]
        self.assertEqual(len(ch.plays), 1)

        b = ev("night.mp3", at_s=0.0, gain=0.5, continues=True)
        bus.emit("shot_load", {"shot": SimpleNamespace(audio_events=[b], fps=30)})
        bus.emit("shot_frames_ready", {})
        frame["v"] = 10
        mixer.update()
        self.assertEqual(len(ch.plays), 1)         # no restart
        self.assertEqual(ch.fadeouts, [])          # and no fade
        self.assertAlmostEqual(ch.volume, 0.5)     # new gain adopted

    def test_unclaimed_bed_fades_on_next_shot(self):
        a = ev("night.mp3", at_s=0.0, fade_out_ms=650)
        mixer, fake, bus, frame = make_mixer([a])
        mixer.update()
        ch = fake.channels[am.BED_CHANNELS[0]]

        bus.emit("shot_load", {"shot": SimpleNamespace(audio_events=[], fps=30)})
        bus.emit("shot_frames_ready", {})
        self.assertEqual(ch.fadeouts, [650])

    def test_continues_without_active_bed_starts_fresh(self):
        b = ev("river.mp3", at_s=0.0, continues=True)
        mixer, fake, bus, frame = make_mixer([b])
        mixer.update()
        ch = fake.channels[am.BED_CHANNELS[0]]
        self.assertEqual(len(ch.plays), 1)   # fork entry mid-chain still gets audio

    def test_master_volume_rescales_bed(self):
        a = ev("night.mp3", at_s=0.0, gain=0.5)
        mixer, fake, bus, frame = make_mixer([a], master=1.0)
        mixer.update()
        ch = fake.channels[am.BED_CHANNELS[0]]
        self.assertAlmostEqual(ch.volume, 0.5)
        bus.emit("master_volume", {"volume": 0.4})
        self.assertAlmostEqual(ch.volume, 0.2)     # gain × master

    def test_bed_eviction_when_all_channels_busy(self):
        events = [ev(f"bed{i}.mp3", at_s=0.0) for i in range(4)]
        mixer, fake, bus, frame = make_mixer(events)
        mixer.update()
        total_plays = sum(len(fake.channels[c].plays) for c in am.BED_CHANNELS)
        self.assertEqual(total_plays, 4)     # oldest evicted, newest placed
        faded = sum(len(fake.channels[c].fadeouts) for c in am.BED_CHANNELS)
        self.assertEqual(faded, 1)


class TestSegmentJumpGuard(unittest.TestCase):
    """FSM forks jump the playback frame straight into a branch segment.
    Anchors skipped by more than the grace window are consumed silently;
    anchors at/near the landing frame still fire."""

    def test_jump_consumes_far_anchors_fires_near_ones(self):
        far  = ev("left_pick.ogg",  role="vo", at_s=21.0)   # frame 630
        near = ev("right_pick.ogg", role="vo", at_s=37.9)   # frame 1137
        mixer, fake, bus, frame = make_mixer([far, near], fps=30)
        mixer.update()                       # frame 0 — establishes reference
        frame["v"] = 1140                    # jump into right_selected
        mixer.update()
        ch = fake.channels.get(am.VO_CHANNEL)
        self.assertEqual(len(ch.plays), 1)   # only the landing-frame line
        self.assertEqual(len(mixer._fired), 2)   # far anchor consumed, not lost

    def test_first_update_of_a_shot_fires_normally(self):
        # No jump reference at shot entry: an intro anchored at 0 must fire
        # even when the first observed frame is already past it.
        e = ev("intro.ogg", role="vo", at_s=0.0)
        mixer, fake, bus, frame = make_mixer([e], fps=30)
        frame["v"] = 200                     # first frame seen mid-intro
        mixer.update()
        self.assertEqual(len(fake.channels[am.VO_CHANNEL].plays), 1)

    def test_continuous_playback_unaffected(self):
        e = ev("crinkle.mp3", role="sfx", at_s=1.0)
        mixer, fake, bus, frame = make_mixer([e], fps=30)
        for f in range(0, 45):               # normal frame-by-frame playback
            frame["v"] = f
            mixer.update()
        plays = [p for c in fake.channels.values()
                 if c.idx in am.ONESHOT_CHANNELS for p in c.plays]
        self.assertEqual(len(plays), 1)


class TestVoEvents(unittest.TestCase):
    """Auntie Liza's lines: channel 0, fire-once, never restarted by a
    `continues` piece, master volume re-stamped over RenderEngine's flat
    stamp, stopped by mixer.stop() only while a vo event owns the channel."""

    def test_vo_plays_once_on_channel_0(self):
        e = ev("bhr_scene_02.mp3", role="vo", at_s=1.0, gain=0.55)
        mixer, fake, bus, frame = make_mixer([e], fps=30)
        mixer.update()                               # before the anchor
        self.assertNotIn(am.VO_CHANNEL, fake.channels)
        frame["v"] = 30
        mixer.update()
        mixer.update()                               # must not re-fire
        ch = fake.channels[am.VO_CHANNEL]
        self.assertEqual(len(ch.plays), 1)
        self.assertEqual(ch.plays[0][1], 0)          # loops=0 — never loops
        self.assertAlmostEqual(ch.volume, 0.55)

    def test_new_line_replaces_previous_on_channel_0(self):
        a = ev("bhr_scene_02.mp3", role="vo", at_s=0.0)
        b = ev("bhr_scene_02__t5000_2000.ogg", role="vo", at_s=2.0)
        mixer, fake, bus, frame = make_mixer([a, b], fps=30)
        mixer.update()
        frame["v"] = 60
        mixer.update()
        self.assertEqual(len(fake.channels[am.VO_CHANNEL].plays), 2)

    def test_continues_piece_hands_over_without_restart(self):
        a = ev("line.ogg", role="vo", at_s=0.0, gain=0.5)
        mixer, fake, bus, frame = make_mixer([a])
        mixer.update()
        ch = fake.channels[am.VO_CHANNEL]
        self.assertEqual(len(ch.plays), 1)

        b = ev("line.ogg", role="vo", at_s=0.0, gain=0.7, continues=True)
        bus.emit("shot_load", {"shot": SimpleNamespace(audio_events=[b], fps=30)})
        bus.emit("shot_frames_ready", {})
        frame["v"] = 20
        mixer.update()
        self.assertEqual(len(ch.plays), 1)           # no restart
        self.assertAlmostEqual(ch.volume, 0.7)       # gain adopted

    def test_continues_piece_after_line_finished_is_consumed_silently(self):
        a = ev("line.ogg", role="vo", at_s=0.0)
        mixer, fake, bus, frame = make_mixer([a])
        mixer.update()
        ch = fake.channels[am.VO_CHANNEL]
        ch.busy = False                              # line ran out in a HOLD

        b = ev("line.ogg", role="vo", at_s=0.0, continues=True)
        bus.emit("shot_load", {"shot": SimpleNamespace(audio_events=[b], fps=30)})
        bus.emit("shot_frames_ready", {})
        frame["v"] = 50
        mixer.update()
        self.assertEqual(len(ch.plays), 1)           # never replayed

    def test_master_volume_restamps_vo_gain(self):
        a = ev("line.ogg", role="vo", at_s=0.0, gain=0.5)
        mixer, fake, bus, frame = make_mixer([a], master=1.0)
        mixer.update()
        ch = fake.channels[am.VO_CHANNEL]
        # RenderEngine would have stamped a flat master here first
        ch.set_volume(0.4)
        bus.emit("master_volume", {"volume": 0.4})
        self.assertAlmostEqual(ch.volume, 0.2)       # gain × master wins

    def test_stop_stops_channel_0_only_while_vo_live(self):
        mixer, fake, bus, frame = make_mixer([ev("night.mp3", at_s=0.0)])
        mixer.update()
        mixer.stop()
        self.assertNotIn(am.VO_CHANNEL, fake.channels)   # ch0 untouched

        a = ev("line.ogg", role="vo", at_s=0.0)
        mixer, fake, bus, frame = make_mixer([a])
        mixer.update()
        mixer.stop()
        self.assertTrue(fake.channels[am.VO_CHANNEL].stopped)


if __name__ == "__main__":
    unittest.main()
