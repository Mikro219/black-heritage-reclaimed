"""Experience Builder export — a fixture .bhrx.json project (the mockup's demo
flow: Intro -> Choice -> Forest/River -> Merge -> Reunion, plus interaction
windows) must export to a scenes tree that satisfies the same wiring
invariants test_sequence.py enforces on the live tree."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import export_experience  # noqa: E402
from engines.detectors import REGISTRY  # noqa: E402
from engines.sequence_loader import load_sequence  # noqa: E402


def fixture_project() -> dict:
    """The mockup 1A demo flow with every export path exercised:
    single-OI playback, multi-OI playback, choice fork, merge, voice window."""
    return {
        "version": 1,
        "name": "Fixture Flow",
        "fps": 30,
        "media": [{"id": "m1", "name": "master.mp4", "duration_s": 60.0}],
        "sounds": [{"id": "s1", "name": "detect.mp3"},
                   {"id": "s2", "name": "night_bed.mp3"},
                   {"id": "s3", "name": "crinkle.mp3"}],
        "settings": {"global_detect_sound": "s1"},
        "start": "b_intro",
        "blocks": [
            {"id": "b_intro", "type": "playback", "name": "Intro", "media": "m1",
             "range_s": [0.0, 6.0], "pos": [0, 0],
             "audio": [
                 {"id": "a1", "sound": "s2", "role": "ambience", "at_s": 0.0,
                  "duration_s": None, "source_offset_s": 0, "gain": 0.4,
                  "fade_in_ms": 300, "fade_out_ms": 600, "sustain": True,
                  "continues": False},
                 {"id": "a2", "sound": "s3", "role": "sfx", "at_s": 2.5,
                  "duration_s": 1.0, "source_offset_s": 0, "gain": 1.0,
                  "fade_in_ms": 0, "fade_out_ms": 0, "sustain": False,
                  "continues": False}],
             "windows": [
                 {"id": "w_reach", "label": "Reach flask", "detector": "forward_reach",
                  "params": {"area_growth_threshold": 0.3},
                  "region": {"shape": "rect", "x": 0.3, "y": 0.1, "w": 0.3, "h": 0.3},
                  "appears_s": 2.0, "duration_s": 2.5}]},
            {"id": "b_choice", "type": "choice", "name": "Choose Direction", "media": "m1",
             "range_s": [6.0, 12.0], "hold": "pause_end", "pos": [0, 0],
             "audio": [
                 {"id": "a3", "sound": "s2", "role": "ambience", "at_s": 0.0,
                  "duration_s": None, "source_offset_s": 0, "gain": 0.4,
                  "fade_in_ms": 0, "fade_out_ms": 600, "sustain": True,
                  "continues": True}],
             "branches": [
                 {"id": "br_l", "window": "w_left", "to": "b_forest", "label": "Go Left"},
                 {"id": "br_r", "window": "w_right", "to": "b_river", "label": "Go Right"},
                 {"id": "br_v", "window": "w_say", "to": "b_river", "label": "Say river"}],
             "timeout": {"seconds": 20, "to": "b_forest"},
             "windows": [
                 {"id": "w_left", "label": "Go Left", "detector": "point_region",
                  "params": {"hold_ms": 700},
                  "region": {"shape": "rect", "x": 0.05, "y": 0.3, "w": 0.25, "h": 0.4},
                  "appears_s": 0.0, "duration_s": None},
                 {"id": "w_right", "label": "Go Right", "detector": "point_region",
                  "params": {"hold_ms": 700},
                  "region": {"shape": "rect", "x": 0.7, "y": 0.3, "w": 0.25, "h": 0.4},
                  "appears_s": 0.0, "duration_s": None},
                 {"id": "w_say", "label": "Say river", "detector": "voice",
                  "params": {"keyword": "river"}, "region": None,
                  "appears_s": 0.0, "duration_s": None}]},
            {"id": "b_forest", "type": "playback", "name": "Forest Path", "media": "m1",
             "range_s": [12.0, 21.0], "pos": [0, 0],
             "windows": [
                 {"id": "w_oi1", "label": "Duck", "detector": "bilateral_lower",
                  "params": {}, "region": None, "appears_s": 1.0, "duration_s": 2.0},
                 {"id": "w_oi2", "label": "Look up", "detector": "presence_bilateral",
                  "params": {"hold_ms": 500}, "region": None,
                  "appears_s": 5.0, "duration_s": 2.0},
                 {"id": "w_voice", "label": "Say go", "detector": "voice",
                  "params": {"keyword": "go"}, "region": None,
                  "appears_s": 7.5, "duration_s": 1.0}]},
            {"id": "b_river", "type": "playback", "name": "River Path", "media": "m1",
             "range_s": [21.0, 29.0], "pos": [0, 0], "windows": []},
            {"id": "b_merge", "type": "merge", "name": "Merge", "pos": [0, 0]},
            {"id": "b_end", "type": "playback", "name": "Reunion", "media": "m1",
             "range_s": [29.0, 40.0], "pos": [0, 0], "windows": []},
        ],
        "edges": [
            {"from": "b_intro", "to": "b_choice"},
            {"from": "b_forest", "to": "b_merge"},
            {"from": "b_river", "to": "b_merge"},
            {"from": "b_merge", "to": "b_end"},
        ],
    }


_HOLD_SEGMENTS = {"intro": [1, 309], "idle_loop": [310, 310],
                  "left_selected": [662, 748], "left_to_right": [748, 820],
                  "right_switch_hold": [820, 820], "right_to_left": [836, 868],
                  "left_switch_hold": [868, 868], "right_selected": [1140, 1254]}


def _confirm_choice_block(hold_segments):
    """A choice block whose voice window has NO branch target (the confirm-
    keyword pattern), optionally with authored hold animation segments."""
    b = {
        "id": "b_c", "type": "choice", "name": "Fork", "media": "m1",
        "range_s": [0.0, 47.1],
        "branches": [
            {"id": "bl", "window": "wl", "to": "x_l", "label": "L"},
            {"id": "br", "window": "wr", "to": "x_r", "label": "R"},
        ],
        "timeout": {"seconds": 60, "to": "x_l"},
        "windows": [
            {"id": "wl", "label": "L", "detector": "point_region",
             "params": {"hold_ms": 600}, "region": None,
             "appears_s": 0, "duration_s": None},
            {"id": "wr", "label": "R", "detector": "point_region",
             "params": {"hold_ms": 600}, "region": None,
             "appears_s": 0, "duration_s": None},
            {"id": "wv", "label": 'Say "go"', "detector": "voice",
             "params": {"keyword": "go"}, "region": None,
             "appears_s": 0, "duration_s": None},
        ],
    }
    if hold_segments:
        b["hold_segments"] = hold_segments
    return b


def _retry_project() -> dict:
    """A mini flow with a wrong-way retry fork (the live shot 37 model):
    picking left plays a redirect and returns to the choice; only right
    advances. The wrong clip block carries audio to exercise the fold."""
    return {
        "version": 1, "name": "Retry Flow", "fps": 30,
        "media": [{"id": "m1", "name": "master.mp4", "duration_s": 60.0}],
        "sounds": [{"id": "s2", "name": "night_bed.mp3"},
                   {"id": "s3", "name": "crinkle.mp3"}],
        "settings": {},
        "start": "b_start",
        "blocks": [
            {"id": "b_start", "type": "playback", "name": "Start", "media": "m1",
             "range_s": [0.0, 6.0], "pos": [0, 0], "windows": []},
            {"id": "b_fork", "type": "choice", "name": "Paw Fork", "media": "m1",
             "range_s": [6.0, 17.0], "hold": "pause_end", "pos": [0, 0],
             "hold_segments": {"intro": [1, 180], "idle_loop": [180, 180],
                               "wrong_path": [181, 330]},
             "branches": [
                 {"id": "bl", "window": "wl", "to": "b_wrong", "label": "Go Left",
                  "retry": True},
                 {"id": "br", "window": "wr", "to": "b_correct",
                  "label": "Go Right"}],
             "timeout": {"seconds": 60, "to": "b_wrong"},
             "windows": [
                 {"id": "wl", "label": "Go Left", "detector": "point_region",
                  "params": {"hold_ms": 600}, "region": None,
                  "appears_s": 0, "duration_s": None},
                 {"id": "wr", "label": "Go Right", "detector": "point_region",
                  "params": {"hold_ms": 600}, "region": None,
                  "appears_s": 0, "duration_s": None}]},
            {"id": "b_wrong", "type": "playback", "name": "Wrong Way",
             "media": "m1", "range_s": [12.0, 17.0], "pos": [0, 0],
             "windows": [],
             "audio": [
                 {"id": "aw1", "sound": "s3", "role": "sfx", "at_s": 1.0,
                  "duration_s": 1.0, "source_offset_s": 0, "gain": 1.0,
                  "fade_in_ms": 0, "fade_out_ms": 0, "sustain": False,
                  "continues": False},
                 {"id": "aw2", "sound": "s2", "role": "music", "at_s": 0.0,
                  "duration_s": None, "source_offset_s": 0, "gain": 0.5,
                  "fade_in_ms": 0, "fade_out_ms": 400, "sustain": True,
                  "continues": True}]},
            {"id": "b_correct", "type": "playback", "name": "Correct Way",
             "media": "m1", "range_s": [17.0, 20.0], "pos": [0, 0],
             "windows": []},
            {"id": "b_merge", "type": "merge", "name": "Merge", "pos": [0, 0]},
            {"id": "b_end", "type": "playback", "name": "End", "media": "m1",
             "range_s": [20.0, 26.0], "pos": [0, 0], "windows": []},
        ],
        "edges": [
            {"from": "b_start", "to": "b_fork"},
            {"from": "b_wrong", "to": "b_merge"},
            {"from": "b_correct", "to": "b_merge"},
            {"from": "b_merge", "to": "b_end"},
        ],
    }


class TestRetryChoiceExport(unittest.TestCase):
    """A branch marked "retry": true is the wrong-way path: its chain emits no
    shots (folded into the choice), the choice FSM loops the redirect back to
    waiting, and only the other side advances (live shot 37 model)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(cls.tmp.name)
        cls.project_path = tmp_path / "retry.bhrx.json"
        with open(cls.project_path, "w", encoding="utf-8") as f:
            json.dump(_retry_project(), f)
        (tmp_path / "night_bed.mp3").write_bytes(b"bed")
        (tmp_path / "crinkle.mp3").write_bytes(b"sfx")
        cls.out = tmp_path / "scenes_generated"
        export_experience.warnings.clear()
        export_experience.export(cls.project_path, cls.out, do_frames=False,
                                 video_override=None, sound_override=None)
        cls.export_warnings = list(export_experience.warnings)
        cls.shots = load_sequence(cls.out, {"fps": 30})
        cls.by_name = {}
        for s in cls.shots:
            meta_path = (cls.out / export_experience.ACT_DIRNAME /
                         f"scene_{s.shot}" / "metadata.json")
            with open(meta_path, encoding="utf-8") as f:
                cls.by_name[json.load(f)["_generated_from"]["name"]] = s

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_retry_branch_emits_no_shot(self):
        self.assertEqual(sorted(self.by_name),
                         ["Correct Way", "End", "Paw Fork", "Start"])
        self.assertFalse(any("unreachable" in w for w in self.export_warnings),
                         "folded retry blocks must not be reported unreachable")

    def test_wrong_pick_loops_back_only_correct_advances(self):
        fork = self.by_name["Paw Fork"]
        fsm = fork.interaction["interaction_fsm"]
        t = fsm["transitions"]
        self.assertIn({"from": "waiting", "on": "point_left",
                       "to": "wrong_path"}, t)
        self.assertIn({"from": "wrong_path", "on": "segment_end",
                       "to": "waiting"}, t)
        self.assertIn({"from": "waiting", "on": "point_right",
                       "to": "confirm_right"}, t)
        self.assertIn({"from": "confirm_right", "on": "segment_end",
                       "to": "__advance__"}, t)
        # the wrong side must have no path to __advance__
        self.assertFalse(any(tr["to"] == "__advance__"
                             and tr["from"] == "wrong_path" for tr in t))
        self.assertEqual(fork.segments["wrong_path"], [181, 330])
        self.assertEqual(fork.segments["idle_loop"], [180, 180])
        self.assertNotIn("detect.mp3", json.dumps(fork.interaction))

    def test_only_correct_branch_is_gated(self):
        fork = self.by_name["Paw Fork"]
        correct = self.by_name["Correct Way"]
        self.assertEqual(correct.play_if,
                         {"shot": fork.shot, "branch": "right"})
        gated = [s for s in self.shots if s.play_if]
        self.assertEqual(len(gated), 1,
                         "timeout must latch the CORRECT side (the only "
                         "gated chain) — a gated wrong chain would loop")
        self.assertIsNone(self.by_name["End"].play_if)

    def test_folded_audio_rebased_beds_dropped(self):
        fork = self.by_name["Paw Fork"]
        events = fork.audio_events or []
        sfx = [e for e in events if e["file"] == "crinkle.mp3"]
        self.assertEqual(len(sfx), 1)
        # b_wrong starts at 12.0, the fork at 6.0 -> clip at 1.0 lands at 7.0
        self.assertEqual(sfx[0]["at_s"], 7.0)
        self.assertFalse(sfx[0]["continues"])
        self.assertFalse(any(e["file"] == "night_bed.mp3" for e in events),
                         "a folded sustain bed must be dropped — the choice "
                         "shot's own bed loops through the redirect")

    def test_sound_only_retry_fallback(self):
        """Without hold_segments the wrong pick is a sound-only bounce on the
        idle frame (RETRY_WRONG_SFX), still returning to waiting."""
        block = _retry_project()["blocks"][1]
        del block["hold_segments"]
        meta = export_experience.choice_metadata(block, "02", 330, 30)
        fsm = meta["interaction"]["interaction_fsm"]
        wrong = fsm["states"]["wrong_path"]
        self.assertEqual(wrong["segment"], "idle_loop")
        self.assertEqual(wrong["on_enter_sfx"],
                         export_experience.RETRY_WRONG_SFX)
        self.assertIn({"from": "wrong_path", "on": "segment_end",
                       "to": "waiting"}, fsm["transitions"])


class TestBlockingDrawExport(unittest.TestCase):
    """Draw windows FREEZE-AND-HOLD on export (live shot-20 idle/play model):
    the FSM holds on the window's FIRST frame (single-frame loop segment)
    until the stroke is detected (oi_<id>) or the per-stroke window expires
    (oi_done escape hatch); only then does the window's animation span play.
    Draw payloads carry neither detect sfx nor green flash. Other windows
    keep play-through."""

    def _block(self, windows):
        return {"id": "b", "type": "playback", "name": "Draw scene",
                "media": "m1", "range_s": [0.0, 20.0], "windows": windows}

    @staticmethod
    def _draw(wid, label, direction, appears_s, extra_params=None):
        params = {"direction": direction, **(extra_params or {})}
        return {"id": wid, "label": label, "detector": "directional_draw",
                "params": params, "region": None,
                "appears_s": appears_s, "duration_s": 2.0}

    def test_draw_windows_freeze_then_play_on_stroke(self):
        wins = [self._draw("w1", "Draw left", "left", 2.0),
                self._draw("w2", "Draw up", "up", 6.0)]
        meta = export_experience.playback_metadata(
            self._block(wins), "01", 600, 30)
        fsm = meta["interaction"]["interaction_fsm"]

        for st_id in ("oi_1", "oi_2"):
            st = fsm["states"][st_id]
            self.assertTrue(st["loop"], st_id)
            self.assertEqual(st["segment"], f"{st_id}_hold")
            self.assertEqual(st["window_ms"],
                             int(export_experience.DRAW_BLOCK_TIMEOUT_S * 1000))
            # hold = the window's FIRST frame only; play = the full span
            hold = meta["segments"][f"{st_id}_hold"]
            span = meta["segments"][st_id]
            self.assertEqual(hold, [span[0], span[0]])
            self.assertFalse(fsm["states"][f"{st_id}_play"]["loop"])
            self.assertEqual(fsm["states"][f"{st_id}_play"]["segment"], st_id)
            # no flash, no ping on the stroke payload
            self.assertNotIn("feedback", st["oi"])
            self.assertNotIn("sfx", st["oi"])
        self.assertEqual(meta["segments"]["oi_1"], [61, 120])

        trans = {(t["from"], t["on"]): t["to"] for t in fsm["transitions"]}
        self.assertEqual(trans[("oi_1", "oi_draw_left")], "oi_1_play")
        self.assertEqual(trans[("oi_1", "oi_done")], "oi_1_play")
        self.assertNotIn(("oi_1", "segment_end"), trans)
        self.assertEqual(trans[("oi_1_play", "segment_end")], "between_1")
        self.assertEqual(trans[("oi_2", "oi_draw_up")], "oi_2_play")
        self.assertEqual(trans[("oi_2_play", "segment_end")], "outro")
        # gap and outro stay play-through
        self.assertFalse(fsm["states"]["between_1"]["loop"])
        self.assertEqual(trans[("between_1", "segment_end")], "oi_2")
        self.assertEqual(trans[("outro", "segment_end")], "__advance__")

        # top-level timeout covers content PLUS both blocking budgets
        content_s = (600 - 61 + 1) / 30
        self.assertGreaterEqual(
            meta["fallback"]["timeout_s"],
            content_s + 2 * export_experience.DRAW_BLOCK_TIMEOUT_S)

    def test_single_draw_window_routes_to_fsm(self):
        """A lone draw window must still block — it takes the FSM path, not
        the non-blocking oi_frame_window playback path."""
        meta = export_experience.playback_metadata(
            self._block([self._draw("w1", "Draw left", "left", 2.0)]),
            "01", 600, 30)
        self.assertEqual(meta["kind"], "interactive")
        fsm = meta["interaction"]["interaction_fsm"]
        self.assertTrue(fsm["states"]["oi_1"]["loop"])
        self.assertIn("oi_1_play", fsm["states"])
        self.assertNotIn("oi_frame_window", meta["interaction"])

    def test_block_timeout_param_overrides_and_is_consumed(self):
        meta = export_experience.playback_metadata(
            self._block([self._draw("w1", "Draw left", "left", 2.0,
                                    {"block_timeout_s": 10})]),
            "01", 600, 30)
        st = meta["interaction"]["interaction_fsm"]["states"]["oi_1"]
        self.assertEqual(st["window_ms"], 10000)
        self.assertNotIn("block_timeout_s", st["oi"]["params"],
                         "exporter-only param must not reach the detector")

    def test_non_draw_windows_stay_play_through(self):
        wins = [{"id": "w1", "label": "Duck", "detector": "bilateral_lower",
                 "params": {}, "region": None, "appears_s": 1.0,
                 "duration_s": 2.0},
                {"id": "w2", "label": "Look", "detector": "presence_bilateral",
                 "params": {}, "region": None, "appears_s": 5.0,
                 "duration_s": 2.0}]
        meta = export_experience.playback_metadata(
            self._block(wins), "01", 600, 30)
        fsm = meta["interaction"]["interaction_fsm"]
        for st in fsm["states"].values():
            self.assertFalse(st.get("loop"))
        ons = {t["on"] for t in fsm["transitions"]}
        self.assertEqual(ons, {"segment_end"})


class TestClipSpeedExport(unittest.TestCase):
    """A builder clip's `speed` (CapCut pitch-with-tempo shift — the draw-
    stroke beep ramp) must produce a dedicated speed-tagged render; chained
    beds ignore speed (their shared render is seek-offset-addressed)."""

    def _events(self, clip_extra, plan=None):
        import unittest.mock as mock
        block = {"id": "b1", "type": "playback", "name": "B",
                 "range_s": [0.0, 10.0],
                 "audio": [{"id": "a1", "sound": "s3", "role": "sfx",
                            "at_s": 1.0, "duration_s": 0.933,
                            "source_offset_s": 0, "gain": 1.0,
                            "fade_in_ms": 0, "fade_out_ms": 0,
                            "sustain": False, "continues": False,
                            **clip_extra}]}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "crinkle.mp3").write_bytes(b"x")
            pool = tmp / "_audio"
            pool.mkdir()
            sounds = {"s3": {"id": "s3", "name": "crinkle.mp3"}}
            fake_render = mock.Mock(
                side_effect=lambda ff, src, off, dur, spd, out:
                    (out.write_bytes(b"r") or True))
            with mock.patch.object(export_experience, "probe_duration",
                                   return_value=1.033), \
                 mock.patch.object(export_experience, "render_trim",
                                   fake_render):
                evs = export_experience.block_audio_events(
                    block, sounds, tmp / "p.bhrx.json", pool, "ffmpeg",
                    {}, plan)
        return evs, fake_render

    def test_speeded_clip_renders_pitch_variant(self):
        evs, fake_render = self._events({"speed": 1.107})
        self.assertIn("_x1p107", evs[0]["file"])
        call = fake_render.call_args
        self.assertAlmostEqual(call.args[4], 1.107)   # speed reaches ffmpeg
        self.assertAlmostEqual(call.args[3], 0.933)   # OUTPUT (timeline) dur

    def test_speed_one_clip_needs_no_render(self):
        # duration ~= natural, offset 0, speed 1 -> ships the original file
        evs, fake_render = self._events({"duration_s": 1.0})
        self.assertEqual(evs[0]["file"], "crinkle.mp3")
        fake_render.assert_not_called()


class TestExperienceExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(cls.tmp.name)
        cls.project_path = tmp_path / "fixture.bhrx.json"
        with open(cls.project_path, "w", encoding="utf-8") as f:
            json.dump(fixture_project(), f)
        # sound files next to the project so the audio exporter resolves them
        (tmp_path / "night_bed.mp3").write_bytes(b"bed")
        (tmp_path / "crinkle.mp3").write_bytes(b"sfx")
        cls.out = tmp_path / "scenes_generated"
        export_experience.warnings.clear()
        export_experience.export(cls.project_path, cls.out, do_frames=False,
                                 video_override=None, sound_override=None)
        cls.export_warnings = list(export_experience.warnings)
        cls.shots = load_sequence(cls.out, {"fps": 30})
        cls.by_id = {s.shot: s for s in cls.shots}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ── structure ──────────────────────────────────────────────────────────

    def test_shot_count_and_order(self):
        # merge emits no shot: intro, choice, forest, river, reunion = 5
        self.assertEqual(len(self.shots), 5)
        names = []
        for s in self.shots:
            meta_path = (self.out / export_experience.ACT_DIRNAME /
                         f"scene_{s.shot}" / "metadata.json")
            with open(meta_path, encoding="utf-8") as f:
                names.append(json.load(f)["_generated_from"]["name"])
        self.assertEqual(names, ["Intro", "Choose Direction", "Forest Path",
                                 "River Path", "Reunion"])

    def test_kinds(self):
        kinds = [s.kind for s in self.shots]
        # intro has ONE window -> stays playback with oi_frame_window;
        # choice -> interactive; forest has 2 gesture windows -> interactive
        self.assertEqual(kinds, ["playback", "interactive", "interactive",
                                 "playback", "playback"])

    def test_branch_gating(self):
        forest, river, reunion = self.shots[2], self.shots[3], self.shots[4]
        choice = self.shots[1]
        self.assertEqual(forest.play_if, {"shot": choice.shot, "branch": "left"})
        self.assertEqual(river.play_if, {"shot": choice.shot, "branch": "right"})
        self.assertIsNone(reunion.play_if, "convergence must clear play_if")

    def test_single_window_playback_uses_oi_frame_window(self):
        intro = self.shots[0]
        inter = intro.interaction
        self.assertEqual(inter["tier"], "OI")
        self.assertEqual(inter["type"], "forward_reach")
        a, b = inter["oi_frame_window"]
        # 2.0s..4.5s @30fps -> frames 61..135 of 180
        self.assertEqual([a, b], [61, 135])
        # screen x 0.3 (w 0.3) -> raw camera x 1-0.3-0.3 (the mirror flip)
        self.assertEqual(inter["params"]["region_rect"]["x"], 0.4)

    def test_choice_fsm_shape(self):
        choice = self.shots[1]
        fsm = choice.interaction["interaction_fsm"]
        self.assertEqual(fsm["gesture_type"], "region")
        self.assertEqual(fsm["initial"], "waiting")
        ons = {t["on"] for t in fsm["transitions"]}
        self.assertIn("point_left", ons)
        self.assertIn("point_right", ons)
        self.assertEqual(fsm["fallback"]["timeout_s"], 20)
        self.assertEqual(choice.interaction["hold_ms"], 700)  # from window params

    def test_flat_choice_confirms_are_silent(self):
        """A flat choice pick advances silently into its gated branch shot —
        no detect.mp3 ping on the confirm states (Bear Paw / Wagon Wheel)."""
        fsm = self.shots[1].interaction["interaction_fsm"]
        for st in ("confirm_left", "confirm_right"):
            self.assertNotIn("on_enter_sfx", fsm["states"][st])

    def test_draw_windows_have_no_detect_sfx(self):
        """directional_draw stroke completions are silent (comet/star-trail/
        green flash are the feedback); other detectors keep the detect ping."""
        draw = {"id": "w", "label": "Draw left", "detector": "directional_draw",
                "params": {"direction": "left"}, "region": None}
        self.assertNotIn("sfx", export_experience.oi_dict(draw, 1))
        point = {"id": "w", "label": "Point", "detector": "point_target_held",
                 "params": {"hold_ms": 400}, "region": None}
        self.assertEqual(export_experience.oi_dict(point, 1)["sfx"],
                         "detect.mp3")

    def test_block_captions_pass_through(self):
        """A block's authored captions flow to shot metadata; rect (screen-space
        placement) is preserved, textless entries dropped, sorted by at_s."""
        block = {"captions": [
            {"id": "c2", "at_s": 5.0, "duration_s": 3, "text": "second"},
            {"id": "c1", "at_s": 1.0, "duration_s": 2, "text": "first",
             "rect": {"x": 0.2, "y": 0.8, "w": 0.6, "h": 0.1}},
            {"id": "c3", "at_s": 9.0, "text": "   "},   # empty -> dropped
        ]}
        caps = export_experience.block_captions(block)
        self.assertEqual([c["text"] for c in caps], ["first", "second"])
        self.assertEqual(caps[0]["rect"]["x"], 0.2)
        self.assertNotIn("rect", caps[1])
        self.assertEqual(export_experience.block_captions({}), [])

    def test_voice_alternative_param_exports_as_payload(self):
        """params.voice_alternative on a gesture window becomes a parallel
        VI declaration in the OI payload (Scene 1 raise-hands / "freedom")
        and never reaches the detector params."""
        win = {"id": "w", "label": "Raise hands", "detector": "presence_bilateral",
               "params": {"hold_ms": 500, "voice_alternative": "freedom"},
               "region": None}
        out = export_experience.oi_dict(win, 1)
        self.assertEqual(out["voice_alternative"]["keywords"], ["freedom"])
        self.assertEqual(out["voice_alternative"]["id"], "freedom")
        self.assertEqual(out["voice_alternative"]["tier"], "cg_alternative")
        self.assertNotIn("voice_alternative", out["params"])
        plain = export_experience.oi_dict(
            {"id": "w", "label": "Raise", "detector": "presence_bilateral",
             "params": {"hold_ms": 500}, "region": None}, 1)
        self.assertNotIn("voice_alternative", plain)

    def test_region_space_conventions(self):
        """The Builder authors regions in SCREEN space. A detection region
        (point windows) must flip to RAW camera space on export (the mirror
        bug: off-centre targets detected on the wrong side); a draw window's
        rect is pure display and exports unflipped as indicator_rect."""
        region = {"shape": "rect", "x": 0.3, "y": 0.12, "w": 0.134, "h": 0.25}
        point = {"id": "w", "label": "Point", "detector": "point_target_held",
                 "params": {"hold_ms": 400}, "region": dict(region)}
        p = export_experience.oi_dict(point, 1)["params"]
        self.assertAlmostEqual(p["region_rect"]["x"], 1.0 - 0.3 - 0.134,
                               places=3)
        self.assertAlmostEqual(p["region_rect"]["y"], 0.12, places=3)
        self.assertNotIn("indicator_rect", p)

        draw = {"id": "w", "label": "Draw", "detector": "directional_draw",
                "params": {"direction": "left"}, "region": dict(region)}
        p = export_experience.oi_dict(draw, 1)["params"]
        self.assertAlmostEqual(p["indicator_rect"]["x"], 0.3, places=3)
        self.assertNotIn("region_rect", p)

    def test_region_screen_to_raw_mirror(self):
        """Builder regions are SCREEN space; the exporter mirrors non-draw
        rects to RAW camera space (x -> 1-x-w) for the detectors."""
        raw = {"x": 0.567, "y": 0.12, "w": 0.134, "h": 0.253}
        screen = {"x": round(1.0 - 0.567 - 0.134, 6), "y": 0.12,
                  "w": 0.134, "h": 0.253}
        win = {"id": "w", "label": "Point", "detector": "point_target_held",
               "params": {"hold_ms": 600}, "region": dict(screen)}
        back = export_experience.oi_dict(win, 1)["params"]["region_rect"]
        for k in ("x", "y", "w", "h"):
            self.assertAlmostEqual(back[k], raw[k], places=3)

    def test_choice_voice_branch_is_spoken_pick(self):
        """The 'Say river' voice branch targets the same block as Go Right, so
        the fork's waiting state gains the keyword and a voice_<kw> transition
        to confirm_right (the shot 09 pattern)."""
        fsm = self.shots[1].interaction["interaction_fsm"]
        self.assertEqual(fsm["states"]["waiting"].get("voice"), "river")
        self.assertIn({"from": "waiting", "on": "voice_river", "to": "confirm_right"},
                      fsm["transitions"])

    def test_playback_voice_window_exports_as_keyword_oi(self):
        """Forest Path's voice window becomes a keyword VI state in the
        play-through FSM (shot 24 pattern with a `keywords` oi payload)."""
        forest = self.shots[2]
        states = forest.interaction["interaction_fsm"]["states"]
        voice_states = [st for st in states.values()
                        if (st.get("oi") or {}).get("keywords")]
        self.assertEqual(len(voice_states), 1)
        self.assertEqual(voice_states[0]["oi"]["keywords"], ["go"])
        self.assertEqual(voice_states[0]["oi"]["mode"], "keyword")
        self.assertFalse(any("left out of the export" in w
                             for w in self.export_warnings),
                         "voice windows must export, not be dropped")

    def test_voice_confirmed_choice_animated_hold(self):
        """A voice window with NO branch target becomes the confirm keyword:
        picks land in selected holds (pick SFX), flips play the switch
        animation (same SFX), only voice_<kw> advances — no confirm states,
        no detect.mp3 (the crossroads 'say go' pattern)."""
        meta = export_experience.choice_metadata(
            _confirm_choice_block(_HOLD_SEGMENTS), "02", 1413, 30)
        fsm = meta["interaction"]["interaction_fsm"]
        self.assertNotIn("confirm_left", fsm["states"])
        self.assertNotIn("detect.mp3", json.dumps(meta))
        sfx = export_experience.CHOICE_HOLD_SFX
        self.assertEqual(fsm["states"]["left_selected"]["on_enter_sfx"], sfx)
        self.assertEqual(fsm["states"]["left_to_right"]["on_enter_sfx"], sfx)
        self.assertEqual(fsm["states"]["right_selected"]["voice"], "go")
        self.assertEqual(meta["segments"]["left_selected"], [662, 748])
        t = fsm["transitions"]
        self.assertIn({"from": "waiting", "on": "point_left",
                       "to": "left_selected"}, t)
        self.assertIn({"from": "left_selected", "on": "point_right",
                       "to": "left_to_right"}, t)
        self.assertIn({"from": "left_to_right", "on": "segment_end",
                       "to": "right_switch_hold"}, t)
        for st in ("left_selected", "right_selected",
                   "left_switch_hold", "right_switch_hold"):
            self.assertIn({"from": st, "on": "voice_go", "to": "__advance__"}, t)
        # a pick alone must never advance
        self.assertFalse(any(tr["to"] == "__advance__" and tr["from"] == "waiting"
                             for tr in t))

    def test_voice_confirmed_choice_sound_only_fallback(self):
        """Without hold_segments the same machine holds on the idle frame."""
        meta = export_experience.choice_metadata(
            _confirm_choice_block(None), "02", 310, 30)
        fsm = meta["interaction"]["interaction_fsm"]
        self.assertNotIn("detect.mp3", json.dumps(meta))
        self.assertEqual(fsm["states"]["left_selected"]["segment"], "idle_loop")
        self.assertTrue(fsm["states"]["left_selected"]["loop"])
        t = fsm["transitions"]
        self.assertIn({"from": "left_selected", "on": "point_right",
                       "to": "right_switch_hold"}, t)
        self.assertIn({"from": "left_selected", "on": "voice_go",
                       "to": "__advance__"}, t)

    def test_shot_map_targets_original_shots(self):
        """shot_map.json maps ORIGINAL master-timeline shot numbers
        (copy_frames.SHOT_FRAMES) into the exported tree. The fixture's media
        covers only the first 40s of the master, so exactly original shot 01
        (t=0.0) resolves — to the Intro block's first frame."""
        with open(self.out / "shot_map.json", encoding="utf-8") as f:
            m = json.load(f)
        shots = m["shots"]
        self.assertEqual(shots["01"]["shot"], "01")
        self.assertEqual(shots["01"]["frame"], 1)
        self.assertEqual(shots["01"]["act"], "act_00_prologue")
        self.assertEqual(len(shots), 1)   # nothing past the fixture's media maps

    def test_playthrough_fsm_timeout_covers_content(self):
        """The runtime HOLD timeout reads the TOP-LEVEL fallback (shot.fallback),
        not the copy nested in interaction_fsm. Without one, the standard
        profile's 30s auto_advance cuts any play-through FSM longer than 30s
        mid-shot (the stretch-block skip-to-choice bug). Every play-through FSM
        shot must carry a top-level fallback covering its full FSM content,
        with no reprompts."""
        checked = 0
        for s in self.shots:
            fsm = (s.interaction or {}).get("interaction_fsm")
            if not fsm or not s.segments or fsm["initial"] == "waiting":
                continue  # choice forks manage their own waiting-hold timeout
            content_s = sum(
                b - a + 1
                for a, b in (s.segments[st["segment"]]
                             for st in fsm["states"].values())
            ) / s.fps
            self.assertGreaterEqual(s.fallback["timeout_s"], content_s)
            self.assertEqual(s.fallback["reprompt_s"], [])
            self.assertEqual(s.fallback["on_timeout"], "auto_advance")
            checked += 1
        self.assertGreater(checked, 0, "fixture must contain a play-through FSM shot")

    # ── the live-tree invariants, applied to the generated tree ────────────

    def test_fsm_states_reference_defined_segments(self):
        for s in self.shots:
            fsm = (s.interaction or {}).get("interaction_fsm")
            if not fsm or not s.segments:
                continue
            for state_id, state in fsm.get("states", {}).items():
                seg = state.get("segment")
                if seg is not None:
                    self.assertIn(seg, s.segments, f"shot {s.shot} state {state_id!r}")

    def test_fsm_transitions_reference_defined_states(self):
        for s in self.shots:
            fsm = (s.interaction or {}).get("interaction_fsm")
            if not fsm:
                continue
            states = set(fsm.get("states", {}))
            self.assertIn(fsm.get("initial"), states, f"shot {s.shot}")
            for t in fsm.get("transitions", []):
                self.assertIn(t.get("from"), states, f"shot {s.shot}: {t}")
                self.assertTrue(t.get("to") == "__advance__" or t.get("to") in states,
                                f"shot {s.shot}: {t}")

    def test_segment_ranges_are_ordered_pairs_and_contiguous(self):
        for s in self.shots:
            for name, rng in (s.segments or {}).items():
                self.assertEqual(len(rng), 2, f"shot {s.shot} segment {name}")
                self.assertLessEqual(rng[0], rng[1], f"shot {s.shot} segment {name}")
                self.assertGreaterEqual(rng[0], 1, f"shot {s.shot} segment {name}")

    def test_play_if_points_at_real_fork_with_interaction(self):
        for s in self.shots:
            if not s.play_if:
                continue
            src = self.by_id.get(s.play_if["shot"])
            self.assertIsNotNone(src, f"shot {s.shot}")
            self.assertIn(s.play_if["branch"], ("left", "right"))
            self.assertIsNotNone(src.interaction, f"fork {src.shot} records no choice")

    def test_detector_types_are_registered(self):
        for s in self.shots:
            inter = s.interaction or {}
            candidates = []
            if inter.get("type"):
                candidates.append(inter["type"])
            for state in (inter.get("interaction_fsm", {}) or {}).get("states", {}).values():
                oi = state.get("oi") or {}
                if oi.get("type"):
                    candidates.append(oi["type"])
            for t in candidates:
                self.assertIn(t, REGISTRY, f"shot {s.shot}: {t!r} not a real detector")

    # (test_scenes_import_round_trips retired Aug 2026 with scenes_to_builder.py
    #  and the live scenes/ tree — the .bhrx is the single source of truth.)

    # ── layered stem audio (audio_events) ──────────────────────────────────

    def test_block_audio_exports_as_audio_events(self):
        intro = self.shots[0]
        events = intro.audio_events
        self.assertEqual(len(events), 2)
        bed = next(e for e in events if e["role"] == "ambience")
        sfx = next(e for e in events if e["role"] == "sfx")
        self.assertTrue(bed["sustain"])
        self.assertFalse(bed["continues"])
        self.assertEqual(bed["fade_in_ms"], 300)
        self.assertAlmostEqual(bed["gain"], 0.4)
        self.assertFalse(sfx["sustain"])
        self.assertAlmostEqual(sfx["at_s"], 2.5)
        # loader resolved both against the exported _audio pool
        for e in events:
            self.assertIsNotNone(e["path"], e["file"])
            self.assertTrue(e["path"].exists())

    def test_audio_pool_holds_referenced_files(self):
        pool = self.out / "_audio"
        self.assertTrue((pool / "night_bed.mp3").exists())
        self.assertTrue((pool / "crinkle.mp3").exists())

    def test_bed_continuity_pair_is_consistent(self):
        intro, choice = self.shots[0], self.shots[1]
        bed = next(e for e in intro.audio_events if e["sustain"])
        cont = next(e for e in choice.audio_events if e["continues"])
        self.assertEqual(bed["file"], cont["file"],
                         "handover match key is the shared file name")

    def test_audio_events_within_shot_span(self):
        for s in self.shots:
            length = None
            if s.segments:
                length = max(rng[1] for rng in s.segments.values()) / (s.fps or 30)
            for e in s.audio_events:
                self.assertGreaterEqual(e["at_s"], 0.0, f"shot {s.shot}")
                if length is not None:
                    self.assertLessEqual(e["at_s"], length + 0.001,
                                         f"shot {s.shot}: {e['file']}")

    def test_offset_bed_chain_shares_one_render(self):
        """Sustain beds linked by `continues` must ship as ONE render (the
        chain head's offset, running to the end of the file) — the mixer's
        handover matches on the file name, so per-block trims would restart
        the bed at every block boundary."""
        import unittest.mock as mock

        project = {
            "sounds": [{"id": "s_bed", "name": "chain_bed.mp3"}],
            "start": "b1",
            "blocks": [
                {"id": "b1", "type": "choice", "range_s": [100.0, 110.0],
                 "audio": [{"id": "a1", "sound": "s_bed", "role": "music",
                            "at_s": 0.0, "duration_s": 10.0,
                            "source_offset_s": 43.2, "gain": 0.1,
                            "fade_in_ms": 1300, "fade_out_ms": 1200,
                            "sustain": True, "continues": False}],
                 "branches": [{"id": "br", "window": "w", "to": "b2"}]},
                {"id": "b2", "type": "playback", "range_s": [131.0, 138.0],
                 "audio": [{"id": "a2", "sound": "s_bed", "role": "music",
                            "at_s": 0.0, "duration_s": 7.0,
                            "source_offset_s": 74.2, "gain": 0.1,
                            "fade_in_ms": 0, "fade_out_ms": 1200,
                            "sustain": True, "continues": True}]},
                {"id": "b3", "type": "playback", "range_s": [200.0, 208.0],
                 "audio": [{"id": "a3", "sound": "s_bed", "role": "music",
                            "at_s": 0.0, "duration_s": 8.0,
                            "source_offset_s": 5.0, "gain": 0.1,
                            "fade_in_ms": 0, "fade_out_ms": 0,
                            "sustain": True, "continues": False}]},
            ],
            "edges": [],
        }

        plans = export_experience.plan_bed_chains(project)
        self.assertIs(plans[("b1", "a1")], plans[("b2", "a2")],
                      "continues piece must inherit the head's plan "
                      "(choice branch is a direct edge)")
        self.assertAlmostEqual(plans[("b1", "a1")]["offset"], 43.2)
        self.assertIsNot(plans[("b3", "a3")], plans[("b1", "a1")],
                         "an unlinked bed is its own chain")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "chain_bed.mp3").write_bytes(b"bed")
            pool = tmp / "_audio"
            pool.mkdir()
            sounds = {"s_bed": {"id": "s_bed", "name": "chain_bed.mp3"}}
            rendered = {}
            fake_render = mock.Mock(
                side_effect=lambda ff, src, off, dur, spd, out:
                    (out.write_bytes(b"r") or True))
            with mock.patch.object(export_experience, "probe_duration",
                                   return_value=120.0), \
                 mock.patch.object(export_experience, "render_trim",
                                   fake_render):
                evs = {}
                for b in project["blocks"]:
                    evs[b["id"]] = export_experience.block_audio_events(
                        b, sounds, tmp / "p.bhrx.json", pool, "ffmpeg",
                        rendered, plans)

            e1, e2, e3 = evs["b1"][0], evs["b2"][0], evs["b3"][0]
            self.assertEqual(e1["file"], e2["file"],
                             "chain pieces must share one render name")
            self.assertNotEqual(e1["file"], "chain_bed.mp3",
                                "an offset head must be a trimmed render")
            self.assertNotEqual(e3["file"], e1["file"],
                                "separate chains render separately")
            # one render per chain, cut head-offset -> end of file
            chain_call = next(c for c in fake_render.call_args_list
                              if c.args[2] == 43.2)
            self.assertAlmostEqual(chain_call.args[3], 120.0 - 43.2)
            # offsets are rebased onto the shared render
            self.assertAlmostEqual(e1["source_offset_s"], 0.0)
            self.assertAlmostEqual(e2["source_offset_s"], 31.0)

    def test_editor_detector_registry_matches_runtime(self):
        """tools/experience_builder/js/detectors.js is hand-synced from the
        runtime REGISTRY — the sync must hold in BOTH directions: no unknown
        types offered, and no runtime detector silently missing from the tool."""
        js = (ROOT / "tools" / "experience_builder" / "js" / "detectors.js").read_text(
            encoding="utf-8")
        import re
        offered = set(re.findall(r'type:\s*"([a-z_]+)"', js))
        offered.discard("voice")
        unknown = offered - set(REGISTRY)
        self.assertFalse(unknown, f"detectors.js offers unknown types: {sorted(unknown)}")
        missing = set(REGISTRY) - offered
        self.assertFalse(missing,
                         f"detectors.js is missing runtime detectors: {sorted(missing)}")


class TestChoicePickAudio(unittest.TestCase):
    """A choice block's authored pick/switch sound (Builder inspector ->
    Pick / switch audio) exports as on_enter_audio with delay / source offset
    / duration, applied at runtime."""

    SOUNDS = {"s3": {"id": "s3", "name": "crinkle.mp3"}}

    def _choice(self, **audio):
        block = _retry_project()["blocks"][1]
        if audio:
            block["choice_audio"] = audio
        return block

    def test_spec_defaults_are_omitted(self):
        spec = export_experience.choice_audio_spec(
            self._choice(sound="s3"), self.SOUNDS)
        self.assertEqual(spec, {"file": "crinkle.mp3"})   # no noisy zero keys

    def test_spec_carries_timing(self):
        spec = export_experience.choice_audio_spec(
            self._choice(sound="s3", delay_s=0.5, source_offset_s=2.25,
                         duration_s=3.0, gain=0.8), self.SOUNDS)
        self.assertEqual(spec, {"file": "crinkle.mp3", "delay_s": 0.5,
                                "source_offset_s": 2.25, "duration_s": 3.0,
                                "gain": 0.8})

    def test_unknown_sound_is_ignored(self):
        self.assertIsNone(export_experience.choice_audio_spec(
            self._choice(sound="nope"), self.SOUNDS))
        self.assertIsNone(export_experience.choice_audio_spec(
            self._choice(), self.SOUNDS))

    def test_states_take_the_clip_but_the_retry_buzzer_survives(self):
        block = self._choice(sound="s3", delay_s=0.4)
        meta = export_experience.choice_metadata(block, "02", 330, 30, self.SOUNDS)
        states = meta["interaction"]["interaction_fsm"]["states"]
        # the correct-side confirm is a detection -> it takes the clip
        confirm = states["confirm_right"]
        self.assertEqual(confirm["on_enter_audio"],
                         {"file": "crinkle.mp3", "delay_s": 0.4})
        self.assertNotIn("on_enter_sfx", confirm)
        # waiting is idle, never a detection
        self.assertNotIn("on_enter_audio", states["waiting"])
        # a wrong-way redirect says "not that way", not "picked" — it keeps its
        # own audio (here the animated redirect's folded clips) and never takes
        # the pick sound
        self.assertNotIn("on_enter_audio", states["wrong_path"])

    def test_sound_only_retry_keeps_its_buzzer(self):
        block = self._choice(sound="s3")
        del block["hold_segments"]          # sound-only wrong-way bounce
        meta = export_experience.choice_metadata(block, "02", 330, 30, self.SOUNDS)
        wrong = meta["interaction"]["interaction_fsm"]["states"]["wrong_path"]
        self.assertEqual(wrong["on_enter_sfx"], export_experience.RETRY_WRONG_SFX)
        self.assertNotIn("on_enter_audio", wrong)

    def test_hold_model_switch_states_replace_the_default_click(self):
        """In the voice-confirmed hold model the pick AND switch states carry
        CHOICE_HOLD_SFX by default; the authored clip replaces it so the beat
        doesn't double up."""
        block = _confirm_choice_block(None)
        block["choice_audio"] = {"sound": "s3", "duration_s": 1.5}
        meta = export_experience.choice_metadata(block, "02", 900, 30, self.SOUNDS)
        states = meta["interaction"]["interaction_fsm"]["states"]
        for sid in ("left_selected", "right_selected"):
            self.assertNotIn("on_enter_sfx", states[sid])
            self.assertEqual(states[sid]["on_enter_audio"],
                             {"file": "crinkle.mp3", "duration_s": 1.5})
        self.assertFalse(any(s.get("on_enter_sfx") == export_experience.CHOICE_HOLD_SFX
                             for s in states.values()))

    def test_clip_file_is_collected_for_shipping(self):
        """on_enter_audio files must be copied into the shot's audio/ dir like
        any other FSM sound, or the runtime can't resolve them."""
        meta = export_experience.choice_metadata(
            self._choice(sound="s3"), "02", 330, 30, self.SOUNDS)
        self.assertIn("crinkle.mp3", export_experience.collect_sfx_names(meta))

    def test_no_audio_authored_keeps_todays_behaviour(self):
        block = self._choice()
        del block["hold_segments"]
        meta = export_experience.choice_metadata(block, "02", 330, 30, {})
        states = meta["interaction"]["interaction_fsm"]["states"]
        self.assertFalse(any("on_enter_audio" in s for s in states.values()))
        self.assertEqual(states["wrong_path"]["on_enter_sfx"],
                         export_experience.RETRY_WRONG_SFX)


if __name__ == "__main__":
    unittest.main()
