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
        "sounds": [{"id": "s1", "name": "detect.mp3"}],
        "settings": {"global_detect_sound": "s1"},
        "start": "b_intro",
        "blocks": [
            {"id": "b_intro", "type": "playback", "name": "Intro", "media": "m1",
             "range_s": [0.0, 6.0], "pos": [0, 0],
             "windows": [
                 {"id": "w_reach", "label": "Reach flask", "detector": "forward_reach",
                  "params": {"area_growth_threshold": 0.3},
                  "region": {"shape": "rect", "x": 0.3, "y": 0.1, "w": 0.3, "h": 0.3},
                  "appears_s": 2.0, "duration_s": 2.5}]},
            {"id": "b_choice", "type": "choice", "name": "Choose Direction", "media": "m1",
             "range_s": [6.0, 12.0], "hold": "pause_end", "pos": [0, 0],
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


class TestExperienceExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(cls.tmp.name)
        cls.project_path = tmp_path / "fixture.bhrx.json"
        with open(cls.project_path, "w", encoding="utf-8") as f:
            json.dump(fixture_project(), f)
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
                         f"shot_{s.shot}" / "metadata.json")
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
        self.assertEqual(inter["params"]["region_rect"]["x"], 0.3)

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

    def test_scenes_import_round_trips(self):
        """scripts/scenes_to_builder.py converts the live scenes/ tree into a
        .bhrx project; exporting that project back must satisfy the wiring
        invariants and preserve the three forks (shots 09 / 37 / 50)."""
        import scenes_to_builder
        import sys as _sys
        with tempfile.TemporaryDirectory() as td:
            proj_path = Path(td) / "bhr.bhrx.json"
            argv = _sys.argv
            _sys.argv = ["scenes_to_builder.py", "--out", str(proj_path)]
            try:
                scenes_to_builder.main()
            finally:
                _sys.argv = argv

            with open(proj_path, encoding="utf-8") as f:
                proj = json.load(f)
            blocks = {b["id"]: b for b in proj["blocks"]}
            self.assertEqual(sum(1 for b in proj["blocks"] if b["type"] == "choice"), 3)
            # every edge / branch endpoint resolves
            for e in proj["edges"]:
                self.assertIn(e["from"], blocks)
                self.assertIn(e["to"], blocks)
            for b in proj["blocks"]:
                for br in b.get("branches") or []:
                    self.assertIn(br["to"], blocks)

            out = Path(td) / "scenes_generated"
            export_experience.warnings.clear()
            export_experience.export(proj_path, out, do_frames=False,
                                     video_override=None, sound_override=None)
            shots = load_sequence(out, {"fps": 30})
            by_id = {s.shot: s for s in shots}
            forks = {s.play_if["shot"] for s in shots if s.play_if}
            self.assertEqual(len(forks), 3, "three forks must survive round-trip")
            for s in shots:
                if s.play_if:
                    self.assertIsNotNone(by_id[s.play_if["shot"]].interaction)

    def test_editor_detector_registry_matches_runtime(self):
        """tools/experience_builder/js/detectors.js is hand-synced from the
        runtime REGISTRY — every non-voice type it offers must exist."""
        js = (ROOT / "tools" / "experience_builder" / "js" / "detectors.js").read_text(
            encoding="utf-8")
        import re
        offered = set(re.findall(r'type:\s*"([a-z_]+)"', js))
        offered.discard("voice")
        unknown = offered - set(REGISTRY)
        self.assertFalse(unknown, f"detectors.js offers unknown types: {sorted(unknown)}")


if __name__ == "__main__":
    unittest.main()
