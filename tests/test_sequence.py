"""Shot sequence / metadata invariants against the REAL scenes/ tree and
config.json — catches wiring mistakes (bad segment refs, dangling FSM
transitions, broken play_if branches) before they reach a playtest."""

import json
import unittest
from pathlib import Path

from engines.sequence_loader import load_sequence

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "scenes"


def _load_config():
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


class TestConfig(unittest.TestCase):
    def test_config_parses_and_has_core_keys(self):
        cfg = _load_config()
        for key in ("timing_defaults", "detection_thresholds", "voice"):
            self.assertIn(key, cfg)
        self.assertIn("[unk]", cfg["voice"]["grammar"])

    def test_depth_band_sane(self):
        d = _load_config().get("depth", {})
        self.assertLess(d.get("player_min_mm", 500), d.get("player_max_mm", 3200))


class TestSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _load_config()
        cls.shots = load_sequence(SCENES, cls.cfg)
        cls.by_id = {s.shot: s for s in cls.shots}

    def test_sequence_loads_and_is_ordered(self):
        self.assertGreater(len(self.shots), 0)
        self.assertEqual(len(self.by_id), len(self.shots), "duplicate shot ids")

    def test_kinds_are_valid(self):
        for s in self.shots:
            self.assertIn(s.kind, ("playback", "interactive"), f"shot {s.shot}")

    def test_fsm_states_reference_defined_segments(self):
        for s in self.shots:
            fsm = (s.interaction or {}).get("interaction_fsm")
            if not fsm or not s.segments:
                continue
            for state_id, state in fsm.get("states", {}).items():
                seg = state.get("segment")
                if seg is not None:
                    self.assertIn(seg, s.segments,
                                  f"shot {s.shot} state {state_id!r}: "
                                  f"segment {seg!r} not in segments")

    def test_fsm_transitions_reference_defined_states(self):
        for s in self.shots:
            fsm = (s.interaction or {}).get("interaction_fsm")
            if not fsm:
                continue
            states = set(fsm.get("states", {}))
            initial = fsm.get("initial")
            if initial is not None:
                self.assertIn(initial, states, f"shot {s.shot}: bad initial")
            for t in fsm.get("transitions", []):
                self.assertIn(t.get("from"), states,
                              f"shot {s.shot}: transition from unknown state {t}")
                to = t.get("to")
                self.assertTrue(to == "__advance__" or to in states,
                                f"shot {s.shot}: transition to unknown state {t}")

    def test_segment_ranges_are_ordered_pairs(self):
        for s in self.shots:
            for name, rng in (s.segments or {}).items():
                self.assertEqual(len(rng), 2, f"shot {s.shot} segment {name}")
                self.assertLessEqual(rng[0], rng[1],
                                     f"shot {s.shot} segment {name}: {rng}")

    def test_play_if_branches_point_at_real_forks(self):
        for s in self.shots:
            if not s.play_if:
                continue
            src = self.by_id.get(s.play_if.get("shot"))
            self.assertIsNotNone(src, f"shot {s.shot}: play_if source missing")
            self.assertIn(s.play_if.get("branch"), ("left", "right"),
                          f"shot {s.shot}: bad play_if branch")
            self.assertIsNotNone(src.interaction,
                                 f"shot {s.shot}: play_if source {src.shot} "
                                 f"has no interaction to record a choice")

    def test_reuse_targets_exist_or_shot_is_pending(self):
        """A reuse_of pointing at a shot cut from the sequence is tolerated only
        while the shot is an asset-less placeholder (e.g. shot 65 -> tracker
        shot 68, superseded by the act-11 collapse into shot 66). If assets ever
        resolve through a dangling reuse, that's a wiring bug."""
        for s in self.shots:
            if s.reuse_of and not s.reuse_self and s.reuse_of not in self.by_id:
                self.assertTrue(s.assets_pending,
                                f"shot {s.shot}: reuse_of {s.reuse_of} not in "
                                f"sequence but assets resolved")

    def test_detector_types_are_registered(self):
        from engines.detectors import REGISTRY
        known_non_detector = {"voice"}   # voice steps route to the voice engine
        for s in self.shots:
            inter = s.interaction or {}
            candidates = []
            if inter.get("type"):
                candidates.append(inter["type"])
            for step in inter.get("chain", []) or []:
                if step.get("type"):
                    candidates.append(step["type"])
            for state in (inter.get("interaction_fsm", {}) or {}).get("states", {}).values():
                oi = state.get("oi") or {}
                if oi.get("type"):
                    candidates.append(oi["type"])
            for t in candidates:
                if t in known_non_detector:
                    continue
                self.assertIn(t, REGISTRY,
                              f"shot {s.shot}: detector type {t!r} not registered")


if __name__ == "__main__":
    unittest.main()
