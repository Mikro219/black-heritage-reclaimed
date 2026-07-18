"""Contract for scripts/capcut_audio.py — the CapCut draft → audio_events
importer: material/name matching, VO skipping, role-by-track mapping, bed-run
merging, boundary splitting with corrected source offsets + continuity flags,
gap dropping, and idempotent --apply-scenes writes."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import capcut_audio as ca  # noqa: E402

US = 1_000_000


def draft(segments_by_track, audios, fades=None):
    """Minimal CapCut draft JSON: segments_by_track is a list of lists of
    (material_id, t0_s, dur_s, src_off_s, volume, fade_ref)."""
    tracks = []
    for segs in segments_by_track:
        tracks.append({"type": "audio", "segments": [
            {"material_id": mid,
             "target_timerange": {"start": int(t0 * US), "duration": int(dur * US)},
             "source_timerange": {"start": int(off * US), "duration": int(dur * US)},
             "volume": vol, "speed": 1.0,
             "extra_material_refs": [fr] if fr else []}
            for mid, t0, dur, off, vol, fr in segs]})
    return {"tracks": tracks,
            "materials": {"audios": audios, "audio_fades": fades or []}}


def write_draft(tmp, doc):
    p = Path(tmp) / "draft.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return p


def audio_dir(tmp, names):
    d = Path(tmp) / "stems"
    d.mkdir(exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"mp3")
    return d


class TestLoadPlacements(unittest.TestCase):
    def test_matching_vo_skip_and_fades(self):
        audios = [
            {"id": "a1", "name": "night.mp3", "type": "extract_music",
             "duration": 60 * US},
            {"id": "a2", "name": "SC20_V1", "type": "video_original_sound",
             "path": "C:/x/SC20_V1.mov", "duration": 30 * US},
            {"id": "a3", "name": "Compound clip28", "type": "music",
             "duration": 10 * US},
            {"id": "a4", "name": "click", "type": "extract_music",   # no extension
             "duration": 2 * US},
        ]
        fades = [{"id": "f1", "fade_in_duration": 333_000,
                  "fade_out_duration": 600_000}]
        doc = draft([[("a1", 0, 10, 0.5, 0.4, "f1"),
                      ("a2", 5, 5, 0, 1.0, None),
                      ("a3", 8, 2, 0, 1.0, None),
                      ("a4", 9, 1, 0, 1.0, None)]], audios, fades)
        with tempfile.TemporaryDirectory() as tmp:
            d = audio_dir(tmp, ["night.mp3", "click.mp3"])
            placements, stats, unmatched, _ = ca.load_placements(
                write_draft(tmp, doc), d, include_vo=False)
        self.assertEqual(stats["placed"], 2)
        self.assertEqual(stats["vo_render"], 1)
        self.assertEqual(stats["unmatched"], 1)
        self.assertIn("Compound clip28", unmatched)
        night = next(p for p in placements if p["name"] == "night.mp3")
        self.assertEqual(night["fade_in_ms"], 333)
        self.assertEqual(night["fade_out_ms"], 600)
        self.assertAlmostEqual(night["src_off"], 0.5)
        self.assertAlmostEqual(night["gain"], 0.4)

    def test_role_by_track_index(self):
        audios = [{"id": "a1", "name": "x.mp3", "type": "extract_music",
                   "duration": 60 * US}]
        seg = [("a1", 0, 5, 0, 1.0, None)]
        doc = draft([seg, [], [], [], seg, [], seg, []], audios)
        with tempfile.TemporaryDirectory() as tmp:
            d = audio_dir(tmp, ["x.mp3"])
            placements, _, _, _ = ca.load_placements(write_draft(tmp, doc), d)
        roles = sorted(p["role"] for p in placements)
        self.assertEqual(roles, ["ambience", "music", "sfx"])


class TestVoImport(unittest.TestCase):
    """VO (video_original_sound) placements resolve to the extracted comp
    audio, gated by the 0.5s duration tolerance; unmapped/missing/re-cut
    sources are skipped with a recorded reason."""

    def _vo_draft(self, name, path, dur_s, t0=5.0, seg_dur=5.0, src_off=2.0,
                  gain=0.55):
        audios = [{"id": "v1", "name": name, "type": "video_original_sound",
                   "path": path, "duration": int(dur_s * US)}]
        return draft([[("v1", t0, seg_dur, src_off, gain, None)]], audios)

    def _load(self, doc, vo_dir, probes):
        """Run load_placements with VO_DIR + probe_duration patched."""
        orig_dir, orig_probe = ca.VO_DIR, ca.probe_duration
        ca.VO_DIR = vo_dir
        ca.probe_duration = lambda p: probes.get(p.name)
        try:
            with tempfile.TemporaryDirectory() as tmp2:
                d = audio_dir(tmp2, [])
                return ca.load_placements(write_draft(tmp2, doc), d)
        finally:
            ca.VO_DIR, ca.probe_duration = orig_dir, orig_probe

    def test_scene_number_maps_to_voice_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            vo = Path(tmp)
            (vo / "bhr_scene_20.mp3").write_bytes(b"mp3")
            doc = self._vo_draft("SC20_V1", "C:/x/SC20_V1.mov", 30.0)
            placements, stats, _, skips = self._load(
                doc, vo, {"bhr_scene_20.mp3": 30.02})
        self.assertEqual(stats["vo_placed"], 1)
        self.assertEqual(skips, [])
        p = placements[0]
        self.assertEqual(p["role"], "vo")
        self.assertEqual(p["name"], "bhr_scene_20.mp3")
        self.assertAlmostEqual(p["src_off"], 2.0)
        self.assertAlmostEqual(p["gain"], 0.55)
        self.assertAlmostEqual(p["natural_s"], 30.02)   # trusts the local file

    def test_recut_comp_outside_tolerance_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            vo = Path(tmp)
            (vo / "bhr_scene_37.mp3").write_bytes(b"mp3")
            doc = self._vo_draft("SC_37_final", "C:/x/SC_37_final.mp4", 30.0)
            placements, stats, _, skips = self._load(
                doc, vo, {"bhr_scene_37.mp3": 12.63})
        self.assertEqual(stats["vo_placed"], 0)
        self.assertEqual(stats["vo_skipped"], 1)
        self.assertEqual(placements, [])
        self.assertIn("re-cut", skips[0]["reason"])

    def test_missing_source_skipped_with_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = self._vo_draft("tc-draft- new", "C:/x/tc-draft- new.mp4", 12.0)
            _, stats, _, skips = self._load(doc, Path(tmp), {})
        self.assertEqual(stats["vo_skipped"], 1)
        self.assertEqual(skips[0]["reason"], "no local voice-line file")
        self.assertAlmostEqual(skips[0]["t0"], 5.0)

    def test_override_maps_fork_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            vo = Path(tmp)
            (vo / "bhr_scene_09-A.mp3").write_bytes(b"mp3")
            orig = dict(ca.VO_OVERRIDES)
            ca.VO_OVERRIDES = {"sc09_v1-optiona.mov": vo / "bhr_scene_09-A.mp3"}
            try:
                doc = self._vo_draft("SC09_V1-OptionA",
                                     "C:/x/SC09_V1-OptionA.mov", 21.83)
                placements, stats, _, _ = self._load(
                    doc, vo, {"bhr_scene_09-A.mp3": 21.85})
            finally:
                ca.VO_OVERRIDES = orig
        self.assertEqual(stats["vo_placed"], 1)
        self.assertEqual(placements[0]["name"], "bhr_scene_09-A.mp3")

    def test_no_vo_flag_restores_stems_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            vo = Path(tmp)
            (vo / "bhr_scene_20.mp3").write_bytes(b"mp3")
            doc = self._vo_draft("SC20_V1", "C:/x/SC20_V1.mov", 30.0)
            orig_dir, orig_probe = ca.VO_DIR, ca.probe_duration
            ca.VO_DIR = vo
            ca.probe_duration = lambda p: 30.0
            try:
                with tempfile.TemporaryDirectory() as tmp2:
                    d = audio_dir(tmp2, [])
                    placements, stats, _, skips = ca.load_placements(
                        write_draft(tmp2, doc), d, include_vo=False)
            finally:
                ca.VO_DIR, ca.probe_duration = orig_dir, orig_probe
        self.assertEqual(placements, [])
        self.assertEqual(stats["vo_render"], 1)
        self.assertEqual(stats["vo_placed"], 0)
        self.assertEqual(skips, [])

    def test_vo_bypasses_bed_merge_and_splits_with_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "bhr_scene_20.mp3"
            src.write_bytes(b"mp3")
            p1 = {"name": "bhr_scene_20.mp3", "path": src,
                  "role": "vo", "track": 0, "t0": 8.0, "dur": 4.0,
                  "src_off": 0.0, "speed": 1.0, "gain": 0.55,
                  "fade_in_ms": 0, "fade_out_ms": 0, "natural_s": 30.0}
            p2 = dict(p1, t0=12.2)  # back-to-back — a bed would merge here
            merged = ca.merge_bed_runs([p1, p2])
            self.assertEqual(len(merged), 2)

            spans = [("01", 0.0, 10.0), ("02", 10.0, 20.0)]
            by_shot, dropped = ca.cut_against_spans([p1], spans)
            self.assertEqual(dropped, 0)
            self.assertFalse(by_shot["01"][0]["continues"])
            self.assertTrue(by_shot["02"][0]["continues"])
            # role survives into the runtime event, sustain stays off
            events = ca.pieces_to_events(by_shot["02"], Path(tmp) / "pool",
                                         None, False, {})
            self.assertEqual(events[0]["role"], "vo")
            self.assertFalse(events[0]["sustain"])
            self.assertTrue(events[0]["continues"])

    def test_video_container_source_always_trims(self):
        p = {"name": "IMG_5032.mov", "path": Path("IMG_5032.mov"),
             "role": "vo", "track": 1, "t0": 0.0, "dur": 18.0, "src_off": 0.0,
             "speed": 1.0, "gain": 1.0, "fade_in_ms": 0, "fade_out_ms": 0,
             "natural_s": 18.05}
        self.assertTrue(ca.needs_trim(p))   # pygame cannot play .mov


class TestMergeBedRuns(unittest.TestCase):
    def _p(self, name, role, track, t0, dur, gain=1.0):
        return {"name": name, "path": Path(name), "role": role, "track": track,
                "t0": t0, "dur": dur, "src_off": 0.0, "speed": 1.0,
                "gain": gain, "fade_in_ms": 0, "fade_out_ms": 100,
                "natural_s": dur}

    def test_back_to_back_beds_merge(self):
        a = self._p("night.mp3", "ambience", 4, 0.0, 10.0)
        b = self._p("night.mp3", "ambience", 4, 10.2, 10.0)   # 0.2s gap
        c = self._p("night.mp3", "ambience", 4, 40.0, 10.0)   # far away
        merged = ca.merge_bed_runs([a, b, c])
        runs = [p for p in merged if p["name"] == "night.mp3"]
        self.assertEqual(len(runs), 2)
        self.assertAlmostEqual(runs[0]["dur"], 20.2)

    def test_sfx_repeats_not_merged(self):
        a = self._p("crinkle.mp3", "sfx", 0, 0.0, 1.0)
        b = self._p("crinkle.mp3", "sfx", 0, 1.1, 1.0)
        self.assertEqual(len(ca.merge_bed_runs([a, b])), 2)


class TestCutAgainstSpans(unittest.TestCase):
    def _p(self, t0, dur, src_off=0.0, role="ambience"):
        return {"name": "bed.mp3", "path": Path("bed.mp3"), "role": role,
                "track": 4, "t0": t0, "dur": dur, "src_off": src_off,
                "speed": 1.0, "gain": 1.0, "fade_in_ms": 0, "fade_out_ms": 0,
                "natural_s": 120.0}

    SPANS = [("01", 0.0, 10.0), ("02", 10.0, 20.0), ("03", 25.0, 30.0)]

    def test_split_with_offsets_and_continuity(self):
        by_span, dropped = ca.cut_against_spans([self._p(5.0, 12.0, src_off=2.0)],
                                                self.SPANS)
        self.assertEqual(dropped, 0)
        p1 = by_span["01"][0]
        p2 = by_span["02"][0]
        self.assertAlmostEqual(p1["at_s"], 5.0)
        self.assertAlmostEqual(p1["piece_dur"], 5.0)
        self.assertAlmostEqual(p1["piece_src_off"], 2.0)
        self.assertFalse(p1["continues"])
        self.assertAlmostEqual(p2["at_s"], 0.0)
        self.assertAlmostEqual(p2["piece_dur"], 7.0)
        self.assertAlmostEqual(p2["piece_src_off"], 7.0)   # 2.0 + 5s consumed
        self.assertTrue(p2["continues"])

    def test_placement_in_gap_dropped(self):
        by_span, dropped = ca.cut_against_spans([self._p(21.0, 3.0)], self.SPANS)
        self.assertEqual(dropped, 1)
        self.assertEqual(sum(len(v) for v in by_span.values()), 0)

    def test_span_bridging_gap_continues_on_far_side(self):
        by_span, _ = ca.cut_against_spans([self._p(15.0, 15.0)], self.SPANS)
        self.assertTrue(by_span["03"][0]["continues"])


class TestApplyScenes(unittest.TestCase):
    def _mini_tree(self, tmp):
        root = Path(tmp) / "scenes"
        for act, shot in [("act_01_test", "01"), ("act_01_test", "02")]:
            d = root / act / f"shot_{shot}"
            d.mkdir(parents=True)
            with open(d / "metadata.json", "w", encoding="utf-8") as f:
                json.dump({"shot": shot, "kind": "playback"}, f)
        return root

    def test_apply_is_idempotent_and_additive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._mini_tree(tmp)
            src = Path(tmp) / "night.mp3"
            src.write_bytes(b"mp3data")
            piece = {"name": "night.mp3", "path": src, "role": "ambience",
                     "track": 4, "t0": 0.0, "dur": 5.0, "src_off": 0.0,
                     "speed": 1.0, "gain": 0.4, "fade_in_ms": 0,
                     "fade_out_ms": 300, "natural_s": 5.0,
                     "span": "01", "at_s": 0.0, "piece_dur": 5.0,
                     "piece_src_off": 0.0, "continues": False}
            by_shot = {"01": [piece]}

            orig_frames = ca.SHOT_FRAMES
            ca.SHOT_FRAMES = [("act_01_test", "01", 1, 150),
                              ("act_01_test", "02", 151, 300)]
            try:
                ca.apply_scenes(by_shot, root, None, False)
                meta1 = json.loads((root / "act_01_test" / "shot_01" /
                                    "metadata.json").read_text())
                ca.apply_scenes(by_shot, root, None, False)
                meta2 = json.loads((root / "act_01_test" / "shot_01" /
                                    "metadata.json").read_text())
            finally:
                ca.SHOT_FRAMES = orig_frames

            self.assertEqual(meta1, meta2)                       # idempotent
            self.assertEqual(meta1["kind"], "playback")          # additive
            self.assertEqual(len(meta1["audio_events"]), 1)
            e = meta1["audio_events"][0]
            self.assertEqual(e["file"], "night.mp3")
            self.assertTrue(e["sustain"])
            self.assertTrue((root / "_audio" / "night.mp3").exists())
            # shot without events: key absent
            meta_02 = json.loads((root / "act_01_test" / "shot_02" /
                                  "metadata.json").read_text())
            self.assertNotIn("audio_events", meta_02)


class TestRealDraftSmoke(unittest.TestCase):
    """The actual delivered draft parses and every placement lands in a shot."""

    DRAFT = ROOT / "assets" / "audio" / "draft_content.json"
    STEMS = ROOT / "assets" / "audio" / "stems"

    def test_real_draft_full_coverage(self):
        if not (self.DRAFT.exists() and self.STEMS.is_dir()):
            self.skipTest("delivered audio assets not present")
        placements, stats, _, vo_skips = ca.load_placements(self.DRAFT,
                                                            self.STEMS)
        self.assertGreater(stats["placed"] - stats["vo_placed"], 300)
        # every draft segment is accounted for: stems placed + VO segments
        # (placed or skipped) + unmatched compound clips
        self.assertEqual((stats["placed"] - stats["vo_placed"])
                         + stats["vo_render"] + stats["unmatched"], 529)
        self.assertEqual(stats["vo_placed"] + stats["vo_skipped"],
                         stats["vo_render"])
        merged = ca.merge_bed_runs(placements)
        spans = [(shot, s0, s1) for _, shot, s0, s1 in ca.shot_spans()]
        by_shot, dropped = ca.cut_against_spans(merged, spans)
        self.assertEqual(dropped, 0)
        self.assertGreater(sum(len(v) for v in by_shot.values()), 400)

    def test_real_draft_vo_resolution(self):
        """The delivered draft's VO maps onto the extracted comp audio: the
        bulk imports, and only the known missing/re-cut sources skip."""
        vo_dir = ROOT / "assets" / "audio" / "voice_lines"
        if not (self.DRAFT.exists() and self.STEMS.is_dir()
                and vo_dir.is_dir()):
            self.skipTest("delivered audio assets not present")
        import shutil as _sh
        if _sh.which("ffprobe") is None:
            self.skipTest("ffprobe not available")
        placements, stats, _, vo_skips = ca.load_placements(self.DRAFT,
                                                            self.STEMS)
        self.assertGreaterEqual(stats["vo_placed"], 85)
        vo = [p for p in placements if p["role"] == "vo"]
        self.assertEqual(len(vo), stats["vo_placed"])
        # fork options resolve to their own comp audio
        names = {p["name"] for p in vo}
        self.assertIn("bhr_scene_09-A.mp3", names)
        self.assertIn("bhr_scene_09-B.mp3", names)
        # the known gaps are skipped, not silently dropped
        skipped_names = {s["name"] for s in vo_skips}
        for missing in ("tc-draft- new.mp4",
                        "Shoofly 29,30,31.mp4"):
            self.assertIn(missing, skipped_names)
        self.assertTrue(any("SC_37" in n for n in skipped_names))


if __name__ == "__main__":
    unittest.main()
