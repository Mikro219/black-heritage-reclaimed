"""Pose-hand fusion contract (engines/pose_hand_filter.py):

VETO       hands nowhere near a trusted pose wrist are dropped — but ONLY when
           both wrists are trusted (full skeleton). Partial skeleton = pass.
LABEL      matched hands take handedness from the pose side, not MediaPipe's
           classifier.
ARBITRATE  a matched hand lagging further than its own size behind the pose
           wrist is a stale track — dropped.
NO POSE    plain-webcam behaviour is byte-for-byte unchanged.
RESCUE     crop/remap geometry for the pose-guided re-inference.
"""

import unittest

from engines import pose_hand_filter as phf
from tests.mocks import LM, make_hand, pose33


def wrong_handed(label):
    return phf.synth_handedness(label)


class TestFilterPassThrough(unittest.TestCase):
    def test_no_pose_is_unchanged(self):
        hands = [make_hand(0.5, 0.5, 0.1, "open")]
        hnd = [wrong_handed("Right")]
        out_l, out_h, stats = phf.filter_hands(hands, hnd, None)
        self.assertIs(out_l, hands)
        self.assertIs(out_h, hnd)
        self.assertEqual(stats, {"dropped": 0, "corrected": 0})

    def test_no_trusted_wrists_is_unchanged(self):
        pose = pose33({15: LM(0.3, 0.8, visibility=0.1),
                       16: LM(0.7, 0.8, visibility=0.1)})
        hands = [make_hand(0.5, 0.5, 0.1, "open")]
        out_l, out_h, _ = phf.filter_hands(hands, [wrong_handed("Left")], pose)
        self.assertIs(out_l, hands)

    def test_partial_skeleton_keeps_unmatched_hand(self):
        # Only the LEFT pose wrist is trusted; a hand far from it (the player's
        # other hand, whose pose wrist dropped out) must survive.
        pose = pose33({15: LM(0.30, 0.50, visibility=0.9),
                       16: LM(0.70, 0.50, visibility=0.1)})
        far_hand = make_hand(0.72, 0.52, 0.10, "open")
        out_l, out_h, stats = phf.filter_hands([far_hand], [wrong_handed("Right")], pose)
        self.assertEqual(len(out_l), 1)
        self.assertEqual(stats["dropped"], 0)
        # unmatched pass-through keeps its original label
        self.assertEqual(out_h[0].classification[0].label, "Right")


class TestVetoAndLabel(unittest.TestCase):
    def setUp(self):
        self.pose = pose33({15: LM(0.30, 0.50, visibility=0.9),
                            16: LM(0.70, 0.50, visibility=0.9)})

    def test_phantom_hand_dropped_with_full_skeleton(self):
        player = make_hand(0.31, 0.51, 0.10, "open")     # at left wrist
        phantom = make_hand(0.95, 0.05, 0.08, "open")    # background corner
        out_l, out_h, stats = phf.filter_hands(
            [player, phantom], [wrong_handed("Left"), wrong_handed("Right")], self.pose)
        self.assertEqual(len(out_l), 1)
        self.assertEqual(stats["dropped"], 1)

    def test_handedness_comes_from_pose_side(self):
        # MediaPipe says "Right", but the hand sits on the pose LEFT wrist.
        hand = make_hand(0.31, 0.51, 0.10, "point")
        out_l, out_h, stats = phf.filter_hands([hand], [wrong_handed("Right")], self.pose)
        self.assertEqual(out_h[0].classification[0].label, "Left")
        self.assertEqual(stats["corrected"], 1)

    def test_two_hands_two_wrists_both_kept_and_labelled(self):
        lh = make_hand(0.31, 0.51, 0.10, "open")
        rh = make_hand(0.69, 0.49, 0.10, "fist")
        out_l, out_h, stats = phf.filter_hands(
            [lh, rh], [wrong_handed("Right"), wrong_handed("Left")], self.pose)
        self.assertEqual(len(out_l), 2)
        self.assertEqual([h.classification[0].label for h in out_h], ["Left", "Right"])

    def test_one_wrist_takes_only_nearest_hand(self):
        # Two detections near the same wrist (double-detection): nearest wins
        # the match; the other is unmatched -> dropped (full skeleton).
        near = make_hand(0.31, 0.50, 0.10, "open")
        dup = make_hand(0.38, 0.55, 0.10, "open")
        out_l, out_h, stats = phf.filter_hands(
            [dup, near], [wrong_handed("Left"), wrong_handed("Left")], self.pose)
        self.assertEqual(len(out_l), 1)
        self.assertAlmostEqual(out_l[0].landmark[0].x, near.landmark[0].x, places=3)

    def test_all_dropped_returns_none(self):
        phantom = make_hand(0.95, 0.05, 0.08, "open")
        out_l, out_h, _ = phf.filter_hands([phantom], [wrong_handed("Left")], self.pose)
        self.assertIsNone(out_l)
        self.assertIsNone(out_h)


class TestArbitration(unittest.TestCase):
    def setUp(self):
        self.pose = pose33({15: LM(0.30, 0.50, visibility=0.9),
                            16: LM(0.70, 0.50, visibility=0.9)})

    def test_small_lagging_hand_dropped(self):
        # A small (far player, ~0.10 bbox) hand whose wrist trails the pose
        # wrist by 0.16: inside match range (0.25) but > 1.3 x its own size
        # (0.13) -> stale/lagging track, dropped.
        # make_hand puts the wrist at (cx, cy + size): cy chosen so the hand
        # wrist lands exactly at y=0.50, x offset 0.16 from the pose wrist.
        lag = make_hand(0.30 + 0.16, 0.50 - 0.05, 0.05, "open")
        out_l, _, stats = phf.filter_hands([lag], [wrong_handed("Left")], self.pose)
        self.assertIsNone(out_l)
        self.assertEqual(stats["dropped"], 1)

    def test_big_hand_same_offset_kept(self):
        # Same 0.16 wrist offset but a big near hand (~0.40 bbox): the offset
        # is well under its own size — plausible geometry, kept.
        big = make_hand(0.30 + 0.16, 0.50 - 0.20, 0.20, "open")
        out_l, _, stats = phf.filter_hands([big], [wrong_handed("Left")], self.pose)
        self.assertEqual(len(out_l), 1)
        self.assertEqual(stats["dropped"], 0)


class TestRescueGeometry(unittest.TestCase):
    def setUp(self):
        # forearm elbow(13) -> wrist(15) length 0.15 normalised
        self.pose = pose33({13: LM(0.30, 0.65, visibility=0.9),
                            15: LM(0.30, 0.50, visibility=0.9)})

    def test_crop_box_centred_and_scaled(self):
        box = phf.wrist_crop_box(self.pose, "Left", 1280, 720)
        self.assertIsNotNone(box)
        x0, y0, x1, y1 = box
        size = x1 - x0
        # forearm 0.15 * scale 2.2 * max(1280,720) = ~422px
        self.assertAlmostEqual(size, int(0.15 * 2.2 * 1280), delta=2)
        self.assertEqual(size, y1 - y0)  # square
        cx = int(0.30 * 1280)
        self.assertLessEqual(abs((x0 + x1) // 2 - cx), 2)

    def test_crop_box_clamped_to_frame(self):
        pose = pose33({13: LM(0.05, 0.20, visibility=0.9),
                       15: LM(0.02, 0.05, visibility=0.9)})   # near corner
        box = phf.wrist_crop_box(pose, "Left", 1280, 720)
        x0, y0, x1, y1 = box
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, 1280)
        self.assertLessEqual(y1, 720)

    def test_low_visibility_wrist_gives_no_box(self):
        pose = pose33({13: LM(0.30, 0.65, visibility=0.9),
                       15: LM(0.30, 0.50, visibility=0.2)})
        self.assertIsNone(phf.wrist_crop_box(pose, "Left", 1280, 720))

    def test_remap_round_trips_crop_centre(self):
        box = (100, 200, 300, 400)   # 200px crop in a 1000x500 frame
        from types import SimpleNamespace
        crop_hand = SimpleNamespace(landmark=[SimpleNamespace(x=0.5, y=0.5, z=-0.1)])
        out = phf.remap_crop_landmarks(crop_hand, box, 1000, 500)
        self.assertAlmostEqual(out.landmark[0].x, 200 / 1000)   # crop centre x = 200px
        self.assertAlmostEqual(out.landmark[0].y, 300 / 500)    # crop centre y = 300px
        self.assertAlmostEqual(out.landmark[0].z, -0.1)


if __name__ == "__main__":
    unittest.main()
