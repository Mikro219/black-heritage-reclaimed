"""Detector behaviour contract — the July 2026 playtest semantics.

Every test here encodes a promise made to the installation: if one goes red,
a scene interaction regressed."""

import time
import unittest

from engines.detectors.rules import (directional_draw, hand_pose,
                                     mouth_proximity_tip, paddle, point_region,
                                     point_target_held, rhythm_bilateral,
                                     run_arms, speed_bilateral, survey, throw)
from tests.mocks import LM, make_hand, pose33


class TestHandPose(unittest.TestCase):
    def test_pointing_hand(self):
        self.assertTrue(hand_pose.is_pointing(make_hand(0.5, 0.5, shape="point")))

    def test_open_palm(self):
        h = make_hand(0.5, 0.5, shape="open")
        self.assertTrue(hand_pose.is_open_palm(h))
        self.assertFalse(hand_pose.is_pointing(h))

    def test_fist(self):
        self.assertTrue(hand_pose.is_fist(make_hand(0.5, 0.5, shape="fist")))


class TestSurveySalute(unittest.TestCase):
    """survey = salute: hold a visible pose wrist at brow height."""

    PARAMS = {"brow_y": 0.45, "salute_hold_ms": 100}

    def test_fires_after_hold_at_brow(self):
        ctx = {"_pose_lm": pose33({15: LM(0.6, 0.30, visibility=0.9)})}
        self.assertFalse(survey.detect([], self.PARAMS, ctx))
        time.sleep(0.12)
        ctx["_pose_lm"] = pose33({15: LM(0.6, 0.30, visibility=0.9)})
        self.assertTrue(survey.detect([], self.PARAMS, ctx))

    def test_hands_down_never_fires(self):
        ctx = {"_pose_lm": pose33()}   # wrists at 0.85
        self.assertFalse(survey.detect([], self.PARAMS, ctx))

    def test_phantom_wrist_ignored(self):
        ctx = {"_pose_lm": pose33({15: LM(0.6, 0.30, visibility=0.2)})}
        self.assertFalse(survey.detect([], self.PARAMS, ctx))


class TestKnockApproach(unittest.TestCase):
    """rhythm_bilateral = the hand gets bigger (pushes at the camera), twice."""

    PARAMS = {"knock_count": 2, "min_growth_pct": 30, "knock_window_ms": 3000,
              "baseline_window_ms": 1200, "refractory_ms": 10}

    def _run(self, sizes):
        ctx, fired = {}, False
        for size in sizes:
            fired = rhythm_bilateral.detect(
                [make_hand(0.5, 0.5, size=size)], self.PARAMS, ctx) or fired
            time.sleep(0.02)
        return fired

    def test_two_pushes_fire(self):
        self.assertTrue(self._run([0.08, 0.08, 0.12, 0.08, 0.08, 0.12]))

    def test_idle_hand_does_not_fire(self):
        self.assertFalse(self._run([0.08, 0.08, 0.085, 0.09, 0.088, 0.09]))

    def test_single_push_does_not_fire(self):
        self.assertFalse(self._run([0.08, 0.08, 0.12, 0.12, 0.12, 0.12]))


class TestSpeedBilateral(unittest.TestCase):
    """speed_bilateral: velocity in screen units per SECOND, dropout-tolerant."""

    PARAMS = {"min_bursts": 3, "burst_window_ms": 3000, "velocity_multiplier": 2.0,
              "idle_velocity": 0.45, "min_active_ms": 50, "use_pose": True}

    def test_fast_shake_fires(self):
        ctx, fired, x = {}, False, 0.3
        for i in range(8):
            x += 0.06 if i % 2 == 0 else -0.06
            ctx["_pose_lm"] = pose33({15: LM(x, 0.5, visibility=0.9),
                                      16: LM(x - 0.2, 0.5, visibility=0.9)})
            fired = speed_bilateral.detect([], self.PARAMS, ctx) or fired
            time.sleep(0.03)
        self.assertTrue(fired)

    def test_slow_drift_does_not_fire(self):
        ctx, fired = {}, False
        for i in range(8):
            ctx["_pose_lm"] = pose33({15: LM(0.3 + i * 0.002, 0.5, visibility=0.9),
                                      16: LM(0.1, 0.5, visibility=0.9)})
            fired = speed_bilateral.detect([], self.PARAMS, ctx) or fired
            time.sleep(0.03)
        self.assertFalse(fired)

    def test_single_visible_wrist_ok(self):
        ctx, fired = {}, False
        for i in range(8):
            x = 0.3 + (0.06 if i % 2 == 0 else -0.06)
            ctx["_pose_lm"] = pose33({15: LM(x, 0.5, visibility=0.9),
                                      16: LM(0.1, 0.5, visibility=0.1)})
            fired = speed_bilateral.detect([], self.PARAMS, ctx) or fired
            time.sleep(0.03)
        self.assertTrue(fired)


class TestPaddle(unittest.TestCase):
    """paddle: hysteresis band + timed strokes (midline jitter must not count)."""

    PARAMS = {"min_strokes": 2, "min_stroke_interval_ms": 10, "stroke_window_ms": 6000}

    def test_two_clear_strokes_fire(self):
        ctx, fired = {}, False
        for y in [0.70, 0.45, 0.70, 0.45, 0.70]:
            ctx["_pose_lm"] = pose33({15: LM(0.6, y, visibility=0.9),
                                      16: LM(0.4, y, visibility=0.9)})
            fired = paddle.detect([], self.PARAMS, ctx) or fired
            time.sleep(0.03)
        self.assertTrue(fired)

    def test_midline_jitter_does_not_fire(self):
        ctx, fired = {}, False
        for i in range(20):   # ±0.01 inside the band around midline 0.575
            y = 0.575 + (0.01 if i % 2 == 0 else -0.01)
            ctx["_pose_lm"] = pose33({15: LM(0.6, y, visibility=0.9),
                                      16: LM(0.4, y, visibility=0.9)})
            fired = paddle.detect([], self.PARAMS, ctx) or fired
        self.assertFalse(fired)


class TestDirectionalDraw(unittest.TestCase):
    """directional_draw: pose-wrist primary; either wrist can draw the stroke."""

    PARAMS = {"direction": "left", "window_ms": 600, "min_displacement": 0.04}

    def test_left_stroke_no_hands_fires(self):
        ctx, fired = {}, False
        for i in range(6):   # raw x increases -> screen dx negative -> LEFT
            ctx["_pose_lm"] = pose33({15: LM(0.4 + i * 0.03, 0.5, visibility=0.9)})
            fired = directional_draw.detect([], self.PARAMS, ctx) or fired
            time.sleep(0.04)
        self.assertTrue(fired)

    def test_wrong_direction_rejected(self):
        ctx, fired = {}, False
        for i in range(6):   # moving DOWN, target left
            ctx["_pose_lm"] = pose33({15: LM(0.4, 0.4 + i * 0.03, visibility=0.9)})
            fired = directional_draw.detect([], self.PARAMS, ctx) or fired
            time.sleep(0.04)
        self.assertFalse(fired)


class TestPointTargetHeld(unittest.TestCase):
    """point_target_held: pointing FINGER (open palms rejected) + pose fallback."""

    PARAMS = {"region_rect": {"x": 0.4, "y": 0.2, "w": 0.3, "h": 0.5}, "hold_ms": 80}

    def test_pointing_finger_in_rect_fires(self):
        ctx = {"_pose_lm": None}
        h = make_hand(0.55, 0.45, shape="point")
        self.assertFalse(point_target_held.detect([h], self.PARAMS, ctx))
        time.sleep(0.1)
        self.assertTrue(point_target_held.detect([h], self.PARAMS, ctx))

    def test_open_palm_rejected(self):
        ctx = {"_pose_lm": None}
        h = make_hand(0.55, 0.45, shape="open")
        self.assertFalse(point_target_held.detect([h], self.PARAMS, ctx))
        time.sleep(0.1)
        self.assertFalse(point_target_held.detect([h], self.PARAMS, ctx))

    def test_pose_wrist_fallback(self):
        ctx = {"_pose_lm": pose33({15: LM(0.55, 0.45, visibility=0.9)})}
        self.assertFalse(point_target_held.detect([], self.PARAMS, ctx))
        time.sleep(0.1)
        self.assertTrue(point_target_held.detect([], self.PARAMS, ctx))


class TestPointRegion(unittest.TestCase):
    """point_region: side selection = a POINT (open palms rejected), and low
    side-reaches (Scene 9 hidden compartment) are accepted."""

    PARAMS = {"directions": ["left", "right"], "hold_ms": 60, "dead_zone_frac": 0.25}

    def _select(self, pose, hands=()):
        ctx = {"_pose_lm": pose}
        point_region.detect(list(hands), self.PARAMS, ctx)
        time.sleep(0.08)
        ctx["_pose_lm"] = pose
        return point_region.detect(list(hands), self.PARAMS, ctx), ctx

    def test_pose_only_side_reach_selects(self):
        fired, ctx = self._select(pose33({15: LM(0.95, 0.55, visibility=0.9)}))
        self.assertTrue(fired)
        self.assertEqual(ctx.get("point_direction"), "left")

    def test_open_palm_at_wrist_rejected(self):
        pose = pose33({15: LM(0.95, 0.55, visibility=0.9)})
        fired, _ = self._select(pose, [make_hand(0.95, 0.55, shape="open")])
        self.assertFalse(fired)

    def test_pointing_hand_at_wrist_accepted(self):
        pose = pose33({15: LM(0.95, 0.55, visibility=0.9)})
        fired, _ = self._select(pose, [make_hand(0.95, 0.55, shape="point")])
        self.assertTrue(fired)

    def test_low_side_reach_accepted(self):
        fired, _ = self._select(pose33({15: LM(0.95, 0.72, visibility=0.9)}))
        self.assertTrue(fired)

    def test_hanging_arm_near_body_rejected(self):
        fired, _ = self._select(pose33({15: LM(0.66, 0.72, visibility=0.9)}))
        self.assertFalse(fired)


class TestThrow(unittest.TestCase):
    def test_survives_blur_mid_stroke(self):
        """Wind-up above shoulder, visibility drops mid-stroke, release below."""
        p = {"max_stroke_ms": 600, "shoulder_margin": 0.03}
        ctx = {}
        ctx["_pose_lm"] = pose33({16: LM(0.35, 0.20, visibility=0.9)})
        self.assertFalse(throw.detect([], p, ctx))
        ctx["_pose_lm"] = pose33({16: LM(0.35, 0.50, visibility=0.2)})
        self.assertFalse(throw.detect([], p, ctx))
        ctx["_pose_lm"] = pose33({16: LM(0.35, 0.60, visibility=0.9)})
        self.assertTrue(throw.detect([], p, ctx))


class TestDrink(unittest.TestCase):
    def test_natural_drink_pose_fires(self):
        """Fingertips at the mouth with the wrist BELOW them (natural drink)."""
        drink = make_hand(0.5, 0.40, size=0.10, shape="fist")
        drink.landmark[8] = LM(0.5, 0.33)
        drink.landmark[0] = LM(0.5, 0.50)
        ctx = {"_pose_lm": pose33()}
        p = {"hold_ms": 60, "proximity_threshold": 0.12}
        self.assertFalse(mouth_proximity_tip.detect([drink], p, ctx))
        time.sleep(0.08)
        self.assertTrue(mouth_proximity_tip.detect([drink], p, ctx))


class TestRunArms(unittest.TestCase):
    def test_chest_height_pump_fires(self):
        """Alternating pump across the WAIST line (shoulder/hip midpoint)."""
        ctx, fired = {}, False
        p = {"min_cycles": 2, "window_ms": 3000}
        for i in range(8):
            hi, lo = (0.45, 0.70) if i % 2 == 0 else (0.70, 0.45)
            ctx["_pose_lm"] = pose33({15: LM(0.6, hi, visibility=0.9),
                                      16: LM(0.4, lo, visibility=0.9)})
            fired = run_arms.detect([], p, ctx) or fired
            time.sleep(0.02)
        self.assertTrue(fired)


if __name__ == "__main__":
    unittest.main()
