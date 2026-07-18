"""Behaviour contract for the POSE-ONLY detector layer (July 2026 rework).

Every detector reads the 33-landmark Pose skeleton from context["_pose_lm"]
(the legacy Hands argument is always []). These tests pin the semantics that
replaced the Hands-model checks: arm extension instead of finger pose,
wrist→index point vectors, and depth / pose-z instead of hand-bbox growth.
If one goes red, a scene interaction regressed.
"""

import time
import unittest

from engines.detectors.rules import (directional_draw, directional_point,
                                     mouth_proximity_tip, paddle,
                                     point_region, point_target_held,
                                     pose_helpers, presence_bilateral,
                                     push_out, reach_and_close,
                                     rhythm_bilateral, run_arms,
                                     speed_bilateral, survey, throw)
from tests.mocks import LM, pose33, seq_sampler


def ctx_for(pose):
    return {"_pose_lm": pose}


def straight_left_arm_pointing(direction_raw_dx=-0.05):
    """Left arm fully extended horizontally, index visible: shoulder (0.6,0.4)
    -> elbow (0.45,0.4) -> wrist (0.30,0.4) -> index just past the wrist."""
    return pose33({13: LM(0.45, 0.40),
                   15: LM(0.30, 0.40, visibility=0.9),
                   19: LM(0.30 + direction_raw_dx, 0.40, visibility=0.9)})


class TestPoseHelpers(unittest.TestCase):
    def test_classify_direction(self):
        self.assertEqual(pose_helpers.classify_direction(1, 0), "right")
        self.assertEqual(pose_helpers.classify_direction(0, -1), "up")
        self.assertEqual(pose_helpers.classify_direction(-0.7, -0.7), "up_left")

    def test_arm_reach_straight_vs_bent(self):
        straight = straight_left_arm_pointing()
        self.assertGreater(pose_helpers.arm_reach_frac(straight, "L"), 0.95)
        bent = pose33({13: LM(0.58, 0.55),
                       15: LM(0.56, 0.42, visibility=0.9)})
        self.assertLess(pose_helpers.arm_reach_frac(bent, "L"), 0.5)

    def test_hand_point_prefers_visible_index(self):
        pose = pose33({19: LM(0.7, 0.9, visibility=0.9)})
        self.assertAlmostEqual(pose_helpers.hand_point(pose, "L").x, 0.7)
        pose_lo = pose33()   # index visibility 0.3 -> falls back to the wrist
        self.assertAlmostEqual(pose_helpers.hand_point(pose_lo, "L").x, 0.65)


class TestSurveySalute(unittest.TestCase):
    def test_salute_hold_fires(self):
        pose = pose33({15: LM(0.60, 0.30, visibility=0.9)})   # wrist at brow
        ctx = ctx_for(pose)
        params = {"salute_hold_ms": 60}
        self.assertFalse(survey.detect([], params, ctx))
        time.sleep(0.08)
        self.assertTrue(survey.detect([], params, ctx))

    def test_hands_down_never_fires(self):
        ctx = ctx_for(pose33())   # wrists at 0.85 — below any brow line
        params = {"salute_hold_ms": 60}
        survey.detect([], params, ctx)
        time.sleep(0.08)
        self.assertFalse(survey.detect([], params, ctx))


class TestPresenceBilateral(unittest.TestCase):
    def test_both_wrists_raised_fires(self):
        pose = pose33({15: LM(0.65, 0.30, visibility=0.9),
                       16: LM(0.35, 0.30, visibility=0.9)})
        ctx = ctx_for(pose)
        params = {"y_threshold": 0.5, "hold_ms": 60}
        presence_bilateral.detect([], params, ctx)
        time.sleep(0.08)
        self.assertTrue(presence_bilateral.detect([], params, ctx))

    def test_one_wrist_low_never_fires(self):
        pose = pose33({15: LM(0.65, 0.30, visibility=0.9)})   # right stays 0.85
        ctx = ctx_for(pose)
        params = {"y_threshold": 0.5, "hold_ms": 60}
        presence_bilateral.detect([], params, ctx)
        time.sleep(0.08)
        self.assertFalse(presence_bilateral.detect([], params, ctx))


class TestDirectionalPoint(unittest.TestCase):
    PARAMS = {"directions": ["right"], "hold_ms": 60}

    def test_extended_arm_pointing_right_fires(self):
        # Raw dx negative -> screen dx positive -> player "right".
        pose = straight_left_arm_pointing(-0.05)
        ctx = ctx_for(pose)
        directional_point.detect([], self.PARAMS, ctx)
        time.sleep(0.08)
        self.assertTrue(directional_point.detect([], self.PARAMS, ctx))
        self.assertEqual(ctx["dominant_direction"], "right")

    def test_bent_arm_rejected(self):
        pose = pose33({13: LM(0.58, 0.55),
                       15: LM(0.56, 0.42, visibility=0.9),
                       19: LM(0.50, 0.42, visibility=0.9)})
        ctx = ctx_for(pose)
        directional_point.detect([], self.PARAMS, ctx)
        time.sleep(0.08)
        self.assertFalse(directional_point.detect([], self.PARAMS, ctx))


class TestPointRegion(unittest.TestCase):
    PARAMS = {"directions": ["left", "right"], "hold_ms": 60}

    def test_side_reach_selects(self):
        pose = pose33({16: LM(0.15, 0.50, visibility=0.9)})   # screen right
        ctx = ctx_for(pose)
        point_region.detect([], self.PARAMS, ctx)
        time.sleep(0.08)
        self.assertTrue(point_region.detect([], self.PARAMS, ctx))
        self.assertEqual(ctx["point_direction"], "right")

    def test_centre_dead_zone_selects_nothing(self):
        pose = pose33({16: LM(0.49, 0.50, visibility=0.9)})
        ctx = ctx_for(pose)
        point_region.detect([], self.PARAMS, ctx)
        time.sleep(0.08)
        self.assertFalse(point_region.detect([], self.PARAMS, ctx))

    def test_raised_hand_near_centre_selects_nothing(self):
        # July 2026 tightening: a raised hand only ~0.3 shoulder-widths off
        # the midline (raw x 0.44, offset 0.06 < 0.45 x 0.2 dead zone) used to
        # select a side under the old 0.15 dead zone — it must not.
        pose = pose33({16: LM(0.44, 0.50, visibility=0.9)})
        ctx = ctx_for(pose)
        point_region.detect([], self.PARAMS, ctx)
        time.sleep(0.08)
        self.assertFalse(point_region.detect([], self.PARAMS, ctx))

    def test_low_reach_accepted_below_raise_gate(self):
        # Wrist below the raise gate but a full shoulder-width from midline
        # (Scene 9 hidden-compartment point at waist height).
        pose = pose33({16: LM(0.15, 0.72, visibility=0.9)})
        ctx = ctx_for(pose)
        point_region.detect([], self.PARAMS, ctx)
        time.sleep(0.08)
        self.assertTrue(point_region.detect([], self.PARAMS, ctx))


class TestPointTargetHeld(unittest.TestCase):
    RECT = {"x": 0.20, "y": 0.20, "w": 0.20, "h": 0.25}

    def test_extended_arm_on_target_fires(self):
        pose = pose33({13: LM(0.45, 0.38),
                       15: LM(0.32, 0.34, visibility=0.9),
                       19: LM(0.30, 0.30, visibility=0.9)})   # index in rect
        ctx = ctx_for(pose)
        params = {"region_rect": self.RECT, "hold_ms": 60}
        point_target_held.detect([], params, ctx)
        time.sleep(0.08)
        self.assertTrue(point_target_held.detect([], params, ctx))

    def test_bent_arm_hover_on_target_fires(self):
        # Pointing at an on-screen box aims the arm at the camera — the 2-D
        # reach projection is foreshortened (~0.48 here), which used to be
        # rejected by min_reach_frac. In-rect hover only needs the light
        # hover_reach_frac gate (July 2026 tutorial fix).
        pose = pose33({13: LM(0.62, 0.60),
                       15: LM(0.32, 0.34, visibility=0.9),
                       19: LM(0.30, 0.30, visibility=0.9)})
        ctx = ctx_for(pose)
        params = {"region_rect": self.RECT, "hold_ms": 60}
        point_target_held.detect([], params, ctx)
        time.sleep(0.08)
        self.assertTrue(point_target_held.detect([], params, ctx))

    def test_folded_arm_on_target_rejected(self):
        # Wrist parked next to the shoulder (reach ~0.07): a hand resting on
        # the chest that happens to overlap the box must not fire.
        pose = pose33({13: LM(0.62, 0.60),
                       15: LM(0.58, 0.42, visibility=0.9),
                       19: LM(0.30, 0.30, visibility=0.9)})
        ctx = ctx_for(pose)
        params = {"region_rect": self.RECT, "hold_ms": 60}
        point_target_held.detect([], params, ctx)
        time.sleep(0.08)
        self.assertFalse(point_target_held.detect([], params, ctx))


class TestKnockApproach(unittest.TestCase):
    """A knock is a push toward the camera — depth drop (Gemini) or pose-z drop."""

    def _pose_one_wrist(self, z=0.0):
        return pose33({15: LM(0.60, 0.55, z, visibility=0.9),
                       16: LM(0.35, 0.85, visibility=0.1)})   # R untrusted

    def test_two_depth_knocks_fire(self):
        params = {"knock_count": 2, "min_depth_delta_mm": 120, "refractory_ms": 0}
        ctx = ctx_for(self._pose_one_wrist())
        ctx["_depth_mm_at"] = seq_sampler([1500, 1500, 1300, 1520, 1330])
        fired = []
        for _ in range(5):
            ctx["_pose_lm"] = self._pose_one_wrist()
            fired.append(rhythm_bilateral.detect([], params, ctx))
        self.assertTrue(any(fired))

    def test_single_push_does_not_fire(self):
        params = {"knock_count": 2, "min_depth_delta_mm": 120, "refractory_ms": 0}
        ctx = ctx_for(self._pose_one_wrist())
        ctx["_depth_mm_at"] = seq_sampler([1500, 1500, 1300, 1300, 1300])
        fired = []
        for _ in range(5):
            ctx["_pose_lm"] = self._pose_one_wrist()
            fired.append(rhythm_bilateral.detect([], params, ctx))
        self.assertFalse(any(fired))

    def test_pose_z_knocks_fire_without_sampler(self):
        params = {"knock_count": 2, "min_pose_z_delta": 0.12, "refractory_ms": 0}
        ctx = {}
        fired = []
        for z in (0.0, 0.0, -0.15, 0.05, -0.10):
            ctx["_pose_lm"] = self._pose_one_wrist(z)
            fired.append(rhythm_bilateral.detect([], params, ctx))
        self.assertTrue(any(fired))


class TestPushOutPoseZ(unittest.TestCase):
    def test_bilateral_z_push_fires(self):
        params = {"window_ms": 800, "min_pose_z_delta": 0.15}
        ctx = {}
        fired = False
        for z in (0.0, 0.0, -0.2):
            ctx["_pose_lm"] = pose33({15: LM(0.65, 0.55, z, visibility=0.9),
                                      16: LM(0.35, 0.55, z, visibility=0.9)})
            fired = push_out.detect([], params, ctx) or fired
        self.assertTrue(fired)

    def test_one_hand_only_does_not_fire(self):
        params = {"window_ms": 800, "min_pose_z_delta": 0.15}
        ctx = {}
        fired = False
        for z in (0.0, 0.0, -0.2):
            ctx["_pose_lm"] = pose33({15: LM(0.65, 0.55, z, visibility=0.9),
                                      16: LM(0.35, 0.55, 0.0, visibility=0.9)})
            fired = push_out.detect([], params, ctx) or fired
        self.assertFalse(fired)


class TestDrink(unittest.TestCase):
    def test_natural_drink_pose_fires(self):
        # Index near the mouth (0.5, 0.34), wrist below it — the tip.
        pose = pose33({19: LM(0.50, 0.36, visibility=0.9),
                       15: LM(0.50, 0.42, visibility=0.9)})
        ctx = ctx_for(pose)
        params = {"hold_ms": 60}
        mouth_proximity_tip.detect([], params, ctx)
        time.sleep(0.08)
        self.assertTrue(mouth_proximity_tip.detect([], params, ctx))

    def test_wrist_above_tip_rejected(self):
        pose = pose33({19: LM(0.50, 0.36, visibility=0.9),
                       15: LM(0.50, 0.25, visibility=0.9)})
        ctx = ctx_for(pose)
        params = {"hold_ms": 60}
        mouth_proximity_tip.detect([], params, ctx)
        time.sleep(0.08)
        self.assertFalse(mouth_proximity_tip.detect([], params, ctx))


class TestDirectionalDraw(unittest.TestCase):
    def test_moving_wrist_draws_left_while_other_is_static(self):
        params = {"direction": "left", "window_ms": 800, "min_displacement": 0.04}
        ctx = {}
        fired = False
        for x in (0.60, 0.66, 0.72):     # raw x grows -> screen moves LEFT
            ctx["_pose_lm"] = pose33({15: LM(x, 0.5, visibility=0.9),
                                      16: LM(0.35, 0.85, visibility=0.9)})
            fired = directional_draw.detect([], params, ctx) or fired
            time.sleep(0.06)
        self.assertTrue(fired)

    def test_static_wrists_never_fire(self):
        params = {"direction": "left", "window_ms": 800, "min_displacement": 0.04}
        ctx = {}
        fired = False
        for _ in range(4):
            ctx["_pose_lm"] = pose33({15: LM(0.6, 0.5, visibility=0.9)})
            fired = directional_draw.detect([], params, ctx) or fired
            time.sleep(0.05)
        self.assertFalse(fired)


class TestThrow(unittest.TestCase):
    def test_windup_and_snap_fires(self):
        params = {"max_stroke_ms": 600}
        above = pose33({16: LM(0.35, 0.20, visibility=0.9)})
        below = pose33({16: LM(0.35, 0.60, visibility=0.9)})
        ctx = ctx_for(above)
        self.assertFalse(throw.detect([], params, ctx))
        ctx["_pose_lm"] = below
        self.assertTrue(throw.detect([], params, ctx))

    def test_slow_lower_is_not_a_throw(self):
        params = {"max_stroke_ms": 50}
        above = pose33({16: LM(0.35, 0.20, visibility=0.9)})
        below = pose33({16: LM(0.35, 0.60, visibility=0.9)})
        ctx = ctx_for(above)
        throw.detect([], params, ctx)
        time.sleep(0.08)   # stroke window expires
        ctx["_pose_lm"] = below
        self.assertFalse(throw.detect([], params, ctx))


class _FakeClock:
    """Injectable stand-in for the `time` module inside a rule: each
    monotonic() read is exact, so velocity tests don't flake on wall-clock
    scheduling jitter (the old sleep(0.03) drove real dt through v = d/dt)."""

    def __init__(self, step=0.03):
        self.now = 1000.0
        self.step = step

    def monotonic(self):
        return self.now

    def tick(self):
        self.now += self.step


class TestSpeedBilateral(unittest.TestCase):
    PARAMS = {"min_bursts": 3, "burst_window_ms": 3000, "velocity_multiplier": 2.0,
              "idle_velocity": 0.45, "min_active_ms": 50}

    def _run(self, positions):
        from unittest import mock
        clock = _FakeClock()
        fired, ctx = False, {}
        with mock.patch.object(speed_bilateral, "time", clock):
            for x in positions:
                ctx["_pose_lm"] = pose33({15: LM(x, 0.5, visibility=0.9)})
                fired = speed_bilateral.detect([], self.PARAMS, ctx) or fired
                clock.tick()
        return fired

    def test_fast_wrist_fires(self):
        xs, x = [], 0.4
        for i in range(8):
            x += 0.06 if i % 2 == 0 else -0.06
            xs.append(x)
        self.assertTrue(self._run(xs))       # 0.06/0.03s = 2.0 units/s > 0.9

    def test_idle_wrist_does_not_fire(self):
        self.assertFalse(self._run([0.4] * 8))


class TestPaddle(unittest.TestCase):
    def _pose_both(self, y):
        return pose33({15: LM(0.65, y, visibility=0.9),
                       16: LM(0.35, y, visibility=0.9)})

    def test_two_full_strokes_fire(self):
        params = {"min_strokes": 2, "min_stroke_interval_ms": 0}
        ctx = {}
        fired = False
        for y in (0.70, 0.45, 0.70, 0.45):   # midline 0.575 ± band
            ctx["_pose_lm"] = self._pose_both(y)
            fired = paddle.detect([], params, ctx) or fired
        self.assertTrue(fired)

    def test_midline_jitter_does_not_fire(self):
        params = {"min_strokes": 2, "min_stroke_interval_ms": 0, "band_frac": 0.12}
        ctx = {}
        fired = False
        for y in (0.57, 0.58, 0.57, 0.58, 0.57):   # inside the hysteresis band
            ctx["_pose_lm"] = self._pose_both(y)
            fired = paddle.detect([], params, ctx) or fired
        self.assertFalse(fired)


class TestRunArms(unittest.TestCase):
    def test_alternating_pump_across_waist_fires(self):
        params = {"min_cycles": 3, "window_ms": 2000}
        ctx = {}
        fired = False
        for i in range(8):   # waist = (0.40 + 0.75) / 2 = 0.575
            hi, lo = (0.45, 0.70) if i % 2 == 0 else (0.70, 0.45)
            ctx["_pose_lm"] = pose33({15: LM(0.65, hi, visibility=0.9),
                                      16: LM(0.35, lo, visibility=0.9)})
            fired = run_arms.detect([], params, ctx) or fired
        self.assertTrue(fired)


class TestReachAndClose(unittest.TestCase):
    def test_reach_then_settle_fires(self):
        params = {"min_pose_z_delta": 0.15, "settle_ms": 60,
                  "settle_velocity": 0.25, "reach_window_ms": 2000}
        ctx = {}

        def frame(z):
            return pose33({15: LM(0.60, 0.50, z, visibility=0.9),
                           16: LM(0.35, 0.85, visibility=0.1)})

        ctx["_pose_lm"] = frame(0.0)
        self.assertFalse(reach_and_close.detect([], params, ctx))
        ctx["_pose_lm"] = frame(-0.2)                       # the reach
        self.assertFalse(reach_and_close.detect([], params, ctx))
        fired = False
        for _ in range(4):                                  # hold still
            time.sleep(0.04)
            ctx["_pose_lm"] = frame(-0.2)
            fired = reach_and_close.detect([], params, ctx) or fired
        self.assertTrue(fired)


if __name__ == "__main__":
    unittest.main()
