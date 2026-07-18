"""Gemini 335 depth-fusion contract: lift / veto / meter / metric / band,
and the depth-preferring paths inside the z-approach detectors."""

import time
import unittest

from engines.depth.fusion import (PoseDepth, metric_point, metric_speed_mm_s,
                                  trusted_landmark)
from engines.detectors.rules import (forward_reach, point_region, push_out,
                                     speed_bilateral, throw)
from tests.mocks import LM, flat_sampler, pose33, seq_sampler, zone_sampler


class TestPoseDepthCore(unittest.TestCase):
    def test_lift_and_torso(self):
        pd = PoseDepth(flat_sampler(1500.0), pose33())
        self.assertTrue(pd.available)
        self.assertEqual(pd.torso_depth_mm(), 1500.0)
        self.assertEqual(pd.landmark_mm(15), 1500.0)
        self.assertEqual(pd.reach_mm(15), 0.0)

    def test_reach_toward_camera(self):
        pd = PoseDepth(zone_sampler([(0.6, 0.7, 0.8, 0.9, 1100.0)], 1500.0), pose33())
        self.assertEqual(pd.reach_mm(15), 400.0)
        self.assertIs(pd.plausible(15), True)

    def test_back_wall_phantom_vetoed(self):
        pose = pose33()
        pd = PoseDepth(zone_sampler([(0.6, 0.7, 0.8, 0.9, 3000.0)], 1500.0), pose)
        self.assertIs(pd.plausible(15), False)
        self.assertFalse(pd.trusted(15, pose[15], 0.5))
        self.assertTrue(pd.trusted(16, pose[16], 0.5))   # other wrist unaffected

    def test_no_sampler_degrades_to_visibility(self):
        pose = pose33()
        pd = PoseDepth(None, pose)
        self.assertFalse(pd.available)
        self.assertIsNone(pd.plausible(15))
        self.assertTrue(pd.trusted(15, pose[15], 0.5))

    def test_helper_with_and_without_fusion(self):
        pose = pose33()
        ctx = {"_pose_depth": PoseDepth(
            zone_sampler([(0.6, 0.7, 0.8, 0.9, 3000.0)], 1500.0), pose)}
        self.assertFalse(trusted_landmark(ctx, 15, pose[15], 0.5))
        self.assertTrue(trusted_landmark({}, 15, pose[15], 0.5))

    def test_metric_geometry(self):
        left = metric_point(0.0, 0.5, 1000.0)
        right = metric_point(1.0, 0.5, 1000.0)
        self.assertTrue(1800 <= right[0] - left[0] <= 1930)   # ~1.86 m at 1 m
        v = metric_speed_mm_s(metric_point(0.5, 0.5, 1000),
                              metric_point(0.5, 0.5, 900), 0.1)
        self.assertAlmostEqual(v, 1000.0)


class TestPointRegionDepthVeto(unittest.TestCase):
    PARAMS = {"directions": ["left", "right"], "hold_ms": 60, "dead_zone_frac": 0.25}

    def _select(self, fusion, pose):
        ctx = {"_pose_lm": pose, "_pose_depth": fusion}
        point_region.detect([], self.PARAMS, ctx)
        time.sleep(0.08)
        return point_region.detect([], self.PARAMS, ctx)

    def test_back_wall_wrist_never_selects(self):
        pose = pose33({15: LM(0.95, 0.55, visibility=0.9)})
        fusion = PoseDepth(zone_sampler([(0.9, 1.0, 0.5, 0.6, 4000.0)], 1500.0), pose)
        self.assertFalse(self._select(fusion, pose))

    def test_consistent_depth_selects(self):
        pose = pose33({15: LM(0.95, 0.55, visibility=0.9)})
        self.assertTrue(self._select(PoseDepth(flat_sampler(1500.0), pose), pose))


class TestThrowDepthDelta(unittest.TestCase):
    PARAMS = {"max_stroke_ms": 600, "shoulder_margin": 0.03,
              "require_growth": True, "min_depth_delta_mm": 250}

    def _throw(self, release_depth_mm):
        above = pose33({16: LM(0.35, 0.20, visibility=0.9)})
        below = pose33({16: LM(0.35, 0.60, visibility=0.9)})
        ctx = {"_pose_lm": above,
               "_pose_depth": PoseDepth(flat_sampler(1500.0), above)}
        throw.detect([], self.PARAMS, ctx)
        ctx["_pose_lm"] = below
        ctx["_pose_depth"] = PoseDepth(
            zone_sampler([(0.3, 0.4, 0.55, 0.65, release_depth_mm)], 1500.0), below)
        return throw.detect([], self.PARAMS, ctx)

    def test_real_400mm_approach_fires(self):
        self.assertTrue(self._throw(1100.0))

    def test_100mm_approach_rejected(self):
        self.assertFalse(self._throw(1400.0))


class TestPushOutDepth(unittest.TestCase):
    """POSE-ONLY: both pose wrists must show a real depth approach."""

    PARAMS = {"min_depth_delta_mm": 200, "window_ms": 500}

    def _pose(self):
        return pose33({15: LM(0.65, 0.55, visibility=0.9),
                       16: LM(0.35, 0.55, visibility=0.9)})

    def test_bilateral_approach_fires(self):
        # wrist_depth_or_z samples L then R each frame -> pairs per frame.
        ctx = {"_pose_lm": self._pose(),
               "_depth_mm_at": seq_sampler([1500, 1500, 1400, 1400, 1250, 1250])}
        fired = False
        for _ in range(3):
            fired = push_out.detect([], self.PARAMS, ctx) or fired
            time.sleep(0.03)
        self.assertTrue(fired)

    def test_static_depth_does_not_fire(self):
        ctx = {"_pose_lm": self._pose(), "_depth_mm_at": flat_sampler(1500.0)}
        fired = False
        for _ in range(5):
            fired = push_out.detect([], self.PARAMS, ctx) or fired
            time.sleep(0.03)
        self.assertFalse(fired)


class TestForwardReachDepth(unittest.TestCase):
    """POSE-ONLY: a single trusted pose wrist approaching the camera."""

    PARAMS = {"min_depth_delta_mm": 180, "window_ms": 900}

    def _pose(self):
        # Only the LEFT wrist is trusted, so the seq sampler feeds one wrist.
        return pose33({15: LM(0.60, 0.50, visibility=0.9),
                       16: LM(0.35, 0.85, visibility=0.1)})

    def test_approach_fires(self):
        ctx = {"_pose_lm": self._pose(),
               "_depth_mm_at": seq_sampler([1500, 1450, 1380, 1290, 1200])}
        fired = False
        for _ in range(5):
            fired = forward_reach.detect([], self.PARAMS, ctx) or fired
        self.assertTrue(fired)

    def test_static_depth_does_not_fire(self):
        ctx = {"_pose_lm": self._pose(), "_depth_mm_at": flat_sampler(1500.0)}
        fired = False
        for _ in range(8):
            fired = forward_reach.detect([], self.PARAMS, ctx) or fired
        self.assertFalse(fired)


class TestSpeedMetric(unittest.TestCase):
    def test_far_player_fires_in_metric_mode(self):
        """0.02 units/frame at 3 m ≈ 3.7 m/s physical — fires the mm/s
        threshold although the normalised path would have missed it."""
        p = {"min_bursts": 3, "burst_window_ms": 3000, "velocity_multiplier": 2.0,
             "idle_velocity": 0.45, "min_active_ms": 50, "use_pose": True,
             "min_speed_mm_s": 800}
        ctx, fired, x = {}, False, 0.4
        for i in range(8):
            x += 0.02 if i % 2 == 0 else -0.02
            pl = pose33({15: LM(x, 0.5, visibility=0.9),
                         16: LM(0.1, 0.5, visibility=0.9)})
            ctx["_pose_lm"] = pl
            ctx["_pose_depth"] = PoseDepth(flat_sampler(3000.0), pl)
            fired = speed_bilateral.detect([], p, ctx) or fired
            time.sleep(0.03)
        self.assertTrue(fired)


if __name__ == "__main__":
    unittest.main()
