"""Contracts pinning the July 2026 codebase-review fixes.

Detector layer: depth/visibility phantom gates (touch_head, forward_point,
spread_arms), rolling-window accumulators with dropout resets
(bilateral_rotation / alternating / arcing), the bilateral_sweep side lock,
and a first arms_crossed contract. Runtime layer: voice events queue instead
of mutating the FSM on the voice thread; the gesture engine disarms on
input_lock and survives pause/resume without expiring its OI window.
"""

import math
import threading
import unittest
from pathlib import Path

from engines.depth.fusion import PoseDepth
from engines.detectors.rules import (arms_crossed, bilateral_alternating,
                                     bilateral_arcing, bilateral_rotation,
                                     bilateral_sweep, forward_point,
                                     spread_arms, touch_head, unravel)
from tests.mocks import LM, pose33, zone_sampler, Bus

ROOT = Path(__file__).resolve().parent.parent


def ctx_for(pose):
    return {"_pose_lm": pose}


# ---------------------------------------------------------------------------
# 1.5 — phantom gates
# ---------------------------------------------------------------------------

class TestTouchHeadGates(unittest.TestCase):
    # pose33 head: ears y=0.28, shoulders y=0.40 -> crown ~(0.5, 0.232) r>=0.07

    def test_visible_wrist_at_crown_fires(self):
        pose = pose33({15: LM(0.50, 0.22, visibility=0.9)})
        self.assertTrue(touch_head.detect([], {"hold_ms": 0}, ctx_for(pose)))

    def test_low_visibility_wrist_at_crown_ignored(self):
        """An occluded wrist's estimated position inside the crown circle must
        not fire — Pose reports positions for landmarks it can't see."""
        pose = pose33({15: LM(0.50, 0.22, visibility=0.2)})
        self.assertFalse(touch_head.detect([], {"hold_ms": 0}, ctx_for(pose)))

    def test_depth_phantom_wrist_vetoed(self):
        pose = pose33({15: LM(0.50, 0.22, visibility=0.9)})
        ctx = ctx_for(pose)
        # wrist zone samples 3km behind the 1.5m torso -> phantom veto
        ctx["_pose_depth"] = PoseDepth(
            zone_sampler([(0.45, 0.55, 0.15, 0.30, 4500.0)], 1500.0), pose)
        self.assertFalse(touch_head.detect([], {"hold_ms": 0}, ctx))


class TestForwardPointGate(unittest.TestCase):
    def _pose(self):
        # right wrist reaching into the top-right player quadrant (raw x 0.3)
        return pose33({16: LM(0.30, 0.30, visibility=0.9, z=-0.4)})

    PARAMS = {"target_region": "top_right_quadrant", "hold_ms": 0,
              "min_pose_z_delta": 0.25}

    def test_reaching_wrist_in_region_fires(self):
        self.assertTrue(forward_point.detect([], self.PARAMS, ctx_for(self._pose())))

    def test_depth_phantom_wrist_vetoed(self):
        pose = self._pose()
        ctx = ctx_for(pose)
        ctx["_pose_depth"] = PoseDepth(
            zone_sampler([(0.25, 0.35, 0.25, 0.35, 4500.0)], 1500.0), pose)
        self.assertFalse(forward_point.detect([], self.PARAMS, ctx))


class TestSpreadArmsGate(unittest.TestCase):
    def _pose(self, l_vis=0.9):
        return pose33({15: LM(0.78, 0.42, visibility=l_vis),
                       16: LM(0.22, 0.42, visibility=0.9)})

    def test_wingspan_fires(self):
        self.assertTrue(spread_arms.detect([], {"hold_ms": 0},
                                           ctx_for(self._pose())))

    def test_low_visibility_wrist_blocks(self):
        self.assertFalse(spread_arms.detect([], {"hold_ms": 0},
                                            ctx_for(self._pose(l_vis=0.2))))

    def test_depth_phantom_wrist_vetoed(self):
        pose = self._pose()
        ctx = ctx_for(pose)
        ctx["_pose_depth"] = PoseDepth(
            zone_sampler([(0.7, 0.9, 0.35, 0.5, 4500.0)], 1500.0), pose)
        self.assertFalse(spread_arms.detect([], {"hold_ms": 0}, ctx))


# ---------------------------------------------------------------------------
# 1.6 — windowed accumulators / side lock
# ---------------------------------------------------------------------------

def _rotating_pose(theta_l, theta_r, r=0.08):
    """Both wrists tracked, index tips at angle theta around each wrist."""
    lw, rw = (0.65, 0.60), (0.35, 0.60)
    return pose33({
        15: LM(*lw, visibility=0.9),
        16: LM(*rw, visibility=0.9),
        19: LM(lw[0] + r * math.cos(theta_l), lw[1] + r * math.sin(theta_l),
               visibility=0.9),
        20: LM(rw[0] + r * math.cos(theta_r), rw[1] + r * math.sin(theta_r),
               visibility=0.9),
    })


class TestBilateralRotation(unittest.TestCase):
    # legacy angle-sweep model (the elbow-line crossing model is the default
    # since July 2026 — its contract is TestBilateralRotationElbowLine below)
    PARAMS = {"min_cycles": 2, "mode": "sweep"}

    def test_deliberate_cycles_fire(self):
        ctx, fired = {}, False
        for i in range(18):                      # 2 full cycles in pi/4 steps
            theta = i * (math.pi / 4)
            ctx["_pose_lm"] = _rotating_pose(theta, theta)
            fired = bilateral_rotation.detect([], self.PARAMS, ctx) or fired
        self.assertTrue(fired)

    def test_jitter_never_accumulates(self):
        """Back-and-forth wiggle sums to ~zero NET rotation — the old
        abs(delta)-forever accumulator fired from tremor alone."""
        ctx, fired = {}, False
        for i in range(200):
            theta = 0.4 if i % 2 == 0 else -0.4
            ctx["_pose_lm"] = _rotating_pose(theta, theta)
            fired = bilateral_rotation.detect([], self.PARAMS, ctx) or fired
        self.assertFalse(fired)

    def test_dropout_resets_angle_state(self):
        ctx = {}
        ctx["_pose_lm"] = _rotating_pose(0.0, 0.0)
        bilateral_rotation.detect([], self.PARAMS, ctx)
        ctx["_pose_lm"] = pose33({15: LM(0.5, 0.5, visibility=0.1),
                                  16: LM(0.5, 0.5, visibility=0.1)})
        bilateral_rotation.detect([], self.PARAMS, ctx)
        self.assertEqual(ctx.get("rotation_prev_angle"), {})
        self.assertEqual(ctx.get("rotation_history"), {})


def _winding_pose(y_l, y_r):
    """Both wrists overridden; elbows stay at the pose33 default y=0.62, so
    the elbow-elbow line is horizontal at 0.62 (shoulder width 0.2 →
    default hysteresis band ±0.024 around it)."""
    return pose33({15: LM(0.68, y_l, visibility=0.9),
                   16: LM(0.32, y_r, visibility=0.9)})


class TestBilateralRotationElbowLine(unittest.TestCase):
    """Default model: each hand must oscillate across the elbow-elbow line."""
    PARAMS = {"min_cycles": 2}

    def _run(self, frames):
        ctx, fired = {}, False
        for y_l, y_r in frames:
            ctx["_pose_lm"] = _winding_pose(y_l, y_r)
            fired = bilateral_rotation.detect([], self.PARAMS, ctx) or fired
        return ctx, fired

    def test_oscillation_across_line_fires(self):
        # 4 crossings per hand = 2 oscillations = min_cycles
        _, fired = self._run([(0.50, 0.50), (0.74, 0.74), (0.50, 0.50),
                              (0.74, 0.74), (0.50, 0.50)])
        self.assertTrue(fired)

    def test_jitter_inside_band_never_fires(self):
        # ±0.01 around the line, inside the ±0.024 hysteresis band
        frames = [(0.61, 0.61) if i % 2 == 0 else (0.63, 0.63)
                  for i in range(200)]
        ctx, fired = self._run(frames)
        self.assertFalse(fired)
        self.assertEqual(ctx.get("rotation_crossings"), {})

    def test_single_hand_oscillation_never_fires(self):
        # right hand parked above the line while the left circles
        frames = [(0.50 if i % 2 == 0 else 0.74, 0.50) for i in range(12)]
        _, fired = self._run(frames)
        self.assertFalse(fired)

    def test_elbow_dropout_resets_state(self):
        ctx, _ = self._run([(0.50, 0.50), (0.74, 0.74), (0.50, 0.50)])
        self.assertTrue(any(ctx.get("rotation_crossings", {}).values()))
        ctx["_pose_lm"] = pose33({13: LM(0.63, 0.62, visibility=0.1),
                                  15: LM(0.68, 0.50, visibility=0.9),
                                  16: LM(0.32, 0.50, visibility=0.9)})
        self.assertFalse(bilateral_rotation.detect([], self.PARAMS, ctx))
        self.assertEqual(ctx.get("rotation_zone"), {})
        self.assertEqual(ctx.get("rotation_crossings"), {})


def _unravel_pose(y, x_l=0.51, x_r=0.49):
    """Both wrists close together (prox OK), circling IN PHASE at height y.
    Elbows stay at the pose33 default y=0.62 → the elbow line sits there,
    default hysteresis band ±0.016 around it."""
    return pose33({15: LM(x_l, y, visibility=0.9),
                   16: LM(x_r, y, visibility=0.9)})


class TestUnravelElbowLine(unittest.TestCase):
    """Scene 11 unravel: per-wrist oscillation across the elbow-elbow line."""
    PARAMS = {"min_cycles": 2}

    def _run(self, ys, ctx=None):
        ctx = {} if ctx is None else ctx
        fired = False
        for y in ys:
            ctx["_pose_lm"] = _unravel_pose(y)
            fired = unravel.detect([], self.PARAMS, ctx) or fired
        return ctx, fired

    def test_in_phase_circles_fire(self):
        """Both hands circling in parallel — the old diff-signal model was
        blind to this (left.y - right.y stays flat, amplitude gate blocked
        forever); elbow-line crossings count each wrist independently."""
        ctx, fired = self._run([0.58, 0.66, 0.58, 0.66, 0.58])
        self.assertTrue(fired)
        self.assertTrue(ctx.get("unravel_fired"))

    def test_one_shot_latch(self):
        ctx, fired = self._run([0.58, 0.66, 0.58, 0.66, 0.58])
        self.assertTrue(fired)
        _, again = self._run([0.66, 0.58, 0.66, 0.58, 0.66], ctx)
        self.assertFalse(again)

    def test_jitter_inside_band_never_fires(self):
        ys = [0.615 if i % 2 == 0 else 0.625 for i in range(200)]
        ctx, fired = self._run(ys)
        self.assertFalse(fired)
        self.assertEqual(ctx.get("unravel_crossings"), {})

    def test_hands_apart_outward_rotation_fires_by_default(self):
        """The scripted gesture holds the hands APART (rotating outward).
        The old default prox_frac 0.25 required near-touching wrists and
        silently blocked the experience (the tuner's laptop_dev override
        masked it); the relaxed 1.75 default must let a hands-apart
        oscillation (~1.2 shoulder-widths) fire."""
        ctx = {}
        fired = False
        for y in (0.58, 0.66, 0.58, 0.66, 0.58):
            ctx["_pose_lm"] = _unravel_pose(y, x_l=0.62, x_r=0.38)
            fired = unravel.detect([], self.PARAMS, ctx) or fired
        self.assertTrue(fired)

    def test_wrists_apart_resets(self):
        """Beyond even the relaxed gate (2× shoulder width) state clears."""
        ctx, _ = self._run([0.58, 0.66, 0.58])
        self.assertTrue(any(ctx.get("unravel_crossings", {}).values()))
        ctx["_pose_lm"] = pose33({15: LM(0.70, 0.58, visibility=0.9),
                                  16: LM(0.30, 0.58, visibility=0.9)})
        self.assertFalse(unravel.detect([], self.PARAMS, ctx))
        self.assertEqual(ctx.get("unravel_crossings"), {})


def _arm(side, wrist_x, wrist_y=0.40, extended=True):
    """Landmark overrides for one straight (or bent) arm."""
    if side == "L":
        sh, el, wr = 11, 13, 15
        sx = 0.60
    else:
        sh, el, wr = 12, 14, 16
        sx = 0.40
    if extended:
        ex, ey = (sx + wrist_x) / 2, wrist_y
    else:
        ex, ey = sx + 0.02, 0.62            # elbow dropped: low reach fraction
    return {sh: LM(sx, 0.40), el: LM(ex, ey),
            wr: LM(wrist_x, wrist_y, visibility=0.9)}


def _l_only(x, **extra):
    """Left arm extended at raw x; right wrist untracked (a hanging straight
    arm scores a high reach fraction, so it must be out of play)."""
    over = {**_arm("L", x), 16: LM(0.35, 0.85, visibility=0.2)}
    over.update(extra)
    return pose33(over)


class TestBilateralSweepSideLock(unittest.TestCase):
    PARAMS = {"min_screen_fraction": 0.33, "direction": "left_to_right",
              "min_reach_frac": 0.6}

    def test_hand_swap_does_not_fire(self):
        """Old bug: 'most extended wrist' re-chosen per frame, so travel was
        measured between two different physical hands."""
        ctx = {}
        # L extended at x=0.30 starts the sweep
        ctx["_pose_lm"] = _l_only(0.30)
        self.assertFalse(bilateral_sweep.detect([], self.PARAMS, ctx))
        self.assertEqual(ctx.get("sweep_side"), "L")
        # dominance flips: L bends, R extends far right — must NOT count as travel
        ctx["_pose_lm"] = pose33({**_arm("L", 0.55, extended=False),
                                  **_arm("R", 0.90)})
        self.assertFalse(bilateral_sweep.detect([], self.PARAMS, ctx))

    def test_single_hand_sweep_fires(self):
        ctx, fired = {}, False
        # stays left of the shoulder (0.6) so the arm keeps a real extension
        for x in (0.15, 0.28, 0.42, 0.55):
            ctx["_pose_lm"] = _l_only(x)
            fired = bilateral_sweep.detect([], self.PARAMS, ctx) or fired
        self.assertTrue(fired)


class TestBilateralAlternatingReset(unittest.TestCase):
    PARAMS = {"min_cycles": 2}

    def _pose(self, lx, rx):
        return pose33({15: LM(lx, 0.6, visibility=0.9),
                       16: LM(rx, 0.6, visibility=0.9)})

    def test_alternating_crossings_fire(self):
        ctx, fired = {}, False
        # midline x = 0.5; alternate hands across it
        seq = [(0.45, 0.35), (0.55, 0.35), (0.55, 0.55), (0.45, 0.55),
               (0.55, 0.55), (0.55, 0.45)]
        for lx, rx in seq:
            ctx["_pose_lm"] = self._pose(lx, rx)
            fired = bilateral_alternating.detect([], self.PARAMS, ctx) or fired
        self.assertTrue(fired)

    def test_dropout_clears_state(self):
        ctx = {}
        ctx["_pose_lm"] = self._pose(0.45, 0.35)
        bilateral_alternating.detect([], self.PARAMS, ctx)
        ctx["_pose_lm"] = pose33({15: LM(0.5, 0.5, visibility=0.1),
                                  16: LM(0.5, 0.5, visibility=0.1)})
        bilateral_alternating.detect([], self.PARAMS, ctx)
        self.assertEqual(ctx.get("alt_prev_x"), {})
        self.assertEqual(ctx.get("alt_strokes"), [])


class TestBilateralArcingReset(unittest.TestCase):
    def _pose(self, y):
        return pose33({15: LM(0.65, y, visibility=0.9),
                       16: LM(0.35, y, visibility=0.9)})

    def test_strokes_fire(self):
        ctx, fired = {}, False
        params = {"strokes": 2}
        for y in (0.40, 0.70, 0.40, 0.70, 0.40):   # two down-then-up cycles
            ctx["_pose_lm"] = self._pose(y)
            fired = bilateral_arcing.detect([], params, ctx) or fired
        self.assertTrue(fired)

    def test_dropout_clears_phase(self):
        ctx = {}
        ctx["_pose_lm"] = self._pose(0.40)
        bilateral_arcing.detect([], {"strokes": 2}, ctx)
        ctx["_pose_lm"] = pose33({15: LM(0.5, 0.5, visibility=0.1),
                                  16: LM(0.5, 0.5, visibility=0.1)})
        bilateral_arcing.detect([], {"strokes": 2}, ctx)
        self.assertEqual(ctx.get("arc_phase"), {})


class TestArmsCrossed(unittest.TestCase):
    def test_crossed_wrists_at_torso_fire(self):
        # midline 0.5: left wrist (15) crosses to LOW raw x, right (16) to HIGH
        pose = pose33({15: LM(0.38, 0.55, visibility=0.9),
                       16: LM(0.62, 0.55, visibility=0.9)})
        self.assertTrue(arms_crossed.detect([], {"hold_ms": 0}, ctx_for(pose)))

    def test_uncrossed_never_fires(self):
        pose = pose33({15: LM(0.65, 0.55, visibility=0.9),
                       16: LM(0.35, 0.55, visibility=0.9)})
        self.assertFalse(arms_crossed.detect([], {"hold_ms": 0}, ctx_for(pose)))

    def test_crossed_above_shoulders_rejected(self):
        pose = pose33({15: LM(0.38, 0.20, visibility=0.9),
                       16: LM(0.62, 0.20, visibility=0.9)})
        self.assertFalse(arms_crossed.detect([], {"hold_ms": 0}, ctx_for(pose)))


# ---------------------------------------------------------------------------
# 1.1 — voice events marshalled to the main thread
# ---------------------------------------------------------------------------

def _synthetic_shots():
    """A minimal playback shot — these tests exercise the player's voice-event
    marshalling, not any scene content."""
    from engines.sequence_loader import Shot
    from pathlib import Path
    return [Shot(shot="01", act="01", kind="playback", audio_lines=[],
                 tracker_type="", tracker_notes="", reuse_of=None,
                 segments=None, interaction=None,
                 fallback={"timeout_s": 300, "reprompt_s": [],
                           "on_timeout": "auto_advance"},
                 play_if=None, fps=30, timing_profile="standard",
                 frames_dir=Path("frames"), audio_dir=None, audio_file=None,
                 audio_events=[], captions=[], assets_pending=False,
                 segments_todo=False, interaction_todo=False,
                 reuse_self=False)]


class TestVoiceEventMarshalling(unittest.TestCase):
    def test_vi_detected_queues_and_drains_on_update(self):
        from engines.shot_sequence_player import ShotSequencePlayer
        bus = Bus()
        player = ShotSequencePlayer(_synthetic_shots(), {"timing_defaults": {}}, bus)
        player.start(0)

        state_before = player._shot_state
        err = []

        def fire_from_thread():
            try:
                bus.emit("vi_detected", {"voice_id": "voice_go", "tier": "cg"})
            except Exception as exc:            # pragma: no cover
                err.append(exc)

        t = threading.Thread(target=fire_from_thread)
        t.start()
        t.join()
        self.assertFalse(err)
        # The handler must ONLY queue — no state mutation on the voice thread.
        self.assertEqual(len(player._pending_vi), 1)
        self.assertEqual(player._shot_state, state_before)

        player.update()                          # main thread drains the queue
        self.assertEqual(len(player._pending_vi), 0)

    def test_paused_player_drops_voice_events(self):
        from engines.shot_sequence_player import ShotSequencePlayer
        player = ShotSequencePlayer(_synthetic_shots(), {"timing_defaults": {}},
                                    Bus())
        player.start(0)
        player.pause()
        player._on_vi_detected({"voice_id": "voice_go"})
        self.assertEqual(len(player._pending_vi), 0)


class TestWarmPoseWhileLocked(unittest.TestCase):
    """Pose is kept warm at a trickle while input is locked.

    Skipping inference entirely during locked playback meant the first
    inference after a lock ran cold (heavy detector + TFLite re-warm), landing
    a frame hitch on the exact frame each gesture prompt appeared."""

    def _engine(self, hz=3.0):
        from engines.gesture_engine import GestureEngine
        g = GestureEngine.__new__(GestureEngine)
        g._input_locked = False
        g._warm_pose_interval = 0.0 if hz <= 0 else 1.0 / hz
        g._last_pose_run = 0.0
        return g

    def test_unlocked_runs_every_frame(self):
        g = self._engine()
        for t in (10.0, 10.01, 10.02):
            self.assertTrue(g._should_run_pose(t))

    def test_locked_runs_at_the_warm_rate(self):
        g = self._engine(hz=3.0)          # one every ~0.333s
        g._should_run_pose(100.0)         # unlocked, stamps the clock
        g._input_locked = True
        self.assertFalse(g._should_run_pose(100.1))   # too soon
        self.assertFalse(g._should_run_pose(100.3))
        self.assertTrue(g._should_run_pose(100.4))    # due
        self.assertFalse(g._should_run_pose(100.5))   # and re-armed

    def test_unlocking_runs_immediately(self):
        """The frame a window opens must not wait out the warm interval."""
        g = self._engine()
        g._input_locked = True
        g._should_run_pose(200.0)
        g._input_locked = False
        self.assertTrue(g._should_run_pose(200.001))

    def test_zero_hz_restores_skip_entirely(self):
        g = self._engine(hz=0)
        g._input_locked = True
        for t in (300.0, 305.0, 900.0):
            self.assertFalse(g._should_run_pose(t))
        g._input_locked = False
        self.assertTrue(g._should_run_pose(901.0))


if __name__ == "__main__":
    unittest.main()
