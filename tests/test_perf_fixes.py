"""Contracts pinning the Aug 2026 performance batch.

Depth layer: raw-uint16 depth frames with the mm scale applied per sampled
patch (no whole-frame float32 conversion on the capture thread), and the lazy
pyorbbecsdk import (importing pose_helpers / fusion must never execute
orbbec_camera, let alone the SDK). Gesture engine: the Orbbec RGB fast path
(no RGB→BGR→RGB round trip; latest_camera_frame keeps its BGR contract by
converting lazily) and a single PoseDepth construction per dispatch even when
the player-band gate rejects. Voice engine: the Vosk model builds on the voice
thread instead of the boot path, and the recognizer only consumes audio while
a VI window is open and input is unlocked (hum DSP stays always-on).
"""

import subprocess
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from engines.depth.orbbec_camera import (ORBBEC_AVAILABLE, OrbbecGemini335,
                                         depth_at, landmark_depth_mm,
                                         try_open_orbbec)
from tests.mocks import LM, flat_sampler, pose33

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1 — depth_at over raw uint16 frames (scale applied after slicing the patch)
# ---------------------------------------------------------------------------

class TestDepthAtScale(unittest.TestCase):
    def test_uint16_frame_scaled_to_mm(self):
        frame = np.full((10, 10), 1500, dtype=np.uint16)
        self.assertEqual(depth_at(frame, 0.5, 0.5, scale=0.25), 375.0)

    def test_float_frame_default_scale_is_identity(self):
        """Old contract: an already-in-mm float frame samples unchanged."""
        frame = np.full((10, 10), 1234.5, dtype=np.float32)
        self.assertAlmostEqual(depth_at(frame, 0.5, 0.5), 1234.5, places=1)

    def test_invalid_patch_returns_none_regardless_of_scale(self):
        frame = np.zeros((10, 10), dtype=np.uint16)
        self.assertIsNone(depth_at(frame, 0.5, 0.5, scale=0.25))
        self.assertIsNone(depth_at(frame, 2.0, 0.5, scale=0.25))  # off frame

    def test_median_ignores_zero_dropouts_then_scales(self):
        frame = np.zeros((10, 10), dtype=np.uint16)
        frame[4:7, 4:7] = 2000
        frame[5, 5] = 0                      # single-pixel dropout in the patch
        self.assertEqual(depth_at(frame, 0.5, 0.5, scale=0.5), 1000.0)

    def test_landmark_depth_mm_passes_scale(self):
        frame = np.full((10, 10), 800, dtype=np.uint16)
        self.assertEqual(landmark_depth_mm(frame, LM(0.5, 0.5), scale=2.0),
                         1600.0)


# ---------------------------------------------------------------------------
# 3 — lazy pyorbbecsdk import
# ---------------------------------------------------------------------------

class TestLazyOrbbecImport(unittest.TestCase):
    def test_pose_helpers_import_does_not_touch_orbbec(self):
        """Importing the detector layer (the path every boot and test run
        pays) must not execute orbbec_camera or pyorbbecsdk. Fresh interpreter
        so this process's own imports can't mask an eager re-export."""
        code = (
            "import sys\n"
            "from engines.detectors.rules import pose_helpers\n"
            "assert 'engines.depth.fusion' in sys.modules\n"
            "assert 'engines.depth.orbbec_camera' not in sys.modules, "
            "'orbbec_camera imported eagerly via engines.depth.__init__'\n"
            "assert 'pyorbbecsdk' not in sys.modules, 'SDK imported eagerly'\n"
            "import engines.depth\n"
            "assert engines.depth.OrbbecCapture is not None\n"
            "assert 'engines.depth.orbbec_camera' in sys.modules\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0,
                         f"stdout={proc.stdout!r} stderr={proc.stderr!r}")

    def test_package_getattr_resolves_orbbec_exports(self):
        import engines.depth as depth_pkg
        self.assertIs(depth_pkg.depth_at, depth_at)
        self.assertIs(depth_pkg.OrbbecGemini335, OrbbecGemini335)
        with self.assertRaises(AttributeError):
            depth_pkg.no_such_export

    @unittest.skipIf(ORBBEC_AVAILABLE, "pyorbbecsdk installed on this host")
    def test_missing_sdk_still_degrades_gracefully(self):
        with self.assertRaises(RuntimeError):
            OrbbecGemini335()
        self.assertIsNone(try_open_orbbec())


# ---------------------------------------------------------------------------
# 2 — Orbbec RGB fast path in the gesture capture loop
# ---------------------------------------------------------------------------

class _FakePoseResults:
    pose_landmarks = None


class _FakePose:
    def __init__(self):
        self.seen = []

    def process(self, rgb):
        self.seen.append(rgb)
        return _FakePoseResults()


class _FakeCap:
    """One frame, then stops the loop."""

    def __init__(self, engine, frame, is_rgb):
        self._engine = engine
        self._frame = frame
        self._reads = 0
        if is_rgb:
            self.frame_is_rgb = True   # attribute present only on the Orbbec shim

    def read(self):
        self._reads += 1
        if self._reads == 1:
            return True, self._frame
        self._engine._capture_running = False
        return False, None


def _capture_engine():
    import threading

    from engines.gesture_engine import GestureEngine
    g = GestureEngine.__new__(GestureEngine)
    g._frame_lock = threading.Lock()
    g._pub_frame = None
    g._pub_frame_is_rgb = False
    g._pub_pose_lm = None
    g._pub_pose_time = 0.0
    g._pub_seq = 0
    g._input_locked = False
    g._warm_pose_interval = 0.0
    g._last_pose_run = 0.0
    g._capture_running = True
    g._pose = _FakePose()
    return g


def _test_frame():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 20
    frame[:, :, 2] = 30
    return frame


class TestCaptureRgbFastPath(unittest.TestCase):
    def test_rgb_source_skips_cvtcolor(self):
        g = _capture_engine()
        frame = _test_frame()
        g._cap = _FakeCap(g, frame, is_rgb=True)
        g._capture_loop()
        self.assertEqual(len(g._pose.seen), 1)
        self.assertIs(g._pose.seen[0], frame)      # fed byte-for-byte, no copy
        self.assertTrue(g._pub_frame_is_rgb)

    def test_bgr_source_still_converted(self):
        g = _capture_engine()
        frame = _test_frame()
        g._cap = _FakeCap(g, frame, is_rgb=False)  # plain cv2.VideoCapture shape
        g._capture_loop()
        self.assertEqual(len(g._pose.seen), 1)
        np.testing.assert_array_equal(g._pose.seen[0], frame[:, :, ::-1])
        self.assertFalse(g._pub_frame_is_rgb)

    def test_latest_camera_frame_converts_rgb_lazily(self):
        """The setup screen's accessor keeps its BGR contract: an RGB-published
        frame comes back channel-flipped; a BGR frame comes back untouched."""
        g = _capture_engine()
        frame = _test_frame()
        g._pub_frame = frame
        g._pub_frame_is_rgb = True
        np.testing.assert_array_equal(g.latest_camera_frame(),
                                      frame[:, :, ::-1])
        g._pub_frame_is_rgb = False
        self.assertIs(g.latest_camera_frame(), frame)   # webcam path: no copy

    def test_latest_camera_frame_none_before_first_read(self):
        g = _capture_engine()
        self.assertIsNone(g.latest_camera_frame())


# ---------------------------------------------------------------------------
# 6 — one PoseDepth per dispatch, even when the player band rejects
# ---------------------------------------------------------------------------

class TestPoseDepthSingleConstruction(unittest.TestCase):
    def _engine(self, sampler):
        from engines.gesture_engine import GestureEngine
        g = GestureEngine.__new__(GestureEngine)
        g._gesture_tuning = {}
        g._warned_missing = set()
        g._pose_stale_s = 0.5
        g._last_pose_lm = pose33()
        g._last_pose_time = time.monotonic()
        g._cap = types.SimpleNamespace(depth_mm_at=sampler)
        g._player_min_mm = 500
        g._player_max_mm = 3200
        g._phantom_behind_mm = 400
        g._player_gated = False
        return g

    def _dispatch(self, g):
        import engines.gesture_engine as ge
        calls = []
        real = ge.PoseDepth

        class Counting(real):
            def __init__(self, *a, **k):
                calls.append(1)
                super().__init__(*a, **k)

        ctx = {}
        with mock.patch.object(ge, "PoseDepth", Counting):
            g._dispatch({"type": "presence_bilateral",
                         "params": {"hold_ms": 0}}, ctx)
        return calls, ctx

    def test_band_reject_builds_one_fusion_and_gates(self):
        g = self._engine(flat_sampler(5000.0))     # torso far outside the band
        calls, ctx = self._dispatch(g)
        self.assertEqual(len(calls), 1)
        self.assertTrue(g._player_gated)
        self.assertIsNone(ctx["_pose_lm"])
        self.assertFalse(ctx["_pose_depth"].available)
        self.assertIsNone(ctx["_pose_depth"].torso_depth_mm())

    def test_in_band_builds_one_fusion_and_keeps_pose(self):
        g = self._engine(flat_sampler(1500.0))
        calls, ctx = self._dispatch(g)
        self.assertEqual(len(calls), 1)
        self.assertFalse(g._player_gated)
        self.assertIsNotNone(ctx["_pose_lm"])
        self.assertTrue(ctx["_pose_depth"].available)


# ---------------------------------------------------------------------------
# 4 + 5 — Vosk off the boot path, and gated to open windows
# ---------------------------------------------------------------------------

def _import_voice_engine():
    """voice_engine imports sounddevice at module top, which raises when
    PortAudio is absent (this venv). Stub just enough for the engine's module
    import; _run is never allowed to open a real stream in these tests."""
    try:
        import sounddevice  # noqa: F401
    except Exception:
        stub = types.ModuleType("sounddevice")
        stub.RawInputStream = None
        sys.modules["sounddevice"] = stub
    import engines.voice_engine as vem
    return vem


class _FakeRecognizer:
    def __init__(self):
        self.accepts = 0
        self.resets = 0

    def AcceptWaveform(self, chunk):
        self.accepts += 1
        return False

    def Result(self):
        return "{}"

    def Reset(self):
        self.resets += 1


def _chunk(value=0, n=400):
    return np.full(n, value, dtype=np.int16).tobytes()


def _window_cfg(window_ms=10000):
    return {"id": "freedom", "keywords": ["freedom"], "mode": "keyword",
            "tier": "cg_alternative", "window_ms": window_ms}


class TestVoskOffBootPath(unittest.TestCase):
    def test_constructor_does_not_build_vosk(self):
        vem = _import_voice_engine()
        with mock.patch.object(vem.VoiceEngine, "_init_vosk") as init:
            engine = vem.VoiceEngine({})
        init.assert_not_called()
        self.assertIsNone(engine._recognizer)

    def test_run_builds_vosk_before_opening_the_stream(self):
        vem = _import_voice_engine()
        # A stream open that raises immediately: _init_vosk can only have run
        # if it precedes the open.
        with mock.patch.object(vem.VoiceEngine, "_init_vosk") as init, \
             mock.patch.object(vem.sd, "RawInputStream",
                               mock.Mock(side_effect=RuntimeError("no mic"))):
            engine = vem.VoiceEngine({})
            init.assert_not_called()
            engine._running = False
            engine._run()
            init.assert_called_once()

    def test_missing_recognizer_tolerated_with_window_open(self):
        vem = _import_voice_engine()
        with mock.patch.object(vem.VoiceEngine, "_init_vosk"):
            engine = vem.VoiceEngine({})
        engine.open_window(_window_cfg())
        engine._process_chunk(_chunk())            # must not raise


class TestVoskWindowGate(unittest.TestCase):
    def _engine(self):
        vem = _import_voice_engine()
        with mock.patch.object(vem.VoiceEngine, "_init_vosk"):
            engine = vem.VoiceEngine({})
        engine._recognizer = _FakeRecognizer()
        return engine

    def test_no_window_skips_recognizer(self):
        engine = self._engine()
        engine._process_chunk(_chunk())
        engine._process_chunk(_chunk())
        self.assertEqual(engine._recognizer.accepts, 0)

    def test_open_window_feeds_and_resets_once(self):
        engine = self._engine()
        engine._process_chunk(_chunk())            # gated — marks idle
        engine.open_window(_window_cfg())
        engine._process_chunk(_chunk())
        engine._process_chunk(_chunk())
        self.assertEqual(engine._recognizer.resets, 1)   # only on idle→active
        self.assertEqual(engine._recognizer.accepts, 2)

    def test_input_lock_gates_recognizer(self):
        engine = self._engine()
        engine.open_window(_window_cfg())
        engine._on_input_lock({"locked": True})
        engine._process_chunk(_chunk())
        self.assertEqual(engine._recognizer.accepts, 0)

    def test_regating_resets_again_on_next_window(self):
        engine = self._engine()
        wid = engine.open_window(_window_cfg())
        engine._process_chunk(_chunk())
        engine.close_window(wid)
        engine._process_chunk(_chunk())            # gated again
        engine.open_window(_window_cfg())
        engine._process_chunk(_chunk())
        self.assertEqual(engine._recognizer.resets, 2)
        self.assertEqual(engine._recognizer.accepts, 2)

    def test_expired_window_closes_the_gate(self):
        engine = self._engine()
        wid = engine.open_window(_window_cfg(window_ms=1))
        time.sleep(0.01)
        engine._process_chunk(_chunk())
        self.assertEqual(engine._recognizer.accepts, 0)
        self.assertFalse(engine.window_open(wid))  # pruned, so pollers re-open

    def test_hum_dsp_runs_while_gated(self):
        """The RMS hum detector must stay always-on — only Vosk is gated."""
        engine = self._engine()
        engine._process_chunk(_chunk(value=8000))  # loud, no window open
        self.assertIsNotNone(engine._hum_start)
        self.assertEqual(engine._recognizer.accepts, 0)


if __name__ == "__main__":
    unittest.main()
