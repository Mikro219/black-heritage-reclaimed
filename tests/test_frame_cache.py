"""FrameCacheManager eviction + warm contract (August 2026).

Closing a multi-GB pack mmap makes the OS tear down every resident page
mapping — measured 944ms for the shot-01 pack — and prioritize() runs on the
main thread at every shot transition, so the close must happen off-thread.
These tests pin the split: the dict pop is synchronous (no reader can obtain
an evicted mmap), the mmap close is asynchronous, and eviction never deletes
the pack file from disk.

The warm tests pin the segment-jump stall fixes: the decode fallback uses
JPEG draft-mode (DCT-domain downscale — the main thread's frame budget can't
afford a full-res decode + bicubic resample at 25-45ms/frame), and
warm_segment() pre-decodes / page-touches off-thread — non-blocking, never
raises, bounded.
"""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import engines.frame_cache as frame_cache
from engines.frame_cache import FrameCacheManager, PACK_NAME


def _make_pack(shot_dir: Path, n=2, w=4, h=4) -> Path:
    frames = shot_dir / "frames"
    frames.mkdir(parents=True)
    pack = shot_dir / PACK_NAME
    mm = np.lib.format.open_memmap(pack, mode="w+", dtype=np.uint8,
                                   shape=(n, h, w, 3))
    mm[:] = 0
    mm.flush()
    del mm
    return pack


class TestEviction(unittest.TestCase):
    def test_evict_pops_synchronously_closes_async_keeps_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packs, dirs = [], []
            for name in ("shot_01", "shot_02"):
                d = root / name
                packs.append(_make_pack(d))
                dirs.append(d / "frames")

            cache = FrameCacheManager(resolution=(4, 4), keep_ahead=0)
            cache._order = list(dirs)
            mm_old = np.load(packs[0], mmap_mode="r")
            mm_new = np.load(packs[1], mmap_mode="r")
            cache._packs = {dirs[0]: mm_old, dirs[1]: mm_new}

            # Move priority to shot_02: shot_01 leaves the window.
            cache._priority_idx = 1
            cache._evict_outside_window()

            # Pop is synchronous — the evicted mmap is unreachable at once.
            self.assertNotIn(dirs[0], cache._packs)
            self.assertIn(dirs[1], cache._packs)

            # Close happens on the background thread shortly after.
            deadline = time.monotonic() + 2.0
            while not mm_old._mmap.closed and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(mm_old._mmap.closed)

            # The kept shot's mmap is untouched, and the evicted pack FILE
            # survives on disk (eviction unmaps, never deletes).
            self.assertFalse(mm_new._mmap.closed)
            self.assertTrue(packs[0].exists())
            mm_new._mmap.close()   # let TemporaryDirectory clean up on Windows

    def test_evict_noop_spawns_no_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "shot_01"
            _make_pack(d)
            cache = FrameCacheManager(resolution=(4, 4), keep_ahead=1)
            cache._order = [d / "frames"]
            cache._priority_idx = 0
            cache._evict_outside_window()   # nothing outside the window
            self.assertEqual(cache._packs, {})


def _make_jpeg_frames(shot_dir: Path, n=3, w=640, h=360) -> Path:
    frames = shot_dir / "frames"
    frames.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (w, h), (i * 40, 80, 120)).save(
            frames / f"frame_{i:04d}.jpg", quality=85)
    return frames


def _wait_warm_idle(cache, deadline_s=5.0):
    """Block the TEST (never the runtime) until every warm thread has drained."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        with cache._lock:
            if not cache._warming:
                return
        time.sleep(0.01)
    raise AssertionError("warm thread did not drain")


class TestWarmSegment(unittest.TestCase):
    def _fallback_cache(self, root: Path, n=3):
        """Cache in decode-fallback mode (paths listed, no pack mmapped)."""
        frames = _make_jpeg_frames(root / "shot_01", n=n)
        cache = FrameCacheManager(resolution=(320, 180))
        cache._ensure_paths(frames)
        return cache, frames

    def test_fallback_decode_exact_size_rgb(self):
        # Change 1: draft-mode fallback still returns RGB bytes at exactly
        # the display resolution (640x360 source -> 320x180 target).
        with tempfile.TemporaryDirectory() as tmp:
            cache, frames = self._fallback_cache(Path(tmp))
            got = cache.get_frame_bytes(frames, 0)
            self.assertIsNotNone(got)
            data, size = got
            self.assertEqual(size, (320, 180))
            self.assertEqual(len(data), 320 * 180 * 3)

    def test_warm_predecodes_and_frame_bytes_consumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, frames = self._fallback_cache(Path(tmp))
            calls = []
            real = cache._decode_frame
            cache._decode_frame = lambda p: (calls.append(p), real(p))[1]

            cache.warm_segment(frames, 0, 1)
            _wait_warm_idle(cache)
            self.assertEqual(len(calls), 2)           # frames 0 and 1 pre-decoded
            self.assertEqual(set(cache._warm_ready[frames]), {0, 1})

            # The main-thread fetch consumes the pre-decode — no second decode —
            # and pops the entry it used.
            got = cache.get_frame_bytes(frames, 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual(got[1], (320, 180))
            self.assertEqual(len(got[0]), 320 * 180 * 3)
            self.assertNotIn(0, cache._warm_ready[frames])

            # An unwarmed frame still decodes on demand.
            cache.get_frame_bytes(frames, 2)
            self.assertEqual(len(calls), 3)

    def test_warm_is_safe_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, frames = self._fallback_cache(Path(tmp), n=3)
            # Unknown dir: never raises, warms nothing.
            cache.warm_segment(Path(tmp) / "no_such" / "frames", 0, 10)
            # Out-of-range span on a known dir: never raises, warms nothing.
            cache.warm_segment(frames, 100, 200)
            _wait_warm_idle(cache)
            self.assertEqual(cache._warm_ready.get(frames, {}), {})
            self.assertEqual(cache._warming, set())

    def test_warm_touches_live_pack_without_closing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "shot_01"
            pack = _make_pack(d, n=4)
            frames = d / "frames"
            cache = FrameCacheManager(resolution=(4, 4))
            mm = np.load(pack, mmap_mode="r")
            cache._packs = {frames: mm}
            cache.warm_segment(frames, 0, 3)      # page-touch path, span clamped
            cache.warm_segment(frames, 0, 99)     # overlap + out-of-range: no crash
            _wait_warm_idle(cache)
            self.assertFalse(mm._mmap.closed)     # warming never evicts/closes
            self.assertEqual(cache._warm_ready, {})   # no fallback dict entries
            mm._mmap.close()   # let TemporaryDirectory clean up on Windows

    def test_warm_ready_dict_is_bounded(self):
        # The per-dir ready dict can never exceed _WARM_READY_CAP, even when a
        # warm span asks for more (caps patched down so the test stays fast).
        old_max, old_cap = frame_cache._WARM_DECODE_MAX, frame_cache._WARM_READY_CAP
        frame_cache._WARM_DECODE_MAX, frame_cache._WARM_READY_CAP = 100, 4
        self.addCleanup(lambda: setattr(frame_cache, "_WARM_DECODE_MAX", old_max))
        self.addCleanup(lambda: setattr(frame_cache, "_WARM_READY_CAP", old_cap))
        with tempfile.TemporaryDirectory() as tmp:
            cache, frames = self._fallback_cache(Path(tmp), n=10)
            cache.warm_segment(frames, 0, 9)
            _wait_warm_idle(cache)
            self.assertLessEqual(len(cache._warm_ready[frames]), 4)


if __name__ == "__main__":
    unittest.main()
