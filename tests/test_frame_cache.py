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

    def test_pack_page_warm_is_paced_past_the_burst(self):
        """A long-span warm must throttle after the burst — an unthrottled
        sweep starves the main thread's own page faults (192ms + sustained
        80-124ms gaps in the exhibition log)."""
        import numpy as np
        from unittest import mock
        from engines import frame_cache
        from engines.frame_cache import FrameCacheManager
        cache = FrameCacheManager((8, 4))
        mm = np.zeros((frame_cache._WARM_BURST_FRAMES + 20, 4, 8, 3), np.uint8)
        sleeps = []
        with mock.patch.object(frame_cache.time, "sleep",
                               side_effect=lambda s: sleeps.append(s)):
            cache._warm_pack_pages(mm, 0, mm.shape[0] - 1)
        self.assertTrue(sleeps, "past the burst the warm must yield the disk")
        self.assertTrue(all(s <= 0.05 for s in sleeps))
        sleeps.clear()
        with mock.patch.object(frame_cache.time, "sleep",
                               side_effect=lambda s: sleeps.append(s)):
            cache._warm_pack_pages(mm, 0, frame_cache._WARM_BURST_FRAMES - 1)
        self.assertFalse(sleeps, "a short burst-sized span stays immediate")

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


class TestIncrementalBuildServe(unittest.TestCase):
    """A pack mid-build serves every frame the writer has passed — a long
    unpacked shot must not run its whole length on the decode fallback."""

    def _cache_with_building(self, tmp, n=6, done=3):
        import numpy as np
        from engines.frame_cache import FrameCacheManager
        frames = Path(tmp) / "frames"
        frames.mkdir()
        for i in range(n):
            Image.new("RGB", (64, 36), (i * 20, 0, 0)).save(
                frames / f"{i:04d}.jpg")
        cache = FrameCacheManager((64, 36))
        cache.prioritize(frames)          # registers paths (no worker thread)
        mm = np.zeros((n, 36, 64, 3), np.uint8)
        written = bytearray(n)
        for i in range(done):
            mm[i, :, :, 1] = 100 + i      # builder-written rows are green-ish
            written[i] = 1
        cache._building[frames] = {"mm": mm, "written": written,
                                   "hint": None, "count": done}
        return cache, frames

    def test_serves_written_rows_without_decoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, frames = self._cache_with_building(Path(tmp))
            decodes = []
            orig = cache._decode_frame
            cache._decode_frame = lambda p: decodes.append(p) or orig(p)
            data, size = cache.get_frame_bytes(frames, 1)
            self.assertEqual(size, (64, 36))
            self.assertEqual(data[1], 101, "must come from the building pack")
            self.assertFalse(decodes, "no fallback decode for a written row")

    def test_fallback_past_cursor_contributes_to_the_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, frames = self._cache_with_building(Path(tmp))
            data, size = cache.get_frame_bytes(frames, 5)   # beyond done=3
            self.assertEqual(size, (64, 36))
            self.assertEqual(data[0], 100, "red gradient = decoded from JPEG")
            b = cache._building[frames]
            self.assertTrue(b["written"][5],
                            "the main-thread decode must land in the pack")
            self.assertEqual(b["count"], 4)
            self.assertEqual(int(b["mm"][5, 0, 0, 0]), 100)

    def test_warm_segment_hints_the_builder_while_building(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, frames = self._cache_with_building(Path(tmp))
            cache.warm_segment(frames, 4, 5)     # unwritten region -> hint
            self.assertNotIn(frames, cache._warming,
                             "no competing decoder thread while building")
            self.assertEqual(cache._building[frames]["hint"], 4)
            cache.warm_segment(frames, 1, 2)     # already written -> no hint
            self.assertEqual(cache._building[frames]["hint"], 4)

    def test_hint_reorders_the_build_and_wraps_to_fill(self):
        from engines import frame_cache
        from engines.frame_cache import FrameCacheManager, PACK_NAME
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "frames"
            frames.mkdir()
            for i in range(6):
                Image.new("RGB", (32, 18)).save(frames / f"{i:04d}.jpg")
            cache = FrameCacheManager((32, 18))
            order = []
            real_open = frame_cache.Image.open

            def spy_open(p):
                order.append(int(Path(p).stem))
                if len(order) == 1:      # a "seek" lands mid-build
                    cache._building[frames]["hint"] = 4
                return real_open(p)

            frame_cache.Image.open = spy_open
            try:
                ok = cache._build_pack(frames / PACK_NAME,
                                       sorted(frames.glob("*.jpg")))
            finally:
                frame_cache.Image.open = real_open
            self.assertTrue(ok)
            self.assertEqual(order, [0, 4, 5, 1, 2, 3],
                             "jump to the hint, then wrap to fill the gap")

    def test_build_registers_progress_and_unregisters_on_finish(self):
        import time as _time
        from engines.frame_cache import FrameCacheManager, PACK_NAME
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "frames"
            frames.mkdir()
            for i in range(4):
                Image.new("RGB", (32, 18)).save(frames / f"{i:04d}.jpg")
            cache = FrameCacheManager((32, 18))
            ok = cache._build_pack(frames / PACK_NAME,
                                   sorted(frames.glob("*.jpg")))
            self.assertTrue(ok)
            self.assertNotIn(frames, cache._building,
                             "finished build must unregister (Windows needs "
                             "the mapping closed before os.replace)")
            self.assertTrue((frames / PACK_NAME).exists())


if __name__ == "__main__":
    unittest.main()
