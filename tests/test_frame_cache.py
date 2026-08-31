"""FrameCacheManager eviction contract (August 2026).

Closing a multi-GB pack mmap makes the OS tear down every resident page
mapping — measured 944ms for the shot-01 pack — and prioritize() runs on the
main thread at every shot transition, so the close must happen off-thread.
These tests pin the split: the dict pop is synchronous (no reader can obtain
an evicted mmap), the mmap close is asynchronous, and eviction never deletes
the pack file from disk.
"""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
