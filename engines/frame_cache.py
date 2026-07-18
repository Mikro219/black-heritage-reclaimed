"""
FrameCacheManager — disk-backed, RAM-bounded frame provider.

Each shot's frames are packed once into a single raw `.npy` array on disk
(`<shot>/framecache.npy`, shape (N, H, W, 3) uint8 at display resolution). On
later runs the pack is memory-mapped instead of re-decoding thousands of PNG/JPG
files — so startup and look-ahead loading cost almost nothing the second time.

Memory is bounded two ways:
  • Only a window of shots (current + `keep_ahead`) keeps an open mmap; shots
    behind / far ahead are evicted. The OS pages mmapped frames in/out on demand.
  • Frames are served as raw bytes; the render engine converts them to pygame
    Surfaces lazily and keeps only a small LRU window of Surfaces (see FrameView).

First-run fallback: until a shot's pack finishes building, frames are decoded
one at a time directly from the source images, so playback can start immediately
while the pack builds in the background.

The build streams frame-by-frame through a memmap, so packing a 5000-frame shot
never holds more than one frame in RAM.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
PACK_NAME = "framecache.npy"


class FrameCacheManager:
    def __init__(self, resolution: tuple[int, int], keep_ahead: int = 2):
        self._w, self._h = resolution
        self._keep_ahead = keep_ahead

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        self._order: list[Path] = []                 # frames dirs, sequence order
        self._paths: dict[Path, list[Path]] = {}     # frames dir -> sorted image paths
        self._packs: dict[Path, np.memmap] = {}      # frames dir -> mmap (N,H,W,3)
        self._priority: Optional[Path] = None
        self._priority_idx: int = 0
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public — main thread
    # ------------------------------------------------------------------

    def resolution(self) -> tuple[int, int]:
        return (self._w, self._h)

    def start(self, ordered_dirs) -> None:
        with self._lock:
            self._order = [Path(d) for d in ordered_dirs]
        # Sweep temp packs orphaned by a previous unclean exit (the worker was
        # mid-build when the process died, so os.replace never ran).
        for d in self._order:
            for stale in d.glob(f"{PACK_NAME}.building*"):
                try:
                    stale.unlink()
                    print(f"[FrameCache] removed stale temp pack: {stale}")
                except OSError:
                    pass
        self._thread = threading.Thread(target=self._worker, daemon=True, name="FrameCache")
        self._thread.start()
        print(f"[FrameCache] started: {len(self._order)} shots with art queued")

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    def prioritize(self, frames_dir) -> None:
        """Mark this shot current: ensure its paths are listed, evict far shots."""
        d = Path(frames_dir)
        self._ensure_paths(d)
        with self._cv:
            self._priority = d
            if d in self._order:
                self._priority_idx = self._order.index(d)
            else:
                self._order.append(d)
                self._priority_idx = len(self._order) - 1
            self._cv.notify_all()
        self._evict_outside_window()

    def is_ready(self, frames_dir) -> bool:
        """True once frames can be served (pack mmapped OR source paths listed)."""
        d = Path(frames_dir)
        with self._lock:
            return d in self._packs or bool(self._paths.get(d))

    def frame_count(self, frames_dir) -> Optional[int]:
        d = Path(frames_dir)
        with self._lock:
            mm = self._packs.get(d)
            if mm is not None:
                return mm.shape[0]
            paths = self._paths.get(d)
            return len(paths) if paths else None

    def pack_ready(self, frames_dir) -> bool:
        """True if the fast mmap pack (not just the decode fallback) is live."""
        with self._lock:
            return Path(frames_dir) in self._packs

    def get_frame_bytes(self, frames_dir, i: int):
        """Return (raw_rgb_bytes, (w, h)) for frame i, or None if unavailable.

        The current (priority) shot is never evicted while it plays, so reading its
        mmap outside the lock is safe.
        """
        d = Path(frames_dir)
        with self._lock:
            mm = self._packs.get(d)
            paths = self._paths.get(d)

        if mm is not None:
            n = mm.shape[0]
            i = max(0, min(i, n - 1))
            return mm[i].tobytes(), (self._w, self._h)

        if paths:
            i = max(0, min(i, len(paths) - 1))
            try:
                img = Image.open(paths[i]).convert("RGB").resize((self._w, self._h))
                return img.tobytes(), (self._w, self._h)
            except Exception as exc:
                print(f"[FrameCache] decode fallback failed {paths[i].name}: {exc}")
                return None
        return None

    # ------------------------------------------------------------------
    # Window / eviction
    # ------------------------------------------------------------------

    def _window(self) -> list[Path]:
        lo = self._priority_idx
        hi = min(len(self._order), self._priority_idx + self._keep_ahead + 1)
        return self._order[lo:hi]

    def _evict_outside_window(self) -> None:
        keep = set(self._window())
        with self._lock:
            for d in list(self._packs):
                if d not in keep:
                    mm = self._packs.pop(d)
                    try:
                        mm._mmap.close()
                    except Exception:
                        pass
                    print(f"[FrameCache] evicted pack {d.parent.name}")

    # ------------------------------------------------------------------
    # Worker thread — builds/mmaps packs for the window
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            with self._cv:
                if self._stop:
                    return
            target = self._next_target()
            if target is None:
                with self._cv:
                    if self._stop:
                        return
                    self._cv.wait(timeout=0.5)
                continue
            self._ensure_pack(target)

    def _next_target(self) -> Optional[Path]:
        """Next windowed shot that doesn't yet have a live mmap pack."""
        for d in self._window():
            with self._lock:
                if d not in self._packs:
                    return d
        return None

    def _pack_path(self, frames_dir: Path) -> Path:
        return frames_dir.parent / PACK_NAME

    def _ensure_paths(self, frames_dir: Path) -> list[Path]:
        with self._lock:
            cached = self._paths.get(frames_dir)
        if cached is not None:
            return cached
        try:
            paths = sorted(
                p for p in frames_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS
            )
        except OSError as exc:
            print(f"[FrameCache] cannot list {frames_dir}: {exc}")
            paths = []
        with self._lock:
            self._paths[frames_dir] = paths
        return paths

    def _ensure_pack(self, frames_dir: Path) -> None:
        paths = self._ensure_paths(frames_dir)
        n = len(paths)
        if n == 0:
            return

        pack = self._pack_path(frames_dir)
        if not pack.exists() or not self._pack_valid(pack, n):
            if not self._build_pack(pack, paths):
                return  # aborted (stop) or build failed — fallback decode stays active

        try:
            mm = np.load(pack, mmap_mode="r")
        except Exception as exc:
            print(f"[FrameCache] mmap load failed {pack.name}: {exc}")
            return

        with self._lock:
            # Only keep if still in the window (priority may have moved on).
            if frames_dir in self._window():
                self._packs[frames_dir] = mm
        print(f"[FrameCache] pack ready {frames_dir.parent.name} ({n} frames)")

    def _pack_valid(self, pack: Path, n: int) -> bool:
        try:
            mm = np.load(pack, mmap_mode="r")
            ok = (mm.shape[0] == n and mm.shape[1] == self._h and mm.shape[2] == self._w)
            del mm
            return bool(ok)
        except Exception:
            return False

    def _build_pack(self, pack: Path, paths: list[Path]) -> bool:
        """Stream frames to a memmapped .npy on disk (one frame in RAM at a time)."""
        n = len(paths)
        tmp = pack.with_suffix(".npy.building")
        print(f"[FrameCache] building pack {pack.parent.name} ({n} frames)...")
        try:
            mm = np.lib.format.open_memmap(
                tmp, mode="w+", dtype=np.uint8, shape=(n, self._h, self._w, 3)
            )
        except Exception as exc:
            print(f"[FrameCache] cannot create pack {tmp}: {exc}")
            return False

        try:
            for i, p in enumerate(paths):
                with self._lock:
                    if self._stop:
                        del mm
                        tmp.unlink(missing_ok=True)
                        return False
                try:
                    img = Image.open(p).convert("RGB").resize((self._w, self._h))
                    mm[i] = np.asarray(img, dtype=np.uint8)
                except Exception as exc:
                    print(f"[FrameCache] skip {p.name}: {exc}")
                    mm[i] = 0
            mm.flush()
            del mm
            import os
            os.replace(tmp, pack)
            return True
        except Exception as exc:
            print(f"[FrameCache] build failed {pack.name}: {exc}")
            try:
                del mm
            except Exception:
                pass
            tmp.unlink(missing_ok=True)
            return False
