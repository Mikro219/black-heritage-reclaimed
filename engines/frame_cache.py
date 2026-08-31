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
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
PACK_NAME = "framecache.npy"

_WARM_DECODE_MAX = 25   # frames pre-decoded per warm_segment call (fallback mode)
_WARM_READY_CAP = 48    # per-dir bound on the pre-decoded ready-bytes dict
_WARM_BURST_FRAMES = 60  # pack-page warm: immediate burst (~2s of frames) ...
_WARM_RATE_FPS = 45.0    # ... then paced at 1.5x playback so disk keeps headroom


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

        self._warm_ready: dict[Path, dict[int, bytes]] = {}  # dir -> {i: rgb bytes} (fallback pre-decodes)
        self._warming: set[Path] = set()                     # dirs with a warm thread in flight

        # Packs mid-build, served INCREMENTALLY:
        #   dir -> {"mm": writer memmap, "written": bytearray(n), "hint": int|None}
        # Readers copy any written frame UNDER the lock and never keep a
        # reference — the builder must be able to close the mapping and
        # os.replace the .building file (Windows refuses the replace while
        # any mapping is open). Without this, a long unpacked shot (scene_01
        # is 9k frames) ran its ENTIRE length on the 30-45ms/frame decode
        # fallback while the build ground on — the first-OI-window freeze.
        # "hint" lets the playhead STEER the build order: a prologue skip
        # jumps thousands of frames past the cursor, and a builder that keeps
        # grinding the skipped region leaves the main thread decoding alone
        # (exhibition log: 20fps pacing + 81-145ms gaps at the first OI after
        # a skip). warm_segment sets the hint; the builder jumps there, then
        # circles back for the skipped rows before finalising.
        self._building: dict[Path, dict] = {}

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

    def get_frame_buffer(self, frames_dir, i: int):
        """Zero-copy view of frame i's raw RGB, or None when no pack is live.

        Returns (buffer, (w, h)) where buffer is the frame's C-contiguous slice
        of the pack mmap — hand it straight to pygame.image.frombuffer(...)
        .convert() and DROP it; it aliases the mmap, which eviction closes.
        The priority shot's pack is never evicted while it plays, so the wrap+
        convert window is safe. Fallback (no pack) callers use get_frame_bytes.
        """
        d = Path(frames_dir)
        with self._lock:
            mm = self._packs.get(d)
        if mm is None:
            return None
        i = max(0, min(i, mm.shape[0] - 1))
        return mm[i], (self._w, self._h)

    def get_frame_bytes(self, frames_dir, i: int):
        """Return (raw_rgb_bytes, (w, h)) for frame i, or None if unavailable.

        The current (priority) shot is never evicted while it plays, so reading its
        mmap outside the lock is safe.
        """
        d = Path(frames_dir)
        with self._lock:
            mm = self._packs.get(d)
            paths = self._paths.get(d)
            if mm is None:
                # Pack still building: serve every frame the writer has
                # already passed. Copy under the lock — the reference must
                # not outlive it (see _building above).
                b = self._building.get(d)
                if (b is not None and 0 <= i < len(b["written"])
                        and b["written"][i]):
                    return bytes(b["mm"][i]), (self._w, self._h)

        if mm is not None:
            n = mm.shape[0]
            i = max(0, min(i, n - 1))
            return mm[i].tobytes(), (self._w, self._h)

        if paths:
            i = max(0, min(i, len(paths) - 1))
            # A warm_segment() pre-decode may already hold this frame — use it
            # (pop, so the dict stays small) instead of decoding on the main thread.
            with self._lock:
                ready = self._warm_ready.get(d)
                data = ready.pop(i, None) if ready else None
            if data is not None:
                return data, (self._w, self._h)
            try:
                data = self._decode_frame(paths[i])
            except Exception as exc:
                print(f"[FrameCache] decode fallback failed {paths[i].name}: {exc}")
                return None
            # Contribute the decode to a pack mid-build: when the playhead and
            # the builder are neck-and-neck, every frame the playhead wins
            # would otherwise be decoded TWICE (once here, once by the
            # builder catching up).
            with self._lock:
                b = self._building.get(d)
                if (b is not None and i < len(b["written"])
                        and not b["written"][i]):
                    try:
                        b["mm"][i] = np.frombuffer(
                            data, np.uint8).reshape(self._h, self._w, 3)
                        b["written"][i] = 1
                        b["count"] += 1
                    except Exception:
                        pass   # size mismatch etc. — builder will redo it
            return data, (self._w, self._h)
        return None

    def warm_segment(self, frames_dir, start_idx: int, end_idx: int) -> None:
        """Warm frames [start_idx, end_idx] (inclusive; the SAME 0-based indices
        get_frame_bytes serves) ahead of the render loop. Non-blocking, never raises.

        Pack live: a throwaway thread touches the frames' mmap pages so a segment
        jump doesn't stall the render loop on cold disk-bound page faults. No pack
        yet (decode fallback — including while a .building pack is mid-build): the
        thread pre-decodes up to _WARM_DECODE_MAX frames into a small ready-bytes
        dict that get_frame_bytes consumes. Unknown dir, empty span, or a warm
        already in flight for the dir: safe no-op.
        """
        try:
            d = Path(frames_dir)
            with self._lock:
                mm = self._packs.get(d)
                paths = self._paths.get(d)
                if mm is None and not paths:
                    return                    # unknown dir — nothing to warm
                b = self._building.get(d) if mm is None else None
                if b is not None:
                    # Steer the builder to the played region instead of
                    # spawning a competing decoder: after a seek (prologue
                    # skip) the sequential cursor can be thousands of frames
                    # behind the playhead.
                    if not (0 <= start_idx < len(b["written"])
                            and b["written"][start_idx]):
                        b["hint"] = max(0, int(start_idx))
                    return
                if d in self._warming:
                    return                    # one warm per dir — no thread pile-up
                self._warming.add(d)

            def _warm():
                try:
                    if mm is not None:
                        self._warm_pack_pages(mm, start_idx, end_idx)
                    else:
                        self._warm_decode(d, paths, start_idx, end_idx)
                except Exception as exc:
                    print(f"[FrameCache] warm_segment failed {d.parent.name}: {exc}")
                finally:
                    with self._lock:
                        self._warming.discard(d)

            threading.Thread(target=_warm, daemon=True,
                             name="FrameCacheWarm").start()
        except Exception:
            with self._lock:
                self._warming.discard(Path(frames_dir))

    # ------------------------------------------------------------------
    # Warm helpers — run on the throwaway warm thread
    # ------------------------------------------------------------------

    def _decode_frame(self, path: Path) -> bytes:
        """Decode one source image to raw RGB bytes at display resolution.

        draft() lets the JPEG decoder downscale in the DCT domain during the
        decode itself (e.g. 1920 -> 960 for a 1280x720 target) — this path runs
        on the MAIN thread's frame budget, where the old full-res decode +
        bicubic resample cost 25-45ms/frame. A cheap BILINEAR pass then fixes
        the exact size. draft() is a no-op for PNGs.
        """
        img = Image.open(path)
        img.draft("RGB", (self._w, self._h))
        img = img.convert("RGB")
        if img.size != (self._w, self._h):
            img = img.resize((self._w, self._h), Image.BILINEAR)
        return img.tobytes()

    def _warm_pack_pages(self, mm: np.memmap, start_idx: int, end_idx: int) -> None:
        """Fault the frames' mmap pages in ahead of the render loop.

        Reads one byte per row into a throwaway: rows stride w*3 bytes apart
        (< the 4KB page at any display width), so every page a frame spans gets
        touched. The mmap reference was grabbed under the lock by the caller;
        the page touches deliberately run outside it.
        """
        lo = max(0, start_idx)
        hi = min(mm.shape[0] - 1, end_idx)
        # PACED: an unthrottled sweep of a long segment (~2.7GB for a 980-frame
        # span at 720p) saturates the disk and STARVES the main thread's own
        # page faults — exhibition log: a 192ms render stall at segment entry
        # then 80-124ms gaps for ~5s while the sweep streamed. Touch a short
        # burst immediately, then stay ahead of the playhead at ~1.5x playback
        # rate so the disk keeps headroom for the render loop.
        burst = _WARM_BURST_FRAMES
        rate = _WARM_RATE_FPS
        t0 = time.monotonic()
        sink = 0
        for k, i in enumerate(range(lo, hi + 1)):
            with self._lock:
                if self._stop:
                    return
            sink += int(mm[i, :, 0, 0].sum())
            if k >= burst:
                target = t0 + (k - burst) / rate
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, 0.05))

    def _warm_decode(self, d: Path, paths: list[Path],
                     start_idx: int, end_idx: int) -> None:
        """Pre-decode a short run of fallback frames into the ready-bytes dict."""
        lo = max(0, start_idx)
        hi = min(len(paths) - 1, end_idx, lo + _WARM_DECODE_MAX - 1)
        if lo > hi:
            return
        with self._lock:
            # A fresh warm supersedes stale pre-decodes for this dir — the
            # render loop has jumped to a new segment.
            self._warm_ready[d] = {}
        for i in range(lo, hi + 1):
            with self._lock:
                if self._stop:
                    return
                ready = self._warm_ready.get(d)
                if ready is None or len(ready) >= _WARM_READY_CAP:
                    return
            try:
                data = self._decode_frame(paths[i])
            except Exception as exc:
                print(f"[FrameCache] warm decode skip {paths[i].name}: {exc}")
                continue
            with self._lock:
                ready = self._warm_ready.get(d)
                if ready is not None and len(ready) < _WARM_READY_CAP:
                    ready[i] = data

    # ------------------------------------------------------------------
    # Window / eviction
    # ------------------------------------------------------------------

    def _window(self) -> list[Path]:
        lo = self._priority_idx
        hi = min(len(self._order), self._priority_idx + self._keep_ahead + 1)
        return self._order[lo:hi]

    def _evict_outside_window(self) -> None:
        """Drop packs behind/far ahead of the priority shot.

        The dict pop happens here (cheap, under the lock, so no reader can
        obtain an evicted mmap), but the actual mmap CLOSE runs on a throwaway
        background thread: unmapping a multi-GB pack makes the OS tear down
        every resident page mapping, which measured ~1ms/frame-in-residence —
        944ms for the shot-01 pack — and used to land as a picture freeze at
        every shot transition (prioritize() runs on the main thread). Only the
        main thread reads packs, and only the priority shot's (always in the
        window), so nothing can touch an mmap once it leaves ``_packs``.
        Eviction never deletes the .npy file — it is re-mmapped next visit.
        """
        keep = set(self._window())
        stale: list[tuple[Path, np.memmap]] = []
        with self._lock:
            for d in list(self._packs):
                if d not in keep:
                    stale.append((d, self._packs.pop(d)))
        if not stale:
            return

        def _close(items=stale):
            for d, mm in items:
                t0 = time.perf_counter()
                try:
                    mm._mmap.close()
                except Exception:
                    pass
                print(f"[FrameCache] evicted pack {d.parent.name} "
                      f"(unmapped off-thread in "
                      f"{(time.perf_counter() - t0) * 1000:.0f}ms; "
                      f"file kept on disk)")

        threading.Thread(target=_close, daemon=True,
                         name="FrameCacheEvict").start()

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
            # Fallback pre-decodes are dead weight once the pack serves frames.
            self._warm_ready.pop(frames_dir, None)
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

        d = pack.parent
        written = bytearray(n)
        state = {"mm": mm, "written": written, "hint": None, "count": 0}
        with self._lock:
            self._building[d] = state   # serve finished rows immediately
        try:
            pos = 0
            while True:
                with self._lock:
                    if self._stop:
                        self._building.pop(d, None)
                        del mm
                        tmp.unlink(missing_ok=True)
                        return False
                    if state["count"] >= n:
                        break   # main-thread contributions can finish it too
                    # The playhead steers the order: jump to a seek target so
                    # the main thread never decodes alone; skipped rows are
                    # filled on the wrap-around before the pack finalises.
                    hint = state["hint"]
                    if hint is not None:
                        state["hint"] = None
                        if 0 <= hint < n and not written[hint]:
                            pos = hint
                if pos >= n:
                    pos = 0
                while written[pos]:
                    pos += 1
                    if pos >= n:
                        pos = 0
                p = paths[pos]
                try:
                    img = Image.open(p).convert("RGB").resize((self._w, self._h))
                    row = np.asarray(img, dtype=np.uint8)
                except Exception as exc:
                    print(f"[FrameCache] skip {p.name}: {exc}")
                    row = None
                with self._lock:
                    if not written[pos]:   # the main thread may have beaten us
                        mm[pos] = row if row is not None else 0
                        written[pos] = 1
                        state["count"] += 1
                pos += 1
            mm.flush()
            # Readers copy under the lock and hold no reference, so popping
            # here guarantees the mapping can close before the replace
            # (Windows refuses os.replace under an open mapping).
            with self._lock:
                self._building.pop(d, None)
            del mm
            import os
            os.replace(tmp, pack)
            return True
        except Exception as exc:
            print(f"[FrameCache] build failed {pack.name}: {exc}")
            with self._lock:
                self._building.pop(d, None)
            try:
                del mm
            except Exception:
                pass
            tmp.unlink(missing_ok=True)
            return False
