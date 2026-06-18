"""
FrameCacheManager — persistent background look-ahead frame cache.

Decodes frames for every shot that has art on disk into an in-memory raw-bytes
cache, working forward from the current shot so upcoming shots are fully loaded
*before* the player reaches them. The old per-shot preloader stopped after the
current (and immediately-next) shot; this keeps going until every shot with art
is cached, using the idle time between interactions.

Frames are decoded with PIL on a background thread as raw RGB bytes. The render
engine converts them to pygame Surfaces on the main thread (pygame surface
creation is kept off worker threads on purpose).

Priority: the current shot first, then every shot after it in sequence order,
then wrap around to fill any earlier ones. Per-shot loading is resumable — a
shot that was preempted mid-load continues from where it stopped.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from PIL import Image

_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
_FLUSH_CHUNK = 200  # frames decoded per batch before publishing to the cache


class FrameCacheManager:
    def __init__(self, resolution: tuple[int, int]):
        self._res = resolution
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        self._raw: dict[Path, list] = {}        # dir -> list[(bytes, (w, h))]
        self._counts: dict[Path, int] = {}      # dir -> total frame count on disk
        self._complete: set[Path] = set()       # dirs fully decoded
        self._order: list[Path] = []            # ordered dirs (shots with art)
        self._priority: Optional[Path] = None   # dir to load first (current shot)
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public — main thread
    # ------------------------------------------------------------------

    def start(self, ordered_dirs) -> None:
        with self._lock:
            self._order = [Path(d) for d in ordered_dirs]
        self._thread = threading.Thread(target=self._worker, daemon=True, name="FrameCache")
        self._thread.start()
        print(f"[FrameCache] started: {len(self._order)} shots with art queued")

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    def prioritize(self, frames_dir) -> None:
        """Load this shot next (it just became current)."""
        d = Path(frames_dir)
        with self._cv:
            self._priority = d
            if d not in self._order:
                self._order.append(d)
            self._cv.notify_all()

    def is_complete(self, frames_dir) -> bool:
        with self._lock:
            return Path(frames_dir) in self._complete

    def loaded_count(self, frames_dir) -> int:
        with self._lock:
            lst = self._raw.get(Path(frames_dir))
            return len(lst) if lst else 0

    def total_count(self, frames_dir) -> Optional[int]:
        with self._lock:
            return self._counts.get(Path(frames_dir))

    def get_raw_slice(self, frames_dir, start: int, count: int) -> list:
        """Return a shallow copy of raw frames [start : start+count] for conversion."""
        with self._lock:
            lst = self._raw.get(Path(frames_dir))
            if not lst:
                return []
            return list(lst[start:start + count])

    def progress(self) -> tuple[int, int]:
        with self._lock:
            return len(self._complete), len(self._order)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _next_target(self) -> Optional[Path]:
        with self._lock:
            if not self._order:
                return None
            start = self._order.index(self._priority) if self._priority in self._order else 0
            # forward from the current shot, then wrap to earlier shots
            for d in self._order[start:] + self._order[:start]:
                if d not in self._complete:
                    return d
            return None

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
                    self._cv.wait(timeout=1.0)   # everything cached — idle until prioritise()
                continue
            self._load_dir(target)

    def _load_dir(self, frames_dir: Path) -> None:
        w, h = self._res
        try:
            paths = sorted(
                p for p in frames_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS
            )
        except OSError as exc:
            print(f"[FrameCache] cannot list {frames_dir}: {exc}")
            with self._lock:
                self._complete.add(frames_dir)
            return

        with self._lock:
            self._counts[frames_dir] = len(paths)
            existing = len(self._raw.get(frames_dir, []))

        if existing >= len(paths):
            with self._lock:
                self._complete.add(frames_dir)
            return

        buf: list = []
        for i in range(existing, len(paths)):
            try:
                img = Image.open(paths[i]).convert("RGB").resize((w, h))
                buf.append((img.tobytes(), (w, h)))
            except Exception as exc:
                print(f"[FrameCache] skip {paths[i].name}: {exc}")
                continue
            if len(buf) >= _FLUSH_CHUNK:
                with self._lock:
                    if self._stop:
                        return
                    pri = self._priority
                    self._raw.setdefault(frames_dir, []).extend(buf)
                buf = []
                # A newly-current shot preempts this one; flush partial and yield.
                if pri is not None and pri != frames_dir and pri not in self._complete:
                    return

        if buf:
            with self._lock:
                self._raw.setdefault(frames_dir, []).extend(buf)

        with self._lock:
            if len(self._raw.get(frames_dir, [])) >= len(paths):
                self._complete.add(frames_dir)
                done, total = len(self._complete), len(self._order)
                print(f"[FrameCache] complete {frames_dir.parent.name} "
                      f"({len(paths)} frames)  [{done}/{total} shots cached]")
