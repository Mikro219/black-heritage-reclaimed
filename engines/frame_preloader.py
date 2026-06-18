"""
FramePreloader — background thread frame loader with initial-batch fast-start.

Phase 1 (initial_batch frames): PIL decode → raw bytes → signal is_ready().
Phase 2 (remaining frames):     continue loading; caller drains via drain().

Usage:
    preloader.start(frames_dir, resolution, initial_batch=90)
    # poll:
    if preloader.is_ready(frames_dir):
        surfaces = preloader.claim(frames_dir)   # main thread: bytes → Surfaces
    # each render tick after claiming:
    new = preloader.drain()
    if new:
        self._frames.extend(new)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from PIL import Image
import pygame

_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
_DRAIN_CHUNK = 30   # how many frames to buffer before releasing to main thread


class FramePreloader:
    def __init__(self):
        self._lock = threading.Lock()

        # Phase 1 buffer
        self._initial_raw: list = []
        self._target: Optional[Path] = None
        self._initial_done = False

        # Phase 2 buffer — drained each tick
        self._continuation_raw: list = []

        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def start(self, frames_dir: Path, resolution: tuple[int, int],
              initial_batch: int = 90) -> None:
        """Begin loading. Signals ready after initial_batch frames, then continues."""
        self._cancel.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cancel.clear()

        with self._lock:
            self._initial_raw       = []
            self._continuation_raw  = []
            self._target            = frames_dir
            self._initial_done      = False

        self._thread = threading.Thread(
            target=self._worker,
            args=(frames_dir, resolution, initial_batch),
            daemon=True,
            name=f"FramePreloader-{frames_dir.name}",
        )
        self._thread.start()
        print(f"[Preloader] started {frames_dir.name}  initial_batch={initial_batch}")

    def is_ready(self, frames_dir: Path) -> bool:
        with self._lock:
            return self._initial_done and self._target == frames_dir

    def claim(self, frames_dir: Path) -> Optional[list]:
        """
        Convert initial-batch raw buffers → pygame.Surfaces and return them.
        Returns None if not ready. Must be called from the main thread.
        """
        with self._lock:
            if not (self._initial_done and self._target == frames_dir):
                return None
            raw                = self._initial_raw
            self._initial_raw  = []

        return [pygame.image.fromstring(data, size, "RGB") for data, size in raw]

    def drain(self) -> list:
        """
        Return any continuation surfaces decoded since the last drain call.
        Call once per render tick after claim(). Main thread only.
        """
        with self._lock:
            if not self._continuation_raw:
                return []
            raw                     = self._continuation_raw
            self._continuation_raw  = []

        return [pygame.image.fromstring(data, size, "RGB") for data, size in raw]

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self, frames_dir: Path, resolution: tuple[int, int],
                initial_batch: int) -> None:
        w, h = resolution
        try:
            paths = sorted(
                p for p in frames_dir.iterdir()
                if p.suffix.lower() in _IMAGE_EXTS
            )
        except OSError as exc:
            print(f"[Preloader] cannot list {frames_dir}: {exc}")
            with self._lock:
                if self._target == frames_dir:
                    self._initial_done = True
            return

        initial: list      = []
        chunk: list        = []
        initial_signalled  = False

        for i, path in enumerate(paths):
            if self._cancel.is_set():
                return
            try:
                img = Image.open(path).convert("RGB").resize((w, h))
                raw_frame = (img.tobytes(), (w, h))
            except Exception as exc:
                print(f"[Preloader] skip {path.name}: {exc}")
                continue

            if not initial_signalled and i < initial_batch:
                initial.append(raw_frame)
                # Signal ready once we hit the batch size (or ran out of frames)
                if i + 1 == initial_batch or i + 1 == len(paths):
                    with self._lock:
                        if self._target == frames_dir and not self._cancel.is_set():
                            self._initial_raw  = initial
                            self._initial_done = True
                            initial_signalled  = True
                    print(f"[Preloader] initial batch ready ({len(initial)} frames)")
            else:
                chunk.append(raw_frame)
                if len(chunk) >= _DRAIN_CHUNK:
                    with self._lock:
                        if self._target == frames_dir and not self._cancel.is_set():
                            self._continuation_raw.extend(chunk)
                    chunk = []

        # Flush tail
        if chunk:
            with self._lock:
                if self._target == frames_dir and not self._cancel.is_set():
                    self._continuation_raw.extend(chunk)

        print(f"[Preloader] all frames loaded for {frames_dir.name}")
