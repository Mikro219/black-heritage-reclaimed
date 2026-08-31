"""
orbbec_camera — Orbbec Gemini 335 adapter (color + aligned depth).

STATUS: WIRED (July 2026). `config.json` key `use_orbbec_camera: true` makes both
main.py and scripts/gesture_tuner.py open the Gemini 335 via `try_open_orbbec()`
(an OrbbecCapture cv2-style shim), falling back to the webcam when the SDK or
camera is unavailable. The gesture engine injects `context["_depth_mm_at"]` so
detectors can sample real wrist depth. Verified against hardware 2026-07-03:
color+depth 720p30, MediaPipe coexistence OK in both import orders (the crash in
github.com/orbbec/pyorbbecsdk#184 does not reproduce with pyorbbecsdk2 2.1.1 +
mediapipe 0.10.14).

Hardware: Orbbec Gemini 335 — active stereo depth (0.25–6m usable), RGB up to
1280×800@30, USB 3.0, works through Orbbec SDK v2.

Install:  py -3.12 -m pip install pyorbbecsdk2 numpy
          (PyPI package is pyorbbecsdk2; the import name is pyorbbecsdk)
Preview:  py -3.12 -m engines.depth.orbbec_camera   (color + colorized depth windows)

Gotcha: only one process can hold the camera — close OrbbecViewer before running
(symptom: OBError "Hardware MFT failed to start streaming ... hardware resources").

Integration plan (when the camera arrives)
------------------------------------------
1. Host profile gains a block:
       "depth_camera": { "enabled": true, "resolution": [1280, 720], "fps": 30 }
2. `main.py` opens OrbbecGemini335 instead of cv2.VideoCapture when enabled; the
   color frame feeds MediaPipe exactly as the webcam frame does today (BGR ndarray,
   same shape), so GestureEngine needs no landmark-side changes.
3. `gesture_engine._capture_loop` keeps the latest aligned depth frame next to the
   published landmarks, and `_dispatch` injects a sampler into detector context:
       context["_depth_mm_at"] = lambda x, y: depth_at(depth_mm, x, y)
   (normalized x/y in RAW camera space — the same space landmarks arrive in, before
   any display mirroring.)
4. Z-proxy detectors grow real-depth params and prefer them when the sampler is
   present, falling back to today's bbox-growth/pose-z behaviour when it isn't:
       throw:          min_depth_delta_mm  (e.g. 250 within the stroke window)
       push_out:       min_depth_delta_mm  (bilateral)
       forward_reach:  min_depth_delta_mm
5. Optional later: depth-gate the whole vision pipeline to the nearest person
   (ignore anything beyond ~2.5m) so passers-by behind the visitor can't steer
   detections — a real concern for the public installation.

Depth conventions
-----------------
- `read()` returns the RAW uint16 depth frame (a copy — the SDK buffer is
  recycled), aligned to the color frame (same pixel grid), 0 = no data
  (shadow/too close/too far). The uint16→mm factor is kept on the camera as
  `depth_scale`; converting the whole 1280×720 frame to float32 mm every frame
  (~3.7MB alloc + multiply on the capture thread) was pure waste when the only
  consumer samples a few small patches.
- `depth_at()` samples a small patch and returns the **median of valid pixels**
  × `scale`, so single-pixel dropouts on a moving hand don't spike the reading.
  Callers holding an already-scaled float frame (tests, old captures) omit
  `scale` and get the identity behaviour.

Color conventions
-----------------
- The Gemini delivers RGB and `read()` keeps it RGB (`OrbbecCapture.frame_is_rgb`
  is True). The old RGB→BGR flip here was immediately undone by the gesture
  engine's BGR→RGB cvtColor for MediaPipe — a full-frame round trip per frame
  for nothing. Consumers that need BGR (cv2.imshow, the camera-setup screen)
  convert at their edge; plain cv2.VideoCapture stays BGR and has no
  `frame_is_rgb` attribute.

SDK import is LAZY (Aug 2026): `pyorbbecsdk` costs ~3s + stdout noise at import
time, and this module used to be dragged into every boot AND test run via
`engines.depth.__init__` → `pose_helpers` → every detector rule. Availability
is now computed with `importlib.util.find_spec` (no module execution); the real
import happens inside the code paths that talk to hardware.

SDK-version note: written against pyorbbecsdk v2 naming. If the installed SDK
differs (v1 used the same names for everything used here except AlignFilter
construction), the preview entry point will surface it immediately — fix in
_build_pipeline / read() only; the public surface (read/depth_at) must not change.
"""

from __future__ import annotations

import importlib.util
import threading
from typing import Optional, Tuple

import numpy as np

# pyorbbecsdk is optional until the hardware arrives. find_spec only scans the
# path finders — it never executes the module, so this stays instant even when
# the SDK is installed (the ~3s import is paid only when a Gemini is opened).
ORBBEC_AVAILABLE = importlib.util.find_spec("pyorbbecsdk") is not None


class OrbbecGemini335:
    """Color + aligned-depth capture from a Gemini 335.

    Usage:
        cam = OrbbecGemini335(resolution=(1280, 720), fps=30)
        cam.start()
        color_rgb, depth_raw = cam.read()  # (H,W,3) uint8 RGB, (H,W) uint16
        mm = depth_at(depth_raw, 0.5, 0.5, scale=cam.depth_scale)
        ...
        cam.stop()
    """

    def __init__(self, resolution: Tuple[int, int] = (1280, 720), fps: int = 30,
                 timeout_ms: int = 100):
        if not ORBBEC_AVAILABLE:
            raise RuntimeError(
                "pyorbbecsdk is not installed. Run: py -3.12 -m pip install pyorbbecsdk"
            )
        self._resolution = resolution
        self._fps = fps
        self._timeout_ms = timeout_ms
        self._pipeline = None
        self._align = None
        # uint16 → millimetres factor of the last depth frame (usually 1.0 on
        # the Gemini 335). Updated per read(); pass to depth_at(..., scale=).
        self.depth_scale: float = 1.0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        # Lazy SDK import — see module docstring. __init__ already verified
        # availability via ORBBEC_AVAILABLE.
        from pyorbbecsdk import (          # type: ignore
            AlignFilter,
            Config,
            OBFormat,
            OBSensorType,
            OBStreamType,
            Pipeline,
        )

        self._pipeline = Pipeline()
        config = Config()
        w, h = self._resolution

        color_profiles = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color = color_profiles.get_video_stream_profile(w, h, OBFormat.RGB, self._fps)
        config.enable_stream(color)

        depth_profiles = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth = depth_profiles.get_video_stream_profile(w, h, OBFormat.Y16, self._fps)
        config.enable_stream(depth)

        self._pipeline.start(config)
        # Hardware-assisted depth-to-color alignment: depth lands on the color grid,
        # so a landmark's (x, y) indexes both frames identically.
        self._align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._align = None

    # -- capture -----------------------------------------------------------

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (color_rgb, depth_raw) or (None, None) on a dropped frameset.

        color_rgb: (H, W, 3) uint8 RGB — the sensor's native order; no per-frame
                   channel flip (consumers needing BGR convert at their edge)
        depth_raw: (H, W)    uint16 sensor units, 0 where invalid; multiply by
                   `self.depth_scale` for millimetres (depth_at does this)
        """
        if self._pipeline is None:
            raise RuntimeError("start() has not been called")

        frames = self._pipeline.wait_for_frames(self._timeout_ms)
        if frames is None:
            return None, None
        if self._align is not None:
            aligned = self._align.process(frames)
            if aligned is not None:
                frames = aligned.as_frame_set()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None, None

        cw, ch = color_frame.get_width(), color_frame.get_height()
        color = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        color = color.reshape((ch, cw, 3)).copy()   # native RGB; copy off SDK buffer

        dw, dh = depth_frame.get_width(), depth_frame.get_height()
        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_raw = depth_raw.reshape((dh, dw)).copy()   # SDK recycles the buffer
        self.depth_scale = float(depth_frame.get_depth_scale())

        return color, depth_raw


# ---------------------------------------------------------------------------
# Sampling helpers (pure functions — usable without the SDK, e.g. in tests)
# ---------------------------------------------------------------------------

def depth_at(depth_mm: np.ndarray, x_norm: float, y_norm: float,
             patch_px: int = 5, scale: float = 1.0) -> Optional[float]:
    """Median valid depth (mm) in a patch around a normalized coordinate.

    x_norm/y_norm are in RAW camera space (0-1, same space MediaPipe landmarks
    arrive in). Returns None when the patch has no valid depth (all zeros or
    coordinate out of frame).

    `scale` converts stored units to millimetres and is applied AFTER slicing
    the patch — so a raw uint16 frame plus its sensor scale costs a multiply on
    a handful of pixels instead of a float32 conversion of the whole frame.
    Pass 1.0 (the default) for a frame already in millimetres (float arrays,
    tests).
    """
    h, w = depth_mm.shape[:2]
    cx = int(round(x_norm * (w - 1)))
    cy = int(round(y_norm * (h - 1)))
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    r = max(0, patch_px // 2)
    patch = depth_mm[max(0, cy - r): cy + r + 1, max(0, cx - r): cx + r + 1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * float(scale)


def landmark_depth_mm(depth_mm: np.ndarray, landmark,
                      patch_px: int = 5, scale: float = 1.0) -> Optional[float]:
    """Depth (mm) at a MediaPipe landmark (any object with .x/.y in 0-1)."""
    return depth_at(depth_mm, landmark.x, landmark.y, patch_px, scale=scale)


# ---------------------------------------------------------------------------
# cv2.VideoCapture-compatible shim + opener
# ---------------------------------------------------------------------------

class OrbbecCapture:
    """Drop-in stand-in for cv2.VideoCapture backed by the Gemini 335.

    read() returns (ret, frame) like a webcam — but the frame is the sensor's
    native RGB (`frame_is_rgb = True`; cv2.VideoCapture has no such attribute
    and stays BGR), so consumers feeding MediaPipe skip a cvtColor and BGR
    consumers convert at their edge. The latest aligned depth frame is kept
    raw (uint16 + scale) under a lock; depth_mm_at(x, y) samples it in mm from
    any thread (the gesture engine injects it into detector context as
    "_depth_mm_at").
    """

    frame_is_rgb = True   # native RGB frames — no per-frame BGR flip

    def __init__(self, resolution: Tuple[int, int] = (1280, 720), fps: int = 30):
        self._cam = OrbbecGemini335(resolution=resolution, fps=fps)
        self._cam.start()
        self._open = True
        self._lock = threading.Lock()
        self._depth: Optional[np.ndarray] = None
        self._depth_scale: float = 1.0

    # -- cv2.VideoCapture surface ------------------------------------------

    def isOpened(self) -> bool:
        return self._open

    def read(self):
        if not self._open:
            return False, None
        color, depth = self._cam.read()
        if color is None:
            return False, None
        if depth is not None:
            with self._lock:
                self._depth = depth
                self._depth_scale = self._cam.depth_scale
        return True, color

    def set(self, prop, value) -> bool:   # resolution/fps are fixed at open
        return False

    def get(self, prop) -> float:
        return 0.0

    def release(self) -> None:
        if self._open:
            self._open = False
            self._cam.stop()

    # -- depth --------------------------------------------------------------

    def depth_mm_at(self, x_norm: float, y_norm: float,
                    patch_px: int = 5) -> Optional[float]:
        """Median depth (mm) at a normalized RAW-camera-space coordinate, from
        the most recent frame. None until the first depth frame arrives."""
        with self._lock:
            d = self._depth
            scale = self._depth_scale
        if d is None:
            return None
        return depth_at(d, x_norm, y_norm, patch_px, scale=scale)


def try_open_orbbec(resolution: Tuple[int, int] = (1280, 720),
                    fps: int = 30,
                    verify_timeout_s: float = 3.0) -> Optional["OrbbecCapture"]:
    """Open the Gemini 335 as an OrbbecCapture, or None if unavailable.

    Callers fall back to a plain webcam on None, so a missing SDK or unplugged
    camera degrades gracefully instead of killing the app. Three failure modes
    are covered:
      1. SDK not installed                       → None immediately.
      2. No Gemini enumerated / open raises      → None (device probe + except).
      3. Pipeline opens but never delivers a
         frame (half-detected device, USB2 port) → None after verify_timeout_s.
    """
    if not ORBBEC_AVAILABLE:
        print("[orbbec] pyorbbecsdk not installed (pip install pyorbbecsdk2) "
              "— falling back to webcam")
        return None

    # Cheap device probe before constructing a pipeline: an SDK with no camera
    # attached can otherwise stall in native code or "open" a device-less
    # pipeline that never produces frames.
    try:
        from pyorbbecsdk import Context  # type: ignore
        if Context().query_devices().get_count() == 0:
            print("[orbbec] no Gemini device detected — falling back to webcam")
            return None
    except Exception:
        pass   # older SDK without Context/query — rely on the open + verify below

    try:
        cap = OrbbecCapture(resolution=resolution, fps=fps)
    except Exception as exc:
        print(f"[orbbec] Gemini 335 open failed: {exc}")
        print("[orbbec] (is OrbbecViewer or another app holding the camera?) "
              "— falling back to webcam")
        return None

    # Verify a real color frame arrives before handing the capture to the
    # engine; otherwise the runtime would sit on a black screen instead of
    # using the webcam.
    import time
    deadline = time.monotonic() + verify_timeout_s
    while time.monotonic() < deadline:
        try:
            ok, frame = cap.read()
        except Exception as exc:
            print(f"[orbbec] Gemini 335 read failed during verification: {exc} "
                  "— falling back to webcam")
            _safe_release(cap)
            return None
        if ok and frame is not None:
            return cap
    print(f"[orbbec] Gemini 335 delivered no frames within {verify_timeout_s:.0f}s "
          "— falling back to webcam")
    _safe_release(cap)
    return None


def _safe_release(cap: "OrbbecCapture") -> None:
    try:
        cap.release()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Standalone preview:  py -3.12 -m engines.depth.orbbec_camera
# ---------------------------------------------------------------------------

def _preview() -> None:
    if not ORBBEC_AVAILABLE:
        print("pyorbbecsdk not installed — run: py -3.12 -m pip install pyorbbecsdk")
        return
    import cv2

    cam = OrbbecGemini335()
    cam.start()
    print("Gemini 335 preview — Q/Esc to quit. Centre-pixel depth shown in title.")
    try:
        while True:
            color, depth = cam.read()
            if color is None:
                continue
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)   # imshow wants BGR
            depth_mm = depth.astype(np.float32) * cam.depth_scale
            vis = cv2.applyColorMap(
                cv2.convertScaleAbs(np.clip(depth_mm, 0, 4000), alpha=255.0 / 4000.0),
                cv2.COLORMAP_JET,
            )
            centre = depth_at(depth, 0.5, 0.5, scale=cam.depth_scale)
            label = f"centre: {centre:.0f} mm" if centre else "centre: --"
            cv2.putText(color, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 80), 2)
            cv2.imshow("Gemini 335 color", color)
            cv2.imshow("Gemini 335 depth", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _preview()
