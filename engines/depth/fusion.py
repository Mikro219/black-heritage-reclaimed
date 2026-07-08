"""
fusion — depth + MediaPipe fusion layer for the Orbbec Gemini 335 (July 2026).

MediaPipe gives landmark POSITIONS but only guesses at depth (its z is a
hip-normalised estimate, and its `visibility` score is a guess about occlusion).
The Gemini 335 gives real per-pixel millimetres aligned to the color frame.
This module fuses the two per frame, without any model retraining:

  1. LIFT      — sample real depth at any landmark: landmark_mm(idx).
  2. VETO      — phantom-landmark rejection: MediaPipe reports positions for
                 occluded/out-of-frame joints; sampling depth at those spots
                 hits the BACKGROUND (metres behind the torso). plausible(idx)
                 turns that into a veto that stacks with the visibility gate.
  3. METER     — reach_mm(idx): how far a joint reaches toward the camera
                 relative to the torso plane. Replaces bbox-growth z-proxies.
  4. METRIC    — metric_point(): normalised (x, y) + depth -> real millimetres,
                 so velocities become mm/s and thresholds stop depending on how
                 far from the camera the player stands.
  5. GATE      — torso_depth_mm(): the tracked person's distance. The gesture
                 engine ignores poses outside the configured player band, so a
                 passer-by 4 m behind the visitor can't steer detections.

Everything degrades to None when the depth sampler is absent (plain webcam) —
callers fall back to the pre-depth behaviour, so laptop_dev keeps working.

Design rule: depth VETOES, it does not rescue. A landmark that fails the
visibility gate stays rejected even if its depth looks fine; a landmark that
passes visibility can additionally be rejected when its depth says "that's the
back wall". This keeps false positives (the installation's real enemy) strictly
lower than the visibility-only baseline.

Config (config.json "depth" section, all optional):
  player_min_mm / player_max_mm — interaction zone for the player band gate.
  phantom_behind_mm — how far behind the torso plane a landmark's sampled depth
                      may sit before it's vetoed as a phantom. Default 400.

FOV constants are the Gemini 335 color camera's (~86° x 55°); override via
metric_point(..., hfov_deg=, vfov_deg=) if the stream profile changes.
"""

from __future__ import annotations

import math
from typing import Optional

# Gemini 335 color stream field of view (approx; good to a few percent, which
# is far tighter than any gesture threshold cares about).
GEMINI335_HFOV_DEG = 86.0
GEMINI335_VFOV_DEG = 55.0

# Pose landmark indices used for the torso plane estimate.
_TORSO_IDX = (11, 12, 23, 24)   # shoulders + hips


class PoseDepth:
    """Per-frame fused view over one Pose result and one depth frame.

    Construct once per dispatch tick; sampling is lazy and cached, so building
    the object costs nothing when no detector asks for depth.
    """

    def __init__(self, depth_mm_at, pose_lm,
                 phantom_behind_mm: float = 400.0, patch_px: int = 5):
        self._sample = depth_mm_at if callable(depth_mm_at) else None
        self._pose = pose_lm
        self._patch = patch_px
        self.phantom_behind_mm = phantom_behind_mm
        self._cache: dict = {}
        self._torso: Optional[float] = None
        self._torso_done = False

    # -- availability --------------------------------------------------------

    @property
    def available(self) -> bool:
        """True when both a depth sampler and a pose are present."""
        return self._sample is not None and self._pose is not None

    # -- lifting -------------------------------------------------------------

    def landmark_mm(self, idx: int) -> Optional[float]:
        """Real depth (mm) at pose landmark idx, or None (no sampler / invalid
        sample / landmark out of frame)."""
        if not self.available:
            return None
        if idx in self._cache:
            return self._cache[idx]
        try:
            lm = self._pose[idx]
        except (IndexError, TypeError):
            self._cache[idx] = None
            return None
        d = None
        if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
            try:
                d = self._sample(lm.x, lm.y, self._patch)
            except TypeError:
                d = self._sample(lm.x, lm.y)
        self._cache[idx] = d
        return d

    def torso_depth_mm(self) -> Optional[float]:
        """Median depth of the visible torso landmarks — the player's distance
        from the camera. None when fewer than two torso samples are valid."""
        if self._torso_done:
            return self._torso
        self._torso_done = True
        samples = [d for d in (self.landmark_mm(i) for i in _TORSO_IDX)
                   if d is not None]
        if len(samples) < 2:
            self._torso = None
        else:
            samples.sort()
            mid = len(samples) // 2
            self._torso = (samples[mid] if len(samples) % 2
                           else (samples[mid - 1] + samples[mid]) / 2)
        return self._torso

    # -- metering ------------------------------------------------------------

    def reach_mm(self, idx: int) -> Optional[float]:
        """How far landmark idx reaches TOWARD the camera from the torso plane
        (positive = closer than the torso). None when either depth is missing."""
        lm_d = self.landmark_mm(idx)
        torso = self.torso_depth_mm()
        if lm_d is None or torso is None:
            return None
        return torso - lm_d

    # -- veto ----------------------------------------------------------------

    def plausible(self, idx: int) -> Optional[bool]:
        """Depth plausibility of landmark idx.

        False — the sampled depth sits more than phantom_behind_mm BEHIND the
                torso plane: MediaPipe put the landmark somewhere the player
                isn't (classic phantom estimate over background).
        True  — depth is consistent with a body part (at, near, or in front
                of the torso).
        None  — unknown (no sampler, invalid sample, no torso estimate);
                caller must fall back to visibility alone.
        """
        lm_d = self.landmark_mm(idx)
        torso = self.torso_depth_mm()
        if lm_d is None or torso is None:
            return None
        return (lm_d - torso) <= self.phantom_behind_mm

    def trusted(self, idx: int, lm, min_visibility: float) -> bool:
        """Combined gate: MediaPipe visibility AND depth plausibility.
        Depth can veto, never rescue."""
        if getattr(lm, "visibility", 1.0) < min_visibility:
            return False
        return self.plausible(idx) is not False


# ---------------------------------------------------------------------------
# Detector-facing helper (safe with or without fusion in context)
# ---------------------------------------------------------------------------

def trusted_landmark(context: dict, idx: int, lm, min_visibility: float) -> bool:
    """Visibility gate + depth veto for a pose landmark. Detectors call this
    instead of a raw visibility comparison; with no fusion in context (webcam)
    it degrades to exactly the old visibility-only check."""
    fusion = context.get("_pose_depth")
    if isinstance(fusion, PoseDepth):
        return fusion.trusted(idx, lm, min_visibility)
    return getattr(lm, "visibility", 1.0) >= min_visibility


def reach_toward_camera_mm(context: dict, idx: int) -> Optional[float]:
    """Convenience: fusion reach_mm(idx) or None when fusion is absent."""
    fusion = context.get("_pose_depth")
    if isinstance(fusion, PoseDepth):
        return fusion.reach_mm(idx)
    return None


# ---------------------------------------------------------------------------
# Metric conversion (normalised image coords + depth -> millimetres)
# ---------------------------------------------------------------------------

def metric_point(x_norm: float, y_norm: float, depth_mm: float,
                 hfov_deg: float = GEMINI335_HFOV_DEG,
                 vfov_deg: float = GEMINI335_VFOV_DEG) -> tuple:
    """(x_mm, y_mm, z_mm) of a normalised image point at a known depth, using a
    pinhole model with the Gemini 335's color FOV. Origin on the optical axis."""
    half_w = depth_mm * math.tan(math.radians(hfov_deg) / 2)
    half_h = depth_mm * math.tan(math.radians(vfov_deg) / 2)
    return ((x_norm - 0.5) * 2 * half_w,
            (y_norm - 0.5) * 2 * half_h,
            depth_mm)


def metric_speed_mm_s(p_prev: tuple, p_now: tuple, dt_s: float) -> Optional[float]:
    """3D speed in mm/s between two metric_point() results. None on bad dt."""
    if dt_s <= 0:
        return None
    return math.dist(p_prev, p_now) / dt_s
