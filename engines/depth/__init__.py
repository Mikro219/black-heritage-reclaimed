"""Depth-camera layer (Orbbec Gemini 335). Not wired into the runtime yet —
see orbbec_camera.py for the integration plan."""

from .orbbec_camera import (          # noqa: F401
    ORBBEC_AVAILABLE,
    OrbbecGemini335,
    depth_at,
    landmark_depth_mm,
)
