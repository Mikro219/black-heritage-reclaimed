"""Depth-camera layer (Orbbec Gemini 335). Enabled via config.json
`use_orbbec_camera` — see orbbec_camera.py.

The orbbec_camera re-exports are LAZY (PEP 562, Aug 2026): this package is
imported by `pose_helpers` — and therefore by every detector rule, every boot
and every test run — but only the hardware openers need orbbec_camera, whose
pyorbbecsdk import costs ~3s + stdout noise when the SDK is installed. The
fusion layer is SDK-free and stays eager.
"""

from .fusion import (                 # noqa: F401
    PoseDepth,
    metric_point,
    metric_speed_mm_s,
    reach_toward_camera_mm,
    trusted_landmark,
)

_ORBBEC_EXPORTS = (
    "ORBBEC_AVAILABLE",
    "OrbbecCapture",
    "OrbbecGemini335",
    "depth_at",
    "landmark_depth_mm",
    "try_open_orbbec",
)


def __getattr__(name):
    """Import orbbec_camera only on first access to one of its exports."""
    if name in _ORBBEC_EXPORTS:
        from . import orbbec_camera
        return getattr(orbbec_camera, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_ORBBEC_EXPORTS))
