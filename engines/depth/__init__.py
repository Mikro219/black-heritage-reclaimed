"""Depth-camera layer (Orbbec Gemini 335). Enabled via config.json
`use_orbbec_camera` — see orbbec_camera.py."""

from .orbbec_camera import (          # noqa: F401
    ORBBEC_AVAILABLE,
    OrbbecCapture,
    OrbbecGemini335,
    depth_at,
    landmark_depth_mm,
    try_open_orbbec,
)
from .fusion import (                 # noqa: F401
    PoseDepth,
    metric_point,
    metric_speed_mm_s,
    reach_toward_camera_mm,
    trusted_landmark,
)
