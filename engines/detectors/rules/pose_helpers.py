"""
pose_helpers — shared MediaPipe Pose landmark helpers for the rule-based
detectors and the render engine's cursors.

POSE-ONLY runtime (July 2026): the Hands model is gone. Every detector reads
the 33-landmark Pose skeleton from context["_pose_lm"]. Pose includes three
hand-adjacent landmarks per side — pinky (17/18), index (19/20), thumb
(21/22) — which stand in for the old fingertip where a "hand point" is
needed. What Pose cannot see is finger curl: open-palm / fist / pointing-
finger classification does not exist in this runtime; detectors are written
to not need it (detect intent, not precision).

Landmark indices used across the codebase:
  0 nose · 2/5 eyes · 7/8 ears · 9/10 mouth corners
  11/12 shoulders · 13/14 elbows · 15/16 wrists
  17/18 pinky · 19/20 index · 21/22 thumb · 23/24 hips

All functions do cheap 2D math on raw (unmirrored) coordinates; the x-flip to
player/screen space happens at the call sites that need it, same as before.
"""

import math

from ...depth.fusion import trusted_landmark

# side key -> (wrist, elbow, shoulder, index, pinky, thumb)
SIDES = {
    "L": {"wrist": 15, "elbow": 13, "shoulder": 11, "index": 19, "pinky": 17, "thumb": 21},
    "R": {"wrist": 16, "elbow": 14, "shoulder": 12, "index": 20, "pinky": 18, "thumb": 22},
}

_DIRS_CW = ["right", "down_right", "down", "down_left",
            "left", "up_left", "up", "up_right"]


def lm(pose_lm, idx):
    """Landmark by index, or None when pose is absent/malformed."""
    if pose_lm is None:
        return None
    try:
        return pose_lm[idx]
    except (IndexError, TypeError):
        return None


def trusted_wrists(pose_lm, context: dict, min_visibility: float = 0.5) -> dict:
    """{"L": wrist_lm, "R": wrist_lm} for wrists passing the visibility gate
    AND (when the Gemini sampler is present) the depth phantom veto."""
    out = {}
    for side, idxs in SIDES.items():
        w = lm(pose_lm, idxs["wrist"])
        if w is not None and trusted_landmark(context, idxs["wrist"], w, min_visibility):
            out[side] = w
    return out


def hand_point(pose_lm, side: str, min_visibility: float = 0.5):
    """The best 'hand tip' Pose offers for a side: the index landmark when it
    passes the visibility gate, else the wrist. Returns a landmark or None."""
    idxs = SIDES[side]
    tip = lm(pose_lm, idxs["index"])
    if tip is not None and getattr(tip, "visibility", 1.0) >= min_visibility:
        return tip
    w = lm(pose_lm, idxs["wrist"])
    if w is not None and getattr(w, "visibility", 1.0) >= min_visibility:
        return w
    return None


def point_vector(pose_lm, side: str):
    """(dx, dy) of the hand's pointing direction in SCREEN space (x flipped):
    wrist→index when the index is usable, else elbow→wrist (forearm line).
    None when neither is available."""
    idxs = SIDES[side]
    w = lm(pose_lm, idxs["wrist"])
    if w is None:
        return None
    tip = lm(pose_lm, idxs["index"])
    if tip is not None and getattr(tip, "visibility", 1.0) >= 0.5:
        return (-(tip.x - w.x), tip.y - w.y)
    el = lm(pose_lm, idxs["elbow"])
    if el is not None:
        return (-(w.x - el.x), w.y - el.y)
    return None


def classify_direction(dx: float, dy: float) -> str:
    """Snap a screen-space (dx, dy) vector to the nearest of 8 compass
    directions (45° sectors)."""
    angle = math.degrees(math.atan2(dy, dx)) % 360
    return _DIRS_CW[int((angle + 22.5) / 45) % 8]


def geometry(pose_lm):
    """Torso geometry dict, or None: midline_x, shoulder_width, shoulder_y,
    hip_y, torso_h — the shared body-relative reference frame."""
    sh_l, sh_r = lm(pose_lm, 11), lm(pose_lm, 12)
    hip_l, hip_r = lm(pose_lm, 23), lm(pose_lm, 24)
    if None in (sh_l, sh_r, hip_l, hip_r):
        return None
    shoulder_y = (sh_l.y + sh_r.y) / 2
    hip_y = (hip_l.y + hip_r.y) / 2
    return {
        "midline_x": (sh_l.x + sh_r.x) / 2,
        "shoulder_width": abs(sh_l.x - sh_r.x),
        "shoulder_y": shoulder_y,
        "hip_y": hip_y,
        "torso_h": max(hip_y - shoulder_y, 0.05),
    }


def arm_reach_frac(pose_lm, side: str) -> float:
    """How extended the arm is: shoulder→wrist distance over the arm's full
    length (shoulder→elbow + elbow→wrist). ~1.0 = arm straight, ~0.5 = folded.
    0.0 when landmarks are missing."""
    idxs = SIDES[side]
    sh, el, w = lm(pose_lm, idxs["shoulder"]), lm(pose_lm, idxs["elbow"]), lm(pose_lm, idxs["wrist"])
    if None in (sh, el, w):
        return 0.0
    upper = math.hypot(el.x - sh.x, el.y - sh.y)
    fore = math.hypot(w.x - el.x, w.y - el.y)
    full = upper + fore
    if full < 1e-6:
        return 0.0
    return math.hypot(w.x - sh.x, w.y - sh.y) / full


def wrist_depth_or_z(context: dict, pose_lm, side: str):
    """(kind, value) for the wrist's camera-distance signal: ("mm", depth) with
    the Gemini sampler, else ("z", pose_z) — pose z is hip-normalised, more
    negative = closer to camera. (None, None) when neither resolves."""
    idxs = SIDES[side]
    w = lm(pose_lm, idxs["wrist"])
    if w is None:
        return None, None
    depth_at = context.get("_depth_mm_at")
    if callable(depth_at):
        d = depth_at(w.x, w.y)
        if d is not None:
            return "mm", d
    return "z", getattr(w, "z", 0.0)
