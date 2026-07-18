"""
bilateral_rotation — both hands making circular motions, ≥ min_cycles.
Used as an alternative unravel model (Scene 11).

ELBOW-LINE CROSSING MODEL (July 2026, default): a winding motion mostly
TRANSLATES the hands in a circle — the hand's own orientation barely changes,
so the old wrist→index angle sweep under-counted real unravels. Instead we
draw the line through the two elbow landmarks (13/14) and count each hand
oscillating across it: a circling hand crosses that line twice per cycle.
The line is body-relative, so the check survives camera angle and the elbows
drifting during the gesture. A hysteresis band (band_frac of shoulder width
around the line) keeps landmark jitter from registering crossings, and
crossings older than the rolling window age away.

The previous net-signed-rotation sweep of the wrist→index vector survives
behind ``mode: "sweep"`` (its jitter-cancelling and dropout-reset semantics
unchanged). Wrist/elbow dropout resets all per-side state in both modes so a
stale zone or angle can't register phantom motion when tracking returns.

Params:
  min_cycles (int): complete oscillations (2 line crossings each) required
                    per hand. Default 2.
  rotation_window_ms (int): rolling window the cycles must occur within.
                            Default 5000.
  band_frac (float): hysteresis half-band around the elbow line, as a
                     fraction of shoulder width — the hand must clear this on
                     BOTH sides of the line for a crossing to count.
                     Default 0.12. (elbow-line mode only)
  mode (str): "elbow_line" (default) | "sweep" (legacy angle-sweep model).
  min_visibility (float): Pose wrist/elbow visibility gate. Default 0.5.

Context keys: rotation_zone, rotation_crossings,
              rotation_prev_angle, rotation_history  (reads: _pose_lm)
"""

import math
import time

from ...depth.fusion import trusted_landmark
from . import pose_helpers


def _side_angle(pose_lm, side: str):
    vec = pose_helpers.point_vector(pose_lm, side)
    if vec is None:
        return None
    return math.atan2(vec[1], vec[0])


def _angle_delta(prev: float, curr: float) -> float:
    d = curr - prev
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _reset(context: dict) -> None:
    context["rotation_prev_angle"] = {}
    context["rotation_history"] = {}
    context["rotation_zone"] = {}
    context["rotation_crossings"] = {}


def _detect_sweep(pose_lm, wrists, params: dict, context: dict) -> bool:
    """Legacy model: net signed rotation of the wrist→index vector."""
    min_cycles = params.get("min_cycles", 2)
    window_s = params.get("rotation_window_ms", 5000) / 1000.0
    now = time.monotonic()

    for side in wrists:
        angle = _side_angle(pose_lm, side)
        if angle is None:
            continue
        prev = context["rotation_prev_angle"].get(side)
        context["rotation_prev_angle"][side] = angle
        if prev is None:
            continue
        hist = context["rotation_history"].setdefault(side, [])
        hist.append((now, _angle_delta(prev, angle)))

    cutoff = now - window_s
    counts = {}
    for side, hist in context["rotation_history"].items():
        while hist and hist[0][0] < cutoff:
            hist.pop(0)
        net = sum(d for _, d in hist)
        counts[side] = int(abs(net) / (2 * math.pi))

    if len(counts) >= 2 and all(v >= min_cycles for v in counts.values()):
        _reset(context)
        return True
    return False


def _detect_elbow_line(pose_lm, wrists, params: dict, context: dict) -> bool:
    """Default model: each hand oscillates across the elbow-elbow line."""
    min_vis = params.get("min_visibility", 0.5)
    el_l = pose_helpers.lm(pose_lm, 13)
    el_r = pose_helpers.lm(pose_lm, 14)
    if (el_l is None or el_r is None
            or not trusted_landmark(context, 13, el_l, min_vis)
            or not trusted_landmark(context, 14, el_r, min_vis)):
        _reset(context)   # no reliable line — stale zones must not carry over
        return False

    dx, dy = el_r.x - el_l.x, el_r.y - el_l.y
    line_len = math.hypot(dx, dy)
    if line_len < 0.03:   # elbows collapsed (body sideways) — geometry useless
        _reset(context)
        return False

    geom = pose_helpers.geometry(pose_lm)
    scale = geom["shoulder_width"] if geom else line_len
    band = params.get("band_frac", 0.12) * max(scale, 0.10)

    min_cycles = params.get("min_cycles", 2)
    window_s = params.get("rotation_window_ms", 5000) / 1000.0
    now = time.monotonic()

    for side in wrists:
        p = pose_helpers.hand_point(pose_lm, side, min_vis)
        if p is None:
            continue
        # signed perpendicular offset of the hand from the elbow line
        off = (dx * (p.y - el_l.y) - dy * (p.x - el_l.x)) / line_len
        zone = context["rotation_zone"].get(side)
        new_zone = 1 if off > band else (-1 if off < -band else zone)
        if zone is not None and new_zone == -zone:
            context["rotation_crossings"].setdefault(side, []).append(now)
        context["rotation_zone"][side] = new_zone

    cutoff = now - window_s
    counts = {}
    for side, hist in context["rotation_crossings"].items():
        while hist and hist[0] < cutoff:
            hist.pop(0)
        counts[side] = len(hist) // 2      # 2 crossings = 1 oscillation

    if len(wrists) >= 2 and all(counts.get(s, 0) >= min_cycles for s in wrists):
        _reset(context)
        return True
    return False


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if len(wrists) < 2:
        _reset(context)   # dropout: stale per-side state must not carry over
        return False

    if "rotation_history" not in context or "rotation_crossings" not in context:
        _reset(context)

    if params.get("mode") == "sweep":
        return _detect_sweep(pose_lm, wrists, params, context)
    return _detect_elbow_line(pose_lm, wrists, params, context)
