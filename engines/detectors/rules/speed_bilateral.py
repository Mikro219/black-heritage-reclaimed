"""
speed_bilateral — hands moving fast (rapid gathering / shaking motion).

Used for Scene 10 `fast_gather` (AL-10-004) and `forward_push`.

REWORKED (July 2026 playtest): the old model summed per-FRAME displacement of
both pose wrists against a fixed per-frame threshold. Two problems: the
threshold silently changed meaning with camera fps (a 15fps Orbbec frame covers
twice the motion of a 30fps one), and requiring BOTH wrists visible meant the
gesture died the moment motion blur dropped one wrist — precisely when the
player is moving fastest. What it's searching for now, concretely: ANY visible
wrist moving faster than idle_velocity × velocity_multiplier (screen units per
SECOND), sustained — min_bursts fast frames spanning at least min_active_ms
inside burst_window_ms.

Params:
  min_bursts (int): fast frames required within burst_window_ms. Default 3.
  burst_window_ms (int): rolling window to accumulate bursts. Default 3000.
  velocity_multiplier (float): threshold as a multiple of idle_velocity. Default 2.0.
  idle_velocity (float): nominal resting wrist speed in screen units/second.
                  Default 0.45 (≈ the old 0.015/frame at 30fps).
  min_active_ms (int): bursts must span at least this long — a single jitter
                  spike can't fire. Default 250.
  use_pose (bool): track Pose wrists (15/16) instead of Hands wrists. Default False.
  min_visibility (float): Pose wrist visibility gate (use_pose mode). Default 0.5.
  min_speed_mm_s (float): METRIC burst threshold used when the Gemini 335 depth
                  sampler provides real wrist depth — the same physical hand
                  speed fires at any distance from the camera. Default 800 mm/s.
                  (The idle_velocity × multiplier path is the webcam fallback.)

Context keys: speed_burst_times, prev_wrist_pos, prev_wrist_time
"""

import math
import time

from ...depth.fusion import PoseDepth, metric_point, trusted_landmark

_MAX_DT_S = 0.25   # gap longer than this = tracking dropout, not motion


def _reset(context: dict) -> None:
    """Signal lost or untrustworthy — restart motion tracking from scratch.
    Clearing burst history matters as much as prev position: bursts accumulated
    before a dropout must not count toward a fire after the signal returns."""
    context["prev_wrist_pos"] = None
    context["prev_wrist_time"] = None
    context["speed_burst_times"] = []


def detect(landmarks, params: dict, context: dict) -> bool:
    # POSE-ONLY (July 2026): always tracks the Pose wrists; the old Hands-wrist
    # mode is gone (use_pose is accepted and ignored).
    # Collect currently-trusted wrist positions keyed by identity, so velocity
    # is always computed against the same physical wrist. With the Gemini depth
    # sampler present, positions carry real depth so speeds become mm/s.
    fusion = context.get("_pose_depth")
    if not isinstance(fusion, PoseDepth):
        fusion = None
    current: dict = {}
    pose_lm = context.get("_pose_lm")
    if pose_lm is None:
        _reset(context)
        return False
    min_visibility = params.get("min_visibility", 0.5)
    for key, idx in (("L", 15), ("R", 16)):
        try:
            w = pose_lm[idx]
        except (IndexError, TypeError):
            continue
        # Phantom low-visibility estimates jitter frame-to-frame and read
        # as motion — skip them (visibility gate + Gemini depth veto), but
        # keep the other wrist if it's trusted.
        if not trusted_landmark(context, idx, w, min_visibility):
            continue
        depth = fusion.landmark_mm(idx) if fusion is not None else None
        current[key] = (w.x, w.y, depth)
    if not current:
        _reset(context)
        return False

    if "speed_burst_times" not in context:
        _reset(context)

    min_bursts      = params.get("min_bursts", 3)
    burst_window_ms = params.get("burst_window_ms", 3000)
    multiplier      = params.get("velocity_multiplier", 2.0)
    idle_velocity   = params.get("idle_velocity", 0.45)
    min_active_ms   = params.get("min_active_ms", 250)
    min_speed_mm_s  = params.get("min_speed_mm_s", 800)
    threshold = idle_velocity * multiplier

    now = time.monotonic()
    prev_pos: dict = context.get("prev_wrist_pos") or {}
    prev_time = context.get("prev_wrist_time")

    if prev_time is not None:
        dt = now - prev_time
        if 0 < dt <= _MAX_DT_S:
            # Fastest currently-tracked wrist. When both frames carry real
            # depth (Gemini 335), speed is METRIC (mm/s) — the same physical
            # shake fires identically at 1 m and at 3 m from the camera. The
            # normalised units/s path is the webcam fallback.
            burst = False
            for key, (cx, cy, cd) in current.items():
                prev = prev_pos.get(key)
                if prev is None:
                    continue
                px, py, pd = prev
                if cd is not None and pd is not None:
                    v = math.dist(metric_point(px, py, pd),
                                  metric_point(cx, cy, cd)) / dt
                    if v >= min_speed_mm_s:
                        burst = True
                else:
                    if math.hypot(cx - px, cy - py) / dt >= threshold:
                        burst = True
            if burst:
                context["speed_burst_times"].append(now)
        elif dt > _MAX_DT_S:
            context["speed_burst_times"] = []   # dropout — stale bursts out

    # Trim expired bursts
    window_s = burst_window_ms / 1000.0
    context["speed_burst_times"] = [
        t for t in context["speed_burst_times"] if now - t <= window_s
    ]
    context["prev_wrist_pos"] = current
    context["prev_wrist_time"] = now

    bursts = context["speed_burst_times"]
    if len(bursts) >= min_bursts and (bursts[-1] - bursts[0]) * 1000 >= min_active_ms:
        context["speed_burst_times"] = []  # reset so it can fire again after cooldown
        return True
    return False
