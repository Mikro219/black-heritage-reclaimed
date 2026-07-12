"""
reach_and_close — single hand reaches forward, then settles (takes hold).
Used for Scene 6 clasp-hand OI (AL-06-009).

POSE-ONLY SEMANTIC CHANGE (July 2026): finger curl is invisible to the Pose
model, so "closes into a grasp" can't be observed directly. The gesture is now
REACH + SETTLE: a wrist approaches the camera (real depth with the Gemini
sampler, pose-z fallback) and then holds nearly still — the natural end of
"reach out and take it". Detect intent, not precision.

Params:
  min_depth_delta_mm (float): wrist depth decrease that counts as the reach
                  (Gemini sampler). Default 150.
  min_pose_z_delta (float): pose-z decrease fallback (more negative = closer).
                  Default 0.15.
  reach_window_ms (int): window for the approach. Default 1500.
  settle_ms (int): stillness required after the reach. Default 350.
  settle_velocity (float): max wrist speed (normalised units/s) that counts as
                  settled. Default 0.25.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: reach_z_hist, reach_reached, reach_settle_since, reach_prev
  (reads: _pose_lm, _depth_mm_at)
"""

import math
import time

from . import pose_helpers


def _reset(context: dict) -> None:
    context["reach_z_hist"] = {}
    context["reach_reached"] = None
    context["reach_settle_since"] = None
    context["reach_prev"] = None


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if not wrists:
        _reset(context)
        return False

    min_mm = params.get("min_depth_delta_mm", 150)
    min_z = params.get("min_pose_z_delta", 0.15)
    window_s = params.get("reach_window_ms", 1500) / 1000.0
    settle_ms = params.get("settle_ms", 350)
    settle_v = params.get("settle_velocity", 0.25)

    now = time.monotonic()
    hist: dict = context.setdefault("reach_z_hist", {})

    # Phase 1 — reach: any wrist approaching the camera within the window.
    reached_side = context.get("reach_reached")
    if reached_side is None:
        for side in wrists:
            kind, val = pose_helpers.wrist_depth_or_z(context, pose_lm, side)
            if kind is None:
                continue
            h = hist.setdefault(side, [])
            h.append((now, kind, val))
            h[:] = [(t, k, v) for t, k, v in h if now - t <= window_s]
            same = [(t, v) for t, k, v in h if k == kind]
            if len(same) >= 2:
                drop = max(v for _, v in same) - val
                if (kind == "mm" and drop >= min_mm) or (kind == "z" and drop >= min_z):
                    context["reach_reached"] = side
                    context["reach_settle_since"] = None
                    context["reach_prev"] = None
                    break
        return False

    # Phase 2 — settle: the reaching wrist holds nearly still.
    wrist = wrists.get(reached_side)
    if wrist is None:
        _reset(context)
        return False
    prev = context.get("reach_prev")
    context["reach_prev"] = (wrist.x, wrist.y, now)
    if prev is None:
        return False
    px, py, pt = prev
    dt = now - pt
    if dt <= 0:
        return False
    speed = math.hypot(wrist.x - px, wrist.y - py) / dt
    if speed <= settle_v:
        if context.get("reach_settle_since") is None:
            context["reach_settle_since"] = now
        if (now - context["reach_settle_since"]) * 1000 >= settle_ms:
            _reset(context)
            return True
    else:
        context["reach_settle_since"] = None
    return False
