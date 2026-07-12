"""
rhythm_bilateral — knocking at the door. Used for Scene 6 `three_knock` (AL-06-007).

A knock is a push toward the camera; the default fires on knock_count (2)
pushes inside knock_window_ms. POSE-ONLY (July 2026): the push is measured at
the Pose wrists — a REAL depth drop of min_depth_delta_mm against the rolling
pulled-back baseline when the Gemini 335 sampler is present, else a Pose z
drop of min_pose_z_delta. After a knock the baseline resets to the current
(near) value, so the hand must pull BACK before the next knock can register —
two knocks require two distinct forward lunges. Either wrist can knock.

The old hand-bbox growth proxy and the legacy short-short-LONG dip mode both
needed the Hands model and are gone (`mode` is accepted and ignored; legacy
dip params in metadata are ignored).

Params:
  knock_count (int): forward pushes required. Default 2.
  knock_window_ms (int): window in which they must all land. Default 2500.
  min_depth_delta_mm (float): depth decrease per knock (sampler). Default 120.
  min_pose_z_delta (float): Pose-z decrease per knock (no sampler). Default 0.12.
  baseline_window_ms (int): rolling window for the pulled-back baseline.
                  Default 1200.
  refractory_ms (int): minimum time between knocks. Default 300.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: knock_times, knock_refractory_until, knock_depth_hist
  (reads: _pose_lm, _depth_mm_at)
"""

import time

from . import pose_helpers


def detect(landmarks, params: dict, context: dict) -> bool:
    knock_count = params.get("knock_count", 2)
    knock_window_ms = params.get("knock_window_ms", 2500)
    min_mm = params.get("min_depth_delta_mm", 120)
    min_z = params.get("min_pose_z_delta", 0.12)
    baseline_ms = params.get("baseline_window_ms", 1200)
    refractory_ms = params.get("refractory_ms", 300)

    now = time.monotonic()
    times: list = context.setdefault("knock_times", [])
    context.setdefault("knock_refractory_until", 0.0)

    # Trim expired knocks so a stale first knock can't pair with a fresh one.
    times[:] = [t for t in times if (now - t) * 1000 <= knock_window_ms]

    pose_lm = context.get("_pose_lm")
    wrists = pose_helpers.trusted_wrists(pose_lm, context,
                                         params.get("min_visibility", 0.5))
    if not wrists:
        return False

    hist: dict = context.setdefault("knock_depth_hist", {})
    cutoff = now - baseline_ms / 1000.0

    new_knock = False
    for side in wrists:
        kind, val = pose_helpers.wrist_depth_or_z(context, pose_lm, side)
        if kind is None:
            continue
        h = hist.setdefault(side, [])
        h.append((now, kind, val))
        h[:] = [(t, k, v) for t, k, v in h if t >= cutoff]
        same = [v for t, k, v in h if k == kind]
        if len(same) < 2:
            continue
        # Baseline = farthest (pulled back) recently: max depth / max pose z.
        baseline = max(same)
        threshold = min_mm if kind == "mm" else min_z
        if (baseline - val) >= threshold and now >= context["knock_refractory_until"]:
            new_knock = True
            # Forget the pulled-back past: the hand must retreat again before
            # the next knock can register.
            h[:] = [(now, kind, val)]
            break

    if new_knock:
        context["knock_refractory_until"] = now + refractory_ms / 1000.0
        times.append(now)
        if len(times) >= knock_count:
            context["knock_times"] = []
            context["knock_depth_hist"] = {}
            return True

    return False
