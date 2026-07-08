"""
rhythm_bilateral — knocking at the door. Used for Scene 6 `three_knock` (AL-06-007).

SIMPLIFIED (July 2026 playtest): the short-short-LONG dip-pattern proved
undiscoverable — players knock TOWARD the screen (a forward punch), which barely
moves the knuckle vertically. Per Mike's punch list, a knock is now "the hand
gets bigger": a forward push toward the camera grows the hand's apparent size.
Default fires on knock_count (2) forward-push knocks inside knock_window_ms.

Knock event (approach mode):
  Track the largest Hands detection's bounding-box area each frame. Keep a short
  rolling history; the baseline is the smallest area seen in the last
  baseline_window_ms. A knock fires when the current area reaches
  baseline * (1 + min_growth_pct/100) — i.e. the hand visibly lunged toward the
  camera. After a knock, the baseline resets to the current (big) area, so the
  hand must pull BACK (area shrinks) before the next knock can register — two
  pushes require two distinct forward lunges.

  When the capture device provides real depth (Orbbec Gemini 335 sampler in
  context["_depth_mm_at"]), a knock instead fires when the hand-wrist depth
  drops by min_depth_delta_mm relative to its rolling max — same push, measured
  in millimetres instead of pixels.

Legacy short-short-LONG mode survives behind `mode: "dip"` (old params
short_window_ms / long_min_ms / long_max_ms / threshold_fraction /
wrist_y_offset unchanged, read only in that mode).

Params (approach mode):
  knock_count (int): forward pushes required. Default 2.
  knock_window_ms (int): window in which they must all land. Default 2500.
  min_growth_pct (float): required bbox-area growth over the rolling baseline.
                  Default 30.
  min_depth_delta_mm (float): required depth decrease per knock when real depth
                  is available. Default 120.
  baseline_window_ms (int): rolling window for the pulled-back baseline. Default 1200.
  refractory_ms (int): minimum time between knocks. Default 300.

Context keys: knock_area_hist, knock_times, knock_refractory_until,
              knock_depth_hist  (legacy dip mode keeps its old keys)
"""

import time

from . import hand_pose


def detect(landmarks, params: dict, context: dict) -> bool:
    if params.get("mode", "approach") == "dip":
        return _detect_dip(landmarks, params, context)
    return _detect_approach(landmarks, params, context)


# ---------------------------------------------------------------------------
# Approach mode (default): knock = hand grows bigger / comes closer, twice.
# ---------------------------------------------------------------------------

def _detect_approach(landmarks, params: dict, context: dict) -> bool:
    knock_count      = params.get("knock_count", 2)
    knock_window_ms  = params.get("knock_window_ms", 2500)
    min_growth       = params.get("min_growth_pct", 30) / 100.0
    min_depth_delta  = params.get("min_depth_delta_mm", 120)
    baseline_ms      = params.get("baseline_window_ms", 1200)
    refractory_ms    = params.get("refractory_ms", 300)

    now = time.monotonic()
    times: list = context.setdefault("knock_times", [])
    context.setdefault("knock_refractory_until", 0.0)

    # Trim expired knocks first so a stale first knock can't pair with a fresh one.
    times[:] = [t for t in times if (now - t) * 1000 <= knock_window_ms]

    new_knock = False

    # ── Preferred signal: real depth at the hand wrist (Orbbec) ─────────────
    depth_at = context.get("_depth_mm_at")
    depth_mm = None
    if callable(depth_at) and landmarks:
        hand = max(landmarks, key=hand_pose.bbox_area)
        hw = hand.landmark[0]
        depth_mm = depth_at(hw.x, hw.y)

    if depth_mm is not None:
        hist: list = context.setdefault("knock_depth_hist", [])
        hist.append((now, depth_mm))
        cutoff = now - baseline_ms / 1000.0
        hist[:] = [(t, d) for t, d in hist if t >= cutoff]
        baseline = max(d for _, d in hist)   # farthest (pulled back) recently
        if (baseline - depth_mm) >= min_depth_delta and now >= context["knock_refractory_until"]:
            new_knock = True
            # Forget the pulled-back past: the hand must retreat again before
            # the next knock can register.
            hist[:] = [(now, depth_mm)]

    # ── Fallback signal: hand bbox area growth (any camera) ─────────────────
    elif landmarks:
        area = max(hand_pose.bbox_area(hand) for hand in landmarks)
        hist = context.setdefault("knock_area_hist", [])
        hist.append((now, area))
        cutoff = now - baseline_ms / 1000.0
        hist[:] = [(t, a) for t, a in hist if t >= cutoff]
        baseline = min(a for _, a in hist)   # smallest (pulled back) recently
        if baseline > 0 and area >= baseline * (1.0 + min_growth) \
                and now >= context["knock_refractory_until"]:
            new_knock = True
            hist[:] = [(now, area)]

    if new_knock:
        context["knock_refractory_until"] = now + refractory_ms / 1000.0
        times.append(now)
        if len(times) >= knock_count:
            context["knock_times"] = []
            context["knock_area_hist"] = []
            context["knock_depth_hist"] = []
            return True

    return False


# ---------------------------------------------------------------------------
# Legacy dip mode (mode: "dip"): short-short-LONG knuckle dips.
# ---------------------------------------------------------------------------

def _detect_dip(landmarks, params: dict, context: dict) -> bool:
    if "dip_count" not in context:
        context["dip_count"] = 0
        context["dip_times"] = []
        context["dip_refractory_until"] = 0.0
        context["knuckle_below_threshold"] = False

    if not landmarks:
        return False

    short_window_ms    = params.get("short_window_ms",    500)
    long_min_ms        = params.get("long_min_ms",         400)
    long_max_ms        = params.get("long_max_ms",        1400)
    threshold_fraction = params.get("threshold_fraction",  0.30)
    refractory_ms      = params.get("refractory_ms",       220)
    wrist_y_offset     = params.get("wrist_y_offset",      0.0)

    now = time.monotonic()

    lm = landmarks[0].landmark
    wrist_y   = lm[0].y
    knuckle_y = lm[5].y  # index MCP
    hand_top_y = min(lm[i].y for i in range(21))
    hand_height = max(wrist_y - hand_top_y, 0.05)
    threshold_y = wrist_y - threshold_fraction * hand_height + wrist_y_offset

    knuckle_below = knuckle_y >= threshold_y
    prev_below = context["knuckle_below_threshold"]
    context["knuckle_below_threshold"] = knuckle_below

    new_knock = False
    if knuckle_below and not prev_below and now >= context["dip_refractory_until"]:
        new_knock = True
        context["dip_refractory_until"] = now + refractory_ms / 1000.0

    count = context["dip_count"]
    times = context["dip_times"]

    if count == 1 and times:
        if (now - times[0]) * 1000 > short_window_ms:
            context["dip_count"] = 0
            context["dip_times"] = []
            count = 0
    elif count == 2 and times:
        if (now - times[1]) * 1000 > long_max_ms:
            context["dip_count"] = 0
            context["dip_times"] = []
            count = 0

    if new_knock:
        if count == 0:
            context["dip_count"] = 1
            context["dip_times"] = [now]
        elif count == 1:
            if (now - times[0]) * 1000 <= short_window_ms:
                context["dip_count"] = 2
                context["dip_times"].append(now)
            else:
                context["dip_count"] = 1
                context["dip_times"] = [now]
        elif count == 2:
            elapsed_ms = (now - times[1]) * 1000
            if elapsed_ms < long_min_ms:
                context["dip_count"] = 0
                context["dip_times"] = []
            elif elapsed_ms <= long_max_ms:
                context["dip_count"] = 0
                context["dip_times"] = []
                return True

    return False
