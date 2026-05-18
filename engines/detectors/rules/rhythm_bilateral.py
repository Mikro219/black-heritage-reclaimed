"""
rhythm_bilateral — three forward knocks in a short-short-LONG pattern.
Used for Scene 6 `three_knock` CG (AL-06-007).

Pattern:  knock — knock ... KNOCK
  Knocks 1 and 2 must arrive within short_window_ms of each other.
  Knock 3 must arrive no sooner than long_min_ms after knock 2, and no
  later than long_max_ms after knock 2.
  Any timing violation resets the sequence to zero.

Params:
  short_window_ms (int): max gap allowed between knock 1 and knock 2. Default 500.
  long_min_ms (int): minimum gap required between knock 2 and knock 3. Default 400.
  long_max_ms (int): maximum gap allowed between knock 2 and knock 3. Default 1400.
  min_push (float): minimum wrist Y delta (downward) to register one knock. Default 0.04.
  refractory_ms (int): minimum time between any two detected knocks. Default 220.

Context keys: knock_count, knock_times, knock_refractory_until, prev_wrist_y
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    if "knock_count" not in context:
        context["knock_count"] = 0
        context["knock_times"] = []
        context["knock_refractory_until"] = 0.0
        context["prev_wrist_y"] = None

    if not landmarks:
        return False

    short_window_ms = params.get("short_window_ms", 500)
    long_min_ms     = params.get("long_min_ms",     400)
    long_max_ms     = params.get("long_max_ms",     1400)
    min_push        = params.get("min_push",         0.04)
    refractory_ms   = params.get("refractory_ms",   220)

    now = time.monotonic()

    # ── Detect a knock event ────────────────────────────────────────────────
    wrist_y = landmarks[0].landmark[0].y
    prev_y = context["prev_wrist_y"]
    context["prev_wrist_y"] = wrist_y

    new_knock = False
    if prev_y is not None and now >= context["knock_refractory_until"]:
        if wrist_y - prev_y >= min_push:   # downward thrust
            new_knock = True
            context["knock_refractory_until"] = now + refractory_ms / 1000.0

    # ── Timeout checks (run before processing new knock) ────────────────────
    count = context["knock_count"]
    times = context["knock_times"]

    if count == 1 and times:
        if (now - times[0]) * 1000 > short_window_ms:
            # Knock 2 arrived too late — restart
            context["knock_count"] = 0
            context["knock_times"] = []
            count = 0

    elif count == 2 and times:
        if (now - times[1]) * 1000 > long_max_ms:
            # Knock 3 never arrived — restart
            context["knock_count"] = 0
            context["knock_times"] = []
            count = 0

    # ── State transitions on new knock ──────────────────────────────────────
    if new_knock:
        if count == 0:
            context["knock_count"] = 1
            context["knock_times"] = [now]

        elif count == 1:
            elapsed_ms = (now - times[0]) * 1000
            if elapsed_ms <= short_window_ms:
                context["knock_count"] = 2
                context["knock_times"].append(now)
            else:
                # Arrived after window expired (timeout already reset count to 0 above,
                # but guard here in case same frame); start fresh sequence
                context["knock_count"] = 1
                context["knock_times"] = [now]

        elif count == 2:
            elapsed_ms = (now - times[1]) * 1000
            if elapsed_ms < long_min_ms:
                # Too soon — accidental tap, restart
                context["knock_count"] = 0
                context["knock_times"] = []
            elif elapsed_ms <= long_max_ms:
                # Perfect long knock — success
                context["knock_count"] = 0
                context["knock_times"] = []
                return True
            # > long_max_ms: timeout check above already reset; won't reach here

    return False
