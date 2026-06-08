"""
rhythm_bilateral — three forward knocks in a short-short-LONG pattern.
Used for Scene 6 `three_knock` CG (AL-06-007).

Pattern:  knock — knock ... KNOCK
  Knocks 1 and 2 must arrive within short_window_ms of each other.
  Knock 3 must arrive no sooner than long_min_ms after knock 2, and no
  later than long_max_ms after knock 2.
  Any timing violation resets the sequence to zero.

Knock geometry (raised threshold):
  A knock fires when the index MCP knuckle (lm5) descends below a threshold line
  positioned 30% above the wrist toward the top of the hand bounding box.
  This is more lenient than the old "knuckle below wrist" check — a shallower
  forward thrust now counts as a knock.

  threshold_y = wrist_y - 0.30 * hand_height
  where hand_height = max(wrist_y - hand_top_y, 0.05)  (0.05 floor prevents
  degenerate case when the hand bbox is nearly flat)

  A knock event fires on the TRANSITION from above-threshold to below-threshold.
  The knuckle must reset above the threshold before the next knock can register.

Params:
  short_window_ms (int): max gap allowed between knock 1 and knock 2. Default 500.
  long_min_ms (int): minimum gap required between knock 2 and knock 3. Default 400.
  long_max_ms (int): maximum gap allowed between knock 2 and knock 3. Default 1400.
  threshold_fraction (float): fraction of hand height above wrist that sets the
                              threshold line. Default 0.30. Increase to fire on
                              even shallower motions; decrease to require more dip.
  refractory_ms (int): minimum time between any two detected knocks. Default 220.
  wrist_y_offset (float): shifts the threshold line up (negative) or down (positive)
                          in normalised screen coords. Default 0.0. Tune with W/S in
                          gesture_tuner.py; persisted to host profile.

Context keys: knock_count, knock_times, knock_refractory_until, knuckle_below_threshold
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    if "knock_count" not in context:
        context["knock_count"] = 0
        context["knock_times"] = []
        context["knock_refractory_until"] = 0.0
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

    # ── Detect a knock event (geometric threshold) ──────────────────────────
    lm = landmarks[0].landmark
    wrist_y   = lm[0].y
    knuckle_y = lm[5].y  # index MCP
    hand_top_y = min(lm[i].y for i in range(21))
    hand_height = max(wrist_y - hand_top_y, 0.05)
    threshold_y = wrist_y - threshold_fraction * hand_height + wrist_y_offset

    # "below threshold" means the knuckle has descended past the threshold line
    knuckle_below = knuckle_y >= threshold_y
    prev_below = context["knuckle_below_threshold"]
    context["knuckle_below_threshold"] = knuckle_below

    new_knock = False
    if knuckle_below and not prev_below and now >= context["knock_refractory_until"]:
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
