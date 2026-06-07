"""
survey — two-phase "shade eyes and scan" gesture. Used for AL-03-004 (Scene 3).

Replaces the single-phase shade_eyes detector, which fired on any hand-at-brow
pose including a visitor resting their hand on their forehead. The two-phase design
requires the hand to actually move at brow height (a scanning motion), enforcing the
storyboard's intent.

Phase 1 (hand-up): wrist landmark (lm0) is above brow_y threshold.
Phase 2 (scan):    while the wrist stays at brow height, the wrist X coordinate
                   changes direction at least once within scan_window_ms.
                   "Changes direction" = the sign of the X velocity reverses.

Both phases must complete in order. A static hand at brow height does NOT fire.

Params:
  brow_y (float): Y threshold for "brow height." Lower value = higher on screen.
                  Default 0.45 (upper ~45% of frame). Override with live pose Y
                  for accurate per-player tracking.
  scan_window_ms (float): time window (ms) allowed to complete the scan phase
                          after the hand first reaches brow height. Default 2000.
  min_x_delta (float): minimum wrist X change (normalised, per direction leg)
                       required to count as a scan. Default 0.06 (6% of frame width).
  hold_ms (float): ms the hand must remain at brow height before the scan phase
                   begins (prevents accidental triggers). Default 200.

Context keys: survey_phase, survey_phase1_since, scan_started_at, scan_x_history,
              scan_direction_changes
"""

import time

_HISTORY_MAX = 60  # keep up to 2 seconds of X samples at 30fps


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks:
        _reset(context)
        return False

    brow_y = params.get("brow_y", 0.45)
    scan_window_ms = params.get("scan_window_ms", 2000)
    min_x_delta = params.get("min_x_delta", 0.06)
    hold_ms = params.get("hold_ms", 200)

    # Use the wrist of any hand (prefer the first visible)
    wrist = landmarks[0].landmark[0]
    wrist_x = wrist.x  # raw camera x (mirroring doesn't matter for delta tracking)
    wrist_y = wrist.y

    at_brow = wrist_y < brow_y  # wrist is at or above brow height
    now = time.monotonic()
    phase = context.get("survey_phase", 0)

    # ── Phase 0 → 1: hand rises to brow level ──────────────────────────────
    if phase == 0:
        if at_brow:
            context["survey_phase"] = 1
            context["survey_phase1_since"] = now
            context["scan_x_history"] = [wrist_x]
        return False

    # ── Phase 1: wait for hold_ms, then start accepting scan data ──────────
    if phase == 1:
        if not at_brow:
            _reset(context)
            return False
        phase1_since = context.get("survey_phase1_since", now)
        context["scan_x_history"].append(wrist_x)
        if len(context["scan_x_history"]) > _HISTORY_MAX:
            context["scan_x_history"].pop(0)

        if (now - phase1_since) * 1000 < hold_ms:
            return False  # still in hold phase

        # Held long enough — move to scan-detection phase
        context["survey_phase"] = 2
        context["scan_started_at"] = now
        context["scan_direction_changes"] = 0
        context["scan_last_direction"] = None
        context["scan_segment_start_x"] = wrist_x
        return False

    # ── Phase 2: detect at least one direction change ───────────────────────
    if phase == 2:
        if not at_brow:
            _reset(context)
            return False

        # Scan window expired?
        scan_started = context.get("scan_started_at", now)
        if (now - scan_started) * 1000 > scan_window_ms:
            _reset(context)
            return False

        seg_start = context.get("scan_segment_start_x", wrist_x)
        dx = wrist_x - seg_start
        last_dir = context.get("scan_last_direction")

        if abs(dx) >= min_x_delta:
            current_dir = "right" if dx > 0 else "left"
            if last_dir is not None and current_dir != last_dir:
                # Direction reversed — that's a scan
                context["survey_phase"] = 3  # done
                return True
            context["scan_last_direction"] = current_dir
            context["scan_segment_start_x"] = wrist_x  # start new segment

    return False


def _reset(context: dict) -> None:
    context["survey_phase"] = 0
    context["survey_phase1_since"] = None
    context["scan_started_at"] = None
    context["scan_x_history"] = []
    context["scan_direction_changes"] = 0
    context["scan_last_direction"] = None
    context["scan_segment_start_x"] = None
