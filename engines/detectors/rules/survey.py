"""
survey — salute: hand held at brow height. Used for AL-03-004 (Scene 3) and the
Shoofly / Bear's Paw scan OIs (shots 30, 36).

SIMPLIFIED (July 2026 playtest): the two-phase "shade eyes and scan" model —
hand at brow PLUS a side-to-side X sweep — proved too hard to discover and too
easy to reset mid-gesture. Per Mike's punch list the gesture is now just a
salute: a visible Pose wrist held at/above brow height for salute_hold_ms.
The old scan phase survives behind `require_scan: true` for shots that ever
want the stricter motion back; existing metadata (scan_window_ms, min_x_delta,
hold_ms) is still accepted and only read in that mode.

POSE-BASED: tracks MediaPipe Pose wrists (landmarks 15/16), not Hands — a hand
at the brow frequently occludes the face and drops out of the Hands model, and
this detector runs in the gesture engine's _NO_HANDS_DETECTORS set. Phantom
low-visibility wrists are ignored. Once a wrist arms at the brow the detector
stays locked to that side, so the hold (or scan history) can't jump between
hands mid-gesture.

Params:
  brow_y (float): Y threshold for "brow height." Lower value = higher on screen.
                  Default 0.45. Overridden with live pose eye-level Y by
                  _inject_pose_params for per-player tracking.
  salute_hold_ms (float): ms the wrist must stay at brow height to fire in
                  salute mode. Default 500. (Independent of the legacy hold_ms,
                  which shot metadata set to 200 as a pre-scan debounce.)
  require_scan (bool): restore the old two-phase behaviour. Default False.
  scan_window_ms / min_x_delta / hold_ms: legacy scan-mode params, unchanged.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: survey_phase, survey_side, survey_phase1_since, scan_started_at,
              scan_x_history, scan_direction_changes
"""

import time

from ...depth.fusion import trusted_landmark

_HISTORY_MAX = 60  # keep up to 2 seconds of X samples at 30fps

_WRISTS = {"L": 15, "R": 16}  # side -> Pose wrist landmark index


def _tracked_wrist(pose_lm, context: dict, min_visibility: float):
    """Return (side, wrist_landmark) or (None, None).

    Before arming (no survey_side yet): the highest visible wrist is the
    candidate. After arming: locked to the side that armed, so the hold/scan
    follows one physical hand. Wrists are gated on visibility AND (when the
    Gemini depth sampler is present) depth plausibility.
    """
    side = context.get("survey_side")
    if side is not None:
        try:
            wrist = pose_lm[_WRISTS[side]]
        except (IndexError, TypeError):
            return None, None
        if not trusted_landmark(context, _WRISTS[side], wrist, min_visibility):
            return None, None
        return side, wrist

    best = (None, None)
    best_y = None
    for s, idx in _WRISTS.items():
        try:
            wrist = pose_lm[idx]
        except (IndexError, TypeError):
            continue
        if not trusted_landmark(context, idx, wrist, min_visibility):
            continue
        if best_y is None or wrist.y < best_y:   # highest on screen wins
            best_y = wrist.y
            best = (s, wrist)
    return best


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    if pose_lm is None:
        _reset(context)
        return False

    brow_y = params.get("brow_y", 0.45)
    min_visibility = params.get("min_visibility", 0.5)

    side, wrist = _tracked_wrist(pose_lm, context, min_visibility)
    if wrist is None:
        _reset(context)
        return False

    at_brow = wrist.y < brow_y  # wrist is at or above brow height
    now = time.monotonic()

    if not params.get("require_scan", False):
        # ── Salute mode (default): hold at brow for salute_hold_ms ──────────
        salute_hold_ms = params.get("salute_hold_ms", 500)
        if not at_brow:
            _reset(context)
            return False
        if context.get("survey_phase", 0) == 0:
            context["survey_phase"] = 1
            context["survey_side"] = side          # lock onto this hand
            context["survey_phase1_since"] = now
            return False
        since = context.get("survey_phase1_since", now)
        if (now - since) * 1000 >= salute_hold_ms:
            _reset(context)   # re-armable after the hand drops and returns
            return True
        return False

    # ── Legacy scan mode (require_scan: true) ───────────────────────────────
    scan_window_ms = params.get("scan_window_ms", 2000)
    min_x_delta = params.get("min_x_delta", 0.06)
    hold_ms = params.get("hold_ms", 200)

    wrist_x = wrist.x  # raw camera x (mirroring doesn't matter for delta tracking)
    phase = context.get("survey_phase", 0)

    # Phase 0 → 1: hand rises to brow level
    if phase == 0:
        if at_brow:
            context["survey_phase"] = 1
            context["survey_side"] = side
            context["survey_phase1_since"] = now
            context["scan_x_history"] = [wrist_x]
        return False

    # Phase 1: wait for hold_ms, then start accepting scan data
    if phase == 1:
        if not at_brow:
            _reset(context)
            return False
        phase1_since = context.get("survey_phase1_since", now)
        context.setdefault("scan_x_history", []).append(wrist_x)
        if len(context["scan_x_history"]) > _HISTORY_MAX:
            context["scan_x_history"].pop(0)

        if (now - phase1_since) * 1000 < hold_ms:
            return False  # still in hold phase

        context["survey_phase"] = 2
        context["scan_started_at"] = now
        context["scan_direction_changes"] = 0
        context["scan_last_direction"] = None
        context["scan_segment_start_x"] = wrist_x
        return False

    # Phase 2: detect at least one direction change
    if phase == 2:
        if not at_brow:
            _reset(context)
            return False

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
                _reset(context)
                return True
            context["scan_last_direction"] = current_dir
            context["scan_segment_start_x"] = wrist_x  # start new segment

    return False


def _reset(context: dict) -> None:
    context["survey_phase"] = 0
    context["survey_side"] = None
    context["survey_phase1_since"] = None
    context["scan_started_at"] = None
    context["scan_x_history"] = []
    context["scan_direction_changes"] = 0
    context["scan_last_direction"] = None
    context["scan_segment_start_x"] = None
