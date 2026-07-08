"""
run_arms — arms pump fast in alternating above/below-waist motion.

Used by: AL-10-010 run_arms (OI tier, Scene 10).

Uses Pose landmarks for waist reference and labeled wrist positions.
Distinguishes "running" from walking via the time constraint: all 3 cycles
must complete within window_ms (default 2000ms). A slow walk would spread
cycles across several seconds; running pumps them through quickly.

Body geometry (Pose landmarks, read from context["_pose_lm"]):
  15 (left_wrist),  16 (right_wrist)  → tracked wrists
  23 (left_hip),    24 (right_hip)    → waist_y = average hip Y

Detection logic:
  waist_y = (hip_l.y + hip_r.y) / 2
  At each frame, compute:
    left_above  = lw.y < waist_y
    right_above = rw.y < waist_y

  An alternating state is (True, False) or (False, True).
  Track state-change timestamps in a circular buffer.
  A "cycle" = two consecutive state transitions (left/right swap positions twice).
  Fire when min_cycles * 2 transitions accumulate within window_ms.

Params:
  min_cycles (int): complete alternating cycles required. Default 3.
  window_ms (float): all cycles must finish within this window. Default 2000.
  reference (str): "waist" (default, shoulder/hip midpoint) or "hips" (legacy
                   hip line) for the alternating reference line.
  waist_y_offset (float): shifts the reference line; tuned via W/S in the tuner.

Fallback (no Pose): return False.

Context keys: run_alt_transitions (list[float of timestamps]), run_prev_alt_state
"""

import time


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    if pose_lm is None:
        return False

    min_cycles = params.get("min_cycles", 3)
    window_s   = params.get("window_ms", 2000) / 1000.0
    required_transitions = min_cycles * 2

    hip_l, hip_r = pose_lm[23], pose_lm[24]
    sh_l, sh_r   = pose_lm[11], pose_lm[12]
    lw, rw       = pose_lm[15], pose_lm[16]

    # Reference line for the alternating pump. The original used the HIP line,
    # which a natural running arm-pump (elbows bent, hands at waist/chest) almost
    # never crosses — that's why the gesture read as "unclear what it's looking
    # for". Default is now the waist: midway between shoulders and hips.
    hip_y      = (hip_l.y + hip_r.y) / 2
    shoulder_y = (sh_l.y + sh_r.y) / 2
    if params.get("reference", "waist") == "hips":
        waist_y = hip_y + params.get("waist_y_offset", 0.0)
    else:
        waist_y = (shoulder_y + hip_y) / 2 + params.get("waist_y_offset", 0.0)
    left_above   = lw.y < waist_y
    right_above  = rw.y < waist_y

    # Only meaningful when wrists are in opposite positions
    alt_state = (left_above, right_above)
    is_alternating = alt_state in {(True, False), (False, True)}

    prev_state = context.get("run_prev_alt_state")
    transitions: list = context.setdefault("run_alt_transitions", [])

    now = time.monotonic()

    if is_alternating and alt_state != prev_state:
        transitions.append(now)
        context["run_prev_alt_state"] = alt_state
    elif not is_alternating:
        context["run_prev_alt_state"] = None

    # Prune transitions outside the window
    context["run_alt_transitions"] = [t for t in transitions if now - t <= window_s]
    transitions = context["run_alt_transitions"]

    if len(transitions) >= required_transitions:
        # Reset so it can fire again within the same OI window if needed
        context["run_alt_transitions"] = []
        context["run_prev_alt_state"] = None
        return True

    return False
