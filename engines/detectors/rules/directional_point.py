"""
directional_point — single hand pointing in a direction, held for hold_ms.
Used for Scene 2 `point_path` CG, Scene 7 `point_fork` CG, Scene 9 `choose_path` CG.

Params:
  directions (list[str]): accepted directions — any subset of the 8 compass points:
                          "left" | "right" | "up" | "down" |
                          "up_left" | "up_right" | "down_left" | "down_right"
                          If absent or empty, all 8 directions are accepted.
  hold_ms (int): ms the point must be held. Default 500.
  target_x (float): optional screen-space x (0–1, player perspective) of the target
                    marker. If provided along with target_y, the fingertip must be
                    within proximity_threshold of the marker to register.
  target_y (float): optional screen-space y (0–1) of the target marker.
  proximity_threshold (float): max normalised distance from marker. Default 0.30.
  min_extension (float): min normalised wrist→index-tip span. Default 0.10.
  require_point_pose (bool): if True (default), the index finger must be extended
                    past the middle finger — a real "point" pose. Rejects relaxed,
                    open, or withdrawing hands whose fingers are extended but not
                    pointing. Set False for legacy any-extended-finger behaviour.
  targets (dict): optional {direction_name: [x, y]} in player screen space (0–1,
                    x flipped so 0=screen-left). When given, the direction is chosen
                    by WHICH target the fingertip is nearest (position), not by the
                    finger-vector angle. The fingertip must be within
                    proximity_threshold of a target, making screen-centre a dead zone
                    so only a clear reach to one side registers.

Approach:
  Wrist (lm 0) → index fingertip (lm 8) vector; snap to the nearest of 8 compass
  directions using angle sectors of 45° each.
  Extension check: tip must be farther from wrist than PIP joint (distance-based,
  works for all directions including down).
  Point-pose check: index tip must be farther from wrist than the middle tip, so
  only a deliberate index point qualifies (the middle finger curls when pointing).

Context keys: point_direction_since, point_direction, dominant_direction
"""

import math
import time

_ALL_DIRECTIONS = ["left", "right", "up", "down",
                   "up_left", "up_right", "down_left", "down_right"]

# 8 compass directions in counterclockwise order starting from +x (right)
_DIRS_CW = ["right", "down_right", "down", "down_left",
            "left", "up_left", "up", "up_right"]


def _classify_direction(dx: float, dy: float) -> str:
    """Snap the (dx, dy) vector to the nearest of 8 compass directions.

    dx/dy are already in player screen-space (x flipped, y down=positive).
    Each direction occupies a 45° sector centred on the named angle.
    """
    angle = math.degrees(math.atan2(dy, dx)) % 360
    return _DIRS_CW[int((angle + 22.5) / 45) % 8]


def detect(landmarks, params: dict, context: dict) -> bool:
    if not landmarks:
        context["point_direction_since"] = None
        context["point_direction"] = None
        return False

    accepted = set(params.get("directions") or _ALL_DIRECTIONS)
    hold_ms = params.get("hold_ms", 500)
    target_x = params.get("target_x")
    target_y = params.get("target_y")
    proximity_threshold = params.get("proximity_threshold", 0.30)
    # Minimum wrist→fingertip span (normalised) for a hand to count as a deliberate
    # point. Filters out a relaxed second hand whose fingers happen to be extended.
    min_extension = params.get("min_extension", 0.10)
    require_point_pose = params.get("require_point_pose", True)
    # Position/target mode: {direction_name: [x, y]} in player screen space (0–1,
    # x flipped so 0=screen-left). When provided, the direction is decided by WHICH
    # target the fingertip is nearest — not by the finger-vector angle. The fingertip
    # must land within proximity_threshold of a target, so screen-centre is a dead
    # zone and only a clear left/right reach registers.
    targets = params.get("targets")

    # Evaluate every detected hand and keep only the single most-extended one.
    # A resting hand must not be able to win the direction: with two hands visible,
    # the previous "first match wins" logic let a steady, relaxed hand accumulate
    # hold-time and fire the wrong direction while the other hand pointed elsewhere.
    best_extension = 0.0
    best_direction = None
    for hand in landmarks:
        lm = hand.landmark
        wrist = lm[0]
        tip = lm[8]   # index fingertip
        pip = lm[6]   # index PIP joint

        # Require extended index finger: tip must be farther from wrist than PIP.
        # Distance-based check works for all directions including downward pointing.
        wrist_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        wrist_pip = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
        if wrist_tip <= wrist_pip:
            continue
        if wrist_tip < min_extension:
            continue   # too small a span to be a deliberate point

        # Require a genuine pointing pose: the index tip must reach farther from the
        # wrist than the middle tip. When pointing, the index extends and the middle
        # curls; a relaxed/open/withdrawing hand has the middle finger as long or
        # longer, so this rejects every non-pointing hand the dominant-hand rule
        # would otherwise latch onto when the real pointing hand leaves frame.
        if require_point_pose:
            middle_tip = lm[12]
            wrist_middle = math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y)
            if wrist_tip <= wrist_middle:
                continue

        screen_tip_x = 1.0 - tip.x   # mirror fingertip to player/screen space
        screen_tip_y = tip.y

        if targets:
            # Position mode: pick the nearest accepted target within proximity.
            hand_dir = None
            nearest = proximity_threshold
            for name, pos in targets.items():
                if name not in accepted:
                    continue
                d = math.hypot(screen_tip_x - pos[0], screen_tip_y - pos[1])
                if d <= nearest:
                    nearest = d
                    hand_dir = name
            if hand_dir is None:
                continue   # fingertip not near any accepted target
            if wrist_tip > best_extension:
                best_extension = wrist_tip
                best_direction = hand_dir
            continue

        # Optional single-target proximity check (legacy single-marker mode).
        if target_x is not None and target_y is not None:
            if math.hypot(screen_tip_x - target_x, screen_tip_y - target_y) > proximity_threshold:
                continue

        # Vector mode: classify by the wrist→fingertip direction.
        # Negate x so directions match the player's mirrored perspective:
        # player points RIGHT → cursor moves right on screen → -raw_dx > 0 → "right"
        dx = -(tip.x - wrist.x)
        dy = tip.y - wrist.y

        if wrist_tip > best_extension:
            best_extension = wrist_tip
            best_direction = _classify_direction(dx, dy)

    # Only the dominant (most-extended) hand's direction is considered. If that
    # direction isn't an accepted option, fire nothing — don't fall back to a
    # weaker hand that may be pointing somewhere valid by accident.
    matched_dir = best_direction if best_direction in accepted else None
    context["dominant_direction"] = best_direction   # for debug overlay

    now = time.monotonic()
    if matched_dir:
        prev_dir = context.get("point_direction")
        if prev_dir != matched_dir:
            # Direction changed — reset timer
            context["point_direction"] = matched_dir
            context["point_direction_since"] = now
        return (now - context["point_direction_since"]) * 1000 >= hold_ms
    else:
        context["point_direction_since"] = None
        context["point_direction"] = None
        return False
