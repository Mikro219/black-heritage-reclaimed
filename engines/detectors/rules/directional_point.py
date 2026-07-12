"""
directional_point — one arm pointing in a direction, held for hold_ms.
Used for Scene 2 `point_path` CG, Scene 7 `point_fork` CG, Scene 9 `choose_path` CG.

POSE-ONLY (July 2026): direction comes from the Pose hand line — wrist→index
(landmarks 19/20) when the index is visible, else the forearm (elbow→wrist).
The old finger-pose checks (index-vs-middle, open-palm rejection) needed the
Hands model and are gone; the deliberate-gesture gate is now ARM EXTENSION:
only an arm reaching at least min_reach_frac of its full length counts, so a
relaxed bent arm can't select. The most-extended arm wins the direction.

Params:
  directions (list[str]): accepted 8-way directions ("left","right","up","down",
                          "up_left","up_right","down_left","down_right").
                          Absent/empty = all 8 accepted.
  hold_ms (int): ms the point must be held. Default 500.
  min_reach_frac (float): shoulder→wrist distance over full arm length required
                          for an arm to count as pointing. Default 0.7.
  targets (dict): optional {direction_name: [x, y]} in player screen space —
                          direction chosen by WHICH target the hand point is
                          nearest (position mode); the point must be within
                          proximity_threshold of a target.
  target_x / target_y / proximity_threshold: legacy single-marker mode. Default
                          proximity 0.30.
  min_visibility (float): Pose wrist visibility gate. Default 0.5.

Context keys: point_direction_since, point_direction, dominant_direction
  (reads: _pose_lm)
"""

import math
import time

from . import pose_helpers

_ALL_DIRECTIONS = ["left", "right", "up", "down",
                   "up_left", "up_right", "down_left", "down_right"]


def detect(landmarks, params: dict, context: dict) -> bool:
    pose_lm = context.get("_pose_lm")
    if pose_lm is None:
        context["point_direction_since"] = None
        context["point_direction"] = None
        return False

    accepted = set(params.get("directions") or _ALL_DIRECTIONS)
    hold_ms = params.get("hold_ms", 500)
    min_reach = params.get("min_reach_frac", 0.7)
    min_visibility = params.get("min_visibility", 0.5)
    proximity_threshold = params.get("proximity_threshold", 0.30)
    target_x = params.get("target_x")
    target_y = params.get("target_y")
    targets = params.get("targets")

    wrists = pose_helpers.trusted_wrists(pose_lm, context, min_visibility)

    best_reach = 0.0
    best_direction = None
    for side in wrists:
        reach = pose_helpers.arm_reach_frac(pose_lm, side)
        if reach < min_reach:
            continue   # bent/relaxed arm — not a deliberate point

        tip = pose_helpers.hand_point(pose_lm, side, min_visibility)
        if tip is None:
            continue
        screen_x, screen_y = 1.0 - tip.x, tip.y

        if targets:
            # Position mode: nearest accepted target within proximity.
            hand_dir, nearest = None, proximity_threshold
            for name, pos in targets.items():
                if name not in accepted:
                    continue
                d = math.hypot(screen_x - pos[0], screen_y - pos[1])
                if d <= nearest:
                    nearest, hand_dir = d, name
            if hand_dir is None:
                continue
            if reach > best_reach:
                best_reach, best_direction = reach, hand_dir
            continue

        if target_x is not None and target_y is not None:
            if math.hypot(screen_x - target_x, screen_y - target_y) > proximity_threshold:
                continue

        vec = pose_helpers.point_vector(pose_lm, side)
        if vec is None:
            continue
        if reach > best_reach:
            best_reach = reach
            best_direction = pose_helpers.classify_direction(*vec)

    matched_dir = best_direction if best_direction in accepted else None
    context["dominant_direction"] = best_direction   # for debug overlay

    now = time.monotonic()
    if matched_dir:
        if context.get("point_direction") != matched_dir:
            context["point_direction"] = matched_dir
            context["point_direction_since"] = now
        return (now - context["point_direction_since"]) * 1000 >= hold_ms

    context["point_direction_since"] = None
    context["point_direction"] = None
    return False
