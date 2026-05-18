"""
scripts/gesture_tuner.py — Live gesture-detection tuning tool for BHR.

Shows a mirrored camera feed with the full hand-skeleton overlay and cycles
through every gesture in the experience. The real detector runs each frame;
its live internal state is shown on screen so you can see exactly what is and
is not being recognised and tune params on the spot.

Usage:
  python scripts/gesture_tuner.py [--profile NAME]

Controls:
  N / Right   next gesture
  P / Left    previous gesture
  R           reset detector context (clears hold timer, trail, etc.)
  + / Up      increase primary param (hold_ms / window_ms)
  - / Down    decrease primary param
  ]           increase secondary param (min_displacement, proximity_threshold)
  [           decrease secondary param
  Q / Esc     quit
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engines.detectors import REGISTRY

# ── MediaPipe hand connections ───────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ── Gesture roster ───────────────────────────────────────────────────────────
# tune_key: which params key to increment with +/-  (None = no tuning)
GESTURES = [
    # ── directional_point ─────────────────────────────────────────────────
    dict(
        name="reach_flask",
        label="OI · Scene 1 · AL-01-002",
        desc="Point in any direction (all 8 compass directions accepted)",
        type="directional_point",
        params={"directions": [], "hold_ms": 400},
        tune_key="hold_ms", tune_step=50, accent=(0, 220, 255), highlights=[0, 8],
    ),
    dict(
        name="point_quilt_block",
        label="OI · Scene 1 · AL-01-010",
        desc="Point at top-left quilt block — any direction for tuning",
        type="directional_point",
        params={"directions": [], "hold_ms": 800},
        tune_key="hold_ms", tune_step=50, accent=(0, 220, 255), highlights=[0, 8],
    ),
    dict(
        name="point_path — LEFT",
        label="CG · Scene 2 · AL-02-007",
        desc="Point LEFT to choose the left path — tip must reach LEFT marker",
        type="directional_point",
        params={"directions": ["left"], "hold_ms": 500,
                "target_x": 0.22, "target_y": 0.50, "proximity_threshold": 0.30},
        tune_key="hold_ms", tune_step=50, accent=(80, 200, 255), highlights=[0, 8],
    ),
    dict(
        name="point_path — RIGHT",
        label="CG · Scene 2 · AL-02-007",
        desc="Point RIGHT to choose the right path — tip must reach RIGHT marker",
        type="directional_point",
        params={"directions": ["right"], "hold_ms": 500,
                "target_x": 0.78, "target_y": 0.50, "proximity_threshold": 0.30},
        tune_key="hold_ms", tune_step=50, accent=(80, 200, 255), highlights=[0, 8],
    ),
    dict(
        name="point_river_marker",
        label="OI · Scene 3 · AL-03-006",
        desc="Point DOWN at the river marker — tip must reach DOWN marker",
        type="directional_point",
        params={"directions": ["down"], "hold_ms": 500,
                "target_x": 0.50, "target_y": 0.78, "proximity_threshold": 0.30},
        tune_key="hold_ms", tune_step=50, accent=(0, 220, 255), highlights=[0, 8],
    ),
    dict(
        name="point_dipper_corner",
        label="CG · Scene 4 · AL-04-008",
        desc="Point UP or RIGHT at the Big Dipper bowl corner",
        type="directional_point",
        params={"directions": ["up","right"], "hold_ms": 500,
                "target_x": 0.70, "target_y": 0.25, "proximity_threshold": 0.30},
        tune_key="hold_ms", tune_step=50, accent=(80, 200, 255), highlights=[0, 8],
    ),
    # ── presence_bilateral ────────────────────────────────────────────────
    dict(
        name="raise_hands",
        label="CG · Scene 1 · AL-01-007",
        desc="Both hands above shoulder height, hold steady",
        type="presence_bilateral",
        params={"y_threshold": "shoulder", "hold_ms": 500},
        tune_key="hold_ms", accent=(100, 255, 100), highlights=[0],
    ),
    dict(
        name="look_up (bilateral raise)",
        label="OI · Scene 3 · AL-03-001",
        desc="Both hands raised to head level",
        type="presence_bilateral",
        params={"y_threshold": "head", "hold_ms": 300},
        tune_key="hold_ms", accent=(100, 255, 100), highlights=[0],
    ),
    # ── directional_head_or_hand ──────────────────────────────────────────
    dict(
        name="cup_ear_listen",
        label="OI · Scene 2 · AL-02-005",
        desc="Hand near ear (ear proximity), curled (2+ fingers bent)",
        type="directional_head_or_hand",
        params={"direction": "ear", "require_curl": True, "hold_ms": 500},
        tune_key="hold_ms", tune_step=50, accent=(200, 180, 255), highlights=[0, 4, 8, 12, 16, 20],
    ),
    dict(
        name="shade_eyes",
        label="CG · Scene 3 · AL-03-004",
        desc="Wrist at brow/forehead height — hand shading eyes",
        type="directional_head_or_hand",
        params={"direction": "brow", "hold_ms": 300},
        tune_key="hold_ms", tune_step=50, accent=(200, 180, 255), highlights=[0, 4, 8, 12, 16, 20],
    ),
    # ── directional_draw (stroke chains) ─────────────────────────────────
    dict(
        name="trace_dipper — stroke 1: LEFT",
        label="CG · Scene 4 · AL-04-007 (1/7)",
        desc="Swipe index finger LEFT  |  +/- = window_ms  |  [/] = min_displacement",
        type="directional_draw",
        params={"direction": "left", "window_ms": 400, "min_displacement": 0.04},
        tune_key="window_ms", tune_step=50,
        tune_key2="min_displacement", tune_step2=0.01,
        accent=(255, 180, 0), highlights=[8],
    ),
    dict(
        name="trace_dipper — stroke 2: DOWN",
        label="CG · Scene 4 · AL-04-007 (2/7)",
        desc="Swipe index finger DOWN  |  +/- = window_ms  |  [/] = min_displacement",
        type="directional_draw",
        params={"direction": "down", "window_ms": 400, "min_displacement": 0.04},
        tune_key="window_ms", tune_step=50,
        tune_key2="min_displacement", tune_step2=0.01,
        accent=(255, 180, 0), highlights=[8],
    ),
    dict(
        name="trace_dipper — stroke 3: RIGHT",
        label="CG · Scene 4 · AL-04-007 (3/7)",
        desc="Swipe index finger RIGHT  |  +/- = window_ms  |  [/] = min_displacement",
        type="directional_draw",
        params={"direction": "right", "window_ms": 400, "min_displacement": 0.04},
        tune_key="window_ms", tune_step=50,
        tune_key2="min_displacement", tune_step2=0.01,
        accent=(255, 180, 0), highlights=[8],
    ),
    dict(
        name="trace_dipper — stroke 4: UP",
        label="CG · Scene 4 · AL-04-007 (4/7)",
        desc="Swipe index finger UP  |  +/- = window_ms  |  [/] = min_displacement",
        type="directional_draw",
        params={"direction": "up", "window_ms": 400, "min_displacement": 0.04},
        tune_key="window_ms", tune_step=50,
        tune_key2="min_displacement", tune_step2=0.01,
        accent=(255, 180, 0), highlights=[8],
    ),
    dict(
        name="trace_dipper — strokes 5-7: UP-LEFT",
        label="CG · Scene 4 · AL-04-007 (5-7/7)",
        desc="Swipe diagonally UP-LEFT  |  +/- = window_ms  |  [/] = min_displacement",
        type="directional_draw",
        params={"direction": "up_left", "window_ms": 400, "min_displacement": 0.04},
        tune_key="window_ms", tune_step=50,
        tune_key2="min_displacement", tune_step2=0.01,
        accent=(255, 180, 0), highlights=[8],
    ),
    dict(
        name="trace_star — stroke 1: UP-RIGHT",
        label="CG · Scene 4 · AL-04-009 (1/5)",
        desc="Swipe diagonally UP-RIGHT  |  +/- = window_ms  |  [/] = min_displacement",
        type="directional_draw",
        params={"direction": "up_right", "window_ms": 400, "min_displacement": 0.04},
        tune_key="window_ms", tune_step=50,
        tune_key2="min_displacement", tune_step2=0.01,
        accent=(255, 220, 80), highlights=[8],
    ),
    # ── mouth_proximity_tip ───────────────────────────────────────────────
    dict(
        name="drink_gesture",
        label="OI · Scene 1 · AL-01-002 step 2",
        desc="Tipping hand-to-mouth: wrist above tip, tip near mouth region",
        type="mouth_proximity_tip",
        params={"hold_ms": 400, "proximity_threshold": 0.12},
        tune_key="hold_ms", accent=(80, 255, 200), highlights=[0, 4, 8],
    ),
    # ── rhythm_bilateral ──────────────────────────────────────────────────
    dict(
        name="three_knock",
        label="CG · Scene 6 · AL-06-007",
        desc="knock-knock...KNOCK  |  +/-: short_window_ms  |  [/]: long_min_ms",
        type="rhythm_bilateral",
        params={"short_window_ms": 500, "long_min_ms": 400, "long_max_ms": 1400,
                "min_push": 0.04, "refractory_ms": 220},
        tune_key="short_window_ms", tune_step=50,
        tune_key2="long_min_ms", tune_step2=50,
        accent=(255, 100, 180), highlights=[0, 8],
    ),
    # ── speed_bilateral ───────────────────────────────────────────────────
    dict(
        name="fast_gather",
        label="CG · Scene 10 · AL-10-004",
        desc="Both hands rapid gathering — speed multiplier ≥2× idle",
        type="speed_bilateral",
        params={"speed_multiplier": 2.0},
        tune_key=None, accent=(255, 80, 80), highlights=[0, 8],
    ),
]

# ── Named thresholds (mirrors presence_bilateral.py) ────────────────────────
_NAMED_Y = {"shoulder": 0.5, "head": 0.3, "waist": 0.7}
_MOUTH_XY = (0.50, 0.78)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def lm_to_px(lm, w, h, mirror=True):
    """Normalised landmark → pixel coords. Mirrors x by default."""
    x = (1 - lm.x) * w if mirror else lm.x * w
    return int(x), int(lm.y * h)


def raw_to_px(nx, ny, w, h, mirror=True):
    """Normalised (x, y) → pixel coords."""
    x = (1 - nx) * w if mirror else nx * w
    return int(x), int(ny * h)


def draw_skeleton(frame, hand_landmarks_list, highlights, accent):
    h, w = frame.shape[:2]
    for hand_lm in hand_landmarks_list:
        lms = hand_lm.landmark
        pts = [lm_to_px(lm, w, h) for lm in lms]

        # Bones
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 80, 80), 1, cv2.LINE_AA)

        # All joints
        for pt in pts:
            cv2.circle(frame, pt, 3, (160, 160, 160), -1, cv2.LINE_AA)

        # Highlighted joints
        for idx in highlights:
            cv2.circle(frame, pts[idx], 7, accent, -1, cv2.LINE_AA)
            cv2.circle(frame, pts[idx], 9, (255, 255, 255), 1, cv2.LINE_AA)


def draw_target_marker(frame, directions, w, h, accent):
    """Draw a glowing crosshair target for the player to point at.

    Position is based on the gesture's accepted directions, in player-perspective
    screen space (left = player's left, right = player's right).
    For multi-direction gestures the marker sits in the center.
    """
    # Map direction → (x_frac, y_frac) in screen space
    DIR_POS = {
        "left":      (0.22, 0.50),
        "right":     (0.78, 0.50),
        "up":        (0.50, 0.22),
        "down":      (0.50, 0.78),
        "up_left":   (0.22, 0.22),
        "up_right":  (0.78, 0.22),
        "down_left": (0.22, 0.78),
        "down_right":(0.78, 0.78),
    }
    dirs = list(directions)
    if len(dirs) == 1:
        pos = DIR_POS.get(dirs[0], (0.50, 0.50))
    elif set(dirs) == {"left", "right"}:
        # Draw two targets, one each side
        for d in dirs:
            draw_target_marker(frame, [d], w, h, accent)
        return
    elif set(dirs) == {"up", "right"}:
        pos = (0.70, 0.25)
    else:
        pos = (0.50, 0.50)  # center for any-direction

    cx, cy = int(pos[0] * w), int(pos[1] * h)
    r = 38

    # Outer glow rings
    for ring_r, alpha in [(r + 16, 0.08), (r + 8, 0.15)]:
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), ring_r, accent, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Crosshair lines
    gap = 10
    arm = 24
    cv2.line(frame, (cx - r, cy), (cx - gap, cy), accent, 2, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy), (cx + r, cy), accent, 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - r), (cx, cy - gap), accent, 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap), (cx, cy + r), accent, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 5, accent, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, accent, 1, cv2.LINE_AA)

    label = "/".join(d.upper() for d in dirs)
    cv2.putText(frame, f"POINT {label}", (cx - 40, cy + r + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent, 1, cv2.LINE_AA)


def draw_direction_arrow(frame, hand_landmarks_list, w, h, accepted_dirs, accent):
    """Draw wrist→tip vector with player-perspective direction label.

    directional_point now flips dx internally, so what it classifies as
    'right' is the player's right on the mirrored display. Labels here
    match that convention directly.
    """
    for hand_lm in hand_landmarks_list:
        lm = hand_lm.landmark
        wrist = lm[0]
        tip = lm[8]
        pip = lm[6]

        # Same distance-based check as the detector: tip must be farther from wrist than pip
        wrist_tip_sq = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
        wrist_pip_sq = (pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2
        extended = wrist_tip_sq > wrist_pip_sq
        wp = lm_to_px(wrist, w, h)
        tp = lm_to_px(tip, w, h)

        # Arrow from wrist to tip (mirrored display space)
        color = accent if extended else (80, 80, 80)
        cv2.arrowedLine(frame, wp, tp, color, 2, cv2.LINE_AA, tipLength=0.3)

        # Player-perspective direction (x flipped, matching detector's 8-way classifier)
        dx = -(tip.x - wrist.x)
        dy = tip.y - wrist.y
        _dirs_cw = ["right", "down_right", "down", "down_left",
                    "left", "up_left", "up", "up_right"]
        angle = math.degrees(math.atan2(dy, dx)) % 360
        direction = _dirs_cw[int((angle + 22.5) / 45) % 8]

        in_set = direction in accepted_dirs
        ext_str = "extended" if extended else "CURLED"
        label_color = accent if (extended and in_set) else (200, 80, 80) if extended else (120, 120, 120)
        cv2.putText(frame, f"{direction.upper()}  ({ext_str})",
                    (tp[0] + 10, tp[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, label_color, 1, cv2.LINE_AA)

        tip_y_str = f"tip.y={tip.y:.2f}  pip.y={pip.y:.2f}  diff={pip.y - tip.y:.3f}"
        cv2.putText(frame, tip_y_str,
                    (wp[0], wp[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (160, 160, 160), 1, cv2.LINE_AA)


def draw_bilateral_threshold(frame, y_threshold_raw, w, h, accent):
    """Horizontal line showing the presence_bilateral y_threshold."""
    if isinstance(y_threshold_raw, str):
        y_val = _NAMED_Y.get(y_threshold_raw, 0.5)
    else:
        y_val = float(y_threshold_raw)
    if y_val <= 0:
        return
    y_px = int(y_val * h)
    cv2.line(frame, (0, y_px), (w, y_px), accent, 1, cv2.LINE_AA)
    cv2.putText(frame, f"threshold ({y_threshold_raw})",
                (8, y_px - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, accent, 1, cv2.LINE_AA)


def draw_draw_trail(frame, context, w, h, accent):
    """Draw the directional_draw motion trail from context."""
    history = context.get("draw_history", [])
    if len(history) < 2:
        return
    pts = [raw_to_px(p[0], p[1], w, h, mirror=True) for p in history]
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = tuple(int(v * alpha) for v in accent)
        cv2.line(frame, pts[i-1], pts[i], c, 2, cv2.LINE_AA)
    # Show displacement vector
    raw_dx = history[-1][0] - history[0][0]
    raw_dy = history[-1][1] - history[0][1]
    screen_dx = -raw_dx
    screen_dy = raw_dy
    mag = math.hypot(screen_dx, screen_dy)
    angle_deg = math.degrees(math.atan2(screen_dy, screen_dx))
    cv2.putText(frame, f"disp={mag:.3f}  angle={angle_deg:.0f}deg",
                (pts[-1][0] + 8, pts[-1][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, accent, 1, cv2.LINE_AA)


def draw_mouth_marker(frame, w, h, accent, context, mouth_x=None, mouth_y=None,
                      threshold=None):
    """Mark the mouth position for mouth_proximity_tip.

    mouth_x/mouth_y: live pose-derived coords (normalised, raw MediaPipe space).
    Falls back to the fixed estimate when pose is not available.
    """
    mx_norm = mouth_x if mouth_x is not None else _MOUTH_XY[0]
    my_norm = mouth_y if mouth_y is not None else _MOUTH_XY[1]
    thresh = threshold if threshold is not None else 0.12

    mx, my = raw_to_px(mx_norm, my_norm, w, h, mirror=True)
    radius_px = int(thresh * w)

    pose_color = (80, 255, 200) if mouth_x is not None else (120, 120, 120)
    label = "mouth (pose)" if mouth_x is not None else "mouth (est.)"

    cv2.circle(frame, (mx, my), radius_px, accent, 1, cv2.LINE_AA)
    cv2.circle(frame, (mx, my), 6, pose_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (mx, my), 8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, label, (mx + 10, my),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, pose_color, 1, cv2.LINE_AA)
    near_since = context.get("tip_near_since")
    if near_since is not None:
        elapsed = (time.monotonic() - near_since) * 1000
        cv2.putText(frame, f"near {elapsed:.0f}ms",
                    (mx + 10, my + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 150), 1, cv2.LINE_AA)


def draw_knock_progress(frame, context, params, w, h, accent):
    """Draw three-knock progress circles with timing window bars."""
    now = time.monotonic()
    knock_count = context.get("knock_count", 0)
    knock_times = context.get("knock_times", [])
    short_window_ms = params.get("short_window_ms", 500)
    long_min_ms = params.get("long_min_ms", 400)
    long_max_ms = params.get("long_max_ms", 1400)

    # Three circles centred on screen
    cy = h // 2 - 50
    spacing = 110
    r = 30
    cx_base = w // 2 - spacing

    for i, label in enumerate(["K1", "K2", "LONG"]):
        cx = cx_base + i * spacing
        filled = knock_count > i
        fill_color = accent if filled else (30, 30, 30)
        ring_color = accent if filled else (80, 80, 80)
        cv2.circle(frame, (cx, cy), r, fill_color, -1)
        cv2.circle(frame, (cx, cy), r, ring_color, 2, cv2.LINE_AA)
        txt_color = (255, 255, 255) if filled else (100, 100, 100)
        cv2.putText(frame, label, (cx - 14, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1, cv2.LINE_AA)

    bar_y = cy + r + 18
    bar_w = 90

    # Short-window countdown bar under K2 while waiting for second knock
    if knock_count == 1 and knock_times:
        elapsed_ms = (now - knock_times[0]) * 1000
        remaining_ms = max(0.0, short_window_ms - elapsed_ms)
        pct_left = remaining_ms / short_window_ms
        bar_x = cx_base + spacing - bar_w // 2
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (50, 50, 50), -1)
        filled_w = int(pct_left * bar_w)
        if filled_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + 12), accent, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (120, 120, 120), 1)
        cv2.putText(frame, f"K2 window: {remaining_ms:.0f}ms",
                    (bar_x + bar_w + 6, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, accent, 1, cv2.LINE_AA)

    # Two-zone bar under LONG while waiting for third knock
    if knock_count == 2 and len(knock_times) >= 2:
        elapsed_ms = (now - knock_times[1]) * 1000
        bar_x = cx_base + 2 * spacing - bar_w // 2
        wait_w = max(1, int(bar_w * long_min_ms / long_max_ms))
        fire_w = bar_w - wait_w
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (50, 50, 50), -1)
        # Wait zone (red-ish)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + wait_w, bar_y + 12), (120, 50, 50), -1)
        # Fire zone (green-ish)
        cv2.rectangle(frame, (bar_x + wait_w, bar_y), (bar_x + bar_w, bar_y + 12), (30, 80, 30), -1)
        # Moving marker
        if elapsed_ms <= long_min_ms:
            wait_pct = elapsed_ms / long_min_ms
            marker_x = bar_x + int(wait_pct * wait_w)
            cv2.line(frame, (marker_x, bar_y - 2), (marker_x, bar_y + 14), (255, 220, 80), 2)
            remaining = long_min_ms - elapsed_ms
            cv2.putText(frame, f"wait: {remaining:.0f}ms",
                        (bar_x + bar_w + 6, bar_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 120, 80), 1, cv2.LINE_AA)
        elif elapsed_ms <= long_max_ms:
            fire_elapsed = elapsed_ms - long_min_ms
            fire_range = long_max_ms - long_min_ms
            fire_pct = fire_elapsed / fire_range
            marker_x = bar_x + wait_w + int(fire_pct * fire_w)
            cv2.line(frame, (marker_x, bar_y - 2), (marker_x, bar_y + 14), accent, 2)
            remaining = long_max_ms - elapsed_ms
            cv2.putText(frame, f"KNOCK! {remaining:.0f}ms left",
                        (bar_x + bar_w + 6, bar_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, accent, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (120, 120, 120), 1)
        # Zone labels
        cv2.putText(frame, "wait", (bar_x + 2, bar_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(frame, "fire", (bar_x + wait_w + 2, bar_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 180, 80), 1, cv2.LINE_AA)


def draw_head_or_hand_zones(frame, direction, w, h, accent,
                            ear_positions=None, brow_y=None):
    """Draw the active zone for directional_head_or_hand in player-perspective space.

    ear_positions: live pose-derived [(x,y), ...] in raw MediaPipe coords.
    brow_y: live pose-derived absolute Y threshold for brow direction.
    """
    overlay = frame.copy()
    if direction == "right":
        # Player's right → wrist.x < 0.30 raw → mirrors to RIGHT side of display
        x_thresh = int(0.30 * w)
        cv2.rectangle(overlay, (w - x_thresh, 0), (w, h), accent, -1)
        cv2.putText(frame, "PLAYER RIGHT zone (wrist.x < 0.30 raw)",
                    (w - x_thresh + 6, h - 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, accent, 1, cv2.LINE_AA)
        cv2.line(frame, (w - x_thresh, 0), (w - x_thresh, h), accent, 1)
    elif direction == "left":
        # Player's left → wrist.x > 0.70 raw → mirrors to LEFT side of display
        x_thresh = int(0.30 * w)
        cv2.rectangle(overlay, (0, 0), (x_thresh, h), accent, -1)
        cv2.putText(frame, "PLAYER LEFT zone (wrist.x > 0.70 raw)",
                    (8, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, accent, 1, cv2.LINE_AA)
        cv2.line(frame, (x_thresh, 0), (x_thresh, h), accent, 1)
    elif direction == "up":
        y_thresh = int(0.35 * h)
        cv2.rectangle(overlay, (0, 0), (w, y_thresh), accent, -1)
        cv2.line(frame, (0, y_thresh), (w, y_thresh), accent, 1)
        cv2.putText(frame, "UP zone (wrist.y < 0.35)",
                    (8, y_thresh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, accent, 1, cv2.LINE_AA)
    elif direction == "brow":
        threshold_y = brow_y if brow_y is not None else 0.38
        y_thresh = int(threshold_y * h)
        cv2.rectangle(overlay, (0, 0), (w, y_thresh), accent, -1)
        cv2.line(frame, (0, y_thresh), (w, y_thresh), accent, 1)
        label = f"BROW zone (pose eye+0.04 = {threshold_y:.3f})" if brow_y is not None \
                else "BROW zone (wrist.y < 0.38, no pose)"
        cv2.putText(frame, label,
                    (8, y_thresh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, accent, 1, cv2.LINE_AA)
    elif direction == "ear":
        # Draw ear proximity circles — use live pose positions when available
        ear_pos = ear_positions if ear_positions is not None else [(0.15, 0.35), (0.85, 0.35)]
        pose_label = "(pose)" if ear_positions is not None else "(estimated)"
        for ex, ey in ear_pos:
            epx = int((1 - ex) * w)  # mirrored
            epy = int(ey * h)
            r = int(0.15 * w)
            cv2.circle(overlay, (epx, epy), r, accent, -1)
            cv2.circle(frame, (epx, epy), r, accent, 1, cv2.LINE_AA)
            cv2.circle(frame, (epx, epy), 5, (80, 255, 200) if ear_positions else (120,120,120),
                       -1, cv2.LINE_AA)
        cv2.putText(frame, f"EAR zones {pose_label} fingertip prox 0.15",
                    (8, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, accent, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)


# ── Text / panel helpers ─────────────────────────────────────────────────────

def put_lines(frame, lines, origin, font_scale=0.55, thickness=1,
              line_height=22, bg=True):
    """Draw a list of (text, color) with a semi-transparent background box."""
    ox, oy = origin
    if not lines:
        return
    # Measure
    non_empty = [t for t, _ in lines if t]
    if not non_empty:
        return
    max_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
                for t in non_empty)
    box_h = len(lines) * line_height + 10
    if bg:
        overlay = frame.copy()
        cv2.rectangle(overlay, (ox - 6, oy - line_height),
                      (ox + max_w + 10, oy + box_h - line_height + 4), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, (text, color) in enumerate(lines):
        if text:
            cv2.putText(frame, text, (ox, oy + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def progress_bar(frame, origin, width, pct, color, label=""):
    x, y = origin
    cv2.rectangle(frame, (x, y), (x + width, y + 12), (50, 50, 50), -1)
    filled = int(pct * width)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + 12), color, -1)
    if label:
        cv2.putText(frame, label, (x + width + 6, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


# ── Hold-timer computation ───────────────────────────────────────────────────

def hold_progress(context, key, hold_ms):
    """Return 0.0-1.0 progress based on a context 'since' timestamp."""
    since = context.get(key)
    if since is None:
        return 0.0
    elapsed = (time.monotonic() - since) * 1000
    return min(elapsed / hold_ms, 1.0) if hold_ms > 0 else 0.0


# ── Pose helpers ─────────────────────────────────────────────────────────────

def _inject_pose_params(gtype: str, params: dict, pose_lm) -> dict:
    """Return a copy of params with live pose-derived values injected.

    Detectors already accept these as optional overrides — no detector code
    changes needed beyond what they already read from params.
    """
    if pose_lm is None:
        return params
    p = dict(params)

    if gtype == "mouth_proximity_tip":
        # Mouth centre from pose landmarks 9 (mouth_left) and 10 (mouth_right)
        p["mouth_x"] = (pose_lm[9].x + pose_lm[10].x) / 2
        p["mouth_y"] = (pose_lm[9].y + pose_lm[10].y) / 2

    elif gtype == "directional_head_or_hand":
        direction = params.get("direction")
        if direction == "ear":
            # Real ear positions from pose landmarks 7 (left_ear) and 8 (right_ear)
            p["ear_positions"] = [
                (pose_lm[7].x, pose_lm[7].y),
                (pose_lm[8].x, pose_lm[8].y),
            ]
        elif direction == "brow":
            # Use eye level (landmarks 2=left_eye, 5=right_eye) + small offset
            eye_y = (pose_lm[2].y + pose_lm[5].y) / 2
            p["brow_y"] = eye_y + 0.04

    elif gtype == "presence_bilateral":
        threshold_name = params.get("y_threshold", "")
        if threshold_name == "shoulder":
            # Actual shoulder midpoint Y from pose landmarks 11 and 12
            p["y_threshold"] = (pose_lm[11].y + pose_lm[12].y) / 2
        elif threshold_name == "head":
            # Eye level from pose
            p["y_threshold"] = (pose_lm[2].y + pose_lm[5].y) / 2

    return p


def draw_pose_markers(frame, pose_lm, w, h):
    """Draw key body landmarks as subtle reference points: mouth, ears, shoulders, eyes."""
    if pose_lm is None:
        return
    COLOR = (160, 160, 80)  # muted yellow

    def mpx(lm):
        """Raw MediaPipe normalised → mirrored pixel."""
        return int((1 - lm.x) * w), int(lm.y * h)

    # Eyes (brow reference)
    for idx in (2, 5):
        cv2.circle(frame, mpx(pose_lm[idx]), 3, COLOR, -1, cv2.LINE_AA)

    # Ears
    for idx, label in ((7, "LE"), (8, "RE")):
        ep = mpx(pose_lm[idx])
        cv2.circle(frame, ep, 5, COLOR, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (ep[0] + 6, ep[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR, 1, cv2.LINE_AA)

    # Mouth centre
    mx = (pose_lm[9].x + pose_lm[10].x) / 2
    my = (pose_lm[9].y + pose_lm[10].y) / 2
    mpx_m = int((1 - mx) * w), int(my * h)
    cv2.circle(frame, mpx_m, 5, COLOR, -1, cv2.LINE_AA)
    cv2.putText(frame, "M", (mpx_m[0] + 6, mpx_m[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR, 1, cv2.LINE_AA)

    # Shoulders — line + midpoint
    sl = mpx(pose_lm[11])
    sr = mpx(pose_lm[12])
    cv2.line(frame, sl, sr, COLOR, 1, cv2.LINE_AA)
    cv2.circle(frame, sl, 4, COLOR, -1, cv2.LINE_AA)
    cv2.circle(frame, sr, 4, COLOR, -1, cv2.LINE_AA)
    mid_y = (sl[1] + sr[1]) // 2
    cv2.line(frame, (0, mid_y), (w, mid_y), (80, 80, 40), 1, cv2.LINE_AA)

    # Pose detected badge
    cv2.putText(frame, "POSE OK", (w - 130, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR, 1, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────────────

def open_camera(config_path):
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    backend = cv2.CAP_DSHOW
    cap = cv2.VideoCapture(0, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def main():
    parser = argparse.ArgumentParser(description="BHR gesture tuner")
    parser.add_argument(
        "--profile", default="laptop_dev",
        help="Host profile to read/write gesture tuning from. "
             "Options: laptop_dev (default), mini_pc_prod. "
             "Example: --profile mini_pc_prod"
    )
    args = parser.parse_args()

    profile_name = args.profile or "laptop_dev"
    profile_path = os.path.join(ROOT, "config", "host_profiles", f"{profile_name}.json")
    _load_tune_params(GESTURES, profile_path)

    config_path = os.path.join(ROOT, "config.json")
    cap = open_camera(config_path)
    if not cap.isOpened():
        print("[tuner] Failed to open camera.", file=sys.stderr)
        sys.exit(1)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=0,          # fastest; plenty accurate for landmark positions
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    gesture_idx = 0
    context = {}
    fired_at = 0.0
    fire_count = 0

    print(f"[tuner] {len(GESTURES)} gestures loaded. N/P to cycle, +/- to tune, R to reset, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Mirror for natural feedback
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # ── MediaPipe ─────────────────────────────────────────────────────
        # Process on UN-mirrored frame so coords match detector expectations
        raw = cv2.flip(frame, 1)
        raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        results = hands.process(raw_rgb)
        lm_list = results.multi_hand_landmarks or []

        pose_results = pose.process(raw_rgb)
        pose_lm = (pose_results.pose_landmarks.landmark
                   if pose_results.pose_landmarks else None)

        gesture = GESTURES[gesture_idx]
        gtype = gesture["type"]
        params = gesture["params"]
        accent = gesture["accent"]
        highlights = gesture["highlights"]

        # Build live_params: params + pose-derived overrides where applicable
        live_params = _inject_pose_params(gtype, params, pose_lm)

        # ── Run detector ──────────────────────────────────────────────────
        detector_fn = REGISTRY.get(gtype)
        fired = False
        if detector_fn and lm_list:
            fired = detector_fn(lm_list, live_params, context)

        if fired:
            fire_count += 1
            fired_at = time.monotonic()

        # ── Draw skeleton ─────────────────────────────────────────────────
        draw_skeleton(frame, lm_list, highlights, accent)

        # ── Pose body markers (always, when pose is available) ────────────
        draw_pose_markers(frame, pose_lm, w, h)

        # ── Gesture-specific overlays ─────────────────────────────────────
        if gtype == "directional_point":
            accepted = set(live_params.get("directions", ["left","right","up","down"]))
            draw_target_marker(frame, accepted, w, h, accent)
            t_x = live_params.get("target_x")
            t_y = live_params.get("target_y")
            if t_x is not None and t_y is not None:
                prox = live_params.get("proximity_threshold", 0.30)
                cx_p = int(t_x * w)
                cy_p = int(t_y * h)
                r_p = int(prox * w)
                cv2.circle(frame, (cx_p, cy_p), r_p, accent, 1, cv2.LINE_AA)
                cv2.putText(frame, f"prox={prox:.2f}", (cx_p - 30, cy_p + r_p + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, accent, 1, cv2.LINE_AA)
            if lm_list:
                draw_direction_arrow(frame, lm_list, w, h, accepted, accent)

        elif gtype == "presence_bilateral":
            # Use live_params so the threshold line reflects the pose-derived shoulder Y
            y_thresh = live_params.get("y_threshold", 0)
            draw_bilateral_threshold(frame, y_thresh, w, h, accent)

        elif gtype == "directional_draw":
            draw_draw_trail(frame, context, w, h, accent)

        elif gtype == "directional_head_or_hand":
            direction = live_params.get("direction", "right")
            ear_pos = live_params.get("ear_positions")
            brow_y_val = live_params.get("brow_y")
            draw_head_or_hand_zones(frame, direction, w, h, accent,
                                    ear_positions=ear_pos, brow_y=brow_y_val)

        elif gtype == "mouth_proximity_tip":
            draw_mouth_marker(frame, w, h, accent, context,
                              mouth_x=live_params.get("mouth_x"),
                              mouth_y=live_params.get("mouth_y"),
                              threshold=live_params.get("proximity_threshold", 0.12))

        elif gtype == "rhythm_bilateral":
            draw_knock_progress(frame, context, live_params, w, h, accent)

        # ── FIRED flash ───────────────────────────────────────────────────
        if time.monotonic() - fired_at < 1.0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 80), -1)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.putText(frame, f"FIRED! (#{fire_count})",
                        (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 80), 4, cv2.LINE_AA)

        # ── Top-left info panel ───────────────────────────────────────────
        tune_key = gesture.get("tune_key")
        tune_val = params.get(tune_key, "—") if tune_key else "—"
        tune_str = f"  [+/-: {tune_key}={tune_val}]" if tune_key else ""
        tune_key2 = gesture.get("tune_key2")
        tune_val2 = params.get(tune_key2, "—") if tune_key2 else "—"
        tune_str2 = f"  [[/]: {tune_key2}={tune_val2}]" if tune_key2 else ""
        top_lines = [
            (f"GESTURE {gesture_idx + 1}/{len(GESTURES)}: {gesture['name']}",
             (255, 255, 255)),
            (gesture["label"], (160, 160, 255)),
            (gesture["desc"][:70], (200, 200, 200)),
            (f"TYPE: {gtype}{tune_str}{tune_str2}", accent),
        ]
        put_lines(frame, top_lines, (10, 26), font_scale=0.55, line_height=24)

        # ── Detection state panel (bottom-left) ──────────────────────────
        state_lines = []
        hold_ms = live_params.get("hold_ms") or live_params.get("window_ms")

        if gtype == "directional_point":
            since_key = "point_direction_since"
            pct = hold_progress(context, since_key, hold_ms or 500)
            dir_now = context.get("point_direction", "--")
            accepted = params.get("directions", [])
            in_set = dir_now in accepted
            state_lines.append((f"dir: {dir_now}  accepted: {accepted}",
                                 accent if (dir_now != "--" and in_set) else (200,80,80) if dir_now != "--" else (120,120,120)))
            # Show proximity distance from target if configured
            t_x = live_params.get("target_x")
            t_y = live_params.get("target_y")
            if t_x is not None and t_y is not None and lm_list:
                tip = lm_list[0].landmark[8]
                sx = 1.0 - tip.x
                dist = math.hypot(sx - t_x, tip.y - t_y)
                prox = live_params.get("proximity_threshold", 0.30)
                ok = dist <= prox
                state_lines.append((f"tip dist={dist:.3f}  thresh={prox:.2f}  {'IN' if ok else 'OUT'}",
                                     accent if ok else (200, 80, 80)))
            state_lines.append(("hold progress:", (200, 200, 200)))

        elif gtype == "presence_bilateral":
            hand_count = len(lm_list)
            state_lines.append((f"hands: {hand_count}", (0,255,80) if hand_count >= 2 else (200,80,80)))
            if lm_list:
                for i, hl in enumerate(lm_list[:2]):
                    wy = hl.landmark[0].y
                    state_lines.append((f"  wrist[{i}].y = {wy:.3f}", accent))
            pct = hold_progress(context, "bilateral_present_since", hold_ms or 500)
            state_lines.append(("hold progress:", (200, 200, 200)))

        elif gtype == "directional_draw":
            hist = context.get("draw_history", [])
            state_lines.append((f"history pts: {len(hist)}", accent))
            if len(hist) >= 2:
                raw_dx = hist[-1][0] - hist[0][0]
                raw_dy = hist[-1][1] - hist[0][1]
                screen_dx = -raw_dx
                screen_dy = raw_dy
                mag = math.hypot(screen_dx, screen_dy)
                ang = math.degrees(math.atan2(screen_dy, screen_dx))
                state_lines.append((f"disp={mag:.3f}  angle={ang:.0f}deg", accent))
            pct = min(len(hist) / 10, 1.0)
            state_lines.append(("buffer fill:", (200, 200, 200)))

        elif gtype == "directional_head_or_hand":
            since = context.get("directional_since")
            pct = hold_progress(context, "directional_since", hold_ms or 500)
            state_lines.append((f"matched: {'YES' if since else 'no'}", accent if since else (120,120,120)))
            if live_params.get("require_curl") and lm_list:
                lm0 = lm_list[0].landmark
                TIP_PIP = [(8,6),(12,10),(16,14),(20,18)]
                curl_count = sum(1 for ti, pi in TIP_PIP if lm0[ti].y >= lm0[pi].y)
                curl_ok = curl_count >= 2
                state_lines.append((f"curl: {curl_count}/4  {'OK' if curl_ok else 'need 2+ curled'}",
                                     accent if curl_ok else (200, 80, 80)))
            pose_src = "pose" if pose_lm else "fixed"
            state_lines.append((f"coords: {pose_src}", (160, 200, 160) if pose_lm else (120,120,120)))
            state_lines.append(("hold progress:", (200, 200, 200)))

        elif gtype == "mouth_proximity_tip":
            near_since = context.get("tip_near_since")
            pct = hold_progress(context, "tip_near_since", hold_ms or 400)
            state_lines.append((f"near mouth: {'YES' if near_since else 'no'}",
                                 accent if near_since else (120,120,120)))
            if pose_lm:
                mx = (pose_lm[9].x + pose_lm[10].x) / 2
                my = (pose_lm[9].y + pose_lm[10].y) / 2
                state_lines.append((f"mouth (pose): x={mx:.3f}  y={my:.3f}", (160, 200, 160)))
            else:
                state_lines.append(("mouth: fixed estimate (no pose)", (120,120,120)))
            if live_params.get("require_curl") and lm_list:
                lm0 = lm_list[0].landmark
                TIP_PIP = [(8,6),(12,10),(16,14),(20,18)]
                curl_count = sum(1 for ti, pi in TIP_PIP if lm0[ti].y >= lm0[pi].y)
                state_lines.append((f"curl: {curl_count}/4  {'OK' if curl_count >= 2 else 'need 2+ curled'}",
                                     accent if curl_count >= 2 else (200, 80, 80)))
            state_lines.append(("hold progress:", (200, 200, 200)))

        elif gtype == "rhythm_bilateral":
            knock_count = context.get("knock_count", 0)
            knock_times = context.get("knock_times", [])
            now_ts = time.monotonic()
            short_window_ms = live_params.get("short_window_ms", 500)
            long_min_ms = live_params.get("long_min_ms", 400)
            long_max_ms = live_params.get("long_max_ms", 1400)
            pct = knock_count / 3.0
            state_lines.append((f"knocks: {knock_count}/3",
                                 accent if knock_count > 0 else (120, 120, 120)))
            if knock_count == 1 and knock_times:
                elapsed_ms = (now_ts - knock_times[0]) * 1000
                remaining = max(0.0, short_window_ms - elapsed_ms)
                ok = remaining > 0
                state_lines.append((f"K2 window: {remaining:.0f}ms left  (max={short_window_ms}ms)",
                                     (255, 200, 80) if ok else (200, 80, 80)))
            elif knock_count == 2 and len(knock_times) >= 2:
                elapsed_ms = (now_ts - knock_times[1]) * 1000
                if elapsed_ms < long_min_ms:
                    remaining = long_min_ms - elapsed_ms
                    state_lines.append((f"wait before K3: {remaining:.0f}ms  (min={long_min_ms}ms)",
                                         (200, 120, 80)))
                else:
                    remaining = max(0.0, long_max_ms - elapsed_ms)
                    state_lines.append((f"KNOCK NOW!  {remaining:.0f}ms left  (max={long_max_ms}ms)",
                                         accent))
            else:
                state_lines.append((f"short_window={short_window_ms}ms  long_min={long_min_ms}ms  long_max={long_max_ms}ms",
                                     (120, 120, 120)))
            min_push = live_params.get("min_push", 0.04)
            refrac = live_params.get("refractory_ms", 220)
            state_lines.append((f"min_push={min_push:.3f}  refractory={refrac}ms",
                                 (120, 120, 120)))
            state_lines.append(("progress:", (200, 200, 200)))

        else:
            # Generic: dump any scalar context values
            pct = 0.0
            for k, v in context.items():
                if not isinstance(v, list):
                    state_lines.append((f"{k}: {v}", accent))
            state_lines.append(("", (0,0,0)))

        panel_y = h - 130
        put_lines(frame, state_lines, (10, panel_y), font_scale=0.50, line_height=22)
        bar_y = panel_y + len(state_lines) * 22
        progress_bar(frame, (10, bar_y), 200, pct, accent,
                     label=f"{int(pct*100)}%")

        # ── Hand-count + profile indicator (top-right) ───────────────────
        hc = len(lm_list)
        hc_color = (0, 255, 80) if hc > 0 else (80, 80, 80)
        cv2.putText(frame, f"hands: {hc}", (w - 130, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, hc_color, 2, cv2.LINE_AA)
        profile_color = (80, 200, 255) if profile_name == "mini_pc_prod" else (160, 160, 255)
        cv2.putText(frame, f"profile: {profile_name}", (w - 220, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, profile_color, 1, cv2.LINE_AA)

        # ── Controls reminder (bottom-right) ─────────────────────────────
        ctrl_lines = [
            ("N/Right: next   P/Left: prev   R: reset   Q: quit", (140, 140, 140)),
            ("+/-: tune param1   [/]: tune param2", (140, 140, 140)),
        ]
        put_lines(frame, ctrl_lines, (w - 310, h - 40), font_scale=0.42,
                  line_height=18, bg=False)

        cv2.imshow("BHR Gesture Tuner", frame)

        # ── Input ─────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # Q or Esc
            break
        elif key in (ord('n'), 83):  # N or Right arrow
            gesture_idx = (gesture_idx + 1) % len(GESTURES)
            context = {}
            fire_count = 0
        elif key in (ord('p'), 81):  # P or Left arrow
            gesture_idx = (gesture_idx - 1) % len(GESTURES)
            context = {}
            fire_count = 0
        elif key == ord('r'):
            context = {}
            fire_count = 0
            print(f"[tuner] Context reset for: {GESTURES[gesture_idx]['name']}")
        elif key in (ord('+'), ord('='), 82):  # + or Up arrow — primary param
            _adjust_tune_param(GESTURES[gesture_idx], 1, +1)
            context = {}
            _save_tune_params(GESTURES, profile_path)
        elif key in (ord('-'), 84):  # - or Down arrow — primary param
            _adjust_tune_param(GESTURES[gesture_idx], 1, -1)
            context = {}
            _save_tune_params(GESTURES, profile_path)
        elif key == ord(']'):  # ] — secondary param increase
            _adjust_tune_param(GESTURES[gesture_idx], 2, +1)
            context = {}
            _save_tune_params(GESTURES, profile_path)
        elif key == ord('['):  # [ — secondary param decrease
            _adjust_tune_param(GESTURES[gesture_idx], 2, -1)
            context = {}
            _save_tune_params(GESTURES, profile_path)

    cap.release()
    hands.close()
    pose.close()
    cv2.destroyAllWindows()


def _load_tune_params(gestures: list, profile_path: str):
    """Apply saved gesture_tuning values from a host profile to the GESTURES list."""
    if not os.path.exists(profile_path):
        print(f"[tuner] Profile not found: {profile_path} — using built-in defaults",
              file=sys.stderr)
        return
    try:
        with open(profile_path) as f:
            profile = json.load(f)
    except Exception as e:
        print(f"[tuner] Could not read {profile_path}: {e}", file=sys.stderr)
        return
    tuning = profile.get("gesture_tuning", {})
    applied = 0
    for gesture in gestures:
        saved = tuning.get(gesture["name"], {})
        for key, val in saved.items():
            if key in gesture["params"]:
                gesture["params"][key] = val
                applied += 1
    print(f"[tuner] {os.path.basename(profile_path)}: "
          f"{applied} tuning override(s) loaded from gesture_tuning")


def _save_tune_params(gestures: list, profile_path: str):
    """Write current tunable params back to the host profile's gesture_tuning section."""
    if not profile_path:
        return
    try:
        with open(profile_path) as f:
            profile = json.load(f)
    except Exception as e:
        print(f"[tuner] Could not read profile for save: {e}", file=sys.stderr)
        return
    tuning = {}
    for gesture in gestures:
        saved = {}
        for key in filter(None, [gesture.get("tune_key"), gesture.get("tune_key2")]):
            if key in gesture["params"]:
                saved[key] = gesture["params"][key]
        if saved:
            tuning[gesture["name"]] = saved
    profile["gesture_tuning"] = tuning
    try:
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"[tuner] Saved to {os.path.basename(profile_path)}")
    except Exception as e:
        print(f"[tuner] Could not save profile: {e}", file=sys.stderr)


def _adjust_tune_param(gesture, param_num: int, direction: int):
    """Adjust a tunable param. param_num: 1=primary (+/-), 2=secondary ([/])."""
    if param_num == 1:
        key = gesture.get("tune_key")
        step = gesture.get("tune_step", 50)
    else:
        key = gesture.get("tune_key2")
        step = gesture.get("tune_step2", 0.01)
    if not key:
        return
    current = gesture["params"].get(key, step * 8)
    new_val = current + direction * step
    new_val = max(step, new_val)
    if isinstance(step, float):
        new_val = round(new_val, 4)
    gesture["params"][key] = new_val
    print(f"[tuner] {gesture['name']}: {key} → {gesture['params'][key]}")


if __name__ == "__main__":
    main()
