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
from engines.depth.fusion import PoseDepth

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
    # ── forward_reach ─────────────────────────────────────────────────────
    dict(
        name="reach_flask",
        label="OI · Scene 1 · AL-01-002",
        desc="Move hand toward screen (Z-approach) in upper-center region  |  +/-: area_growth_threshold  |  [/]: window_frames",
        type="forward_reach",
        params={"area_growth_threshold": 0.30, "window_frames": 12, "target_region": "upper_center"},
        tune_key="area_growth_threshold", tune_step=0.05,
        tune_key2="window_frames", tune_step2=1,
        accent=(0, 220, 255), highlights=[0],
    ),
    # ── forward_point ─────────────────────────────────────────────────────
    dict(
        name="point_quilt_block",
        label="OI · Scene 1 · AL-01-010",
        desc="Extend index finger toward top-left quadrant, hold 500ms  |  +/-: hold_ms",
        type="forward_point",
        params={"target_region": "top_left_quadrant", "hold_ms": 500},
        tune_key="hold_ms", tune_step=50, accent=(0, 220, 255), highlights=[0, 8],
    ),
    # ── point_region (pose wrist L/R of body midline) ─────────────────────
    dict(
        name="point_path — LEFT",
        label="CG · Scene 2 · AL-02-007 (shot 09)",
        desc="Raise a hand to your LEFT of body midline & hold (pose region)  |  +/-: hold_ms  |  [/]: dead_zone_frac",
        type="point_region",
        params={"directions": ["left"], "hold_ms": 500, "dead_zone_frac": 0.15},
        tune_key="hold_ms", tune_step=50,
        tune_key2="dead_zone_frac", tune_step2=0.02,
        accent=(80, 200, 255), highlights=[0],
    ),
    dict(
        name="point_path — RIGHT",
        label="CG · Scene 2 · AL-02-007 (shot 09)",
        desc="Raise a hand to your RIGHT of body midline & hold (pose region)  |  +/-: hold_ms  |  [/]: dead_zone_frac",
        type="point_region",
        params={"directions": ["right"], "hold_ms": 500, "dead_zone_frac": 0.15},
        tune_key="hold_ms", tune_step=50,
        tune_key2="dead_zone_frac", tune_step2=0.02,
        accent=(80, 200, 255), highlights=[0],
    ),
    dict(
        name="point_river_marker",
        label="OI · Scene 3 · AL-03-006",
        desc="Extend index finger toward lower-third of screen (river), hold 500ms  |  +/-: hold_ms",
        type="forward_point",
        params={"target_region": "lower_third", "hold_ms": 500},
        tune_key="hold_ms", tune_step=50, accent=(0, 220, 255), highlights=[0, 8],
    ),
    dict(
        name="point_dipper_corner",
        label="CG · Scene 4 · AL-04-008",
        desc="Extend index finger toward top-right quadrant (dipper bowl corner), hold 500ms  |  +/-: hold_ms",
        type="forward_point",
        params={"target_region": "top_right_quadrant", "hold_ms": 500},
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
        desc="Wrist near either ear — no curl required, either hand  |  +/-: hold_ms  |  [/]: ear_radius",
        type="directional_head_or_hand",
        params={"direction": "ear", "hold_ms": 250, "ear_radius": 0.22},
        tune_key="hold_ms", tune_step=50,
        tune_key2="ear_radius", tune_step2=0.02,
        accent=(200, 180, 255), highlights=[0],
    ),
    # ── survey ────────────────────────────────────────────────────────────
    dict(
        name="survey",
        label="CG · Scene 3 · AL-03-004",
        desc="SALUTE: hold hand at brow  |  +/-: salute_hold_ms  |  [/]: min_x_delta (scan mode)",
        type="survey",
        params={"brow_y": 0.45, "hold_ms": 200, "salute_hold_ms": 500,
                "scan_window_ms": 2000, "min_x_delta": 0.06},
        tune_key="salute_hold_ms", tune_step=100,
        tune_key2="min_x_delta", tune_step2=0.005, tune_min2=0.005,
        accent=(200, 180, 255), highlights=[0],
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
        desc="Tipping hand-to-mouth: tips near mouth, wrist at/below tips",
        type="mouth_proximity_tip",
        params={"hold_ms": 400, "proximity_threshold": 0.12},
        tune_key="hold_ms", accent=(80, 255, 200), highlights=[0, 4, 8],
    ),
    # ── rhythm_bilateral ──────────────────────────────────────────────────
    dict(
        name="three_knock",
        label="CG · Scene 6 · AL-06-007",
        desc="KNOCK = push hand at camera (grows bigger), x2  |  +/-: min_growth_pct  |  [/]: knock_window_ms",
        type="rhythm_bilateral",
        params={"knock_count": 2, "min_growth_pct": 30, "knock_window_ms": 2500,
                "baseline_window_ms": 1200, "refractory_ms": 300},
        tune_key="min_growth_pct", tune_step=5,
        tune_key2="knock_window_ms", tune_step2=250,
        accent=(255, 100, 180), highlights=[0, 5],
    ),
    # ── speed_bilateral ───────────────────────────────────────────────────
    dict(
        name="fast_gather",
        label="CG · Scene 10 · AL-10-004",
        desc="Fast hand shaking/gathering (Pose wrists, units/s)  |  +/-: velocity_multiplier  |  [/]: min_bursts",
        type="speed_bilateral",
        params={"velocity_multiplier": 2.0, "min_bursts": 3, "burst_window_ms": 3000,
                "idle_velocity": 0.45, "use_pose": True},
        tune_key="velocity_multiplier", tune_step=0.25,
        tune_key2="min_bursts", tune_step2=1,
        accent=(255, 80, 80), highlights=[0, 8],
    ),
    # ── touch_head ────────────────────────────────────────────────────────
    dict(
        name="touch_temple",
        label="CG · Scene 5 · AL-05-011  (one hand, Pose)",
        desc="Touch temple/head (1 hand, Pose wrist)  |  +/-: radius_scale",
        type="touch_head",
        params={"hands_required": 1, "hold_ms": 500, "target": "temple", "use_pose": True, "radius_scale": 1.0},
        tune_key="radius_scale", tune_step=0.1,
        accent=(255, 220, 80), highlights=[0],
    ),
    dict(
        name="hands_to_head",
        label="CG · Scene 8 · AL-08-005  (two hands)",
        desc="Both hands to crown — put on hat  |  +/-: radius_scale",
        type="touch_head",
        params={"hands_required": 2, "hold_ms": 500, "radius_scale": 2.0},
        tune_key="radius_scale", tune_step=0.1,
        accent=(255, 220, 80), highlights=[0],
    ),
    # ── arms_crossed ─────────────────────────────────────────────────────
    dict(
        name="arms_crossed",
        label="CG · Scene 8 · AL-08-006",
        desc="Both wrists cross body midline at torso height  |  +/-: hold_ms  |  [/]: dead_zone_frac",
        type="arms_crossed",
        params={"hold_ms": 500, "dead_zone_frac": 0.05},
        tune_key="hold_ms", tune_step=50,
        tune_key2="dead_zone_frac", tune_step2=0.01,
        accent=(200, 120, 255), highlights=[0, 15, 16],
    ),
    # ── push_out ──────────────────────────────────────────────────────────
    dict(
        name="forward_push",
        label="CG · Scene 10 · AL-10-009  (urgent)",
        desc="Both hands thrust forward — bbox area growth bilateral  |  +/-: min_growth_pct  |  [/]: window_ms",
        type="push_out",
        params={"min_growth_pct": 50, "window_ms": 250},
        tune_key="min_growth_pct", tune_step=5,
        tune_key2="window_ms", tune_step2=25,
        accent=(255, 80, 80), highlights=[0],
    ),
    dict(
        name="launch_push",
        label="CG · Scene 11 · AL-11-012  (deliberate)",
        desc="Both hands launch forward — moderate velocity  |  +/-: min_growth_pct  |  [/]: window_ms",
        type="push_out",
        params={"min_growth_pct": 35, "window_ms": 400},
        tune_key="min_growth_pct", tune_step=5,
        tune_key2="window_ms", tune_step2=25,
        accent=(255, 140, 60), highlights=[0],
    ),
    # ── run_arms ──────────────────────────────────────────────────────────
    dict(
        name="run_arms",
        label="OI · Scene 10 · AL-10-010",
        desc="Fast alternating arm pump — ≥3 cycles within 2s  |  +/-: window_ms  |  [/]: min_cycles  |  W/S: waist_y_offset",
        type="run_arms",
        params={"min_cycles": 3, "window_ms": 2000, "waist_y_offset": 0.0},
        tune_key="window_ms", tune_step=100,
        tune_key2="min_cycles", tune_step2=1,
        accent=(255, 200, 40), highlights=[0, 15, 16],
    ),
    # ── unravel ───────────────────────────────────────────────────────────
    dict(
        name="unravel",
        label="CG · Scene 11 · AL-11-011",
        desc="Hands circle across the ELBOW LINE — ≥2 crossings-pairs per wrist  |  +/-: prox_frac (↑ = hands may be farther apart)  |  [/]: band_frac (↓ = smaller circles count)",
        type="unravel",
        params={"min_cycles": 2, "window_ms": 3000, "band_frac": 0.08, "prox_frac": 0.25},
        tune_key="prox_frac", tune_step=0.05,
        tune_key2="band_frac", tune_step2=0.01,
        accent=(80, 200, 255), highlights=[0, 15, 16],
    ),
    # ── paddle ────────────────────────────────────────────────────────────
    dict(
        name="paddle",
        label="CG · Scene 11 · AL-11-013",
        desc="Both wrists arc same side — 4 strokes shoulder→waist  |  +/-: min_strokes  |  [/]: sync_window_ms  |  W/S: waist_y_offset",
        type="paddle",
        params={"min_strokes": 4, "sync_window_ms": 400, "waist_y_offset": 0.0},
        tune_key="min_strokes", tune_step=1,
        tune_key2="sync_window_ms", tune_step2=50,
        accent=(80, 255, 160), highlights=[0, 11, 12, 15, 16],
    ),
    # ── throw ─────────────────────────────────────────────────────────────
    dict(
        name="throw",
        label="OI · Scene 11 · throw_rope (act 11 one-shot, shot 66)",
        desc="Wind up above shoulder, snap below the release line to fire  |  +/-: max_stroke_ms  |  [/]: shoulder_margin  |  W/S: release_y_offset",
        type="throw",
        params={"max_stroke_ms": 600, "shoulder_margin": 0.03, "release_y_offset": 0.0,
                "require_growth": False, "min_growth_pct": 40, "min_pose_z_delta": 0.2},
        tune_key="max_stroke_ms", tune_step=50,
        tune_key2="shoulder_margin", tune_step2=0.01,
        accent=(255, 120, 200), highlights=[0, 11, 12, 15, 16],
    ),
]

# ── Named thresholds (mirrors presence_bilateral.py) ────────────────────────
_NAMED_Y = {"shoulder": 0.5, "head": 0.3, "waist": 0.7}
_MOUTH_XY = (0.50, 0.78)

_WAIST_STEP = 0.01  # W/S key step for waist_y_offset


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


def _draw_track_histories(context) -> dict:
    """directional_draw keeps one history per tracked point (pose wrists draw
    independently). Normalise the legacy list form to a single-track dict."""
    hist = context.get("draw_history")
    if isinstance(hist, dict):
        return hist
    if isinstance(hist, list) and hist:
        return {"track": hist}
    return {}


def draw_draw_trail(frame, context, w, h, accent):
    """Draw the directional_draw motion trail(s) from context."""
    for key, history in _draw_track_histories(context).items():
        if len(history) < 2:
            continue
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
        cv2.putText(frame, f"{key}: disp={mag:.3f}  angle={angle_deg:.0f}deg",
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
    """Approach-knock panel: one circle per required push-toward-camera knock,
    plus a live approach bar (wrist depth / pose-z delta vs the pulled-back
    baseline — pose-only runtime)."""
    now = time.monotonic()
    knock_count_req = params.get("knock_count", 2)
    knock_window_ms = params.get("knock_window_ms", 2500)
    knock_times = [t for t in context.get("knock_times", [])
                   if (now - t) * 1000 <= knock_window_ms]

    cy = h // 2 - 50
    spacing = 110
    r = 30
    cx_base = w // 2 - spacing * (knock_count_req - 1) // 2

    for i in range(knock_count_req):
        cx = cx_base + i * spacing
        filled = len(knock_times) > i
        fill_color = accent if filled else (30, 30, 30)
        ring_color = accent if filled else (80, 80, 80)
        cv2.circle(frame, (cx, cy), r, fill_color, -1)
        cv2.circle(frame, (cx, cy), r, ring_color, 2, cv2.LINE_AA)
        txt_color = (255, 255, 255) if filled else (100, 100, 100)
        cv2.putText(frame, f"K{i + 1}", (cx - 14, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1, cv2.LINE_AA)

    bar_y = cy + r + 18
    bar_w = 220
    bar_x = w // 2 - bar_w // 2

    # Live approach bar: wrist depth (Gemini) or pose-z delta vs the rolling
    # pulled-back baseline, whichever side is furthest along.
    hist = context.get("knock_depth_hist") or {}
    best_pct, best_label = None, ""
    for side, samples in hist.items():
        if len(samples) < 2:
            continue
        kind = samples[-1][1]
        same = [v for _, k, v in samples if k == kind]
        baseline, cur = max(same), samples[-1][2]
        need = (params.get("min_depth_delta_mm", 120) if kind == "mm"
                else params.get("min_pose_z_delta", 0.12))
        pct = max(0.0, min(1.5, (baseline - cur) / max(1e-6, need)))
        if best_pct is None or pct > best_pct:
            best_pct = pct
            unit = "mm" if kind == "mm" else "z"
            best_label = (f"{side} approach {(baseline - cur):+.0f}{unit}  "
                          f"need {need:.0f}{unit}" if kind == "mm" else
                          f"{side} approach {(baseline - cur):+.2f}{unit}  need {need:.2f}{unit}")
    if best_pct is not None:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (50, 50, 50), -1)
        filled_w = int(min(1.0, best_pct) * bar_w)
        color = accent if best_pct >= 1.0 else (90, 160, 90)
        if filled_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + 12), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (120, 120, 120), 1)
        cv2.putText(frame, best_label,
                    (bar_x, bar_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "push your hand toward the camera",
                    (bar_x, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (150, 150, 150), 1, cv2.LINE_AA)

    # Countdown until the accumulated knocks expire
    if knock_times:
        remaining_ms = knock_window_ms - (now - knock_times[0]) * 1000
        cv2.putText(frame, f"window: {max(0.0, remaining_ms):.0f}ms",
                    (bar_x, bar_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    accent, 1, cv2.LINE_AA)


def draw_head_or_hand_zones(frame, direction, w, h, accent,
                            ear_positions=None, brow_y=None, ear_radius=None):
    """Draw the active zone for directional_head_or_hand in player-perspective space.

    ear_positions: live pose-derived [(x,y), ...] in raw MediaPipe coords.
    brow_y: live pose-derived absolute Y threshold for brow direction.
    ear_radius: normalised wrist-to-ear detection radius (default 0.22).
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
        radius = ear_radius if ear_radius is not None else 0.22
        pose_label = "(pose)" if ear_positions is not None else "(estimated)"
        for ex, ey in ear_pos:
            epx = int((1 - ex) * w)  # mirrored
            epy = int(ey * h)
            r = int(radius * w)
            cv2.circle(overlay, (epx, epy), r, accent, -1)
            cv2.circle(frame, (epx, epy), r, accent, 1, cv2.LINE_AA)
            cv2.circle(frame, (epx, epy), 5, (80, 255, 200) if ear_positions else (120, 120, 120),
                       -1, cv2.LINE_AA)
        cv2.putText(frame, f"EAR zones {pose_label}  wrist prox {radius:.2f}",
                    (8, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, accent, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)


def draw_forward_reach_info(frame, context, params, lm_list, w, h, accent):
    """Overlay for forward_reach: target region + bbox-area growth indicator."""
    overlay = frame.copy()
    region_key = params.get("target_region", "upper_center")
    _REACH_REGIONS = {
        "upper_center": (0.30, 0.0, 0.70, 0.40),
        "upper_left":   (0.0,  0.0, 0.50, 0.40),
        "upper_right":  (0.50, 0.0, 1.0,  0.40),
        "center":       (0.33, 0.33, 0.67, 0.67),
    }
    if region_key in _REACH_REGIONS:
        x0, y0, x1, y1 = _REACH_REGIONS[region_key]
        px0, py0 = int(x0 * w), int(y0 * h)
        px1, py1 = int(x1 * w), int(y1 * h)
        cv2.rectangle(overlay, (px0, py0), (px1, py1), accent, -1)
        cv2.rectangle(frame, (px0, py0), (px1, py1), accent, 2, cv2.LINE_AA)
        cv2.putText(frame, f"target: {region_key}", (px0 + 6, py0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, accent, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

    # Approach bar: wrist depth (Gemini) / pose-z drop vs the rolling max.
    hist = context.get("reach_depth_history") or {}
    best = None   # (pct, label)
    for side, samples in hist.items():
        if len(samples) < 2:
            continue
        kind = samples[-1][1]
        same = [v for _, k, v in samples if k == kind]
        drop = max(same) - samples[-1][2]
        need = (params.get("min_depth_delta_mm", 180) if kind == "mm"
                else params.get("min_pose_z_delta", 0.18))
        pct = min(max(drop / max(1e-6, need), 0.0), 1.0)
        label = (f"{side} reach {drop:+.0f}mm / {need:.0f}mm" if kind == "mm"
                 else f"{side} reach {drop:+.2f}z / {need:.2f}z")
        if best is None or pct > best[0]:
            best = (pct, label)
    if best is not None:
        pct, label = best
        bar_x, bar_y = 10, h // 2
        bar_w = 220
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (50, 50, 50), -1)
        filled = int(pct * bar_w)
        color = (0, 255, 80) if pct >= 1.0 else accent
        if filled > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 14), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (120, 120, 120), 1)
        cv2.putText(frame, label, (bar_x + bar_w + 8, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color if pct >= 1.0 else (200, 200, 200), 1, cv2.LINE_AA)


def draw_forward_point_info(frame, params, lm_list, w, h, accent):
    """Overlay for forward_point: target quadrant + finger extension indicator."""
    overlay = frame.copy()
    region_key = params.get("target_region", "")
    _POINT_REGIONS = {
        "top_left_quadrant":  (0.0, 0.0, 0.5, 0.5),
        "top_right_quadrant": (0.5, 0.0, 1.0, 0.5),
        "lower_third":        (0.0, 0.67, 1.0, 1.0),
        "center":             (0.25, 0.25, 0.75, 0.75),
    }
    if region_key in _POINT_REGIONS:
        x0, y0, x1, y1 = _POINT_REGIONS[region_key]
        px0, py0 = int(x0 * w), int(y0 * h)
        px1, py1 = int(x1 * w), int(y1 * h)
        cv2.rectangle(overlay, (px0, py0), (px1, py1), accent, -1)
        cv2.rectangle(frame, (px0, py0), (px1, py1), accent, 2, cv2.LINE_AA)
        cv2.putText(frame, f"target: {region_key}", (px0 + 6, py0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, accent, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

    # Finger extension indicator per hand
    for hand in lm_list:
        lm = hand.landmark
        wrist, tip, mcp = lm[0], lm[8], lm[5]
        wrist_tip_sq = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
        wrist_mcp_sq = (mcp.x - wrist.x) ** 2 + (mcp.y - wrist.y) ** 2
        extended = wrist_tip_sq > wrist_mcp_sq

        # Draw tip dot: bright if extended, dim if curled
        tx_px = int((1 - tip.x) * w)
        ty_px = int(tip.y * h)
        dot_color = accent if extended else (80, 80, 80)
        cv2.circle(frame, (tx_px, ty_px), 9, dot_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (tx_px, ty_px), 11, (255, 255, 255), 1, cv2.LINE_AA)
        ext_label = "extended" if extended else "CURLED"
        cv2.putText(frame, ext_label, (tx_px + 14, ty_px + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, dot_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"tip px={tip.x:.2f} py={tip.y:.2f}",
                    (tx_px + 14, ty_px + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
        break  # first hand only


def draw_survey_info(frame, context, params, pose_lm, w, h, accent):
    """Overlay for survey: brow threshold line + phase indicator + scan direction.
    Pose-based: wrist dots come from Pose landmarks 15/16, same as the detector."""
    brow_y = params.get("brow_y", 0.45)
    min_x_delta = params.get("min_x_delta", 0.06)
    phase = context.get("survey_phase", 0)

    # Brow threshold line
    by_px = int(brow_y * h)
    cv2.line(frame, (0, by_px), (w, by_px), accent, 1, cv2.LINE_AA)
    cv2.putText(frame, f"brow_y={brow_y:.2f}  (wrist must be ABOVE this line)",
                (8, by_px - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, accent, 1, cv2.LINE_AA)

    # Phase label
    phase_labels = {0: "Phase 0: waiting for wrist above brow",
                    1: "Phase 1: holding at brow — scan will open",
                    2: "Phase 2: SCAN  ← move wrist left/right →",
                    3: "FIRED"}
    phase_color = {0: (120, 120, 120), 1: (255, 200, 80), 2: accent, 3: (0, 255, 80)}
    label = phase_labels.get(phase, f"Phase {phase}")
    color = phase_color.get(phase, accent)
    cv2.putText(frame, label, (8, by_px + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Scan direction arrows when in phase 2
    if phase == 2:
        cx, cy = w // 2, by_px - 40
        arrow_len = int(min_x_delta * w) + 20
        cv2.arrowedLine(frame, (cx, cy), (cx - arrow_len, cy), accent, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(frame, (cx, cy), (cx + arrow_len, cy), accent, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(frame, f"min_x_delta={min_x_delta:.2f}", (cx - 50, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
        # Show last scan direction
        last_dir = context.get("scan_last_direction")
        if last_dir:
            cv2.putText(frame, f"last scan dir: {last_dir}",
                        (8, by_px + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, accent, 1, cv2.LINE_AA)

    # Live wrist indicators — POSE wrists (15/16), matching the detector.
    # The tracked (locked) side draws large in the accent colour.
    if pose_lm is not None:
        tracked_side = context.get("survey_side")
        for side, idx in (("L", 15), ("R", 16)):
            wr = pose_lm[idx]
            if getattr(wr, "visibility", 1.0) < params.get("min_visibility", 0.5):
                continue
            wx_px = int((1 - wr.x) * w)
            wy_px = int(wr.y * h)
            above = wr.y < brow_y
            is_tracked = side == tracked_side
            dot_color = accent if (above or is_tracked) else (80, 80, 80)
            cv2.circle(frame, (wx_px, wy_px), 10 if is_tracked else 7,
                       dot_color, -1, cv2.LINE_AA)
            label = f"{side}{' [tracked]' if is_tracked else ''}  y={wr.y:.3f}"
            cv2.putText(frame, label, (wx_px + 14, wy_px + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, dot_color, 1, cv2.LINE_AA)


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

    elif gtype == "survey":
        # Inject live brow_y from eye level — same derivation as directional_head_or_hand brow
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


def draw_touch_head_info(frame, context, params, lm_list, pose_lm, w, h, accent):
    """Show crown target circle and wrist proximity dots."""
    import math as _math
    if pose_lm is not None:
        ear_l, ear_r = pose_lm[7],  pose_lm[8]
        sh_l,  sh_r  = pose_lm[11], pose_lm[12]
        head_cx = (ear_l.x + ear_r.x) / 2
        head_cy = (ear_l.y + ear_r.y) / 2
        shoulder_y = (sh_l.y + sh_r.y) / 2
        ear_sh = shoulder_y - head_cy
        crown_y = head_cy - 0.4 * max(ear_sh, 0.05)
        crown_x = head_cx
        sh_width = abs(sh_l.x - sh_r.x)
        radius = max(0.15 * sh_width, 0.07)
    else:
        crown_x, crown_y, radius = 0.50, 0.10, 0.10
    # Mirror X for display
    cx_px = int((1 - crown_x) * w)
    cy_px = int(crown_y * h)
    r_px  = int(radius * w)
    cv2.circle(frame, (cx_px, cy_px), r_px, accent, 2, cv2.LINE_AA)
    cv2.putText(frame, "crown", (cx_px - 22, cy_px - r_px - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, accent, 1, cv2.LINE_AA)
    # Draw wrist dots with IN/OUT colour
    if lm_list:
        for hand in lm_list:
            wx, wy = hand.landmark[0].x, hand.landmark[0].y
            dist = _math.hypot(wx - crown_x, wy - crown_y)
            color = accent if dist < radius else (200, 80, 80)
            px, py = int((1 - wx) * w), int(wy * h)
            cv2.circle(frame, (px, py), 8, color, -1)


def draw_arms_crossed_info(frame, context, pose_lm, w, h, accent):
    """Show midline, shoulder/hip band, and wrist crossing status."""
    if pose_lm is None:
        return
    sh_l, sh_r  = pose_lm[11], pose_lm[12]
    hip_l, hip_r = pose_lm[23], pose_lm[24]
    lw, rw       = pose_lm[15], pose_lm[16]

    midline_x  = (sh_l.x + sh_r.x) / 2
    shoulder_y = (sh_l.y + sh_r.y) / 2
    hip_y      = (hip_l.y + hip_r.y) / 2

    # Midline (mirrored)
    mx_px = int((1 - midline_x) * w)
    sy_px = int(shoulder_y * h)
    hy_px = int(hip_y * h)
    cv2.line(frame, (mx_px, sy_px), (mx_px, hy_px), accent, 2, cv2.LINE_AA)
    cv2.putText(frame, "midline", (mx_px + 4, sy_px - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, accent, 1, cv2.LINE_AA)
    # Shoulder / hip bands
    cv2.line(frame, (0, sy_px), (w, sy_px), (100, 100, 100), 1)
    cv2.line(frame, (0, hy_px), (w, hy_px), (100, 100, 100), 1)

    sh_width = abs(sh_l.x - sh_r.x)
    dz = 0.05 * max(sh_width, 0.10)
    crossed_l = lw.x < (midline_x - dz)
    crossed_r = rw.x > (midline_x + dz)

    for pose_lm_idx, label, crossed in [(15, "L", crossed_l), (16, "R", crossed_r)]:
        lm = pose_lm[pose_lm_idx]
        color = accent if crossed else (200, 80, 80)
        px, py = int((1 - lm.x) * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 10, color, -1)
        cv2.putText(frame, label, (px - 6, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_push_out_info(frame, context, params, lm_list, w, h, accent):
    """Per-wrist approach bars: depth (Gemini) / pose-z drop vs threshold."""
    hist = context.get("push_depth_history") or {}
    for i, side in enumerate(("L", "R")):
        samples = hist.get(side) or []
        drop, need, label_val = 0.0, 1.0, "--"
        if len(samples) >= 2:
            kind = samples[-1][1]
            same = [v for _, k, v in samples if k == kind]
            drop = max(same) - samples[-1][2]
            need = (params.get("min_depth_delta_mm", 200) if kind == "mm"
                    else params.get("min_pose_z_delta", 0.15))
            label_val = (f"{drop:+.0f}mm" if kind == "mm" else f"{drop:+.2f}z")
        pct = min(max(drop / max(1e-6, need), 0.0), 1.0)
        bar_x = 12 + i * 180
        bar_y = h - 60
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 150, bar_y + 18), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(150 * pct), bar_y + 18), accent, -1)
        cv2.putText(frame, f"{side}: {label_val}", (bar_x, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, accent, 1, cv2.LINE_AA)


def draw_run_arms_info(frame, context, params, pose_lm, w, h, accent):
    """Show waist line and alternating-state counters."""
    if pose_lm is None:
        return
    hip_l, hip_r = pose_lm[23], pose_lm[24]
    sh_l, sh_r = pose_lm[11], pose_lm[12]
    # Match run_arms.py: default reference is the WAIST (shoulder/hip midpoint),
    # not the hip line — natural arm-pumping crosses it.
    hip_y = (hip_l.y + hip_r.y) / 2
    shoulder_y = (sh_l.y + sh_r.y) / 2
    if params.get("reference", "waist") == "hips":
        waist_y = hip_y + params.get("waist_y_offset", 0.0)
    else:
        waist_y = (shoulder_y + hip_y) / 2 + params.get("waist_y_offset", 0.0)
    wy_px = int(waist_y * h)
    cv2.line(frame, (0, wy_px), (w, wy_px), accent, 2, cv2.LINE_AA)
    offset_val = params.get("waist_y_offset", 0.0)
    offset_str = f"  offset={offset_val:+.2f}  [W/S to adjust]" if offset_val != 0.0 else "  [W/S to adjust]"
    cv2.putText(frame, f"waist{offset_str}", (10, wy_px - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, accent, 1, cv2.LINE_AA)

    transitions = context.get("run_alt_transitions", [])
    min_cycles = params.get("min_cycles", 3)
    required = min_cycles * 2
    cv2.putText(frame, f"transitions: {len(transitions)}/{required}",
                (10, wy_px + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent, 1, cv2.LINE_AA)

    # Wrist markers
    for idx, label in [(15, "L"), (16, "R")]:
        lm = pose_lm[idx]
        above = lm.y < waist_y
        color = (80, 200, 80) if above else (200, 120, 80)
        px, py = int((1 - lm.x) * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 10, color, -1)
        cv2.putText(frame, label, (px - 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def draw_unravel_info(frame, context, params, pose_lm, w, h, accent):
    """Elbow-line crossing model: draw the elbow-elbow line + hysteresis band,
    wrist dots, and the per-wrist crossing counts the detector fires on."""
    if pose_lm is None:
        return
    sh_l, sh_r   = pose_lm[11], pose_lm[12]
    el_l, el_r   = pose_lm[13], pose_lm[14]
    hip_l, hip_r = pose_lm[23], pose_lm[24]
    lw, rw       = pose_lm[15], pose_lm[16]

    shoulder_y = (sh_l.y + sh_r.y) / 2
    hip_y      = (hip_l.y + hip_r.y) / 2
    import math as _math
    wrist_dist = _math.hypot(lw.x - rw.x, lw.y - rw.y)
    sh_width   = abs(sh_l.x - sh_r.x)
    prox_ok    = wrist_dist < params.get("prox_frac", 0.25) * max(sh_width, 0.10)

    # Shoulder / hip bands (torso constraint)
    cv2.line(frame, (0, int(shoulder_y * h)), (w, int(shoulder_y * h)), (100, 100, 100), 1)
    cv2.line(frame, (0, int(hip_y * h)),      (w, int(hip_y * h)),      (100, 100, 100), 1)

    # Elbow line + hysteresis band, extended across the frame (mirrored x).
    dx, dy = el_r.x - el_l.x, el_r.y - el_l.y
    line_len = _math.hypot(dx, dy)
    if line_len >= 0.03:
        band = params.get("band_frac", 0.08) * max(sh_width, 0.10)
        ux, uy = dx / line_len, dy / line_len          # along the line
        nx, ny = -uy, ux                               # normal
        cx, cy = (el_l.x + el_r.x) / 2, (el_l.y + el_r.y) / 2
        for off, color, thick in ((0.0, accent, 2),
                                  (band, (120, 120, 120), 1),
                                  (-band, (120, 120, 120), 1)):
            ax, ay = cx + nx * off - ux, cy + ny * off - uy
            bx, by = cx + nx * off + ux, cy + ny * off + uy
            cv2.line(frame, (int((1 - ax) * w), int(ay * h)),
                     (int((1 - bx) * w), int(by * h)), color, thick, cv2.LINE_AA)

    # Wrist dots (red when the proximity constraint is failing)
    for idx in [15, 16]:
        lm = pose_lm[idx]
        color = accent if prox_ok else (200, 80, 80)
        px, py = int((1 - lm.x) * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 10, color, -1)

    # Per-wrist elbow-line crossings — the exact counts the detector fires on.
    crossings = context.get("unravel_crossings", {})
    need = params.get("min_cycles", 2) * 2
    cv2.putText(frame,
                f"prox: {'OK' if prox_ok else 'FAR'}  crossings  "
                f"L: {len(crossings.get('L', []))}/{need}  "
                f"R: {len(crossings.get('R', []))}/{need}",
                (12, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent, 1, cv2.LINE_AA)


def draw_throw_info(frame, context, params, pose_lm, lm_list, w, h, accent):
    """Per-arm shoulder line with the arm/release margin band; wrist dots show
    the wind-up (ARMED) state read from context['throw_armed']."""
    if pose_lm is None:
        return
    margin = params.get("shoulder_margin", 0.03)
    release_offset = params.get("release_y_offset", 0.0)
    armed = context.get("throw_armed", {})

    for side, wrist_i, shoulder_i in (("L", 15, 11), ("R", 16, 12)):
        sh = pose_lm[shoulder_i]
        wr = pose_lm[wrist_i]
        sh_px = int((1 - sh.x) * w)
        arm_y = int((sh.y - margin) * h)                     # wind-up (true shoulder)
        rel_y = int((sh.y + release_offset + margin) * h)    # release (W/S to shift)
        x0, x1 = max(sh_px - 160, 0), min(sh_px + 160, w)
        cv2.line(frame, (x0, arm_y), (x1, arm_y), accent, 2, cv2.LINE_AA)
        cv2.line(frame, (x0, rel_y), (x1, rel_y), (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, "arm", (x0 + 4, arm_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, accent, 1, cv2.LINE_AA)
        cv2.putText(frame, "release", (x0 + 4, rel_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        wx, wy = int((1 - wr.x) * w), int(wr.y * h)
        is_armed = side in armed
        cv2.circle(frame, (wx, wy), 10, accent if is_armed else (160, 160, 160), -1)
        if is_armed:
            cv2.putText(frame, f"{side} ARMED", (wx + 14, wy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, accent, 1, cv2.LINE_AA)

    offset_str = (f"  release offset={release_offset:+.3f}  [W/S]"
                  if release_offset != 0.0 else "  [W/S: release line offset]")
    cv2.putText(frame, f"wind up ABOVE the arm line, snap BELOW the release line{offset_str}",
                (12, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, accent, 1, cv2.LINE_AA)


def draw_paddle_info(frame, context, params, pose_lm, w, h, accent):
    """Show midline, stroke count, and same-side status."""
    if pose_lm is None:
        return
    sh_l, sh_r   = pose_lm[11], pose_lm[12]
    hip_l, hip_r = pose_lm[23], pose_lm[24]
    lw, rw       = pose_lm[15], pose_lm[16]

    midline_x    = (sh_l.x + sh_r.x) / 2
    shoulder_y   = (sh_l.y + sh_r.y) / 2
    waist_offset = params.get("waist_y_offset", 0.0)
    hip_y        = (hip_l.y + hip_r.y) / 2 + waist_offset
    # Match paddle.py: strokes are counted against the shoulder/hip MIDLINE with
    # a hysteresis band — draw exactly what the detector evaluates.
    mid_y        = (shoulder_y + hip_y) / 2
    band         = params.get("band_frac", 0.12) * max(hip_y - shoulder_y, 0.05)

    mx_px = int((1 - midline_x) * w)
    my_px = int(mid_y * h)
    b_px  = max(1, int(band * h))

    cv2.line(frame, (mx_px, 0), (mx_px, h), (80, 80, 80), 1)
    cv2.line(frame, (0, my_px - b_px), (w, my_px - b_px), (110, 110, 110), 1)
    cv2.line(frame, (0, my_px), (w, my_px), accent, 2)
    cv2.line(frame, (0, my_px + b_px), (w, my_px + b_px), (110, 110, 110), 1)
    offset_str = f"  offset={waist_offset:+.2f}" if waist_offset != 0.0 else ""
    cv2.putText(frame, f"stroke midline (±band){offset_str}", (4, my_px - b_px - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, accent, 1, cv2.LINE_AA)

    stroke_count = len(context.get("paddle_stroke_times", []))
    min_strokes  = params.get("min_strokes", 4)
    cv2.putText(frame, f"strokes: {stroke_count}/{min_strokes}",
                (12, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent, 1, cv2.LINE_AA)

    for idx, crossed_key in ((15, "paddle_l_crossed"), (16, "paddle_r_crossed")):
        lm = pose_lm[idx]
        color = accent if context.get(crossed_key) else (140, 140, 140)
        px, py = int((1 - lm.x) * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 10, color, -1)


def draw_point_region_info(frame, context, params, pose_lm, w, h, accent):
    """Show body midline, centre dead-zone band, hip gate, and wrist region status."""
    if pose_lm is None:
        cv2.putText(frame, "POSE NONE — cannot fire", (12, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 80, 80), 2, cv2.LINE_AA)
        return
    sh_l, sh_r   = pose_lm[11], pose_lm[12]
    hip_l, hip_r = pose_lm[23], pose_lm[24]
    lw, rw       = pose_lm[15], pose_lm[16]

    midline_x      = (sh_l.x + sh_r.x) / 2
    shoulder_width = abs(sh_l.x - sh_r.x)
    hip_y          = (hip_l.y + hip_r.y) / 2
    dead           = params.get("dead_zone_frac", 0.15) * max(shoulder_width, 0.10)
    accepted       = set(params.get("directions") or ["left", "right"])
    require_raised = params.get("require_raised", True)

    # Midline + dead-zone band (mirrored to display space)
    mx_px   = int((1 - midline_x) * w)
    dead_px = int(dead * w)
    overlay = frame.copy()
    cv2.rectangle(overlay, (mx_px - dead_px, 0), (mx_px + dead_px, h), (120, 120, 120), -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.line(frame, (mx_px, 0), (mx_px, h), accent, 1, cv2.LINE_AA)
    cv2.putText(frame, "midline", (mx_px + 4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, accent, 1, cv2.LINE_AA)

    # Side labels
    cv2.putText(frame, "LEFT" if "left" in accepted else "left (off)",
                (mx_px - dead_px - 80, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                accent if "left" in accepted else (90, 90, 90), 2, cv2.LINE_AA)
    cv2.putText(frame, "RIGHT" if "right" in accepted else "right (off)",
                (mx_px + dead_px + 10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                accent if "right" in accepted else (90, 90, 90), 2, cv2.LINE_AA)

    # Hip gate line (wrist must be ABOVE this to count when require_raised)
    if require_raised:
        hy_px = int(hip_y * h)
        cv2.line(frame, (0, hy_px), (w, hy_px), (160, 160, 80), 1, cv2.LINE_AA)
        cv2.putText(frame, "hip gate (raise above)", (8, hy_px - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 80), 1, cv2.LINE_AA)

    # Wrist dots — colour by whether they'd register
    for wrist, label in ((lw, "L"), (rw, "R")):
        screen_x = 1.0 - wrist.x
        offset   = screen_x - (1 - midline_x)
        raised   = (not require_raised) or wrist.y < hip_y
        side     = "left" if offset < 0 else "right"
        active   = raised and abs(offset) >= dead and side in accepted
        color    = (80, 255, 80) if active else (200, 80, 80)
        px, py   = int((1 - wrist.x) * w), int(wrist.y * h)
        cv2.circle(frame, (px, py), 10, color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (px - 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


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

    # Orbbec Gemini 335 first when enabled in config.json (use_orbbec_camera).
    # Same selection logic as main.py; falls back to the webcam below.
    if cfg.get("use_orbbec_camera"):
        try:
            from engines.depth.orbbec_camera import try_open_orbbec
            cap = try_open_orbbec(resolution=(1280, 720), fps=30)
            if cap is not None:
                print("[tuner] Using Orbbec Gemini 335 (color + depth)")
                return cap
        except Exception as e:
            print(f"[tuner] Orbbec import/open error: {e}", file=sys.stderr)
        print("[tuner] Orbbec unavailable — falling back to webcam")

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
    config_path = os.path.join(ROOT, "config.json")
    _load_tune_params(GESTURES, profile_path)

    cap = open_camera(config_path)
    if not cap.isOpened():
        print("[tuner] Failed to open camera.", file=sys.stderr)
        sys.exit(1)

    # POSE-ONLY (July 2026): the Hands model is gone from the runtime; the
    # tuner mirrors that — one Pose graph, detectors get landmarks=[].
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=1,
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
        lm_list = []   # legacy Hands slot — always empty (pose-only runtime)

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
        # Inject pose landmarks so body-relative detectors (touch_head, arms_crossed,
        # run_arms, unravel, paddle) can read context["_pose_lm"] the same way
        # the runtime's gesture_engine does. Same for the depth sampler when the
        # capture device is the Orbbec shim.
        context["_pose_lm"] = pose_lm
        _sampler = getattr(cap, "depth_mm_at", None)
        context["_depth_mm_at"] = _sampler
        # Depth fusion view (same as the runtime injects): phantom veto, reach
        # metering and metric velocity all read from context["_pose_depth"].
        context["_pose_depth"] = PoseDepth(_sampler, pose_lm)

        detector_fn = REGISTRY.get(gtype)
        fired = False
        if detector_fn:
            fired = detector_fn(lm_list, live_params, context)

        if fired:
            fire_count += 1
            fired_at = time.monotonic()

        # ── Draw skeleton ─────────────────────────────────────────────────
        draw_skeleton(frame, lm_list, highlights, accent)

        # ── Pose body markers (always, when pose is available) ────────────
        draw_pose_markers(frame, pose_lm, w, h)

        # ── Depth fusion badge (Gemini 335 present) ───────────────────────
        if _sampler is not None:
            _fus = context["_pose_depth"]
            _torso = _fus.torso_depth_mm() if _fus.available else None
            if _torso is not None:
                _reach_bits = []
                for _s, _wi in (("L", 15), ("R", 16)):
                    _r = _fus.reach_mm(_wi)
                    if _r is not None:
                        _reach_bits.append(f"{_s}{_r:+.0f}")
                _label = f"DEPTH torso {_torso:.0f}mm"
                if _reach_bits:
                    _label += "  reach " + " ".join(_reach_bits)
            else:
                _label = "DEPTH on (no torso lock)"
            cv2.putText(frame, _label, (12, h - 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 255), 1, cv2.LINE_AA)

        # ── Gesture-specific overlays ─────────────────────────────────────
        if gtype == "forward_reach":
            draw_forward_reach_info(frame, context, live_params, lm_list, w, h, accent)

        elif gtype == "forward_point":
            draw_forward_point_info(frame, live_params, lm_list, w, h, accent)

        elif gtype == "survey":
            draw_survey_info(frame, context, live_params, pose_lm, w, h, accent)

        elif gtype == "directional_point":
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

        elif gtype == "point_region":
            draw_point_region_info(frame, context, live_params, pose_lm, w, h, accent)

        elif gtype == "presence_bilateral":
            y_thresh = live_params.get("y_threshold", 0)
            draw_bilateral_threshold(frame, y_thresh, w, h, accent)

        elif gtype == "directional_draw":
            draw_draw_trail(frame, context, w, h, accent)

        elif gtype == "directional_head_or_hand":
            direction = live_params.get("direction", "right")
            ear_pos = live_params.get("ear_positions")
            brow_y_val = live_params.get("brow_y")
            ear_r = live_params.get("ear_radius")
            draw_head_or_hand_zones(frame, direction, w, h, accent,
                                    ear_positions=ear_pos, brow_y=brow_y_val,
                                    ear_radius=ear_r)

        elif gtype == "mouth_proximity_tip":
            draw_mouth_marker(frame, w, h, accent, context,
                              mouth_x=live_params.get("mouth_x"),
                              mouth_y=live_params.get("mouth_y"),
                              threshold=live_params.get("proximity_threshold", 0.12))

        elif gtype == "rhythm_bilateral":
            draw_knock_progress(frame, context, live_params, w, h, accent)

        elif gtype == "touch_head":
            draw_touch_head_info(frame, context, live_params, lm_list, pose_lm, w, h, accent)

        elif gtype == "arms_crossed":
            draw_arms_crossed_info(frame, context, pose_lm, w, h, accent)

        elif gtype == "push_out":
            draw_push_out_info(frame, context, live_params, lm_list, w, h, accent)

        elif gtype == "run_arms":
            draw_run_arms_info(frame, context, live_params, pose_lm, w, h, accent)

        elif gtype == "unravel":
            draw_unravel_info(frame, context, live_params, pose_lm, w, h, accent)

        elif gtype == "paddle":
            draw_paddle_info(frame, context, live_params, pose_lm, w, h, accent)

        elif gtype == "throw":
            draw_throw_info(frame, context, live_params, pose_lm, lm_list, w, h, accent)

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

        if gtype == "forward_reach":
            hist = context.get("reach_depth_history") or {}
            pct = 0.0
            for side, samples in hist.items():
                if len(samples) < 2:
                    continue
                kind = samples[-1][1]
                same = [v for _, k, v in samples if k == kind]
                drop = max(same) - samples[-1][2]
                need = (live_params.get("min_depth_delta_mm", 180) if kind == "mm"
                        else live_params.get("min_pose_z_delta", 0.18))
                spct = min(max(drop / max(1e-6, need), 0.0), 1.0)
                pct = max(pct, spct)
                unit = "mm" if kind == "mm" else "z"
                fmt = f"{drop:+.0f}{unit} / {need:.0f}{unit}" if kind == "mm" \
                    else f"{drop:+.2f}{unit} / {need:.2f}{unit}"
                state_lines.append((f"{side} approach: {fmt}  {'FIRE' if spct >= 1.0 else '...'}",
                                     accent if spct >= 1.0 else (200, 200, 200)))
            if not state_lines:
                state_lines.append(("waiting for wrist depth/z samples...", (120, 120, 120)))
            fired_flag = context.get("forward_reach_fired", False)
            state_lines.append((f"fired_flag: {'FIRED — reset (R) to retry' if fired_flag else 'waiting'}",
                                 accent if fired_flag else (120, 120, 120)))
            state_lines.append(("reach progress:", (200, 200, 200)))

        elif gtype == "forward_point":
            pct = hold_progress(context, "forward_point_since", hold_ms or 500)
            since = context.get("forward_point_since")
            region_key = live_params.get("target_region", "")
            state_lines.append((f"region: {region_key}", (200, 200, 200)))
            # Pose+depth readout (arm reaching toward the camera).
            _sampler = context.get("_depth_mm_at")
            if pose_lm is not None:
                need_mm = live_params.get("min_reach_depth_mm", 300)
                need_z = live_params.get("min_pose_z_delta", 0.25)
                for _side, _wi, _si in (("L", 15, 11), ("R", 16, 12)):
                    _wr, _sh = pose_lm[_wi], pose_lm[_si]
                    if getattr(_wr, "visibility", 1.0) < live_params.get("min_visibility", 0.5):
                        continue
                    _wmm = _smm = None
                    if callable(_sampler):
                        _wmm = _sampler(_wr.x, _wr.y)
                        _smm = _sampler(_sh.x, _sh.y)
                    if _wmm is not None and _smm is not None:
                        _reach = _smm - _wmm
                        _ok = _reach >= need_mm
                        state_lines.append((f"{_side} reach: {_reach:+.0f}mm  need {need_mm}mm  {'OK' if _ok else '...'}  [depth]",
                                             accent if _ok else (160, 160, 160)))
                    else:
                        _zd = getattr(_sh, "z", 0.0) - getattr(_wr, "z", 0.0)
                        _ok = _zd >= need_z
                        state_lines.append((f"{_side} reach: z-delta={_zd:+.2f}  need {need_z:.2f}  {'OK' if _ok else '...'}  [pose-z]",
                                             accent if _ok else (160, 160, 160)))
            state_lines.append((f"on target since: {'YES' if since else 'no'}",
                                 accent if since else (120, 120, 120)))
            state_lines.append(("hold progress:", (200, 200, 200)))

        elif gtype == "survey":
            phase = context.get("survey_phase", 0)
            if live_params.get("require_scan", False):
                phase_labels = {0: "waiting for wrist above brow",
                                1: "holding at brow",
                                2: "scanning ← →",
                                3: "DONE"}
                pct = phase / 3.0
            else:
                # Salute mode (default): phase 1 = holding at brow toward fire.
                phase_labels = {0: "waiting for wrist above brow",
                                1: "SALUTE — holding at brow"}
                pct = hold_progress(context, "survey_phase1_since",
                                    live_params.get("salute_hold_ms", 500)) if phase == 1 else 0.0
            state_lines.append((f"phase {phase}: {phase_labels.get(phase, '?')}",
                                 accent if phase > 0 else (120, 120, 120)))
            brow_y_live = live_params.get("brow_y", 0.45)
            pose_src = "pose" if pose_lm else "fixed"
            state_lines.append((f"brow_y={brow_y_live:.3f} ({pose_src})  min_x_delta={live_params.get('min_x_delta', 0.06):.2f}",
                                 (160, 200, 160) if pose_lm else (120, 120, 120)))
            tracked = context.get("survey_side")
            if tracked:
                state_lines.append((f"tracked wrist: {tracked} (pose)", (160, 200, 160)))
            if phase == 2:
                last_dir = context.get("scan_last_direction")
                seg_start = context.get("scan_segment_start_x")
                if pose_lm and tracked and seg_start is not None:
                    cur_x = pose_lm[15 if tracked == "L" else 16].x
                    delta = abs(cur_x - seg_start)
                    state_lines.append((f"scan dx={delta:.3f}  last_dir: {last_dir or '—'}",
                                         accent if delta >= live_params.get("min_x_delta", 0.06) else (200, 200, 200)))
            state_lines.append(("phase progress:", (200, 200, 200)))

        elif gtype == "directional_point":
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

        elif gtype == "point_region":
            pct = hold_progress(context, "point_direction_since", hold_ms or 500)
            dir_now = context.get("point_direction") or "--"
            accepted = live_params.get("directions", ["left", "right"])
            in_set = dir_now in accepted
            state_lines.append((f"region: {dir_now}  accepted: {accepted}",
                                 accent if (dir_now != "--" and in_set) else (200, 80, 80) if dir_now != "--" else (120, 120, 120)))
            if pose_lm:
                midline_x = (pose_lm[11].x + pose_lm[12].x) / 2
                sh_width  = abs(pose_lm[11].x - pose_lm[12].x)
                dead      = live_params.get("dead_zone_frac", 0.15) * max(sh_width, 0.10)
                state_lines.append((f"midline={midline_x:.3f}  dead_zone={dead:.3f} (frac {live_params.get('dead_zone_frac', 0.15):.2f})",
                                     (160, 160, 160)))
            else:
                state_lines.append(("POSE NONE — cannot fire", (200, 80, 80)))
            state_lines.append(("hold progress:", (200, 200, 200)))

        elif gtype == "presence_bilateral":
            if pose_lm is not None:
                vis = [pose_lm[i] for i in (15, 16)
                       if getattr(pose_lm[i], "visibility", 1.0) >= 0.5]
                state_lines.append((f"pose wrists: {len(vis)}",
                                    (0, 255, 80) if len(vis) >= 2 else (200, 80, 80)))
                for i, wl in enumerate(vis[:2]):
                    state_lines.append((f"  wrist[{i}].y = {wl.y:.3f}", accent))
            else:
                state_lines.append(("POSE NONE — cannot fire", (200, 80, 80)))
            pct = hold_progress(context, "bilateral_present_since", hold_ms or 500)
            state_lines.append(("hold progress:", (200, 200, 200)))

        elif gtype == "directional_draw":
            tracks = _draw_track_histories(context)
            total_pts = sum(len(hh) for hh in tracks.values())
            state_lines.append((f"tracks: {len(tracks)}  history pts: {total_pts}", accent))
            for tkey, hist in tracks.items():
                if len(hist) >= 2:
                    screen_dx = -(hist[-1][0] - hist[0][0])
                    screen_dy = hist[-1][1] - hist[0][1]
                    mag = math.hypot(screen_dx, screen_dy)
                    ang = math.degrees(math.atan2(screen_dy, screen_dx))
                    state_lines.append((f"{tkey}: disp={mag:.3f}  angle={ang:.0f}deg", accent))
            pct = min(total_pts / 10, 1.0)
            state_lines.append(("buffer fill:", (200, 200, 200)))

        elif gtype == "directional_head_or_hand":
            since = context.get("directional_since")
            pct = hold_progress(context, "directional_since", hold_ms or 500)
            direction = live_params.get("direction", "up")
            state_lines.append((f"matched: {'YES' if since else 'no'}", accent if since else (120,120,120)))
            if direction == "ear" and lm_list:
                ear_pos = live_params.get("ear_positions", [(0.15, 0.35), (0.85, 0.35)])
                ear_radius = live_params.get("ear_radius", 0.22)
                best_dist = float("inf")
                for hand in lm_list:
                    wx, wy = hand.landmark[0].x, hand.landmark[0].y
                    for ex, ey in ear_pos:
                        d = math.hypot(wx - ex, wy - ey)
                        best_dist = min(best_dist, d)
                in_range = best_dist < ear_radius
                state_lines.append((f"wrist→ear: {best_dist:.3f}  radius={ear_radius:.2f}  {'IN' if in_range else 'OUT'}",
                                     accent if in_range else (200, 80, 80)))
                pose_src = "pose" if pose_lm else "fixed"
                state_lines.append((f"ear positions: {pose_src}", (160, 200, 160) if pose_lm else (120,120,120)))
            else:
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
            now_ts = time.monotonic()
            knock_count_req = live_params.get("knock_count", 2)
            knock_window_ms = live_params.get("knock_window_ms", 2500)
            min_growth = live_params.get("min_growth_pct", 30)
            knock_times = [t for t in context.get("knock_times", [])
                           if (now_ts - t) * 1000 <= knock_window_ms]
            knocks = len(knock_times)
            pct = knocks / max(1, knock_count_req)
            state_lines.append((f"knocks: {knocks}/{knock_count_req}  "
                                 f"(push hand toward camera)",
                                 accent if knocks > 0 else (120, 120, 120)))
            hist = context.get("knock_area_hist") or []
            if hist:
                baseline = min(a for _, a in hist)
                area = hist[-1][1]
                growth_pct = (area / baseline - 1.0) * 100 if baseline > 0 else 0.0
                ok = growth_pct >= min_growth
                state_lines.append((f"growth {growth_pct:+.0f}%  need {min_growth:.0f}%",
                                     accent if ok else (160, 160, 160)))
            else:
                state_lines.append(("no hand in frame", (200, 80, 80)))
            if knock_times:
                remaining = max(0.0, knock_window_ms - (now_ts - knock_times[0]) * 1000)
                state_lines.append((f"window: {remaining:.0f}ms left  (max={knock_window_ms}ms)",
                                     (255, 200, 80)))
            refrac = live_params.get("refractory_ms", 300)
            state_lines.append((f"refractory={refrac}ms  baseline_window="
                                 f"{live_params.get('baseline_window_ms', 1200)}ms",
                                 (120, 120, 120)))
            state_lines.append(("progress:", (200, 200, 200)))

        elif gtype == "touch_head":
            since = context.get("touch_head_since")
            pct = hold_progress(context, "touch_head_since", hold_ms or 500)
            hands_req = live_params.get("hands_required", 1)
            state_lines.append((f"hands_required: {hands_req}", (200, 200, 200)))
            pose_src = "pose" if pose_lm else "fixed"
            state_lines.append((f"crown geometry: {pose_src}", (160, 200, 160) if pose_lm else (120, 120, 120)))
            if lm_list and pose_lm:
                from engines.detectors.rules.touch_head import _target_and_radius
                cx, cy, r = _target_and_radius(pose_lm, "crown")
                import math as _m
                in_region = sum(1 for hand in lm_list
                                if _m.hypot(hand.landmark[0].x - cx, hand.landmark[0].y - cy) < r)
                state_lines.append((f"wrists in region: {in_region}/{hands_req}",
                                     accent if in_region >= hands_req else (200, 80, 80)))
            state_lines.append((f"holding: {'YES' if since else 'no'}",
                                 accent if since else (120, 120, 120)))
            state_lines.append(("hold progress:", (200, 200, 200)))

        elif gtype == "arms_crossed":
            since = context.get("arms_crossed_since")
            pct = hold_progress(context, "arms_crossed_since", hold_ms or 500)
            if pose_lm:
                sh_l, sh_r   = pose_lm[11], pose_lm[12]
                hip_l, hip_r = pose_lm[23], pose_lm[24]
                lw, rw       = pose_lm[15], pose_lm[16]
                midline_x    = (sh_l.x + sh_r.x) / 2
                sh_width     = abs(sh_l.x - sh_r.x)
                dz           = live_params.get("dead_zone_frac", 0.05) * max(sh_width, 0.10)
                crossed_l    = lw.x < (midline_x - dz)
                crossed_r    = rw.x > (midline_x + dz)
                shoulder_y   = (sh_l.y + sh_r.y) / 2
                hip_y        = (hip_l.y + hip_r.y) / 2
                l_torso      = shoulder_y <= lw.y <= hip_y
                r_torso      = shoulder_y <= rw.y <= hip_y
                state_lines.append((f"L crossed: {'YES' if crossed_l else 'no'}  R crossed: {'YES' if crossed_r else 'no'}",
                                     accent if (crossed_l and crossed_r) else (200, 80, 80)))
                state_lines.append((f"L torso: {'YES' if l_torso else 'no'}  R torso: {'YES' if r_torso else 'no'}",
                                     accent if (l_torso and r_torso) else (200, 80, 80)))
                state_lines.append((f"midline_x={midline_x:.3f}  dead_zone={dz:.3f}", (160, 160, 160)))
            else:
                state_lines.append(("POSE NONE — cannot fire", (200, 80, 80)))
            state_lines.append(("hold progress:", (200, 200, 200)))

        elif gtype == "push_out":
            fired_flag = context.get("push_fired", False)
            histories = context.get("push_area_history", [[], []])
            min_growth = live_params.get("min_growth_pct", 50) / 100.0
            pct = 0.0
            for i, h_list in enumerate(histories[:2]):
                if len(h_list) >= 2:
                    oldest = h_list[0][1]
                    current = h_list[-1][1]
                    ratio = (current / oldest - 1.0) if oldest > 1e-6 else 0.0
                    ok = ratio >= min_growth
                    pct = max(pct, min(ratio / min_growth, 1.0))
                    state_lines.append((f"H{i}: growth={ratio:+.1%}  need={min_growth:.0%}  {'OK' if ok else '...'}",
                                         accent if ok else (200, 200, 200)))
                else:
                    state_lines.append((f"H{i}: {len(h_list)} samples (filling window)",
                                         (120, 120, 120)))
            state_lines.append((f"fired_flag: {'FIRED — reset (R) to retry' if fired_flag else 'waiting'}",
                                 accent if fired_flag else (120, 120, 120)))
            state_lines.append(("bilateral growth progress:", (200, 200, 200)))

        elif gtype == "run_arms":
            transitions = context.get("run_alt_transitions", [])
            min_cycles = live_params.get("min_cycles", 3)
            required = min_cycles * 2
            pct = min(len(transitions) / required, 1.0) if required > 0 else 0.0
            state_lines.append((f"transitions: {len(transitions)}/{required}",
                                 accent if len(transitions) >= required else (200, 200, 200)))
            alt_state = context.get("run_prev_alt_state")
            if pose_lm:
                _wo = live_params.get("waist_y_offset", 0.0)
                _hip_y = (pose_lm[23].y + pose_lm[24].y) / 2
                _sh_y = (pose_lm[11].y + pose_lm[12].y) / 2
                if live_params.get("reference", "waist") == "hips":
                    waist_y = _hip_y + _wo
                else:
                    waist_y = (_sh_y + _hip_y) / 2 + _wo
                lw, rw = pose_lm[15], pose_lm[16]
                above_l = lw.y < waist_y
                above_r = rw.y < waist_y
                state_lines.append((f"L {'above' if above_l else 'below'}  R {'above' if above_r else 'below'}  waist_y={waist_y:.3f} (offset={_wo:+.2f})",
                                     accent if (above_l != above_r) else (120, 120, 120)))
            else:
                state_lines.append(("POSE NONE — cannot fire", (200, 80, 80)))
            state_lines.append(("cycle progress:", (200, 200, 200)))

        elif gtype == "unravel":
            # Elbow-line model: the detector fires on per-wrist crossings in
            # unravel_crossings (the legacy unravel_diff_history key is dead —
            # it made this counter sit at 0 while the detector fired).
            xing = context.get("unravel_crossings", {}) or {}
            crossings = min(len(xing.get("L", [])), len(xing.get("R", [])))
            min_cycles = live_params.get("min_cycles", 2)
            pct = min(crossings / max(min_cycles * 2, 1), 1.0)
            state_lines.append((f"elbow-line crossings (min L/R): {crossings}/{min_cycles*2}",
                                 accent if crossings >= min_cycles * 2 else (200, 200, 200)))
            if pose_lm:
                sh_l, sh_r = pose_lm[11], pose_lm[12]
                # Mirror detector: prefer hand wrists, fall back to pose
                if len(lm_list) >= 2:
                    _sh = sorted(lm_list, key=lambda h: h.landmark[0].x)
                    lw = _sh[0].landmark[0]
                    rw = _sh[1].landmark[0]
                    wrist_src = "hand"
                else:
                    lw, rw = pose_lm[15], pose_lm[16]
                    wrist_src = "pose"
                import math as _m
                wrist_dist = _m.hypot(lw.x - rw.x, lw.y - rw.y)
                sh_width = abs(sh_l.x - sh_r.x)
                prox_ok = wrist_dist < live_params.get("prox_frac", 0.25) * max(sh_width, 0.10)
                state_lines.append((f"wrist_dist={wrist_dist:.3f}  prox: {'OK' if prox_ok else 'FAR'}  [{wrist_src}]",
                                     accent if prox_ok else (200, 80, 80)))
                if diffs:
                    amp = max(diffs) - min(diffs)
                    hip_y = (pose_lm[23].y + pose_lm[24].y) / 2
                    sh_hip = max(hip_y - (sh_l.y + sh_r.y) / 2, 0.10)
                    min_amp = live_params.get("min_amplitude_frac", 0.10) * sh_hip
                    state_lines.append((f"p-p amp={amp:.3f}  need={min_amp:.3f}  {'OK' if amp >= min_amp else '...'}",
                                         accent if amp >= min_amp else (200, 80, 80)))
            else:
                state_lines.append(("POSE NONE — cannot fire", (200, 80, 80)))
            state_lines.append(("winding progress:", (200, 200, 200)))

        elif gtype == "paddle":
            stroke_count = len(context.get("paddle_stroke_times", []))
            min_strokes  = live_params.get("min_strokes", 4)
            pct = min(stroke_count / min_strokes, 1.0) if min_strokes > 0 else 0.0
            state_lines.append((f"strokes: {stroke_count}/{min_strokes}",
                                 accent if stroke_count >= min_strokes else (200, 200, 200)))
            if pose_lm:
                l_crossed = context.get("paddle_l_crossed", False)
                r_crossed = context.get("paddle_r_crossed", False)
                l_above = context.get("paddle_l_above")
                r_above = context.get("paddle_r_above")
                _wo = live_params.get("waist_y_offset", 0.0)
                state_lines.append((f"L {'above' if l_above else 'below'}"
                                     f"{' CROSSED' if l_crossed else ''}   "
                                     f"R {'above' if r_above else 'below'}"
                                     f"{' CROSSED' if r_crossed else ''}   offset={_wo:+.2f}",
                                     accent if (l_crossed and r_crossed) else (160, 160, 200)))
                state_lines.append(("stroke = both wrists cross the midline "
                                     "(hysteresis band, ≥250ms apart)", (120, 120, 120)))
            else:
                state_lines.append(("POSE NONE — cannot fire", (200, 80, 80)))
            state_lines.append(("stroke progress:", (200, 200, 200)))

        elif gtype == "throw":
            armed = context.get("throw_armed", {})
            require_growth = live_params.get("require_growth", False)
            max_stroke_s = live_params.get("max_stroke_ms", 600) / 1000.0
            pct = 0.0
            if pose_lm:
                now_t = time.monotonic()
                for side, wrist_i in (("L", 15), ("R", 16)):
                    wr = pose_lm[wrist_i]
                    st = armed.get(side)
                    if st is None:
                        state_lines.append((f"{side}: raise hand above shoulder line to arm",
                                             (120, 120, 120)))
                        continue
                    elapsed = now_t - st["t"]
                    if not require_growth:
                        # Lenient mode: crossing below the line fires — armed = ready.
                        pct = max(pct, 1.0)
                        state_lines.append((f"{side}: ARMED — snap below the line to FIRE  "
                                             f"(window {elapsed:.2f}s/{max_stroke_s:.2f}s after leaving)",
                                             accent))
                        continue
                    # Strict mode readout: depth delta preferred, pose-z fallback.
                    fus = context.get("_pose_depth")
                    depth_now = fus.landmark_mm(wrist_i) if fus is not None and fus.available else None
                    if st.get("depth") is not None and depth_now is not None:
                        dd = st["depth"] - depth_now
                        need_mm = live_params.get("min_depth_delta_mm", 250)
                        pct = max(pct, min(max(dd, 0.0) / need_mm, 1.0))
                        state_lines.append((f"{side}: ARMED {elapsed:.2f}s/{max_stroke_s:.2f}s  "
                                             f"depth-delta={dd:+.0f}mm  need={need_mm:.0f}mm  [Gemini]",
                                             accent))
                    else:
                        zd = st["z"] - getattr(wr, "z", 0.0)
                        need_z = live_params.get("min_pose_z_delta", 0.2)
                        pct = max(pct, min(max(zd, 0.0) / need_z, 1.0))
                        state_lines.append((f"{side}: ARMED {elapsed:.2f}s/{max_stroke_s:.2f}s  "
                                             f"z-delta={zd:+.2f}  need={need_z:.2f}  [pose z]",
                                             accent))
            else:
                state_lines.append(("POSE NONE — cannot fire", (200, 80, 80)))
            state_lines.append(("throw progress:", (200, 200, 200)))

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

        # ── Pose-presence + profile indicator (top-right) ─────────────────
        pose_color = (0, 255, 80) if pose_lm is not None else (80, 80, 80)
        cv2.putText(frame, "pose: OK" if pose_lm is not None else "pose: --",
                    (w - 130, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2, cv2.LINE_AA)
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
        elif key == ord('w'):  # W — raise offset line (smaller y = higher on screen)
            _cur_g = GESTURES[gesture_idx]
            for _ok in ("waist_y_offset", "wrist_y_offset", "release_y_offset"):
                if _ok in _cur_g["params"]:
                    _cur_g["params"][_ok] = round(_cur_g["params"][_ok] - _WAIST_STEP, 3)
                    print(f"[tuner] {_cur_g['name']}: {_ok}={_cur_g['params'][_ok]:+.3f}")
            context = {}
            _save_tune_params(GESTURES, profile_path)
        elif key == ord('s'):  # S — lower offset line
            _cur_g = GESTURES[gesture_idx]
            for _ok in ("waist_y_offset", "wrist_y_offset", "release_y_offset"):
                if _ok in _cur_g["params"]:
                    _cur_g["params"][_ok] = round(_cur_g["params"][_ok] + _WAIST_STEP, 3)
                    print(f"[tuner] {_cur_g['name']}: {_ok}={_cur_g['params'][_ok]:+.3f}")
            context = {}
            _save_tune_params(GESTURES, profile_path)

    cap.release()
    pose.close()
    cv2.destroyAllWindows()


def _load_tune_params(gestures: list, profile_path: str) -> None:
    """Apply saved tuning values from the host profile's gesture_tuning section."""
    tuning: dict = {}
    if os.path.exists(profile_path):
        try:
            with open(profile_path) as f:
                profile = json.load(f)
            tuning = profile.get("gesture_tuning", {})
        except Exception as e:
            print(f"[tuner] Could not read {profile_path}: {e}", file=sys.stderr)
    else:
        print(f"[tuner] Profile not found: {profile_path} — using built-in defaults",
              file=sys.stderr)

    applied = 0
    for gesture in gestures:
        saved = tuning.get(gesture["name"], {})
        for key, val in saved.items():
            if key in gesture["params"]:
                gesture["params"][key] = val
                applied += 1

    print(f"[tuner] {os.path.basename(profile_path)}: "
          f"{applied} tuning override(s) loaded")

    # Explicitly confirm offset params so resets are visible on startup.
    _OFFSET_KEYS = ("waist_y_offset", "wrist_y_offset", "release_y_offset")
    for gesture in gestures:
        for key in _OFFSET_KEYS:
            if key in gesture["params"]:
                val = gesture["params"][key]
                tag = f"{val:+.3f}" if val != 0.0 else "0.000 (default)"
                print(f"[tuner]   {gesture['name']}.{key} = {tag}")


def _save_tune_params(gestures: list, profile_path: str) -> None:
    """Write current tunable params to the host profile's gesture_tuning section."""
    if not profile_path:
        return
    try:
        with open(profile_path) as f:
            profile = json.load(f)
    except Exception as e:
        print(f"[tuner] Could not read profile for save: {e}", file=sys.stderr)
        profile = {}
    tuning: dict = {}
    for gesture in gestures:
        saved = dict(gesture["params"])
        if saved:
            tuning[gesture["name"]] = saved
    profile["gesture_tuning"] = tuning
    try:
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"[tuner] Saved gesture params → {os.path.basename(profile_path)}")
    except Exception as e:
        print(f"[tuner] Could not save profile: {e}", file=sys.stderr)


def _adjust_tune_param(gesture, param_num: int, direction: int):
    """Adjust a tunable param. param_num: 1=primary (+/-), 2=secondary ([/]).

    The lower clamp defaults to one step (a param can't reach 0 accidentally),
    but a gesture can declare tune_min / tune_min2 to allow finer floors — e.g.
    survey's min_x_delta can be tuned below 0.01 for subtle pose-wrist scans.
    """
    if param_num == 1:
        key = gesture.get("tune_key")
        step = gesture.get("tune_step", 50)
        floor = gesture.get("tune_min", step)
    else:
        key = gesture.get("tune_key2")
        step = gesture.get("tune_step2", 0.01)
        floor = gesture.get("tune_min2", step)
    if not key:
        return
    current = gesture["params"].get(key, step * 8)
    new_val = current + direction * step
    new_val = max(floor, new_val)
    if isinstance(step, float):
        new_val = round(new_val, 4)
    gesture["params"][key] = new_val
    print(f"[tuner] {gesture['name']}: {key} → {gesture['params'][key]}")


if __name__ == "__main__":
    main()
