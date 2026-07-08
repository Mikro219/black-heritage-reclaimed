"""
hand_pose — shared hand-shape classification helpers for the rule-based detectors
and the render engine's hand-icon cursors.

All functions take a MediaPipe Hands detection (object with .landmark[21]) and do
cheap 2D landmark math only. Finger extension uses the same tip-vs-PIP-distance
test as point_target_held: a finger is extended when its TIP is farther from the
wrist than its PIP joint.

The thumb is deliberately ignored everywhere — its extension state is unreliable
across hand rotations and doesn't change what any caller needs to know.

Classification contract (used by detectors that must tell an open palm from a
pointing finger, per the July 2026 playtest):
  is_pointing:  index extended, at most one of middle/ring/pinky extended.
  is_open_palm: at least three of index/middle/ring/pinky extended.
  is_fist:      none of index/middle/ring/pinky extended.
A hand can be none of the three (e.g. two fingers up) — callers treat that as
"indeterminate" and should not reject on it.
"""

import math

# (tip, pip) landmark index pairs for index / middle / ring / pinky
_FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]


def extended_fingers(hand) -> list:
    """[index, middle, ring, pinky] extension booleans (tip farther from wrist
    than PIP, direction-agnostic)."""
    lm = hand.landmark
    wrist = lm[0]
    out = []
    for tip_i, pip_i in _FINGERS:
        tip, pip = lm[tip_i], lm[pip_i]
        tip_sq = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
        pip_sq = (pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2
        out.append(tip_sq > pip_sq)
    return out


def is_pointing(hand) -> bool:
    """Index finger extended with the rest (mostly) curled."""
    ext = extended_fingers(hand)
    return ext[0] and sum(ext[1:]) <= 1


def is_open_palm(hand) -> bool:
    ext = extended_fingers(hand)
    return sum(ext) >= 3


def is_fist(hand) -> bool:
    ext = extended_fingers(hand)
    return sum(ext) == 0


def bbox_area(hand) -> float:
    """Normalised bounding-box area (w × h) of a Hands detection — the shared
    monocular z-proxy (a hand approaching the camera grows on screen)."""
    lm = hand.landmark
    xs = [l.x for l in lm]
    ys = [l.y for l in lm]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def nearest_hand(landmarks, x: float, y: float, max_dist: float = 0.25):
    """The Hands detection whose wrist is nearest raw-space (x, y), or None when
    nothing is close enough to plausibly be the same physical hand. Used to match
    a Pose wrist against the Hands model's view of that hand."""
    best = None
    best_dist = max_dist
    for hand in landmarks or []:
        hw = hand.landmark[0]
        d = math.hypot(hw.x - x, hw.y - y)
        if d <= best_dist:
            best_dist = d
            best = hand
    return best
