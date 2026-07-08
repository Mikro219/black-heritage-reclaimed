"""
pose_hand_filter — Pose-guided validation of MediaPipe Hands output.

MediaPipe Hands and Pose run side by side but don't talk to each other: any
hand-like detection anywhere in the frame (a passer-by's hand, a face, a quilt
pattern) reaches every detector, and the Hands handedness classifier is
unreliable in mirrored installations. The Pose skeleton knows where the
player's wrists actually are — this module uses that to clean the Hands
output before it is published to detectors and the cursor layer:

  VETO      a Hands detection that isn't near any trusted pose wrist is
            dropped — but only when BOTH pose wrists are trusted (a confident,
            full skeleton). With a partial skeleton, unmatched hands pass
            through untouched: pose vetoes, it never rescues (the same
            philosophy as the depth fusion layer).
  LABEL     a matched hand takes its handedness from the pose side it matched
            (pose landmark 15 = player's LEFT wrist, 16 = RIGHT), replacing
            the Hands classifier's guess.
  ARBITRATE a matched hand whose wrist sits further from the pose wrist than
            arbitration_scale x the hand's own size is a stale/lagging track
            (motion blur mid-gesture) — worse than no hand. Dropped.
  RESCUE    (optional, config "pose_hand".rescue) when Pose sees a wrist that
            Hands missed entirely, a second Hands inference runs on a crop
            around the pose wrist — MediaPipe Holistic's trick à la carte,
            paid only on miss frames. The crop/remap geometry lives here;
            the inference call stays in the gesture engine.

Everything in this module is pure landmark math — no MediaPipe imports — so
the behaviour contract is pinned by tests/test_pose_hand_filter.py without
hardware.
"""

from types import SimpleNamespace

POSE_L_WRIST, POSE_R_WRIST = 15, 16
POSE_L_ELBOW, POSE_R_ELBOW = 13, 14

_SIDES = (("Left", POSE_L_WRIST), ("Right", POSE_R_WRIST))


def _dist(ax, ay, bx, by) -> float:
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _hand_size(hand) -> float:
    """Largest bounding-box dimension of the hand, normalised units."""
    xs = [lm.x for lm in hand.landmark]
    ys = [lm.y for lm in hand.landmark]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def trusted_wrists(pose_lm, min_visibility: float = 0.5) -> dict:
    """{"Left": landmark, "Right": landmark} for pose wrists above the
    visibility gate. Empty dict when pose is absent/malformed."""
    out = {}
    if pose_lm is None:
        return out
    for label, idx in _SIDES:
        try:
            lm = pose_lm[idx]
        except (IndexError, TypeError):
            continue
        if getattr(lm, "visibility", 1.0) >= min_visibility:
            out[label] = lm
    return out


def synth_handedness(label: str):
    """A stand-in for MediaPipe's classification result — same access shape
    (`.classification[0].label`) as the protobuf the render layer reads."""
    return SimpleNamespace(
        classification=[SimpleNamespace(label=label, score=1.0, index=0)])


def filter_hands(landmarks, handedness, pose_lm, *,
                 max_match_dist: float = 0.25,
                 arbitration_scale: float = 1.3,
                 min_visibility: float = 0.5):
    """Return (landmarks, handedness, stats) with pose-vetoed hands removed and
    handedness corrected from the pose skeleton.

    Plain-webcam / no-pose behaviour is unchanged: with no trusted pose wrists
    the input passes straight through. stats = {"dropped": n, "corrected": n}.
    """
    stats = {"dropped": 0, "corrected": 0}
    if not landmarks:
        return landmarks, handedness, stats
    wrists = trusted_wrists(pose_lm, min_visibility)
    if not wrists:
        return landmarks, handedness, stats

    hand_list = list(landmarks)
    hand_orig = list(handedness) if handedness else [None] * len(hand_list)
    while len(hand_orig) < len(hand_list):
        hand_orig.append(None)

    # Greedy nearest assignment: closest (hand, pose-wrist) pairs first, one
    # hand per wrist. Distances measured wrist-to-wrist (hand landmark 0).
    pairs = []
    for i, hand in enumerate(hand_list):
        hw = hand.landmark[0]
        for label, pw in wrists.items():
            d = _dist(hw.x, hw.y, pw.x, pw.y)
            if d <= max_match_dist:
                pairs.append((d, i, label))
    pairs.sort()
    match_of = {}          # hand index -> (label, dist)
    taken_sides = set()
    for d, i, label in pairs:
        if i in match_of or label in taken_sides:
            continue
        match_of[i] = (label, d)
        taken_sides.add(label)

    full_skeleton = len(wrists) == 2
    out_lms, out_hnd = [], []
    for i, hand in enumerate(hand_list):
        match = match_of.get(i)
        if match is None:
            if full_skeleton:
                stats["dropped"] += 1          # phantom: nowhere near either wrist
                continue
            out_lms.append(hand)               # partial skeleton: benefit of the doubt
            out_hnd.append(hand_orig[i])
            continue
        label, d = match
        if d > arbitration_scale * max(_hand_size(hand), 1e-6):
            # Matched, but lagging its own size behind the pose wrist — a stale
            # track mid-fast-gesture. Trust pose; drop the hand this frame.
            stats["dropped"] += 1
            continue
        orig = hand_orig[i]
        orig_label = None
        try:
            orig_label = orig.classification[0].label
        except (AttributeError, IndexError, TypeError):
            pass
        if orig_label != label:
            stats["corrected"] += 1
        out_lms.append(hand)
        out_hnd.append(synth_handedness(label))

    if not out_lms:
        return None, None, stats               # MediaPipe's "no hands" convention
    return out_lms, out_hnd, stats


# ---------------------------------------------------------------------------
# Rescue geometry (crop around a pose wrist, remap crop landmarks to frame)
# ---------------------------------------------------------------------------

def wrist_crop_box(pose_lm, side: str, frame_w: int, frame_h: int, *,
                   scale: float = 2.2, min_px: int = 96,
                   min_visibility: float = 0.5):
    """Pixel crop (x0, y0, x1, y1) centred on the pose wrist of `side`
    ("Left"/"Right"), sized from the forearm (elbow->wrist) length so it scales
    with player distance. None when the wrist/elbow aren't trustworthy."""
    wrist_i, elbow_i = ((POSE_L_WRIST, POSE_L_ELBOW) if side == "Left"
                        else (POSE_R_WRIST, POSE_R_ELBOW))
    try:
        wrist, elbow = pose_lm[wrist_i], pose_lm[elbow_i]
    except (IndexError, TypeError):
        return None
    if getattr(wrist, "visibility", 1.0) < min_visibility:
        return None
    forearm = _dist(wrist.x, wrist.y, elbow.x, elbow.y)
    size = max(int(forearm * scale * max(frame_w, frame_h)), min_px)
    cx, cy = int(wrist.x * frame_w), int(wrist.y * frame_h)
    half = size // 2
    x0 = max(0, min(frame_w - size, cx - half))
    y0 = max(0, min(frame_h - size, cy - half))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(frame_w, x0 + size)
    y1 = min(frame_h, y0 + size)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def remap_crop_landmarks(hand, box, frame_w: int, frame_h: int):
    """Hand landmarks detected inside `box` (crop-normalised) -> a hand-like
    object with landmarks in FULL-frame normalised coordinates."""
    x0, y0, x1, y1 = box
    cw, ch = (x1 - x0) / frame_w, (y1 - y0) / frame_h
    ox, oy = x0 / frame_w, y0 / frame_h
    lms = [SimpleNamespace(x=ox + lm.x * cw,
                           y=oy + lm.y * ch,
                           z=getattr(lm, "z", 0.0))
           for lm in hand.landmark]
    return SimpleNamespace(landmark=lms)
