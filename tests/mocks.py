"""Shared landmark / depth / bus mocks for the BHR test suite."""


class LM:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class Hand:
    def __init__(self, landmarks):
        self.landmark = landmarks


def pose33(overrides=None):
    """A standing MediaPipe Pose: shoulders at y=0.40, hips 0.75, wrists down
    at the sides (y=0.85), eyes at 0.30. Override any index with a dict."""
    lms = [LM(0.5, 0.5) for _ in range(33)]
    base = {11: LM(0.6, 0.40), 12: LM(0.4, 0.40),    # shoulders
            23: LM(0.58, 0.75), 24: LM(0.42, 0.75),  # hips
            15: LM(0.65, 0.85, visibility=0.9),      # left wrist
            16: LM(0.35, 0.85, visibility=0.9),      # right wrist
            2: LM(0.52, 0.30), 5: LM(0.48, 0.30),    # eyes
            9: LM(0.51, 0.34), 10: LM(0.49, 0.34)}   # mouth
    base.update(overrides or {})
    for i, lm in base.items():
        lms[i] = lm
    return lms


def make_hand(cx, cy, size=0.12, shape="point"):
    """21-landmark Hands detection. shape: "point" (index out, others curled),
    "open" (all out), "fist" (all curled)."""
    lms = [LM(cx, cy) for _ in range(21)]
    lms[0] = LM(cx, cy + size)               # wrist below the palm
    fingers = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
               "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
    xoff = {"index": -0.02, "middle": 0.0, "ring": 0.02, "pinky": 0.04}
    for name, (mcp, pip, dip, tip) in fingers.items():
        x = cx + xoff[name]
        extended = shape == "open" or (shape == "point" and name == "index")
        lms[mcp] = LM(x, cy + size * 0.4)
        if extended:
            lms[pip] = LM(x, cy - size * 0.2)
            lms[dip] = LM(x, cy - size * 0.6)
            lms[tip] = LM(x, cy - size)
        else:  # curled: tip folds back toward the wrist, closer than pip
            lms[pip] = LM(x, cy + size * 0.1)
            lms[dip] = LM(x, cy + size * 0.35)
            lms[tip] = LM(x, cy + size * 0.55)
    lms[4] = LM(cx - 0.04, cy + size * 0.2)  # thumb tip, neutral
    return Hand(lms)


# -- depth samplers (context["_depth_mm_at"] stand-ins) ----------------------

def flat_sampler(mm):
    """Everything in frame at one depth."""
    return lambda x, y, patch=5: mm


def zone_sampler(zones, default=3000.0):
    """zones: list of (x0, x1, y0, y1, mm) — first hit wins, else default."""
    def sample(x, y, patch=5):
        for x0, x1, y0, y1, mm in zones:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return mm
        return default
    return sample


def seq_sampler(seq):
    """Returns successive values from seq per call; None when exhausted."""
    it = iter(seq)
    return lambda x, y, patch=5: next(it, None)


# -- event bus ---------------------------------------------------------------

class Bus:
    """Minimal EventBus stand-in that also logs every emit for assertions."""

    def __init__(self):
        self.subs = {}
        self.log = []

    def subscribe(self, name, fn):
        self.subs.setdefault(name, []).append(fn)

    def emit(self, name, data):
        self.log.append((name, data))
        for fn in self.subs.get(name, []):
            fn(data)
