"""
scripts/voice_tuner.py — Live voice detection tuning tool for BHR.

Shows a live OpenCV window with a rolling audio waveform, RMS level meter,
hum detection state, and Vosk keyword recognition. Cycles through every
voice interaction in the experience so each can be tested at multiple
microphone distances.

Usage:
  python scripts/voice_tuner.py [--profile NAME]

Controls:
  N / Right   next voice cue
  P / Left    previous voice cue
  Space       open a detection window (or close if already open)
  R           reset (close window, clear history)
  D           cycle test distance label (0.5 m → 1 m → 1.5 m → 2 m → 2.5 m → 3 m)
  + / Up      increase hum_rms_threshold
  - / Down    decrease hum_rms_threshold
  ] / [       increase / decrease hum_min_duration_ms
  Q / Esc     quit

Tuning values are saved to the host profile (default: laptop_dev) under
the "voice_tuning" key, matching the keys voice_engine.py reads from
config["voice"]. Swap values into config.json["voice"] for production.
"""

import argparse
import collections
import json
import math
import os
import queue
import struct
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import sounddevice as sd
except ImportError:
    print("[voice_tuner] sounddevice not installed. Run: pip install sounddevice",
          file=sys.stderr)
    sys.exit(1)

try:
    import vosk as _vosk_module
    _VOSK_AVAILABLE = True
except ImportError:
    _VOSK_AVAILABLE = False

VOSK_MODEL_PATH = os.path.join(ROOT, "models", "vosk-model-small-en-us-0.15")
SAMPLE_RATE   = 16000
BLOCKSIZE     = 4000   # 250 ms per audio chunk

_BHR_GRAMMAR = [
    "freedom", "go", "north", "follow the gourd",
    "shoofly", "yes", "water", "ready", "now", "[unk]",
]

DISTANCES = ["0.5 m", "1 m", "1.5 m", "2 m", "2.5 m", "3 m"]

# ── Voice cue roster ─────────────────────────────────────────────────────────
# accent: BGR colour used for the cue's highlights
VOICE_CUES = [
    dict(
        name="freedom",
        label="CG-alt · Scene 1 · AL-01-007",
        desc="Say 'freedom' — CG alternative to raise_hands",
        mode="keyword",
        keywords=["freedom"],
        tier="cg_alternative",
        window_ms=10000,
        accent=(0, 220, 255),
    ),
    dict(
        name="say_go",
        label="CG-req · Scene 2 · AL-02-008",
        desc="Say 'go' — required voice step after pointing the path",
        mode="keyword",
        keywords=["go"],
        tier="cg_required",
        window_ms=10000,
        accent=(0, 180, 255),
    ),
    dict(
        name="north_keyword",
        label="VI-react · Scene 3 · AL-03-008",
        desc="Say 'north' — reaction VI, no failure mode",
        mode="keyword",
        keywords=["north"],
        tier="reaction",
        window_ms=10000,
        accent=(100, 255, 200),
    ),
    dict(
        name="hum_or_gourd",
        label="CG · Scene 4 · AL-04-010  [MULTI-MODE]",
        desc="Hum ≥ hum_min_duration_ms  OR  say 'follow the gourd'",
        mode="hum",          # multi-mode: also accepts the keyword below
        keywords=["follow the gourd"],
        tier="cg_required",
        window_ms=15000,
        accent=(100, 255, 100),
    ),
    dict(
        name="shoofly_whisper",
        label="CG-alt · Scene 6 · AL-06-007  [WHISPER]",
        desc="Say (whisper) 'shoofly' — no volume gate; any Vosk match counts",
        mode="whisper",
        keywords=["shoofly"],
        tier="cg_alternative",
        window_ms=10000,
        accent=(255, 100, 220),
    ),
    dict(
        name="say_ready",
        label="VI-react · Scene 8 · AL-08-010",
        desc="Say 'ready' — line swap trigger (scene advances regardless)",
        mode="keyword",
        keywords=["ready"],
        tier="reaction",
        window_ms=10000,
        accent=(255, 200, 80),
    ),
    dict(
        name="ready_now",
        label="VI-react · Scene 10 · AL-10-009",
        desc="Say 'ready' or 'now' — reaction to forward_push",
        mode="keyword",
        keywords=["ready", "now"],
        tier="reaction",
        window_ms=10000,
        accent=(255, 160, 50),
    ),
    dict(
        name="hum_only",
        label="DSP test · Hum detection standalone",
        desc="Sustained hum only — calibrates hum_rms_threshold without keyword",
        mode="hum",
        keywords=[],
        tier="cg_required",
        window_ms=30000,
        accent=(100, 255, 100),
    ),
]

# ── Default tunable params ────────────────────────────────────────────────────
_DEFAULTS = {
    "hum_rms_threshold":   0.020,
    "hum_min_duration_ms": 500,
}

TUNE_STEP_RMS = 0.002   # +/- step for hum_rms_threshold
TUNE_STEP_DUR = 50      # +/- step for hum_min_duration_ms (ms)


# ── Profile I/O ──────────────────────────────────────────────────────────────

def _load_tune_params(params: dict, profile_path: str) -> None:
    if not os.path.exists(profile_path):
        return
    try:
        with open(profile_path) as f:
            profile = json.load(f)
    except Exception as e:
        print(f"[voice_tuner] Could not read {profile_path}: {e}", file=sys.stderr)
        return
    saved = profile.get("voice_tuning", {})
    for k in params:
        if k in saved:
            params[k] = saved[k]
    print(f"[voice_tuner] {os.path.basename(profile_path)}: "
          f"{len(saved)} voice_tuning override(s) loaded")


def _save_tune_params(params: dict, profile_path: str) -> None:
    if not profile_path:
        return
    try:
        with open(profile_path) as f:
            profile = json.load(f)
    except Exception as e:
        print(f"[voice_tuner] Could not read profile for save: {e}", file=sys.stderr)
        return
    profile["voice_tuning"] = dict(params)
    try:
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"[voice_tuner] Saved voice_tuning → {os.path.basename(profile_path)}")
    except Exception as e:
        print(f"[voice_tuner] Could not save profile: {e}", file=sys.stderr)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def put_lines(frame, lines, origin, font_scale=0.55, thickness=1,
              line_height=22, bg=True):
    ox, oy = origin
    if not lines:
        return
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


def horiz_bar(frame, origin, width, height, pct, color, bg_color=(50, 50, 50), label=""):
    x, y = origin
    cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)
    filled = int(pct * width)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)
    if label:
        cv2.putText(frame, label, (x + width + 8, y + height - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


def draw_waveform(frame, raw_samples: np.ndarray, rect, threshold_rms: float,
                  accent, label="waveform"):
    """Draw an oscilloscope-style waveform strip with the RMS threshold line."""
    rx, ry, rw, rh = rect
    # background
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (18, 18, 18), -1)
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (60, 60, 60), 1)

    mid_y = ry + rh // 2

    if len(raw_samples) == 0:
        cv2.putText(frame, "no audio", (rx + 8, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
        return

    # Downsample to rw display columns (take abs envelope max per bin)
    n = len(raw_samples)
    bin_size = max(1, n // rw)
    n_bins = rw
    envelope = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        start = i * bin_size
        end = min(start + bin_size, n)
        if start < n:
            envelope[i] = np.max(np.abs(raw_samples[start:end])) / 32768.0

    # Scale to half-height
    half_h = (rh // 2) - 4
    for i in range(n_bins):
        amp_px = int(envelope[i] * half_h)
        x = rx + i
        # Colour: green if below threshold-proportional, yellow if near, red if loud
        norm = envelope[i]
        if norm < threshold_rms * 0.8:
            col = (40, 140, 40)
        elif norm < threshold_rms * 2.0:
            col = accent
        else:
            col = (80, 80, 255)
        cv2.line(frame, (x, mid_y - amp_px), (x, mid_y + amp_px), col, 1)

    # Centre line
    cv2.line(frame, (rx, mid_y), (rx + rw, mid_y), (50, 50, 50), 1)

    # RMS threshold line: convert RMS to approximate peak amplitude
    # RMS of white noise ≈ peak/sqrt(2), but we just show it proportionally
    thresh_px = int(threshold_rms * half_h * 4)  # scale for visibility
    thresh_px = min(thresh_px, half_h - 2)
    ty_top = mid_y - thresh_px
    ty_bot = mid_y + thresh_px
    cv2.line(frame, (rx, ty_top), (rx + rw, ty_top), (0, 220, 220), 1)
    cv2.line(frame, (rx, ty_bot), (rx + rw, ty_bot), (0, 220, 220), 1)
    cv2.putText(frame, f"thr", (rx + 4, ty_top - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1, cv2.LINE_AA)

    cv2.putText(frame, label, (rx + 6, ry + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)


def draw_rms_meter(frame, rms: float, smooth_rms: float, threshold: float,
                   origin, width, height, accent):
    """Vertical RMS level + smoothed RMS + threshold marker."""
    x, y = origin
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (18, 18, 18), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), 1)

    # Smooth bar (faint)
    smooth_pct = min(smooth_rms / max(threshold * 5, 0.001), 1.0)
    sm_h = int(smooth_pct * height)
    if sm_h > 0:
        cv2.rectangle(frame, (x, y + height - sm_h), (x + width, y + height),
                      (50, 80, 50), -1)

    # Instant bar
    pct = min(rms / max(threshold * 5, 0.001), 1.0)
    bar_h = int(pct * height)
    if bar_h > 0:
        col = (40, 200, 40) if rms < threshold else accent
        cv2.rectangle(frame, (x, y + height - bar_h), (x + width, y + height),
                      col, -1)

    # Threshold line
    thresh_pct = min(threshold / max(threshold * 5, 0.001), 1.0)
    ty = y + height - int(thresh_pct * height)
    cv2.line(frame, (x, ty), (x + width, ty), (0, 220, 220), 2)

    # Labels
    cv2.putText(frame, "RMS", (x, y + height + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{rms:.4f}", (x, y + height + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, accent if rms >= threshold else (140, 140, 140),
                1, cv2.LINE_AA)


def draw_hum_meter(frame, hum_start, hum_min_duration_ms: float,
                   rms: float, threshold: float, origin, width, height, accent):
    """Hum sustain progress bar — fills as hum duration accumulates."""
    x, y = origin
    now = time.monotonic()

    # Section label
    cv2.putText(frame, "HUM SUSTAIN", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

    if hum_start is not None and rms >= threshold:
        elapsed_ms = (now - hum_start) * 1000
        pct = min(elapsed_ms / hum_min_duration_ms, 1.0)
        col = accent if pct < 1.0 else (80, 255, 80)
        horiz_bar(frame, (x, y), width, height, pct, col,
                  label=f"{elapsed_ms:.0f} / {hum_min_duration_ms:.0f} ms")
        if pct >= 1.0:
            cv2.putText(frame, "FIRED!", (x + 4, y + height - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 80), 1, cv2.LINE_AA)
    else:
        # Idle bar — show "above threshold?" indicator
        active = rms >= threshold
        horiz_bar(frame, (x, y), width, height, 0.0,
                  (80, 80, 80) if not active else (100, 200, 100),
                  label="0 ms")
        status = f"RMS {rms:.4f} {'≥' if active else '<'} thr {threshold:.4f}"
        col = (100, 200, 100) if active else (120, 120, 120)
        cv2.putText(frame, status, (x, y + height + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)


def draw_vosk_panel(frame, last_text: str, last_time: float,
                    keywords: list, matched: bool, window_open: bool,
                    origin, width, height, accent):
    """Keyword recognition panel — shows last Vosk output and match status."""
    x, y = origin
    now = time.monotonic()

    cv2.rectangle(frame, (x, y), (x + width, y + height), (18, 18, 18), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), 1)

    # Section title
    cv2.putText(frame, "KEYWORD", (x + 8, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

    # Expected keywords
    kw_str = "  /  ".join(f"'{k}'" for k in keywords) if keywords else "(none)"
    cv2.putText(frame, f"expect: {kw_str}", (x + 8, y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    # Window status
    win_col = (0, 200, 100) if window_open else (80, 80, 80)
    win_str = "WINDOW: OPEN" if window_open else "WINDOW: closed (Space to open)"
    cv2.putText(frame, win_str, (x + 8, y + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, win_col, 1, cv2.LINE_AA)

    if not _VOSK_AVAILABLE:
        cv2.putText(frame, "Vosk not installed", (x + 8, y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1, cv2.LINE_AA)
        return

    # Last heard text
    age_ms = (now - last_time) * 1000 if last_text else None
    if last_text and age_ms < 3000:
        alpha = max(0.3, 1.0 - age_ms / 3000)
        col = tuple(int(c * alpha) for c in (accent if matched else (200, 80, 80)))
        cv2.putText(frame, f"HEARD: {last_text.upper()}",
                    (x + 8, y + height // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, col, 2, cv2.LINE_AA)
        match_str = "MATCH" if matched else "no match"
        match_col = accent if matched else (200, 80, 80)
        cv2.putText(frame, match_str, (x + 8, y + height // 2 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, match_col, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "---", (x + 8, y + height // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (60, 60, 60), 2, cv2.LINE_AA)


def draw_fire_log(frame, fire_log: list, origin, width):
    """Recent detection log (last 6 entries)."""
    x, y = origin
    cv2.putText(frame, "DETECTIONS:", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
    for i, (ts, matched_str, dist, trigger) in enumerate(reversed(fire_log[-6:])):
        age = time.monotonic() - ts
        alpha = max(0.4, 1.0 - age / 30.0)
        col = tuple(int(200 * alpha) for _ in range(3))
        row = f"  [{trigger}] '{matched_str}'  @ {dist}  ({age:.1f}s ago)"
        cv2.putText(frame, row, (x, y + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BHR voice tuner")
    parser.add_argument(
        "--profile", default="laptop_dev",
        help="Host profile to read/write voice tuning from "
             "(options: laptop_dev, mini_pc_prod). Default: laptop_dev"
    )
    args = parser.parse_args()

    profile_name = args.profile or "laptop_dev"
    profile_path = os.path.join(ROOT, "config", "host_profiles", f"{profile_name}.json")

    tune = dict(_DEFAULTS)
    _load_tune_params(tune, profile_path)

    # ── Vosk setup ────────────────────────────────────────────────────────
    recognizer = None
    if _VOSK_AVAILABLE and os.path.exists(VOSK_MODEL_PATH):
        import vosk as _vosk
        model = _vosk.Model(VOSK_MODEL_PATH)
        recognizer = _vosk.KaldiRecognizer(model, SAMPLE_RATE,
                                           json.dumps(_BHR_GRAMMAR))
        print("[voice_tuner] Vosk ready.")
    elif not _VOSK_AVAILABLE:
        print("[voice_tuner] vosk not installed — keyword detection disabled. "
              "Run: pip install vosk")
    else:
        print(f"[voice_tuner] Vosk model not found at {VOSK_MODEL_PATH}")

    # ── Audio queue ───────────────────────────────────────────────────────
    audio_q: queue.Queue = queue.Queue()

    def _audio_callback(indata, frames, time_info, status):
        audio_q.put(bytes(indata))

    # ── State ─────────────────────────────────────────────────────────────
    cue_idx       = 0
    dist_idx      = 0
    window_open   = False
    window_start  = 0.0
    fire_count    = 0
    fired_at      = 0.0     # monotonic timestamp of last fire (for flash)
    fire_log: list = []

    # Audio state
    WAVE_SAMPLES = SAMPLE_RATE * 3   # 3 seconds of raw samples
    raw_buf = collections.deque(maxlen=WAVE_SAMPLES)
    rms         = 0.0
    smooth_rms  = 0.0
    hum_start   = None   # monotonic timestamp when current hum began

    # Vosk state
    last_vosk_text  = ""
    last_vosk_time  = 0.0
    last_matched    = False  # did last Vosk result match current cue?

    # ── Open sounddevice stream ───────────────────────────────────────────
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=BLOCKSIZE,
        callback=_audio_callback,
    )
    stream.start()
    print(f"[voice_tuner] Mic open. {len(VOICE_CUES)} cues loaded. "
          f"Q to quit, Space to open window, N/P to cycle cues.")

    # ── OpenCV window ─────────────────────────────────────────────────────
    WIN = "BHR Voice Tuner"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    while True:
        cue = VOICE_CUES[cue_idx]
        accent = cue["accent"]
        mode = cue["mode"]
        keywords = cue["keywords"]
        hum_threshold  = tune["hum_rms_threshold"]
        hum_min_dur_ms = tune["hum_min_duration_ms"]

        # ── Process all pending audio chunks ──────────────────────────────
        while not audio_q.empty():
            try:
                chunk = audio_q.get_nowait()
            except queue.Empty:
                break

            # Unpack to int16 samples
            n_samples = len(chunk) // 2
            samples = struct.unpack(f"{n_samples}h", chunk)
            raw_buf.extend(samples)

            # RMS
            rms_chunk = math.sqrt(sum(s * s for s in samples) / n_samples) / 32768.0
            rms = rms_chunk
            smooth_rms = smooth_rms * 0.85 + rms * 0.15

            # ── Hum detection ──────────────────────────────────────────
            if rms >= hum_threshold:
                if hum_start is None:
                    hum_start = time.monotonic()
                elif window_open and (time.monotonic() - hum_start) * 1000 >= hum_min_dur_ms:
                    # Hum fire — check cue accepts hum
                    if mode == "hum":
                        hum_start = None   # reset for next hum
                        fire_count += 1
                        fired_at = time.monotonic()
                        fire_log.append((fired_at, "hum", DISTANCES[dist_idx], "hum"))
                        window_open = False
                        print(f"[voice_tuner] HUM fired! cue={cue['name']}")
            else:
                hum_start = None

            # ── Vosk keyword spotting ──────────────────────────────────
            if recognizer and recognizer.AcceptWaveform(chunk):
                import json as _json
                result = _json.loads(recognizer.Result())
                text = result.get("text", "").strip().lower()
                if text and text != "[unk]":
                    last_vosk_text = text
                    last_vosk_time = time.monotonic()
                    # Match against current cue's keywords
                    cue_keywords_lower = [k.lower() for k in keywords]
                    kw_match = any(kw in text for kw in cue_keywords_lower)
                    last_matched = kw_match
                    if kw_match and window_open:
                        # Keyword fires in keyword, whisper, and multi-mode hum windows
                        fire_count += 1
                        fired_at = time.monotonic()
                        fire_log.append((fired_at, text, DISTANCES[dist_idx], mode))
                        window_open = False
                        print(f"[voice_tuner] KEYWORD fired! '{text}' "
                              f"cue={cue['name']}")

        # ── Auto-expire window ─────────────────────────────────────────────
        window_ms = cue.get("window_ms", 10000)
        if window_open and (time.monotonic() - window_start) * 1000 > window_ms:
            window_open = False
            print(f"[voice_tuner] Window expired: {cue['name']}")

        # ── Build frame ───────────────────────────────────────────────────
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        h, w = frame.shape[:2]

        # ── Top info panel ─────────────────────────────────────────────────
        tier_col = {"cg_required": (80, 80, 255), "cg_alternative": (0, 200, 255),
                    "reaction": (100, 255, 200)}.get(cue["tier"], (200, 200, 200))
        mode_col = {"keyword": (0, 220, 255), "hum": (100, 255, 100),
                    "whisper": (255, 100, 220)}.get(mode, (200, 200, 200))
        top_lines = [
            (f"CUE {cue_idx + 1}/{len(VOICE_CUES)}: {cue['name']}",
             (255, 255, 255)),
            (cue["label"], (160, 160, 255)),
            (cue["desc"][:80], (200, 200, 200)),
            (f"mode: {mode.upper()}   tier: {cue['tier']}   window: {window_ms}ms",
             mode_col),
        ]
        put_lines(frame, top_lines, (10, 26), font_scale=0.55, line_height=24)

        # ── Distance + profile indicator (top-right) ───────────────────────
        dist_str = DISTANCES[dist_idx]
        cv2.putText(frame, f"dist: {dist_str}  [D to cycle]",
                    (w - 280, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (180, 255, 180), 1, cv2.LINE_AA)
        profile_color = (80, 200, 255) if profile_name == "mini_pc_prod" else (160, 160, 255)
        cv2.putText(frame, f"profile: {profile_name}",
                    (w - 220, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, profile_color, 1, cv2.LINE_AA)
        tune_str = (f"+/-: hum_thr={hum_threshold:.4f}   "
                    f"[/]: hum_dur={hum_min_dur_ms:.0f}ms")
        cv2.putText(frame, tune_str, (w - 480, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)

        # ── Waveform (y 125–290) ───────────────────────────────────────────
        raw_arr = np.array(raw_buf, dtype=np.int16) if raw_buf else np.array([], dtype=np.int16)
        wave_rect = (0, 125, w, 165)
        draw_waveform(frame, raw_arr, wave_rect, hum_threshold, accent,
                      label=f"~3s audio buffer  |  {rms:.4f} RMS")

        # ── RMS meter (right of middle row, narrow vertical bar) ───────────
        rms_x = w - 60
        rms_y = 300
        rms_h = 200
        draw_rms_meter(frame, rms, smooth_rms, hum_threshold,
                       (rms_x, rms_y), 40, rms_h, accent)

        # ── Window countdown bar ───────────────────────────────────────────
        win_y = 300
        if window_open:
            elapsed_ms = (time.monotonic() - window_start) * 1000
            remaining_pct = max(0.0, 1.0 - elapsed_ms / window_ms)
            remaining_ms = max(0.0, window_ms - elapsed_ms)
            win_col = (0, 200, 100) if remaining_pct > 0.3 else (0, 120, 255)
            horiz_bar(frame, (10, win_y), 600, 16, remaining_pct, win_col,
                      label=f"window: {remaining_ms:.0f}ms remaining")
        else:
            horiz_bar(frame, (10, win_y), 600, 16, 0.0, (60, 60, 60),
                      label="window closed  (Space to open)")

        # ── Hum meter (left column, y 330–430) ────────────────────────────
        draw_hum_meter(frame, hum_start, hum_min_dur_ms,
                       rms, hum_threshold,
                       (10, 350), 580, 24, accent)

        # ── Vosk panel (centre, y 395–520) ────────────────────────────────
        draw_vosk_panel(frame, last_vosk_text, last_vosk_time,
                        keywords, last_matched, window_open,
                        (10, 395), 870, 155, accent)

        # ── Fire log (right section, y 500–640) ───────────────────────────
        if fire_log:
            draw_fire_log(frame, fire_log, (w - 560, 510), 540)

        # ── State panel (bottom-left, y 565–660) ──────────────────────────
        state_lines = [
            (f"fires: {fire_count}   last fire: "
             + (f"{(time.monotonic() - fired_at):.1f}s ago" if fired_at else "none"),
             accent if fire_count > 0 else (120, 120, 120)),
            (f"hum_rms_threshold = {hum_threshold:.4f}   "
             f"hum_min_duration_ms = {hum_min_dur_ms:.0f}",
             (160, 160, 160)),
            (f"vosk: {'ready' if recognizer else 'UNAVAILABLE'}   "
             f"grammar: {len(_BHR_GRAMMAR) - 1} keywords + [unk]",
             (100, 200, 100) if recognizer else (80, 80, 200)),
        ]
        put_lines(frame, state_lines, (10, 575), font_scale=0.47, line_height=22)

        # ── Controls (bottom-right) ────────────────────────────────────────
        ctrl_lines = [
            ("N/Right: next   P/Left: prev   Space: open window   R: reset", (140, 140, 140)),
            ("D: distance   +/-: hum_thr   [/]: hum_dur   Q: quit", (140, 140, 140)),
        ]
        put_lines(frame, ctrl_lines, (w - 530, h - 40), font_scale=0.42,
                  line_height=18, bg=False)

        # ── FIRED flash (full-screen green overlay) ────────────────────────
        if time.monotonic() - fired_at < 1.2:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 80), -1)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.putText(frame, f"FIRED! (#{fire_count})",
                        (w // 2 - 160, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 80), 5, cv2.LINE_AA)

        cv2.imshow(WIN, frame)

        # ── Keyboard input ────────────────────────────────────────────────
        key = cv2.waitKey(16) & 0xFF   # ~60fps display

        if key in (ord('q'), 27):
            break

        elif key in (ord('n'), 83):    # N or Right arrow
            cue_idx = (cue_idx + 1) % len(VOICE_CUES)
            window_open = False
            fire_count = 0
            last_vosk_text = ""
            last_matched = False
            hum_start = None

        elif key in (ord('p'), 81):    # P or Left arrow
            cue_idx = (cue_idx - 1) % len(VOICE_CUES)
            window_open = False
            fire_count = 0
            last_vosk_text = ""
            last_matched = False
            hum_start = None

        elif key == ord(' '):          # Space — toggle window
            if window_open:
                window_open = False
                hum_start = None
                print(f"[voice_tuner] Window closed manually: {cue['name']}")
            else:
                window_open = True
                window_start = time.monotonic()
                hum_start = None
                print(f"[voice_tuner] Window opened: {cue['name']}  "
                      f"({window_ms}ms)")

        elif key == ord('r'):          # R — reset
            window_open = False
            fire_count = 0
            fired_at = 0.0
            last_vosk_text = ""
            last_matched = False
            hum_start = None
            print(f"[voice_tuner] Reset: {cue['name']}")

        elif key == ord('d'):          # D — cycle distance label
            dist_idx = (dist_idx + 1) % len(DISTANCES)
            print(f"[voice_tuner] Distance: {DISTANCES[dist_idx]}")

        elif key in (ord('+'), ord('='), 82):   # + or Up — increase hum threshold
            tune["hum_rms_threshold"] = round(
                tune["hum_rms_threshold"] + TUNE_STEP_RMS, 4)
            _save_tune_params(tune, profile_path)
            print(f"[voice_tuner] hum_rms_threshold → {tune['hum_rms_threshold']:.4f}")

        elif key in (ord('-'), 84):    # - or Down — decrease hum threshold
            tune["hum_rms_threshold"] = round(
                max(TUNE_STEP_RMS, tune["hum_rms_threshold"] - TUNE_STEP_RMS), 4)
            _save_tune_params(tune, profile_path)
            print(f"[voice_tuner] hum_rms_threshold → {tune['hum_rms_threshold']:.4f}")

        elif key == ord(']'):          # ] — increase hum duration
            tune["hum_min_duration_ms"] = tune["hum_min_duration_ms"] + TUNE_STEP_DUR
            _save_tune_params(tune, profile_path)
            print(f"[voice_tuner] hum_min_duration_ms → {tune['hum_min_duration_ms']:.0f}ms")

        elif key == ord('['):          # [ — decrease hum duration
            tune["hum_min_duration_ms"] = max(
                TUNE_STEP_DUR, tune["hum_min_duration_ms"] - TUNE_STEP_DUR)
            _save_tune_params(tune, profile_path)
            print(f"[voice_tuner] hum_min_duration_ms → {tune['hum_min_duration_ms']:.0f}ms")

    stream.stop()
    stream.close()
    cv2.destroyAllWindows()
    print("[voice_tuner] Done.")


if __name__ == "__main__":
    main()
