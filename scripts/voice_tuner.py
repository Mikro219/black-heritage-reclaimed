"""
scripts/voice_tuner.py — Live voice detection tuning tool for BHR.

Click-driven UI — all controls are on-screen buttons.

Usage:
  py -3.12 scripts/voice_tuner.py

Tuning values are saved to config.json's "host" section under the
"voice_tuning" key. Swap values into config.json["voice"] for production.
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
    __import__("vosk")
    _VOSK_AVAILABLE = True
except ImportError:
    _VOSK_AVAILABLE = False

VOSK_MODEL_PATH = os.path.join(ROOT, "models", "vosk-model-small-en-us-0.15")
SAMPLE_RATE = 16000
BLOCKSIZE   = 4000   # 250 ms per audio chunk

_BHR_GRAMMAR = [
    "freedom", "go", "north", "follow the gourd",
    "shoofly", "yes", "water", "ready", "now", "[unk]",
]

DISTANCES = ["0.5 m", "1 m", "1.5 m", "2 m", "2.5 m", "3 m"]

# ── Voice cue roster ──────────────────────────────────────────────────────────
VOICE_CUES = [
    dict(name="freedom",       label="CG-alt · Scene 1 · AL-01-007",
         desc="Say 'freedom' — CG alternative to raise_hands",
         mode="keyword",  keywords=["freedom"],       tier="cg_alternative", window_ms=10000,
         accent=(0, 220, 255)),
    dict(name="say_go",        label="CG-req · Scene 2 · AL-02-008",
         desc="Say 'go' — required voice step after pointing the path",
         mode="keyword",  keywords=["go"],             tier="cg_required",    window_ms=10000,
         accent=(0, 180, 255)),
    dict(name="north_keyword", label="VI-react · Scene 3 · AL-03-008",
         desc="Say 'north' — reaction VI, no failure mode",
         mode="keyword",  keywords=["north"],          tier="reaction",       window_ms=10000,
         accent=(100, 255, 200)),
    dict(name="hum_or_gourd",  label="CG · Scene 4 · AL-04-010  [MULTI-MODE]",
         desc="Hum >= hum_min_duration_ms  OR  say 'follow the gourd'",
         mode="hum",      keywords=["follow the gourd"], tier="cg_required",  window_ms=15000,
         accent=(100, 255, 100)),
    dict(name="shoofly_whisper", label="CG-alt · Scene 6 · AL-06-007  [WHISPER]",
         desc="Whisper 'shoofly' — rejected if peak dBFS exceeds whisper_max",
         mode="whisper",  keywords=["shoofly"],        tier="cg_alternative", window_ms=10000,
         accent=(255, 100, 220)),
    dict(name="say_ready",     label="VI-react · Scene 8 · AL-08-010",
         desc="Say 'ready' — line swap trigger (scene advances regardless)",
         mode="keyword",  keywords=["ready"],          tier="reaction",       window_ms=10000,
         accent=(255, 200, 80)),
    dict(name="ready_now",     label="VI-react · Scene 10 · AL-10-009",
         desc="Say 'ready' or 'now' — reaction to forward_push",
         mode="keyword",  keywords=["ready", "now"],   tier="reaction",       window_ms=10000,
         accent=(255, 160, 50)),
    dict(name="hum_only",      label="DSP test · Hum detection standalone",
         desc="Sustained hum only — calibrates hum_rms_threshold without keyword",
         mode="hum",      keywords=[],                 tier="cg_required",    window_ms=30000,
         accent=(100, 255, 100)),
]

# ── Default tunable params ────────────────────────────────────────────────────
_DEFAULTS = {
    "hum_rms_threshold":      0.020,
    "hum_min_duration_ms":    500,
    "whisper_max_volume_dbfs": -25.0,
}

TUNE_STEP_RMS     = 0.002
TUNE_STEP_DUR     = 50
TUNE_STEP_WHISPER = 1.0


# ── Profile I/O ───────────────────────────────────────────────────────────────

def _load_tune_params(params: dict, profile_path: str) -> None:
    if not os.path.exists(profile_path):
        return
    try:
        with open(profile_path, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"[voice_tuner] Could not read {profile_path}: {e}", file=sys.stderr)
        return
    saved = config.get("host", {}).get("voice_tuning", {})
    for k in params:
        if k in saved:
            params[k] = saved[k]
    print(f"[voice_tuner] {os.path.basename(profile_path)}: "
          f"{len(saved)} voice_tuning override(s) loaded")


def _save_tune_params(params: dict, profile_path: str) -> None:
    if not profile_path:
        return
    try:
        with open(profile_path, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"[voice_tuner] Could not read config.json for save — NOT saving: {e}",
              file=sys.stderr)
        return
    config.setdefault("host", {})["voice_tuning"] = dict(params)
    try:
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[voice_tuner] Could not save config.json: {e}", file=sys.stderr)


# ── Layout + colour constants ─────────────────────────────────────────────────
W, H   = 1280, 720
PAD    = 14

# Colours (BGR)
C_BG   = (22, 22, 24)
C_SEP  = (52, 52, 56)
C_HI   = (240, 240, 240)
C_MID  = (170, 170, 175)
C_DIM  = (100, 100, 106)
C_BTN  = (52, 52, 58)
C_BTN_HOVER = (80, 80, 88)
C_BTN_FLASH = (60, 180, 60)
C_BTN_DANGER = (50, 35, 70)
C_BTN_ACTIVE = (35, 70, 35)
C_RED  = (70,  70, 200)
C_YEL  = (60, 200, 200)
C_GRN  = (60, 180, 60)


# ── Button helpers ────────────────────────────────────────────────────────────

def _btn(x, y, w, h, label, key, color=None):
    return {"rect": (x, y, w, h), "label": label, "key": key,
            "color": color or C_BTN}


def _draw_btn(frame, btn, mx, my, flash=False):
    x, y, w, h = btn["rect"]
    hover = x <= mx <= x + w and y <= my <= y + h
    bg = C_BTN_FLASH if flash else (C_BTN_HOVER if hover else btn["color"])
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    border = (180, 180, 190) if hover else (70, 70, 76)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)
    fs, th = 0.46, 1
    tw, tbase = cv2.getTextSize(btn["label"], cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0]
    tx = x + (w - tw) // 2
    ty = y + (h + tbase) // 2
    cv2.putText(frame, btn["label"], (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, fs, C_HI, th, cv2.LINE_AA)


def _hit(btn, px, py):
    x, y, w, h = btn["rect"]
    return x <= px <= x + w and y <= py <= y + h


# ── Drawing primitives ────────────────────────────────────────────────────────

def _sep(frame, y, x0=0, x1=W, color=C_SEP):
    cv2.line(frame, (x0, y), (x1, y), color, 1)


def _vsep(frame, x, y0=0, y1=H, color=C_SEP):
    cv2.line(frame, (x, y0), (x, y1), color, 1)


def _label(frame, text, x, y, color=C_DIM, scale=0.40):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _text(frame, text, x, y, color=C_HI, scale=0.55, bold=False):
    th = 2 if bold else 1
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def _hbar(frame, x, y, bw, bh, pct, fill, label=""):
    pct = max(0.0, min(pct, 1.0))
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (32, 32, 36), -1)
    if pct > 0:
        cv2.rectangle(frame, (x, y), (x + int(pct * bw), y + bh), fill, -1)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), C_SEP, 1)
    if label:
        _label(frame, label, x + bw + 6, y + bh - 2)


# ── Section drawing ───────────────────────────────────────────────────────────

def _draw_header(frame, cue, cue_idx, accent):
    """Top section: cue identity and metadata."""
    # Background tint
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 112), (28, 28, 32), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    tier_col = {"cg_required": C_RED, "cg_alternative": C_YEL,
                "reaction": C_GRN}.get(cue["tier"], C_MID)
    mode_col = {"keyword": (0, 200, 240), "hum": (80, 220, 80),
                "whisper": (220, 100, 200)}.get(cue["mode"], C_MID)

    # Cue counter + name
    counter = f"CUE {cue_idx + 1} / {len(VOICE_CUES)}"
    _label(frame, counter, PAD, 22, C_DIM, 0.42)
    _text(frame, cue["name"], PAD + 100, 22, accent, 0.68, bold=True)

    # Scene label
    _text(frame, cue["label"], PAD, 48, (160, 160, 220), 0.48)

    # Description
    _label(frame, cue["desc"][:90], PAD, 70, C_MID, 0.43)

    # Mode / tier / window
    cv2.putText(frame, "mode:", (PAD, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, cue["mode"].upper(), (PAD + 42, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, mode_col, 1, cv2.LINE_AA)
    cv2.putText(frame, "tier:", (PAD + 130, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, cue["tier"], (PAD + 170, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, tier_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"window: {cue['window_ms']}ms", (PAD + 370, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)


def _draw_waveform(frame, raw_buf, rms, hum_threshold, y0, y1, accent):
    """Oscilloscope strip across full width."""
    rw, rh = W, y1 - y0
    mid_y  = y0 + rh // 2
    half_h = rh // 2 - 4

    raw_arr = np.array(raw_buf, dtype=np.int16) if raw_buf else np.zeros(1, dtype=np.int16)
    n = len(raw_arr)
    bin_size = max(1, n // rw)
    for i in range(rw):
        s = i * bin_size
        e = min(s + bin_size, n)
        if s >= n:
            break
        amp = float(np.max(np.abs(raw_arr[s:e]))) / 32768.0
        amp_px = int(amp * half_h)
        norm = amp
        if norm < hum_threshold * 0.8:
            col = (40, 100, 40)
        elif norm < hum_threshold * 2.0:
            col = accent
        else:
            col = (100, 100, 240)
        cv2.line(frame, (i, mid_y - amp_px), (i, mid_y + amp_px), col, 1)

    # Centre axis
    cv2.line(frame, (0, mid_y), (W, mid_y), (40, 40, 44), 1)

    # Threshold band lines
    thresh_px = min(int(hum_threshold * half_h * 4), half_h - 2)
    cv2.line(frame, (0, mid_y - thresh_px), (W, mid_y - thresh_px), (0, 160, 160), 1)
    cv2.line(frame, (0, mid_y + thresh_px), (W, mid_y + thresh_px), (0, 160, 160), 1)
    _label(frame, f"hum thr  RMS {rms:.4f}", PAD, y0 + 14, (0, 160, 160), 0.38)


def _draw_rms_section(frame, rms, hum_threshold,
                      hum_start, hum_min_dur_ms, x0, y0, bw, accent):
    """Left meter column: RMS bar + hum sustain bar."""
    # ── RMS bar ──────────────────────────────────────────────────────────────
    _label(frame, "RMS LEVEL", x0, y0 + 14, C_DIM)
    rms_pct = min(rms / max(hum_threshold * 5, 0.001), 1.0)
    rms_col = accent if rms >= hum_threshold else C_GRN
    _hbar(frame, x0, y0 + 18, bw, 16, rms_pct, rms_col)

    # Threshold marker on RMS bar
    thresh_px = int(min(1.0 / 5.0, 1.0) * bw)  # threshold is at 1/5 of full scale
    tx = x0 + thresh_px
    cv2.line(frame, (tx, y0 + 14), (tx, y0 + 38), (0, 200, 200), 2)
    _label(frame, f"{rms:.4f}", x0 + bw + 8, y0 + 31, rms_col if rms >= hum_threshold else C_DIM)

    # ── Hum sustain bar ───────────────────────────────────────────────────────
    _label(frame, "HUM SUSTAIN", x0, y0 + 60, C_DIM)
    now = time.monotonic()
    if hum_start is not None and rms >= hum_threshold:
        elapsed_ms = (now - hum_start) * 1000
        pct = min(elapsed_ms / hum_min_dur_ms, 1.0)
        col = (80, 255, 80) if pct >= 1.0 else accent
        _hbar(frame, x0, y0 + 64, bw, 16, pct, col,
              label=f"{elapsed_ms:.0f} / {hum_min_dur_ms:.0f} ms")
        if pct >= 1.0:
            _text(frame, "HUM DETECTED", x0, y0 + 100, (80, 255, 80), 0.55, bold=True)
    else:
        active = rms >= hum_threshold
        col = (100, 200, 100) if active else (50, 50, 54)
        _hbar(frame, x0, y0 + 64, bw, 16, 0.0, col)
        status = f"RMS {'≥' if active else '<'} thr ({hum_threshold:.4f})"
        _label(frame, status, x0, y0 + 96, (100, 200, 100) if active else C_DIM)


def _draw_dbfs_section(frame, peak_dbfs, whisper_max, x0, y0, bw, mode):
    """Right meter column: peak dBFS bar with whisper ceiling."""
    _label(frame, "PEAK dBFS", x0, y0 + 14, C_DIM)
    DB_LO, DB_HI = -60.0, 0.0
    pct = max(0.0, min((peak_dbfs - DB_LO) / (DB_HI - DB_LO), 1.0))
    too_loud = peak_dbfs > whisper_max
    bar_col = C_RED if too_loud else C_GRN
    _hbar(frame, x0, y0 + 18, bw, 16, pct, bar_col)

    # Ceiling line
    ceil_pct = max(0.0, min((whisper_max - DB_LO) / (DB_HI - DB_LO), 1.0))
    cx = x0 + int(ceil_pct * bw)
    cv2.line(frame, (cx, y0 + 14), (cx, y0 + 38), (60, 200, 220), 2)
    _label(frame, f"max={whisper_max:.0f}", cx + 4, y0 + 12, (60, 200, 220), 0.36)

    # Value + status
    val_col = C_RED if too_loud else C_DIM
    status  = "TOO LOUD" if (too_loud and mode == "whisper") else f"{peak_dbfs:.1f} dBFS"
    _label(frame, status, x0 + bw + 8, y0 + 31, val_col)

    if mode != "whisper":
        _label(frame, "(ceiling active on whisper cues only)", x0, y0 + 54, C_DIM, 0.36)
    else:
        cue_status = "WOULD REJECT" if too_loud else "OK — within ceiling"
        _label(frame, cue_status, x0, y0 + 54, C_RED if too_loud else C_GRN)


def _draw_keyword_panel(frame, last_vosk_text, last_vosk_time,
                        keywords, last_matched, window_open, window_start, window_ms,
                        accent, x0, y0, pw, ph):
    """Vosk keyword recognition panel."""
    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (26, 26, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), C_SEP, 1)

    iy = y0 + 22
    _label(frame, "KEYWORD DETECTION", x0 + PAD, iy, C_DIM)
    iy += 24

    # Expected keywords
    kw_str = "  /  ".join(f"'{k}'" for k in keywords) if keywords else "(none)"
    _label(frame, f"expect: {kw_str}", x0 + PAD, iy, C_MID, 0.43)
    iy += 22

    # Window countdown
    now = time.monotonic()
    bar_w = pw - PAD * 2
    if window_open:
        elapsed   = (now - window_start) * 1000
        remaining = max(0.0, window_ms - elapsed)
        rem_pct   = remaining / window_ms
        win_col   = (60, 200, 80) if rem_pct > 0.3 else (60, 120, 220)
        _label(frame, f"window: {remaining:.0f} ms remaining", x0 + PAD, iy, win_col, 0.38)
        _hbar(frame, x0 + PAD, iy + 4, bar_w, 12, rem_pct, win_col)
    else:
        _label(frame, "window closed", x0 + PAD, iy, C_DIM, 0.38)
        _hbar(frame, x0 + PAD, iy + 4, bar_w, 12, 0.0, C_SEP)
    iy += 26

    if not _VOSK_AVAILABLE:
        _text(frame, "Vosk not installed", x0 + PAD, iy + 20, C_RED, 0.50)
        return

    # Last heard
    age_ms = (now - last_vosk_time) * 1000 if last_vosk_text else None
    if last_vosk_text and age_ms < 3500:
        alpha = max(0.35, 1.0 - age_ms / 3500)
        col   = accent if last_matched else C_RED
        col   = tuple(int(c * alpha) for c in col)
        _text(frame, f"HEARD:  {last_vosk_text.upper()}", x0 + PAD, iy + 26,
              col, 0.72, bold=True)
        match_str = "MATCH" if last_matched else "no match"
        _label(frame, match_str, x0 + PAD, iy + 48, accent if last_matched else C_RED, 0.42)
    else:
        _label(frame, "—  listening  —", x0 + PAD, iy + 26, C_DIM, 0.48)


def _draw_fire_log(frame, fire_log, x0, y0):
    """Last few detections."""
    if not fire_log:
        return
    _label(frame, "RECENT DETECTIONS", x0, y0, C_DIM)
    now = time.monotonic()
    for i, (ts, matched_str, dist, trigger) in enumerate(reversed(fire_log[-4:])):
        age  = now - ts
        frac = max(0.35, 1.0 - age / 20.0)
        col  = tuple(int(180 * frac) for _ in range(3))
        row  = f"[{trigger}]  '{matched_str}'  @ {dist}   {age:.0f}s ago"
        _label(frame, row, x0, y0 + 16 + i * 18, col, 0.40)


def _draw_param_buttons(frame, btns, tune, mx, my, flash_key, flash_until):
    """Draw the three tuning groups with their value labels between the ±buttons."""
    now = time.monotonic()

    groups = [
        ("hum_thr_dec",  "hum_thr_inc",  "HUM THRESHOLD",
         f"{tune['hum_rms_threshold']:.4f}"),
        ("hum_dur_dec",  "hum_dur_inc",  "HUM DURATION",
         f"{tune['hum_min_duration_ms']:.0f} ms"),
        ("whisper_dec",  "whisper_inc",  "WHISPER MAX",
         f"{tune['whisper_max_volume_dbfs']:.0f} dBFS"),
    ]

    for dec_key, inc_key, group_label, val_str in groups:
        dec_btn = btns[dec_key]
        inc_btn = btns[inc_key]

        dx, dy, dw, dh = dec_btn["rect"]
        _label(frame, group_label, dx, dy - 5, C_DIM, 0.36)

        for btn in (dec_btn, inc_btn):
            flash = btn["key"] == flash_key and now < flash_until
            _draw_btn(frame, btn, mx, my, flash)

        # Value label between the two buttons
        ix = inc_btn["rect"][0]
        mid_x = dx + dw + 4
        mid_w = ix - mid_x
        val_tw = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        vx = mid_x + (mid_w - val_tw) // 2
        vy = dy + dh // 2 + 6
        _text(frame, val_str, vx, vy, C_HI, 0.48)


def _draw_action_buttons(frame, btns_list, mx, my, flash_key, flash_until):
    now = time.monotonic()
    for btn in btns_list:
        flash = btn["key"] == flash_key and now < flash_until
        _draw_btn(frame, btn, mx, my, flash)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BHR voice tuner")
    parser.parse_args()

    profile_path = os.path.join(ROOT, "config.json")   # tuning lives in ["host"]

    tune = dict(_DEFAULTS)
    _load_tune_params(tune, profile_path)

    # ── Vosk ──────────────────────────────────────────────────────────────────
    recognizer = None
    if _VOSK_AVAILABLE and os.path.exists(VOSK_MODEL_PATH):
        import vosk as _vosk
        model      = _vosk.Model(VOSK_MODEL_PATH)
        recognizer = _vosk.KaldiRecognizer(model, SAMPLE_RATE, json.dumps(_BHR_GRAMMAR))
        print("[voice_tuner] Vosk ready.")
    elif not _VOSK_AVAILABLE:
        print("[voice_tuner] vosk not installed — keyword detection disabled.")
    else:
        print(f"[voice_tuner] Vosk model not found at {VOSK_MODEL_PATH}")

    # ── Audio queue ───────────────────────────────────────────────────────────
    audio_q: queue.Queue = queue.Queue()

    def _audio_callback(indata, *_):
        audio_q.put(bytes(indata))

    # ── Runtime state ─────────────────────────────────────────────────────────
    cue_idx      = 0
    dist_idx     = 0
    window_open  = False
    window_start = 0.0
    fire_count   = 0
    fired_at     = 0.0
    fire_log: list = []

    WAVE_SAMPLES = SAMPLE_RATE * 3
    raw_buf      = collections.deque(maxlen=WAVE_SAMPLES)
    rms          = 0.0
    smooth_rms   = 0.0
    peak_dbfs    = -120.0
    hum_start    = None

    last_vosk_text = ""
    last_vosk_time = 0.0
    last_matched   = False

    # ── Mic stream ────────────────────────────────────────────────────────────
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=BLOCKSIZE, callback=_audio_callback,
    )
    stream.start()
    print(f"[voice_tuner] Mic open. {len(VOICE_CUES)} cues. Click buttons to control.")

    # ── Mouse state ───────────────────────────────────────────────────────────
    _mouse = {"x": 0, "y": 0, "clicked": False, "cx": 0, "cy": 0}

    def _on_mouse(event, x, y, *_):
        _mouse["x"] = x
        _mouse["y"] = y
        if event == cv2.EVENT_LBUTTONDOWN:
            _mouse["clicked"] = True
            _mouse["cx"] = x
            _mouse["cy"] = y

    WIN = "BHR Voice Tuner"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)
    cv2.setMouseCallback(WIN, _on_mouse)

    flash_key   = None
    flash_until = 0.0

    # ── Layout constants ───────────────────────────────────────────────────────
    # Vertical bands (y positions)
    Y_HEADER_BOT  = 112
    Y_WAVE_TOP    = Y_HEADER_BOT + 2
    Y_WAVE_BOT    = Y_WAVE_TOP + 108
    Y_METERS_TOP  = Y_WAVE_BOT + 6
    Y_METERS_BOT  = Y_METERS_TOP + 108
    Y_KW_TOP      = Y_METERS_BOT + 6
    Y_KW_BOT      = Y_KW_TOP + 142
    Y_LOG_TOP     = Y_KW_BOT + 6
    Y_LOG_BOT     = Y_LOG_TOP + 76
    Y_PARAM_TOP   = Y_LOG_BOT + 8
    Y_PARAM_BOT   = Y_PARAM_TOP + 62
    Y_ACTION_TOP  = Y_PARAM_BOT + 8
    Y_ACTION_BOT  = Y_ACTION_TOP + 54
    # Total: ~Y_ACTION_BOT ≈ 650, leaving ~70px for status strip at bottom

    # Metre column split
    METER_SPLIT = W // 2 - 10
    METER_L_W   = METER_SPLIT - PAD * 2
    METER_R_X   = METER_SPLIT + 20
    METER_R_W   = W - METER_R_X - PAD

    # ── Button definitions (rebuilt each frame for dynamic labels) ─────────────
    # We define them in the loop below, but static tuning buttons can be pre-built.

    # Param buttons: [−]  value  [+]  for each of 3 params
    # Spread evenly across width.
    PB_H  = 40
    PB_W  = 36   # ± button width
    PB_MW = 110  # middle value display width

    group_w = (W - PAD * 2) // 3
    g_starts = [PAD + i * group_w for i in range(3)]

    def _param_btns(gi, dec_key, inc_key):
        x0 = g_starts[gi]
        py = Y_PARAM_TOP + 18
        return (
            _btn(x0, py, PB_W, PB_H, "-", dec_key),
            _btn(x0 + PB_W + PB_MW + 8, py, PB_W, PB_H, "+", inc_key),
        )

    btn_hum_thr_dec, btn_hum_thr_inc = _param_btns(0, "hum_thr_dec", "hum_thr_inc")
    btn_hum_dur_dec, btn_hum_dur_inc = _param_btns(1, "hum_dur_dec", "hum_dur_inc")
    btn_wsp_dec,     btn_wsp_inc     = _param_btns(2, "whisper_dec", "whisper_inc")

    PARAM_BTNS_DICT = {
        "hum_thr_dec": btn_hum_thr_dec, "hum_thr_inc": btn_hum_thr_inc,
        "hum_dur_dec": btn_hum_dur_dec, "hum_dur_inc": btn_hum_dur_inc,
        "whisper_dec": btn_wsp_dec,     "whisper_inc": btn_wsp_inc,
    }

    running = True
    while running:
        cue            = VOICE_CUES[cue_idx]
        accent         = cue["accent"]
        mode           = cue["mode"]
        keywords       = cue["keywords"]
        hum_threshold  = tune["hum_rms_threshold"]
        hum_min_dur_ms = tune["hum_min_duration_ms"]
        whisper_max    = tune["whisper_max_volume_dbfs"]
        window_ms      = cue.get("window_ms", 10000)

        # ── Process audio chunks ───────────────────────────────────────────────
        while not audio_q.empty():
            try:
                chunk = audio_q.get_nowait()
            except queue.Empty:
                break

            n_samples = len(chunk) // 2
            if n_samples == 0:
                continue
            samples = struct.unpack(f"{n_samples}h", chunk)
            raw_buf.extend(samples)

            rms_val    = math.sqrt(sum(s * s for s in samples) / n_samples) / 32768.0
            rms        = rms_val
            smooth_rms = smooth_rms * 0.85 + rms * 0.15
            peak_lin   = max(abs(s) for s in samples) / 32768.0
            peak_dbfs  = 20.0 * math.log10(peak_lin) if peak_lin > 1e-6 else -120.0

            # Hum detection
            if rms >= hum_threshold:
                if hum_start is None:
                    hum_start = time.monotonic()
                elif window_open and (time.monotonic() - hum_start) * 1000 >= hum_min_dur_ms:
                    if mode == "hum":
                        hum_start = None
                        fire_count += 1
                        fired_at = time.monotonic()
                        fire_log.append((fired_at, "hum", DISTANCES[dist_idx], "hum"))
                        window_open = False
                        print(f"[voice_tuner] HUM fired! cue={cue['name']}")
            else:
                hum_start = None

            # Vosk keyword spotting
            if recognizer and recognizer.AcceptWaveform(chunk):
                import json as _j
                result = _j.loads(recognizer.Result())
                text   = result.get("text", "").strip().lower()
                if text and text != "[unk]":
                    last_vosk_text = text
                    last_vosk_time = time.monotonic()
                    kw_lower  = [k.lower() for k in keywords]
                    kw_match  = any(kw in text for kw in kw_lower)
                    whisper_rejected = (mode == "whisper" and peak_dbfs > whisper_max)
                    last_matched = kw_match and not whisper_rejected
                    if kw_match and window_open and not whisper_rejected:
                        fire_count += 1
                        fired_at = time.monotonic()
                        fire_log.append((fired_at, text, DISTANCES[dist_idx], mode))
                        window_open = False
                        print(f"[voice_tuner] KEYWORD fired! '{text}' cue={cue['name']}")

        # Auto-expire window
        if window_open and (time.monotonic() - window_start) * 1000 > window_ms:
            window_open = False

        # ── Build dynamic buttons ──────────────────────────────────────────────
        AB_H  = 48
        AB_Y  = Y_ACTION_TOP + 3
        btn_prev   = _btn(PAD, AB_Y, 100, AB_H, "<  PREV", "prev")
        btn_next   = _btn(PAD + 108, AB_Y, 100, AB_H, "NEXT  >", "next")
        btn_dist   = _btn(PAD + 235, AB_Y, 148, AB_H,
                          f"DIST: {DISTANCES[dist_idx]}", "dist", color=(40, 55, 55))
        win_label  = "CLOSE WINDOW" if window_open else "OPEN WINDOW"
        win_color  = C_BTN_ACTIVE if window_open else C_BTN
        btn_window = _btn(PAD + 410, AB_Y, 158, AB_H, win_label, "window", color=win_color)
        btn_reset  = _btn(PAD + 588, AB_Y, 90, AB_H, "RESET", "reset")
        btn_quit   = _btn(W - PAD - 90, AB_Y, 90, AB_H, "QUIT", "quit", color=C_BTN_DANGER)

        action_btns = [btn_prev, btn_next, btn_dist, btn_window, btn_reset, btn_quit]
        all_btns    = action_btns + list(PARAM_BTNS_DICT.values())

        # ── Consume click ──────────────────────────────────────────────────────
        mx, my = _mouse["x"], _mouse["y"]
        clicked = _mouse["clicked"]
        _mouse["clicked"] = False
        cx, cy = _mouse["cx"], _mouse["cy"]

        if clicked:
            for btn in all_btns:
                if not _hit(btn, cx, cy):
                    continue
                k = btn["key"]
                flash_key   = k
                flash_until = time.monotonic() + 0.20

                if k == "prev":
                    cue_idx = (cue_idx - 1) % len(VOICE_CUES)
                    window_open = False; fire_count = 0
                    last_vosk_text = ""; last_matched = False; hum_start = None
                elif k == "next":
                    cue_idx = (cue_idx + 1) % len(VOICE_CUES)
                    window_open = False; fire_count = 0
                    last_vosk_text = ""; last_matched = False; hum_start = None
                elif k == "dist":
                    dist_idx = (dist_idx + 1) % len(DISTANCES)
                elif k == "window":
                    if window_open:
                        window_open = False; hum_start = None
                    else:
                        window_open = True; window_start = time.monotonic(); hum_start = None
                elif k == "reset":
                    window_open = False; fire_count = 0; fired_at = 0.0
                    last_vosk_text = ""; last_matched = False; hum_start = None
                elif k == "quit":
                    running = False
                elif k == "hum_thr_dec":
                    tune["hum_rms_threshold"] = round(
                        max(TUNE_STEP_RMS, tune["hum_rms_threshold"] - TUNE_STEP_RMS), 4)
                    _save_tune_params(tune, profile_path)
                elif k == "hum_thr_inc":
                    tune["hum_rms_threshold"] = round(
                        tune["hum_rms_threshold"] + TUNE_STEP_RMS, 4)
                    _save_tune_params(tune, profile_path)
                elif k == "hum_dur_dec":
                    tune["hum_min_duration_ms"] = max(
                        TUNE_STEP_DUR, tune["hum_min_duration_ms"] - TUNE_STEP_DUR)
                    _save_tune_params(tune, profile_path)
                elif k == "hum_dur_inc":
                    tune["hum_min_duration_ms"] += TUNE_STEP_DUR
                    _save_tune_params(tune, profile_path)
                elif k == "whisper_dec":
                    tune["whisper_max_volume_dbfs"] -= TUNE_STEP_WHISPER
                    _save_tune_params(tune, profile_path)
                elif k == "whisper_inc":
                    tune["whisper_max_volume_dbfs"] = min(
                        0.0, tune["whisper_max_volume_dbfs"] + TUNE_STEP_WHISPER)
                    _save_tune_params(tune, profile_path)
                break

        if not running:
            break

        # ── Render ─────────────────────────────────────────────────────────────
        frame = np.full((H, W, 3), C_BG, dtype=np.uint8)

        # Header
        _draw_header(frame, cue, cue_idx, accent)
        _sep(frame, Y_HEADER_BOT)

        # Profile badge (top-right)
        _label(frame, "tuning: config.json [host]", W - 220, 20, (130, 130, 200), 0.40)
        vosk_ok = recognizer is not None
        _label(frame, f"vosk: {'ready' if vosk_ok else 'unavailable'}",
               W - 200, 38, (80, 200, 80) if vosk_ok else C_RED, 0.40)

        # Waveform
        _draw_waveform(frame, raw_buf, rms, hum_threshold,
                       Y_WAVE_TOP, Y_WAVE_BOT, accent)
        _sep(frame, Y_WAVE_BOT)

        # Meters row
        _vsep(frame, METER_SPLIT, Y_METERS_TOP, Y_METERS_BOT)
        _draw_rms_section(frame, rms, hum_threshold,
                          hum_start, hum_min_dur_ms,
                          PAD, Y_METERS_TOP, METER_L_W, accent)
        _draw_dbfs_section(frame, peak_dbfs, whisper_max,
                           METER_R_X, Y_METERS_TOP, METER_R_W, mode)
        _sep(frame, Y_METERS_BOT)

        # Keyword panel
        _draw_keyword_panel(frame, last_vosk_text, last_vosk_time,
                            keywords, last_matched, window_open,
                            window_start, window_ms, accent,
                            PAD, Y_KW_TOP, W - PAD * 2, Y_KW_BOT - Y_KW_TOP)
        _sep(frame, Y_KW_BOT)

        # Fire log
        _draw_fire_log(frame, fire_log, PAD, Y_LOG_TOP + 14)
        _sep(frame, Y_LOG_BOT)

        # Param buttons section
        _draw_param_buttons(frame, PARAM_BTNS_DICT, tune, mx, my, flash_key, flash_until)
        _sep(frame, Y_PARAM_BOT + 4)

        # Action buttons
        _draw_action_buttons(frame, action_btns, mx, my, flash_key, flash_until)

        # Status strip (bottom)
        status_y = Y_ACTION_BOT + 20
        fires_col = accent if fire_count > 0 else C_DIM
        fires_str = (f"fires: {fire_count}   "
                     f"last: {(time.monotonic() - fired_at):.0f}s ago"
                     if fired_at else f"fires: {fire_count}")
        _label(frame, fires_str, PAD, status_y, fires_col, 0.42)
        gram_str = f"grammar: {len(_BHR_GRAMMAR) - 1} keywords + [unk]"
        _label(frame, gram_str, W - 300, status_y, C_DIM, 0.38)

        # FIRED flash — green border pulse
        now = time.monotonic()
        if now - fired_at < 1.2:
            alpha = 0.25 * max(0.0, 1.0 - (now - fired_at) / 1.2)
            if alpha > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (W, H), (60, 255, 60), -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            # FIRED text centred in the keyword panel
            mid_y = (Y_KW_TOP + Y_KW_BOT) // 2 + 10
            _text(frame, "FIRED!", W // 2 - 80, mid_y,
                  (80, 255, 80), 1.8, bold=True)

        cv2.imshow(WIN, frame)

        # ESC only — all other input is via buttons
        if cv2.waitKey(16) & 0xFF == 27:
            running = False

    stream.stop()
    stream.close()
    cv2.destroyAllWindows()
    print("[voice_tuner] Done.")


if __name__ == "__main__":
    main()
