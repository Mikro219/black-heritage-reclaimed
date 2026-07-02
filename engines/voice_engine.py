"""
VoiceEngine — Vosk keyword spotting, hum detection, and DSP routines.

Primary interface:
    open_window(vi_config) -> window_id   Open a VI window scoped to a dialogue cue.
    close_window(window_id)               Close a window early (e.g. on scene transition).
    subscribe(callback)                   Register a callback(VoiceEvent) for direct consumers.
    start() / stop()                      Lifecycle.
    debug_info() -> dict                  For the debug overlay.

A single sounddevice input callback fans raw audio to two consumers:
  1. Vosk KaldiRecognizer — grammar-constrained keyword spotting.
  2. DSP layer — RMS-based hum detection (Scene 4).

Multi-mode windows (e.g. Scene 4 AL-04-010 — hum OR "follow the gourd"):
    Set mode="hum" in vi_config and also populate keywords. The window accepts
    either trigger. If mode="keyword" (or "whisper"), only keyword matching fires.

Whisper mode (Scene 6 "Shoofly"): per-cue upper volume bound via max_volume_dbfs.
Set max_volume_dbfs: -25 on the shoofly cue to reject normal-volume speech (only a
true whispered "shoofly" passes). Any other VI cue that omits max_volume_dbfs gets
no upper bound — the hum check and all other cues are unaffected.

Keywords detected outside an open window are logged at INFO for on-site analytics.

Fallback to whisper.cpp (tiny.en / base.en) is documented but not built. Switch
only if Vosk accuracy is insufficient after on-site grammar tuning — the interface
boundary (open_window / subscribe) is designed so the swap requires no callers
to change.
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import sounddevice as sd

try:
    from vosk import KaldiRecognizer, Model as VoskModel
    _VOSK_AVAILABLE = True
except ImportError:
    _VOSK_AVAILABLE = False

log = logging.getLogger(__name__)

# Frozen (PyInstaller) builds: __file__ lives inside the bundled _internal dir, so
# resolve the model relative to the exe's folder instead (models/ is deployed
# next to BHR.exe — see scripts/build_exe.py).
if getattr(sys, "frozen", False):
    _APP_ROOT = Path(sys.executable).parent
else:
    _APP_ROOT = Path(__file__).parent.parent
VOSK_MODEL_PATH = _APP_ROOT / "models" / "vosk-model-small-en-us-0.15"

_DEFAULT_GRAMMAR = [
    "freedom", "go", "north", "follow the gourd",
    "shoofly", "yes", "water", "ready", "now", "[unk]",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class VoiceEvent:
    window_id: str
    vi_id: str
    tier: str
    trigger: str    # "keyword" | "hum"
    matched: str    # the keyword text, or "hum"


@dataclass
class _Window:
    window_id: str
    vi_config: dict
    open_time: float
    window_ms: float
    fired: bool = False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VoiceEngine:
    def __init__(self, config: dict, event_bus=None):
        self.config = config
        self.event_bus = event_bus
        self._voice_cfg: dict = config.get("voice", {})
        self._sample_rate: int = 16000
        self._input_locked: bool = False
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._mic_active: bool = False

        # windows: window_id → _Window; protected by _lock
        self._windows: dict[str, _Window] = {}
        self._lock = threading.Lock()

        # External subscribers called in addition to event_bus
        self._subscribers: list[Callable[[VoiceEvent], None]] = []

        # Global per-keyword debounce to prevent rapid double-fires
        self._debounce: dict[str, float] = {}
        self._debounce_ms: float = self._voice_cfg.get("debounce_ms", 1500)

        # Hum detection: wall-clock timestamp when continuous hum began
        self._hum_start: Optional[float] = None

        # Audio queue between sounddevice callback and processing thread
        self._audio_q: queue.Queue[bytes] = queue.Queue()

        # Debug: last keyword/hum heard (cleared after 2s in debug_info)
        self._last_heard: Optional[str] = None
        self._last_heard_time: float = 0.0

        # Vosk
        self._recognizer: Optional[object] = None
        self._init_vosk()

        # Wire up event bus listeners if a bus was provided
        if event_bus:
            event_bus.subscribe("dialogue_cue",   self._on_dialogue_cue)
            event_bus.subscribe("vi_chain_step",  self._on_vi_chain_step)  # shot-driven model
            event_bus.subscribe("vi_window_open", self._on_vi_window_open)  # frame-gated OI path
            event_bus.subscribe("input_lock",     self._on_input_lock)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def open_window(self, vi_config: dict) -> str:
        """Open a VI detection window. Returns an opaque window_id."""
        window_id = str(uuid.uuid4())
        timing = self.config.get("timing_defaults", {})
        window_ms = vi_config.get("window_ms", timing.get("voice_window_ms", 10000))
        win = _Window(
            window_id=window_id,
            vi_config=vi_config,
            open_time=time.monotonic(),
            window_ms=float(window_ms),
        )
        with self._lock:
            self._windows[window_id] = win
        log.info("VI window opened: id=%s vi_id=%s mode=%s keywords=%s",
                 window_id[:8], vi_config.get("id"), vi_config.get("mode"), vi_config.get("keywords"))
        return window_id

    def close_window(self, window_id: str) -> None:
        """Close a window early (scene transition, input lock, etc.)."""
        with self._lock:
            if window_id in self._windows:
                del self._windows[window_id]
                log.info("VI window closed early: id=%s", window_id[:8])

    def subscribe(self, callback: Callable[[VoiceEvent], None]) -> None:
        """Register a callback invoked whenever a VI fires. Used by the test CLI."""
        self._subscribers.append(callback)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="VoiceEngine")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def debug_info(self) -> dict:
        """Snapshot for the debug overlay — safe to call from the main thread."""
        now = time.monotonic()
        with self._lock:
            active_win = next(iter(self._windows.values()), None)
        age_ms = (now - self._last_heard_time) * 1000 if self._last_heard else None
        last_heard = self._last_heard if (age_ms is not None and age_ms < 2000) else None
        return {
            "mic_active": self._mic_active,
            "active_window": active_win.vi_config.get("id") if active_win else None,
            "last_heard": last_heard,
        }

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def _init_vosk(self) -> None:
        if not _VOSK_AVAILABLE:
            log.warning("vosk not installed — keyword detection disabled. Run: pip install vosk")
            return
        if not VOSK_MODEL_PATH.exists():
            log.warning(
                "Vosk model not found at %s — run scripts/setup_voice.py to download it.",
                VOSK_MODEL_PATH,
            )
            return
        grammar = self._voice_cfg.get("grammar") or _DEFAULT_GRAMMAR
        model = VoskModel(str(VOSK_MODEL_PATH))
        self._recognizer = KaldiRecognizer(model, self._sample_rate, json.dumps(grammar))
        log.info("Vosk ready. Grammar: %s", grammar)

    # -----------------------------------------------------------------------
    # Audio loop (runs on dedicated thread)
    # -----------------------------------------------------------------------

    def _run(self) -> None:
        mic_cfg = self.config.get("_profile", {}).get("microphone", {})
        sample_rate: int = mic_cfg.get("sample_rate", self._sample_rate)
        channels: int = mic_cfg.get("channels", 1)
        device = mic_cfg.get("device_name_match") or None

        def _callback(indata, frames, time_info, status):
            self._audio_q.put(bytes(indata))

        try:
            with sd.RawInputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                device=device,
                callback=_callback,
                blocksize=4000,
            ):
                self._mic_active = True
                log.info("Mic open (device=%s rate=%d)", device or "default", sample_rate)
                while self._running:
                    try:
                        chunk = self._audio_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    self._process_chunk(chunk)
        except Exception as exc:
            log.error("Audio stream error: %s", exc)
        finally:
            self._mic_active = False
            log.info("Mic closed.")

    def _process_chunk(self, chunk: bytes) -> None:
        import math
        import struct

        # DSP: RMS + peak amplitude for hum detection and volume gating
        samples = struct.unpack(f"{len(chunk) // 2}h", chunk)
        rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0
        peak = max(abs(s) for s in samples) / 32768.0
        # Convert peak to dBFS (−inf guard: clamp below −120 dB)
        peak_dbfs = 20.0 * math.log10(peak) if peak > 1e-6 else -120.0
        self._check_hum(rms)

        # Vosk keyword spotting
        if self._recognizer and self._recognizer.AcceptWaveform(chunk):
            result = json.loads(self._recognizer.Result())
            text = result.get("text", "").strip().lower()
            if text and text != "[unk]":
                self._last_heard = text
                self._last_heard_time = time.monotonic()
                self._try_fire_windows("keyword", text, peak_dbfs=peak_dbfs)

    # -----------------------------------------------------------------------
    # Hum detection (DSP)
    # -----------------------------------------------------------------------

    def _check_hum(self, rms: float) -> None:
        threshold: float = self._voice_cfg.get("hum_rms_threshold", 0.02)
        min_dur_ms: float = self._voice_cfg.get("hum_min_duration_ms", 500)

        if rms >= threshold:
            if self._hum_start is None:
                self._hum_start = time.monotonic()
            elif (time.monotonic() - self._hum_start) * 1000 >= min_dur_ms:
                self._hum_start = None  # reset so next sustained hum fires again
                self._last_heard = "hum"
                self._last_heard_time = time.monotonic()
                self._try_fire_windows("hum", "hum", peak_dbfs=0.0)
        else:
            self._hum_start = None

    # -----------------------------------------------------------------------
    # Window matching and event emission
    # -----------------------------------------------------------------------

    def _try_fire_windows(self, mode: str, matched: str, peak_dbfs: float = 0.0) -> None:
        if self._input_locked:
            return

        now = time.monotonic()
        to_fire: list[tuple[str, _Window]] = []

        with self._lock:
            # Expire stale windows
            expired = [
                wid for wid, w in self._windows.items()
                if (now - w.open_time) * 1000 > w.window_ms
            ]
            for wid in expired:
                log.info("VI window expired: id=%s vi_id=%s", wid[:8], self._windows[wid].vi_config.get("id"))
                del self._windows[wid]

            # Find the first matching, un-fired window
            for wid, win in list(self._windows.items()):
                if win.fired:
                    continue
                if not _window_matches(win.vi_config, mode, matched):
                    continue
                # Per-cue upper volume bound: reject keyword matches that are too loud.
                # Cue metadata takes precedence; falls back to config voice.whisper_max_volume_dbfs
                # when the cue's mode is "whisper" and no per-cue value is set.
                max_vol = win.vi_config.get("max_volume_dbfs")
                if max_vol is None and win.vi_config.get("mode") == "whisper":
                    max_vol = self._voice_cfg.get("whisper_max_volume_dbfs")
                if max_vol is not None and mode == "keyword" and peak_dbfs > max_vol:
                    log.info(
                        "VI rejected (too loud): vi_id=%s matched=%r peak_dbfs=%.1f max=%.1f",
                        win.vi_config.get("id"), matched, peak_dbfs, max_vol,
                    )
                    continue
                last_fired = self._debounce.get(matched, 0.0)
                if (now - last_fired) * 1000 < self._debounce_ms:
                    log.debug("VI debounced: matched=%r", matched)
                    continue
                self._debounce[matched] = now
                win.fired = True
                del self._windows[wid]
                to_fire.append((wid, win))
                break  # one event per detection

        if not to_fire:
            log.info("VI no match: mode=%s matched=%r (no open window or debounced)", mode, matched)
            return

        for wid, win in to_fire:
            event = VoiceEvent(
                window_id=wid,
                vi_id=win.vi_config.get("id", ""),
                tier=win.vi_config.get("tier", "reaction"),
                trigger=mode,
                matched=matched,
            )
            log.info("VI fired: vi_id=%s trigger=%s matched=%r tier=%s",
                     event.vi_id, event.trigger, event.matched, event.tier)
            self._emit(event)

    def _emit(self, event: VoiceEvent) -> None:
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception as exc:
                log.error("Subscriber error: %s", exc)
        if self.event_bus:
            self.event_bus.emit("vi_detected", {
                "voice_id": event.vi_id,
                "tier": event.tier,
                "trigger": event.trigger,
                "matched": event.matched,
            })

    # -----------------------------------------------------------------------
    # Event bus listeners
    # -----------------------------------------------------------------------

    def _on_dialogue_cue(self, data: dict) -> None:
        """
        Legacy path: opens VI windows from NarrationEngine dialogue_cue events.
        Used when running the old SceneManager-based runtime.
        In the shot-driven model, VI windows are opened via _on_vi_chain_step instead.
        """
        cue = data.get("cue", {})
        interaction = cue.get("interaction", {})
        vi_target = interaction.get("voice_alternative") or interaction.get("voice_required")
        if vi_target:
            self.open_window(vi_target)

    def _on_vi_chain_step(self, data: dict) -> None:
        """
        Shot-driven model: opens a VI window when ShotSequencePlayer arms a voice
        chain step.  The step dict has {type, keyword, required, mode?}.

        Replaces the dialogue_cue → voice_alternative/voice_required path for shots
        wired in the new shot-driven runtime.
        """
        step = data.get("step", {})
        keyword = step.get("keyword", "").strip()
        if not keyword:
            return
        required = bool(step.get("required", False))
        vi_config = {
            "id":         keyword,
            "keywords":   [keyword],
            "mode":       step.get("mode", "keyword"),
            "tier":       "cg_required" if required else "cg_alternative",
            "window_ms":  self.config.get("timing_defaults", {}).get("voice_window_ms", 10000),
        }
        self.open_window(vi_config)
        log.info("VI window opened via vi_chain_step: keyword=%r required=%s", keyword, required)

    def _on_vi_window_open(self, data: dict) -> None:
        """Shot player emits this when a frame-gated OI window opens with a voice_alternative."""
        vi_config = data.get("vi_config")
        if not vi_config:
            return
        wid = self.open_window(vi_config)
        log.info("VI window opened via vi_window_open: id=%s keywords=%s",
                 vi_config.get("id"), vi_config.get("keywords"))
        print(f"[VoiceEngine] window open: id={vi_config.get('id')!r}  "
              f"keywords={vi_config.get('keywords')}  wid={wid[:8]}")

    def _on_input_lock(self, data: dict) -> None:
        locked: bool = data.get("locked", False)
        self._input_locked = locked
        if locked:
            with self._lock:
                count = len(self._windows)
                self._windows.clear()
            if count:
                log.info("Input locked — cleared %d open VI window(s).", count)


# ---------------------------------------------------------------------------
# Module-level helper (no self needed)
# ---------------------------------------------------------------------------

def _window_matches(vi_config: dict, mode: str, matched: str) -> bool:
    """Return True if the detection (mode, matched) satisfies the window's vi_config.

    Multi-mode rule: mode="hum" with a non-empty keywords list means the window
    accepts either a hum OR a matching keyword — used for Scene 4 AL-04-010.
    """
    keywords = [k.lower() for k in vi_config.get("keywords", [])]
    window_mode: str = vi_config.get("mode", "keyword")

    if window_mode == "hum":
        if mode == "hum":
            return True
        # Multi-mode: also accept a keyword match
        return mode == "keyword" and matched.lower() in keywords

    # keyword / whisper — whisper is plain keyword detection per May 2026 PM call
    return mode == "keyword" and matched.lower() in keywords


# ---------------------------------------------------------------------------
# Test CLI  — python -m engines.voice_engine --test-scene 1
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="VoiceEngine isolated test harness")
    parser.add_argument("--test-scene", metavar="N", type=int, default=1,
                        help="Scene number to simulate (default: 1)")
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    engine = VoiceEngine(config)  # no event_bus — standalone mode

    def _on_event(event: VoiceEvent) -> None:
        print(f"\n{'='*60}")
        print(f"  VOICE EVENT FIRED")
        print(f"  vi_id   : {event.vi_id}")
        print(f"  tier    : {event.tier}")
        print(f"  trigger : {event.trigger}")
        print(f"  matched : {event.matched!r}")
        print(f"{'='*60}\n")

    engine.subscribe(_on_event)
    engine.start()

    scene = args.test_scene

    if scene == 1:
        print("\n--- Scene 1 test: AL-01-007 'raise_hands / freedom' ---")
        print("Listening passively for 3 seconds (nothing should fire yet)...\n")
        time.sleep(3)

        print("Opening AL-01-007 voice window (keyword='freedom', 10s window)...")
        wid = engine.open_window({
            "id": "freedom",
            "keywords": ["freedom"],
            "mode": "keyword",
            "tier": "cg_alternative",
            "window_ms": 10000,
        })
        print(f"Window open (id={wid[:8]}...). Say 'freedom' within 10 seconds.\n")
        time.sleep(11)
        print("Window expired. Test complete.")

    else:
        print(f"No test defined for scene {scene}. Supported: 1")

    engine.stop()
