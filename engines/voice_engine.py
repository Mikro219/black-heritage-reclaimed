"""
VoiceEngine — keyword spotting, hum detection, and whisper detection.

Routes detected voice events through the EventBus. The keyword spotter
library is TBD (Vosk / Picovoice / openWakeWord); this stub defines the
interface. Hum and whisper detection use RMS thresholding via DSP.

Modes:
  keyword  — string match against a list of keywords
  hum      — any sustained vocal ≥ hum_min_duration_ms above hum_rms_threshold
  whisper  — low-volume RMS between whisper_rms_threshold and hum_rms_threshold
"""

import threading
import time
from typing import Optional


class VoiceEngine:
    def __init__(self, config: dict, event_bus: "EventBus"):
        self.config = config
        self.event_bus = event_bus
        self._voice_cfg = config.get("voice", {})
        self._active_vi: Optional[dict] = None
        self._vi_open_time: Optional[float] = None
        self._whisper_mode_active = False
        self._input_locked = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.event_bus.subscribe("dialogue_cue", self._on_dialogue_cue)
        self.event_bus.subscribe("input_lock", self._on_input_lock)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        # Stub: replace with real audio stream + spotter integration
        while self._running:
            time.sleep(0.05)
            # TODO: read audio chunk, run RMS / keyword spotter, call _handle_detection

    def _handle_detection(self, mode: str, matched: str):
        if self._input_locked or not self._active_vi:
            return

        window_ms = self.config["timing_defaults"].get("voice_window_ms", 10000)
        if self._vi_open_time:
            elapsed = (time.monotonic() - self._vi_open_time) * 1000
            if elapsed > window_ms:
                self._active_vi = None
                self._vi_open_time = None
                return

        vi = self._active_vi
        if mode != vi.get("mode"):
            return

        keywords = vi.get("keywords", [])
        if mode == "keyword" and matched.lower() not in [k.lower() for k in keywords]:
            return

        self.event_bus.emit("vi_detected", {"voice_id": vi["id"], "tier": vi.get("tier")})
        self._active_vi = None
        self._vi_open_time = None

    def _on_dialogue_cue(self, data: dict):
        cue = data.get("cue", {})
        interaction = cue.get("interaction", {})
        voice_alt = interaction.get("voice_alternative")
        voice_req = interaction.get("voice_required")
        vi_target = voice_alt or voice_req
        if vi_target:
            self._active_vi = vi_target
            self._vi_open_time = time.monotonic()
            # Activate whisper mode only for whisper-mode VIs
            self._whisper_mode_active = vi_target.get("mode") == "whisper"

    def _on_input_lock(self, data: dict):
        self._input_locked = data.get("locked", False)
