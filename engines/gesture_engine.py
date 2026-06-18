"""
GestureEngine — MediaPipe Hands + GRLib gesture detection.

Consumes camera frames, runs hand landmark detection, and dispatches to
detector modules in engines/detectors/ via a registry. Each detector
receives (landmarks, params, context) where context is a persistent dict
scoped to the active interaction — used for cross-frame state like hold
timers, velocity history, and path recordings.
"""

import cv2
import mediapipe as mp
from typing import Optional
import threading
import time

from engines.detectors import REGISTRY


# Detectors that read MediaPipe Pose landmarks (body-relative). Pose inference is
# the most expensive call, so the capture thread only runs it when one of these
# is the active CG/OI detector.
_POSE_PRIMARY = {"arms_crossed", "run_arms", "unravel", "paddle"}


class GestureEngine:
    def __init__(self, config: dict, event_bus: "EventBus"):
        self.config = config
        self.event_bus = event_bus
        self._thresholds = config.get("detection_thresholds", {})

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=self._thresholds.get("min_detection_confidence", 0.6),
            min_tracking_confidence=self._thresholds.get("min_tracking_confidence", 0.5),
        )

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._last_pose_lm = None
        self._last_pose_time: float = 0.0
        self._pose_stale_s: float = 0.5  # stale if no valid update for >500ms

        # --- Capture / inference thread ---------------------------------------
        # The camera read + MediaPipe inference run on a dedicated worker thread so
        # the render loop is never blocked on detection (CLAUDE.md performance rule).
        # The worker publishes the latest landmarks into _pub_* under _frame_lock;
        # the main thread snapshots them in update() and runs detector dispatch.
        self._cap = None
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_running = False
        self._frame_lock = threading.Lock()
        self._pub_hand_landmarks = None
        self._pub_handedness = None
        self._pub_pose_lm = None
        self._pub_pose_time: float = 0.0
        self._pub_seq: int = 0            # increments on each new inference result
        self._last_consumed_seq: int = -1  # last seq the main thread dispatched

        self._active_cg: Optional[dict] = None
        self._active_cg_context: dict = {}
        self._active_oi: Optional[dict] = None
        self._active_oi_context: dict = {}
        self._oi_open_time: Optional[float] = None
        self._cooldown_until: float = 0.0
        self._last_landmarks = None
        self._last_handedness = None
        self._input_locked = False
        self._last_fired: Optional[str] = None
        self._last_fired_time: float = 0.0
        self._warned_missing: set = set()   # suppress repeated "not in registry" noise

        self.event_bus.subscribe("cg_window_open", self._on_cg_window_open)
        self.event_bus.subscribe("oi_window_open", self._on_oi_window_open)
        self.event_bus.subscribe("input_lock", self._on_input_lock)

    def start_capture(self, cap) -> None:
        """Take ownership of the camera and start the capture/inference worker."""
        self._cap = cap
        self._capture_running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="GestureCapture"
        )
        self._capture_thread.start()
        print("[GestureEngine] capture thread started")

    def _pose_needed(self) -> bool:
        """True if the active CG/OI detector reads Pose landmarks."""
        cg_type = self._active_cg.get("type", "") if self._active_cg else ""
        oi_type = self._active_oi.get("type", "") if self._active_oi else ""
        return cg_type in _POSE_PRIMARY or oi_type in _POSE_PRIMARY

    def _capture_loop(self) -> None:
        """Worker: read camera, run MediaPipe, publish landmarks. Never touches the bus."""
        while self._capture_running:
            cap = self._cap
            if cap is None:
                break
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)   # camera hiccup — don't spin at 100% CPU
                continue

            # Always drain the camera to keep the USB pipeline fresh, but skip the
            # expensive inference while input is locked (playback/transitions need
            # no detection — matches the old early-return behaviour).
            if self._input_locked:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self._hands.process(rgb)

            # Pose only when a body-relative detector is armed (#3 — halves cost
            # during the hands-only holds that make up most of acts 1–4).
            run_pose = self._pose_needed()
            pose_lm = None
            if run_pose:
                pose_results = self._pose.process(rgb)
                if pose_results.pose_landmarks:
                    pose_lm = pose_results.pose_landmarks.landmark

            now = time.monotonic()
            with self._frame_lock:
                self._pub_hand_landmarks = hand_results.multi_hand_landmarks
                self._pub_handedness     = hand_results.multi_handedness
                if run_pose and pose_lm is not None:
                    self._pub_pose_lm   = pose_lm
                    self._pub_pose_time = now
                self._pub_seq += 1

    def update(self) -> None:
        """Main thread: dispatch detectors on the newest published landmarks.

        Cheap landmark math only — camera read and MediaPipe inference happen on
        the capture thread, so this never stalls the render loop.
        """
        if self._input_locked:
            return

        # Snapshot the latest inference result under the lock.
        with self._frame_lock:
            seq        = self._pub_seq
            landmarks  = self._pub_hand_landmarks
            handedness = self._pub_handedness
            pose_lm    = self._pub_pose_lm
            pose_time  = self._pub_pose_time

        self._last_landmarks  = landmarks
        self._last_handedness = handedness
        self._last_pose_lm    = pose_lm
        self._last_pose_time  = pose_time

        # No new inference since last update — skip dispatch (avoids re-processing
        # an identical landmark frame when render runs faster than inference).
        if seq == self._last_consumed_seq:
            return
        self._last_consumed_seq = seq

        # Pose-primary detectors can fire without hand landmarks; everything else
        # needs hands present.
        has_hands = bool(landmarks)
        active_cg_type = self._active_cg.get("type", "") if self._active_cg else ""
        needs_dispatch = has_hands or active_cg_type in _POSE_PRIMARY
        if not needs_dispatch:
            return

        now = time.monotonic()
        if now < self._cooldown_until:
            return

        if self._active_cg:
            if self._dispatch(self._active_cg, landmarks, self._active_cg_context):
                # Save before clearing so event callbacks can re-arm without being wiped
                gesture_id = self._active_cg["id"]
                choice     = self._active_cg_context.get("point_direction")
                self._active_cg         = None
                self._active_cg_context = {}
                self._emit_cg(gesture_id, choice)

        oi_window_ms = self.config["timing_defaults"].get("oi_window_ms", 6000)
        if self._active_oi and self._oi_open_time:
            elapsed = (now - self._oi_open_time) * 1000
            if elapsed <= oi_window_ms:
                if self._dispatch(self._active_oi, landmarks, self._active_oi_context):
                    self._emit_oi(self._active_oi["id"])
                    self._active_oi = None
                    self._active_oi_context = {}
                    self._oi_open_time = None
            else:
                self._active_oi = None
                self._active_oi_context = {}
                self._oi_open_time = None

    def hands_detected(self) -> bool:
        return bool(self._last_landmarks)

    def _on_cg_window_open(self, data: dict):
        self._active_cg = data.get("interaction")
        self._active_cg_context = {}

    def _on_oi_window_open(self, data: dict):
        self._active_oi = data.get("interaction")
        self._active_oi_context = {}
        self._oi_open_time = time.monotonic()
        oi = self._active_oi or {}
        print(f"[GestureEngine] OI window open: id={oi.get('id')!r}  type={oi.get('type')!r}  "
              f"in_registry={oi.get('type') in REGISTRY}")

    def _on_input_lock(self, data: dict):
        self._input_locked = data.get("locked", False)

    def _emit_cg(self, gesture_id: str, choice: str | None = None):
        cooldown = self._thresholds.get("gesture_cooldown_ms", 600) / 1000
        self._cooldown_until = time.monotonic() + cooldown
        label = f"CG:{gesture_id}" + (f"({choice})" if choice else "")
        self._last_fired = label
        self._last_fired_time = time.monotonic()
        event = {"gesture_id": gesture_id}
        if choice:
            event["choice"] = choice
        self.event_bus.emit("cg_detected", event)

    def _emit_oi(self, gesture_id: str):
        self._last_fired = f"OI:{gesture_id}"
        self._last_fired_time = time.monotonic()
        print(f"[GestureEngine] OI DETECTED: {gesture_id}")
        self.event_bus.emit("oi_detected", {"gesture_id": gesture_id})

    def debug_info(self) -> dict:
        now = time.monotonic()
        last = self._last_fired if (now - self._last_fired_time) < 2.0 else None
        cg = self._active_cg
        oi = self._active_oi
        recording = self._active_cg_context.get("shape_recording")
        pose_age = now - self._last_pose_time
        if self._last_pose_lm and pose_age < self._pose_stale_s:
            pose_status = "OK"
        elif self._last_pose_lm:
            pose_status = f"STALE({pose_age:.1f}s)"
        else:
            pose_status = "NONE"
        return {
            "active_cg": f"{cg['id']} ({cg['type']})" if cg else None,
            "active_oi": f"{oi['id']} ({oi['type']})" if oi else None,
            "active_oi_params": oi.get("params", {}) if oi else None,
            "last_fired": last,
            "recording_pts": len(recording) if recording else 0,
            "pose_status": pose_status,
            "point_dir": self._active_cg_context.get("dominant_direction"),
        }

    def _dispatch(self, interaction: dict, landmarks, context: dict) -> bool:
        detector_type = interaction.get("type")
        params = interaction.get("params", {})
        detector_fn = REGISTRY.get(detector_type)
        if detector_fn is None:
            if detector_type not in self._warned_missing:
                self._warned_missing.add(detector_type)
                print(f"[GestureEngine] WARNING: detector type {detector_type!r} not in registry. "
                      f"Known types: {sorted(REGISTRY)}")
            return False
        # Inject current pose landmarks under reserved context key.
        # Body-relative detectors read context["_pose_lm"]; existing detectors ignore it.
        pose_age = time.monotonic() - self._last_pose_time
        context["_pose_lm"] = self._last_pose_lm if pose_age < self._pose_stale_s else None
        return detector_fn(landmarks, params, context)

    def close(self):
        # Stop the worker before tearing down the MediaPipe graphs / camera so the
        # worker can't call .process() on a closed graph or read a released cap.
        self._capture_running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        self._hands.close()
        self._pose.close()
