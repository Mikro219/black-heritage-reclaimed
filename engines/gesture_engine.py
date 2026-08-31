"""
GestureEngine — MediaPipe Pose gesture detection.

POSE-ONLY (July 2026): the runtime runs a single model — MediaPipe Pose
(33 body landmarks, includes per-side pinky/index/thumb hand points). The
Hands model is gone; every detector in engines/detectors/rules/ reads the
skeleton from context["_pose_lm"] (plus the Gemini 335 depth fusion in
context["_pose_depth"] / context["_depth_mm_at"]).

The camera read + Pose inference run on a dedicated worker thread; the main
thread snapshots the newest skeleton in update() and dispatches the active
CG/OI detector. Each detector receives (landmarks, params, context) — the
`landmarks` argument is the legacy Hands slot and is always [] now; context
is a persistent dict scoped to the active interaction (hold timers, velocity
history, stroke recordings).
"""

import cv2
import mediapipe as mp
from typing import Optional
import threading
import time

from engines.detectors import REGISTRY
from engines.depth.fusion import PoseDepth


def _inject_pose_params(gtype: str, params: dict, pose_lm) -> dict:
    """Return a copy of params with live pose-derived values injected.

    Mirrors scripts/gesture_tuner.py::_inject_pose_params so the runtime computes
    the same per-player thresholds the tuner shows on-screen, instead of using the
    fixed values baked into shot metadata. Detectors already accept these as
    optional overrides, so no detector code changes are needed.

    Returns params unchanged when pose is unavailable — detectors then fall back to
    their fixed defaults (e.g. survey brow_y=0.45, presence_bilateral "shoulder").
    """
    if pose_lm is None:
        return params
    try:
        if gtype == "survey":
            # Live brow line from eye level (landmarks 2=left_eye, 5=right_eye).
            eye_y = (pose_lm[2].y + pose_lm[5].y) / 2
            p = dict(params)
            p["brow_y"] = eye_y + 0.04
            return p

        if gtype == "mouth_proximity_tip":
            p = dict(params)
            p["mouth_x"] = (pose_lm[9].x + pose_lm[10].x) / 2
            p["mouth_y"] = (pose_lm[9].y + pose_lm[10].y) / 2
            return p

        if gtype == "directional_head_or_hand":
            direction = params.get("direction")
            if direction == "ear":
                p = dict(params)
                p["ear_positions"] = [
                    (pose_lm[7].x, pose_lm[7].y),
                    (pose_lm[8].x, pose_lm[8].y),
                ]
                return p
            if direction == "brow":
                eye_y = (pose_lm[2].y + pose_lm[5].y) / 2
                p = dict(params)
                p["brow_y"] = eye_y + 0.04
                return p
            return params

        if gtype == "presence_bilateral":
            threshold_name = params.get("y_threshold", "")
            if threshold_name == "shoulder":
                p = dict(params)
                p["y_threshold"] = (pose_lm[11].y + pose_lm[12].y) / 2
                return p
            if threshold_name == "head":
                p = dict(params)
                p["y_threshold"] = (pose_lm[2].y + pose_lm[5].y) / 2
                return p
            return params
    except (IndexError, AttributeError, TypeError):
        # Malformed/partial pose result — fall back to fixed params.
        return params

    return params


class GestureEngine:
    def __init__(self, config: dict, event_bus: "EventBus"):
        self.config = config
        self.event_bus = event_bus
        self._thresholds = config.get("detection_thresholds", {})
        self._gesture_tuning: dict = config.get("_profile", {}).get("gesture_tuning", {})

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=self._thresholds.get("min_detection_confidence", 0.5),
            min_tracking_confidence=self._thresholds.get("min_tracking_confidence", 0.5),
        )
        self._last_pose_lm = None
        self._last_pose_time: float = 0.0
        self._pose_stale_s: float = 0.5  # stale if no valid update for >500ms

        # --- Capture / inference thread ---------------------------------------
        # The camera read + Pose inference run on a dedicated worker thread so
        # the render loop is never blocked on detection (CLAUDE.md performance rule).
        # The worker publishes the latest skeleton into _pub_* under _frame_lock;
        # the main thread snapshots it in update() and runs detector dispatch.
        self._cap = None
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_running = False
        self._frame_lock = threading.Lock()
        self._pub_pose_lm = None
        self._pub_pose_time: float = 0.0
        self._pub_frame = None            # latest camera frame, native channel
        self._pub_frame_is_rgb = False    # order (setup screen converts lazily)
        self._pub_seq: int = 0            # increments on each new inference result
        self._last_consumed_seq: int = -1  # last seq the main thread dispatched

        self._active_cg: Optional[dict] = None
        self._active_cg_context: dict = {}
        self._active_oi: Optional[dict] = None
        self._active_oi_context: dict = {}
        self._oi_open_time: Optional[float] = None
        self._active_oi_window_ms: float = config.get("timing_defaults", {}).get("oi_window_ms", 6000)
        self._cooldown_until: float = 0.0
        self._input_locked = False
        # Keep-warm inference rate while input is LOCKED. DEFAULT 0 (skip
        # entirely, the long-standing behaviour). Warming at a trickle was
        # tried against the "frames hitch when a window opens" report and made
        # it WORSE on the exhibition build: at a low rate MediaPipe's tracking
        # ROI is always stale, so every warm frame re-runs the heavy detector —
        # continuous cost during playback for no cold-start saving. Left as a
        # knob rather than deleted, but do not enable it without measuring.
        try:
            hz = float(config.get("warm_pose_hz", 0.0) or 0.0)
        except (TypeError, ValueError):
            hz = 0.0
        self._warm_pose_interval: float = 0.0 if hz <= 0 else 1.0 / hz
        self._last_pose_run: float = 0.0
        self._paused = False
        self._pause_started: float = 0.0
        self._last_fired: Optional[str] = None
        self._last_fired_time: float = 0.0
        self._warned_missing: set = set()   # suppress repeated "not in registry" noise

        # Depth fusion (Gemini 335): player interaction band + phantom veto
        # thresholds. All optional — absent depth degrades to pose-only behaviour.
        depth_cfg = config.get("depth", {})
        self._player_min_mm = depth_cfg.get("player_min_mm", 500)
        self._player_max_mm = depth_cfg.get("player_max_mm", 3200)
        self._phantom_behind_mm = depth_cfg.get("phantom_behind_mm", 400)
        self._player_gated = False   # debug: pose currently rejected by the band

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

    def _should_run_pose(self, now: float) -> bool:
        """Whether this camera frame gets a Pose inference.

        Unlocked: every frame (detectors need it). Locked: only every
        `_warm_pose_interval` seconds — enough to keep MediaPipe's graph and
        tracking state hot so the first inference after a window opens doesn't
        pay the cold-start cost as a frame hitch, but ~1/10th the CPU of full
        rate. `warm_pose_hz: 0` in config restores the old skip-entirely
        behaviour."""
        if not self._input_locked:
            self._last_pose_run = now
            return True
        if self._warm_pose_interval <= 0.0:
            return False
        if now - self._last_pose_run < self._warm_pose_interval:
            return False
        self._last_pose_run = now
        return True

    def _capture_loop(self) -> None:
        """Worker: read camera, run Pose, publish the skeleton. Never touches the bus."""
        inf_mute_until = 0.0   # rate limit for the slow-inference report
        while self._capture_running:
            cap = self._cap
            if cap is None:
                break
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)   # camera hiccup — don't spin at 100% CPU
                continue

            # Native channel order: cv2.VideoCapture delivers BGR; the Orbbec
            # shim delivers the sensor's RGB (frame_is_rgb = True) so MediaPipe
            # gets it without a per-frame cvtColor round trip.
            src_is_rgb = bool(getattr(cap, "frame_is_rgb", False))

            # Publish the raw frame regardless of lock state — the camera-setup
            # screen needs the live view before the experience starts. Reference
            # only, no copy: cap.read() returns a fresh array each call. The
            # frame is published in its NATIVE order plus a flag;
            # latest_camera_frame() converts to BGR lazily, so the conversion
            # is only paid while the setup screen is actually reading frames.
            with self._frame_lock:
                self._pub_frame = frame
                self._pub_frame_is_rgb = src_is_rgb

            # Always drain the camera to keep the USB pipeline fresh. While
            # input is locked we need no detections, but we do keep the model
            # warm at a trickle — see _should_run_pose.
            if not self._should_run_pose(time.monotonic()):
                continue

            rgb = frame if src_is_rgb else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_inf = time.perf_counter()
            pose_results = self._pose.process(rgb)
            # Measured: process() releases the GIL, so a slow inference here
            # can NOT stall the render loop — but the first inference after a
            # locked stretch runs 2-6x the steady cost (stale tracking ROI ->
            # heavy detector + CPU clocks ramping from idle). Reported so a
            # visible stutter at window-open can be correlated (or cleared).
            inf_ms = (time.perf_counter() - t_inf) * 1000.0
            if inf_ms >= 150.0:
                now_rep = time.monotonic()
                if now_rep >= inf_mute_until:
                    inf_mute_until = now_rep + 1.0
                    print(f"[perf] pose inference {inf_ms:.0f}ms "
                          f"(capture thread — does not block the picture)",
                          flush=True)
            pose_lm = None
            if pose_results.pose_landmarks:
                pose_lm = pose_results.pose_landmarks.landmark

            now = time.monotonic()
            with self._frame_lock:
                if pose_lm is not None:
                    self._pub_pose_lm = pose_lm
                    self._pub_pose_time = now
                self._pub_seq += 1

    def pause(self) -> None:
        """Freeze the engine's wall-clock state. Idempotent.

        update() isn't driven while the app is paused, but the OI-window expiry
        and detector cooldown are monotonic timestamps — without shifting them
        on resume, a pause mid-OI silently expires the window (the player shifts
        its own timers; this is the gesture-side counterpart)."""
        if self._paused:
            return
        self._paused = True
        self._pause_started = time.monotonic()

    def resume(self) -> None:
        if not self._paused:
            return
        delta = time.monotonic() - self._pause_started
        if self._oi_open_time is not None:
            self._oi_open_time += delta
        self._cooldown_until += delta
        # Detector hold state (context "*_since" timestamps, histories) can't be
        # shifted from here — clear it so a hold must be re-performed after the
        # pause instead of completing across it ("detect intent").
        self._active_cg_context.clear()
        self._active_oi_context.clear()
        self._paused = False

    def latest_camera_frame(self):
        """Newest raw BGR camera frame (or None before the first read). Used by
        the camera-setup screen; published even while input is locked.

        The frame is stored in its source's native channel order (the Orbbec
        shim is RGB); the BGR contract of this accessor is preserved by
        converting HERE, so the cost lands only on the setup screen's reads
        instead of on every captured frame."""
        with self._frame_lock:
            frame = self._pub_frame
            is_rgb = self._pub_frame_is_rgb
        if frame is not None and is_rgb:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def fresh_pose_lm(self):
        """Pose landmarks for the render overlay, or None if none were detected
        recently. `_pub_pose_lm` is never cleared once set (it only updates when a
        pose is found), so `_last_pose_lm` alone would keep drawing the last skeleton
        forever. Gate it on the same stale window the detectors use."""
        if self._last_pose_lm is None:
            return None
        if time.monotonic() - self._last_pose_time >= self._pose_stale_s:
            return None
        return self._last_pose_lm

    def update(self) -> None:
        """Main thread: dispatch detectors on the newest published skeleton.

        Cheap landmark math only — camera read and Pose inference happen on
        the capture thread, so this never stalls the render loop.
        """
        if self._input_locked:
            return

        # Snapshot the latest inference result under the lock.
        with self._frame_lock:
            seq       = self._pub_seq
            pose_lm   = self._pub_pose_lm
            pose_time = self._pub_pose_time

        self._last_pose_lm = pose_lm
        self._last_pose_time = pose_time

        # No new inference since last update — skip dispatch (avoids re-processing
        # an identical skeleton when render runs faster than inference).
        if seq == self._last_consumed_seq:
            return
        self._last_consumed_seq = seq

        now = time.monotonic()
        if now < self._cooldown_until:
            return

        if self._active_cg:
            if self._dispatch(self._active_cg, self._active_cg_context):
                # Save before clearing so event callbacks can re-arm without being wiped
                gesture_id = self._active_cg["id"]
                choice     = self._active_cg_context.get("point_direction")
                self._active_cg         = None
                self._active_cg_context = {}
                self._emit_cg(gesture_id, choice)

        oi_window_ms = self._active_oi_window_ms
        if self._active_oi and self._oi_open_time:
            elapsed = (now - self._oi_open_time) * 1000
            if elapsed <= oi_window_ms:
                if self._dispatch(self._active_oi, self._active_oi_context):
                    self._emit_oi(self._active_oi["id"])
                    self._active_oi = None
                    self._active_oi_context = {}
                    self._oi_open_time = None
            else:
                self._active_oi = None
                self._active_oi_context = {}
                self._oi_open_time = None

    def _on_cg_window_open(self, data: dict):
        self._active_cg = data.get("interaction")
        self._active_cg_context = {}

    def _on_oi_window_open(self, data: dict):
        self._active_oi = data.get("interaction")
        self._active_oi_context = {}
        self._oi_open_time = time.monotonic()
        # Honor a per-window duration (FSM OI states pass their own); fall back to
        # the global default so existing single-OI shots are unchanged.
        self._active_oi_window_ms = data.get(
            "window_ms", self.config["timing_defaults"].get("oi_window_ms", 6000)
        )
        oi = self._active_oi or {}
        print(f"[GestureEngine] OI window open: id={oi.get('id')!r}  type={oi.get('type')!r}  "
              f"window_ms={self._active_oi_window_ms}  in_registry={oi.get('type') in REGISTRY}")

    def _on_input_lock(self, data: dict):
        self._input_locked = data.get("locked", False)
        if self._input_locked:
            # Disarm on lock, mirroring VoiceEngine (which clears its windows):
            # otherwise a window armed before the lock keeps dispatching — and
            # keeps the window-scoped cursors on screen — after the transition.
            self._active_cg = None
            self._active_cg_context = {}
            self._active_oi = None
            self._active_oi_context = {}

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
            # Raw type/params of the live window (CG wins) — the render engine's
            # hand-icon cursors and interaction indicators key off these.
            "active_type": (cg or oi or {}).get("type"),
            "active_params": (cg or oi or {}).get("params", {}) if (cg or oi) else {},
            "input_locked": self._input_locked,
            "last_fired": last,
            "recording_pts": len(recording) if recording else 0,
            "pose_status": pose_status,
            "player_gated": self._player_gated,
            "point_dir": self._active_cg_context.get("dominant_direction"),
        }

    def _dispatch(self, interaction: dict, context: dict) -> bool:
        detector_type = interaction.get("type")
        # Overlay tuned params saved by scripts/gesture_tuner.py. The tuner keys its
        # saved params by gesture *name*, which doesn't always equal the shot's
        # interaction *id* (e.g. id "scan_survey" vs tuner name "survey"). Look up by
        # id first, then fall back to the detector type so tuned values still apply.
        tuned = (self._gesture_tuning.get(interaction.get("id", ""))
                 or self._gesture_tuning.get(detector_type, {}))
        params = {**interaction.get("params", {}), **tuned}
        detector_fn = REGISTRY.get(detector_type)
        if detector_fn is None:
            if detector_type not in self._warned_missing:
                self._warned_missing.add(detector_type)
                print(f"[GestureEngine] WARNING: detector type {detector_type!r} not in registry. "
                      f"Known types: {sorted(REGISTRY)}")
            return False
        # Inject current pose landmarks under reserved context key.
        pose_age = time.monotonic() - self._last_pose_time
        pose_lm = self._last_pose_lm if pose_age < self._pose_stale_s else None
        # Real depth sampler (mm) when the capture device provides one (Orbbec
        # Gemini 335 shim). None on plain webcams — detectors treat it as absent.
        depth_sampler = getattr(self._cap, "depth_mm_at", None)
        context["_depth_mm_at"] = depth_sampler

        # Depth fusion: lift/veto/meter over this pose + the latest depth frame.
        # Player band gate: when real depth says the tracked torso is outside the
        # interaction zone, the pose belongs to a passer-by (or the projection
        # wall) — drop it for this dispatch so background people can't steer
        # detections. Sampling is lazy, so this costs ~4 depth patch reads only
        # while a window is armed.
        fusion = PoseDepth(depth_sampler, pose_lm,
                           phantom_behind_mm=self._phantom_behind_mm)
        self._player_gated = False
        if fusion.available:
            torso = fusion.torso_depth_mm()
            if torso is not None and not (self._player_min_mm <= torso
                                          <= self._player_max_mm):
                self._player_gated = True
                pose_lm = None
                # Reuse the object with its pose nulled (available -> False,
                # every sample path returns None) instead of constructing a
                # second PoseDepth that discards the first's patch cache.
                # _torso is cleared too so a direct torso_depth_mm() matches
                # the no-pose contract.
                fusion._pose = None
                fusion._torso = None
        context["_pose_lm"] = pose_lm
        context["_pose_depth"] = fusion
        # Compute live pose-derived thresholds (brow line, shoulder height, mouth/ear
        # positions) so detection matches the tuner instead of using fixed values.
        params = _inject_pose_params(detector_type, params, pose_lm)
        # The legacy Hands slot in the detector signature is always empty now.
        return detector_fn([], params, context)

    def close(self):
        # Stop the worker before tearing down the MediaPipe graph / camera so the
        # worker can't call .process() on a closed graph or read a released cap.
        self._capture_running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        self._pose.close()
