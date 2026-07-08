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
from engines.depth.fusion import PoseDepth
from engines import pose_hand_filter


# Detectors that read MediaPipe Pose landmarks. Pose inference is the most
# expensive call, so the capture thread only runs it when one of these is the
# active CG/OI detector.
#
# Includes the detectors that need live pose-derived params injected (survey,
# directional_head_or_hand, presence_bilateral, mouth_proximity_tip) so the
# runtime computes the same per-player thresholds the gesture tuner does, rather
# than falling back to fixed metadata values. See _inject_pose_params below.
_POSE_DETECTORS = {"arms_crossed", "run_arms", "unravel", "paddle",
                   "mouth_proximity_tip", "touch_head",
                   "survey", "directional_head_or_hand", "presence_bilateral",
                   "point_region", "throw", "spread_arms", "forward_point",
                   "directional_draw", "point_target_held", "speed_bilateral"}

# Pose-only detectors that can fire WITHOUT hand landmarks present (they read body
# landmarks, not hands). Used to decide whether to dispatch when no hands are seen.
# point_region reads pose wrists, so a hand reaching out to the side of the body
# still registers even when MediaPipe Hands loses the finger pose there.
# touch_head (use_pose) reads pose wrists too, so both hands raised to the crown
# (AL-08-005 hands_to_head) still register when Hands drops them to head occlusion.
_NO_HANDS_DETECTORS = {"arms_crossed", "run_arms", "unravel", "paddle", "point_region",
                       "touch_head", "throw", "spread_arms", "survey", "forward_point",
                       "directional_draw", "point_target_held", "speed_bilateral"}


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
        self._active_oi_window_ms: float = config.get("timing_defaults", {}).get("oi_window_ms", 6000)
        self._cooldown_until: float = 0.0
        self._last_landmarks = None
        self._last_handedness = None
        self._input_locked = False
        self._last_fired: Optional[str] = None
        self._last_fired_time: float = 0.0
        self._warned_missing: set = set()   # suppress repeated "not in registry" noise

        # Depth fusion (Gemini 335): player interaction band + phantom veto
        # thresholds. All optional — absent depth degrades to pre-depth behaviour.
        depth_cfg = config.get("depth", {})
        self._player_min_mm = depth_cfg.get("player_min_mm", 500)
        self._player_max_mm = depth_cfg.get("player_max_mm", 3200)
        self._phantom_behind_mm = depth_cfg.get("phantom_behind_mm", 400)
        self._player_gated = False   # debug: pose currently rejected by the band

        # Pose-hand fusion: pose skeleton validates the Hands output (phantom
        # veto, handedness from pose side, stale-track arbitration) and can
        # optionally rescue a missed hand via a cropped re-inference. With no
        # fresh pose the filter passes everything through unchanged.
        ph_cfg = config.get("pose_hand", {})
        self._ph_filter = ph_cfg.get("filter", True)
        self._ph_rescue = ph_cfg.get("rescue", False)
        self._ph_match_dist = ph_cfg.get("max_match_dist", 0.25)
        self._ph_arb_scale = ph_cfg.get("arbitration_scale", 1.3)
        self._ph_pose_every_n = max(1, ph_cfg.get("pose_every_n", 2))
        self._ph_stats = {"dropped": 0, "corrected": 0, "rescued": 0}
        self._rescue_hands = None      # lazy second Hands instance (crop inference)
        self._capture_frame_i = 0

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

    @staticmethod
    def _reads_pose(interaction) -> bool:
        if not interaction:
            return False
        return (interaction.get("type", "") in _POSE_DETECTORS
                or bool(interaction.get("params", {}).get("use_pose")))

    @staticmethod
    def _runs_without_hands(interaction) -> bool:
        """Detectors that can fire with no hand landmarks (read body/pose only)."""
        if not interaction:
            return False
        return (interaction.get("type", "") in _NO_HANDS_DETECTORS
                or bool(interaction.get("params", {}).get("use_pose")))

    def _pose_needed(self) -> bool:
        """True if the active CG/OI detector reads Pose landmarks (type or use_pose)."""
        return self._reads_pose(self._active_cg) or self._reads_pose(self._active_oi)

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
            self._capture_frame_i += 1

            # Pose runs when a body-relative detector is armed (full rate), or —
            # with the pose-hand filter on — at a throttled cadence so the
            # skeleton can validate the Hands output during hands-only holds.
            run_pose = self._pose_needed() or (
                self._ph_filter
                and self._capture_frame_i % self._ph_pose_every_n == 0)
            pose_lm = None
            if run_pose:
                pose_results = self._pose.process(rgb)
                if pose_results.pose_landmarks:
                    pose_lm = pose_results.pose_landmarks.landmark

            now = time.monotonic()
            landmarks  = hand_results.multi_hand_landmarks
            handedness = hand_results.multi_handedness

            # Freshest skeleton for the filter: this frame's if pose ran, else
            # the last published one while still inside the stale window.
            # (_pub_* are written only by this thread — reading unlocked is safe.)
            filter_pose = pose_lm
            if filter_pose is None and self._pub_pose_lm is not None \
                    and now - self._pub_pose_time < self._pose_stale_s:
                filter_pose = self._pub_pose_lm

            if self._ph_rescue and filter_pose is not None:
                landmarks, handedness = self._rescue_missed_hands(
                    rgb, landmarks, handedness, filter_pose)

            if self._ph_filter and filter_pose is not None:
                landmarks, handedness, stats = pose_hand_filter.filter_hands(
                    landmarks, handedness, filter_pose,
                    max_match_dist=self._ph_match_dist,
                    arbitration_scale=self._ph_arb_scale)
                self._ph_stats["dropped"] += stats["dropped"]
                self._ph_stats["corrected"] += stats["corrected"]

            with self._frame_lock:
                self._pub_hand_landmarks = landmarks
                self._pub_handedness     = handedness
                if run_pose and pose_lm is not None:
                    self._pub_pose_lm   = pose_lm
                    self._pub_pose_time = now
                self._pub_seq += 1

    def _rescue_missed_hands(self, rgb, landmarks, handedness, pose_lm):
        """Pose sees a wrist Hands missed -> re-run Hands on a crop around the
        pose wrist and remap the result to frame coordinates (the MediaPipe
        Holistic trick, paid only on miss frames)."""
        wrists = pose_hand_filter.trusted_wrists(pose_lm)
        if not wrists:
            return landmarks, handedness
        hands = list(landmarks) if landmarks else []
        hnd   = list(handedness) if handedness else []
        h, w = rgb.shape[:2]
        for side, pw in wrists.items():
            near = any(
                pose_hand_filter._dist(hd.landmark[0].x, hd.landmark[0].y,
                                       pw.x, pw.y) <= self._ph_match_dist
                for hd in hands)
            if near:
                continue
            box = pose_hand_filter.wrist_crop_box(pose_lm, side, w, h)
            if box is None:
                continue
            if self._rescue_hands is None:
                self._rescue_hands = self._mp_hands.Hands(
                    static_image_mode=True, max_num_hands=1,
                    min_detection_confidence=0.5)
            x0, y0, x1, y1 = box
            crop = rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            result = self._rescue_hands.process(crop)
            if not result.multi_hand_landmarks:
                continue
            hands.append(pose_hand_filter.remap_crop_landmarks(
                result.multi_hand_landmarks[0], box, w, h))
            hnd.append(pose_hand_filter.synth_handedness(side))
            self._ph_stats["rescued"] += 1
        return (hands or None), (hnd or None)

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
        needs_dispatch = (has_hands
                          or self._runs_without_hands(self._active_cg)
                          or self._runs_without_hands(self._active_oi))
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

        oi_window_ms = self._active_oi_window_ms
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
        if not self._pose_needed():
            # Pose isn't running right now — stale data isn't misleading, but NONE is clearer.
            pose_status = "NONE"
        elif self._last_pose_lm and pose_age < self._pose_stale_s:
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
            "pose_hand": dict(self._ph_stats) if self._ph_filter else None,
            "point_dir": self._active_cg_context.get("dominant_direction"),
        }

    def _dispatch(self, interaction: dict, landmarks, context: dict) -> bool:
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
        # Body-relative detectors read context["_pose_lm"]; existing detectors ignore it.
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
        # while a pose-reading window is armed.
        fusion = PoseDepth(depth_sampler, pose_lm,
                           phantom_behind_mm=self._phantom_behind_mm)
        self._player_gated = False
        if fusion.available:
            torso = fusion.torso_depth_mm()
            if torso is not None and not (self._player_min_mm <= torso
                                          <= self._player_max_mm):
                self._player_gated = True
                pose_lm = None
                fusion = PoseDepth(depth_sampler, None,
                                   phantom_behind_mm=self._phantom_behind_mm)
        context["_pose_lm"] = pose_lm
        context["_pose_depth"] = fusion
        # Compute live pose-derived thresholds (brow line, shoulder height, mouth/ear
        # positions) so detection matches the tuner instead of using fixed values.
        params = _inject_pose_params(detector_type, params, pose_lm)
        # MediaPipe reports None (not []) when no hands are visible, and the no-hands
        # pose path dispatches anyway — normalise so detectors never see None.
        return detector_fn(landmarks or [], params, context)

    def close(self):
        # Stop the worker before tearing down the MediaPipe graphs / camera so the
        # worker can't call .process() on a closed graph or read a released cap.
        self._capture_running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        self._hands.close()
        self._pose.close()
        if self._rescue_hands is not None:
            self._rescue_hands.close()
