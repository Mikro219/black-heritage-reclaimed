# Vision Pipeline — Optimization Notes & Alternatives

Status: analysis / decision record, June 2026. Companion to `CLAUDE.md` (Tech Stack,
Performance Budget). Nothing here is wired in yet except where marked DONE.

## Where we are

- **mediapipe 0.10.14**, legacy *Solutions* API (`mp.solutions.hands` / `mp.solutions.pose`),
  CPU (XNNPACK), `model_complexity=1` for both, one capture+inference worker thread
  (`gesture_engine._capture_loop`), Pose gated to shots that need it (`_pose_needed()`).
- Budget target (CLAUDE.md): Hands ~15–25% of one core at 30fps, Pose on the same
  thread, on a Ryzen 7 7640HS (mini PC) with no discrete GPU.

The pipeline is healthy today. The items below are ordered by payoff-per-risk.

## Quick wins inside the current stack

1. **Feed Pose a downscaled frame.** Both graphs currently get the full 1280×720 RGB
   frame. MediaPipe resizes internally (Pose detector runs at 224×224, Hands at
   192×192), but the resize of a big frame is our cost. Passing Pose a
   `cv2.resize(rgb, (640, 360))` copy cuts its preprocessing roughly 4× with no
   accuracy loss that matters at exhibition distance (landmarks are normalized, so
   nothing downstream changes).
2. **Alternate Pose frames.** During HOLDs where both graphs run, running Pose every
   2nd frame (15Hz) is imperceptible for the body-scale gestures we detect (500ms+
   holds, multi-second strokes) and halves the Pose cost. The 500ms staleness window
   (`_pose_stale_s`) already tolerates this.
3. **Skip Hands during pose-only holds.** `_pose_needed()` gates Pose, but Hands runs
   always. Shots whose armed detectors are all in `_NO_HANDS_DETECTORS` (run_arms,
   paddle, point_region holds…) could skip the Hands graph symmetrically. Worth ~15%
   of a core during those holds.
4. **`static_image_mode` stays False** (it is) and confidence thresholds stay at
   0.6/0.5 — raising them to reduce jitter costs re-detection storms; the per-detector
   visibility/hold gates added in June are the right layer for jitter.

## Migrating to the MediaPipe Tasks API (recommended mid-term)

The legacy Solutions API we use has been deprecated upstream since 2023 (still ships,
frozen). The replacement is **MediaPipe Tasks** (`mediapipe.tasks.python.vision`):
`HandLandmarker`, `PoseLandmarker`, `GestureRecognizer` with downloadable `.task`
bundles.

Why bother:
- **LIVE_STREAM mode** gives an async callback pipeline with built-in frame-drop
  behaviour — closer to what `_capture_loop` hand-rolls today.
- **Model choice per task**: `pose_landmarker_lite/full/heavy.task` — lite is faster
  than legacy complexity-1 at similar accuracy for large subjects (our case: one
  visitor, full body, close range).
- Legacy API gets no fixes; Tasks is where upstream performance work lands.

Cost: moderate. The landmark output shape is the same 21/33-point scheme, but result
objects differ (`hand_landmarks[i][j].x` instead of `.landmark[j].x`, visibility →
`presence`/`visibility` fields on pose only), so `shared_landmarks`-style adaptation
is needed at the `_capture_loop` boundary — detectors themselves can stay untouched
if we adapt to the legacy shape there. Keep `gesture_tuner.py` in sync (CLAUDE.md
rule: tuner and engine must match).

**Recommendation:** do this as one focused PR after launch-critical wiring is done,
behind a `vision_backend: "legacy" | "tasks"` host-profile flag so the mini PC can be
A/B'd on-site.

## GestureRecognizer (Tasks) as a Layer A upgrade

`GestureRecognizer` ships canned gestures (open palm, closed fist, pointing up,
thumbs up/down, victory, ILY) and supports custom training via **Model Maker** with a
few hundred images per class. That maps well onto our Layer A (GRLib) vocabulary —
directional points, open-palm presence, fist closure.

- If GRLib shadow logs underperform on-site, this is the drop-in classifier to try
  before hand-rolling more Layer B rules: same MediaPipe runtime, no new dependency.
- The authority-flip mechanism in shot metadata (`detector_authority`) already
  supports swapping classifiers without code changes.

## Offloading to the iGPU (only if CPU pressure appears)

The Radeon 760M sits idle today. Two practical routes on Windows:

- **ONNX Runtime + DirectML EP**: run a pose model (MoveNet Lightning ~192×192, very
  fast; or RTMPose-t/s for better accuracy) as ONNX on the iGPU. This removes Pose
  from the CPU budget entirely. Hands stays on MediaPipe.
- MediaPipe's own GPU delegate is effectively unavailable for Python-on-Windows;
  don't plan around it.

Cost: new dependency, new landmark schema (COCO 17-keypoint for MoveNet/RTMPose vs
MediaPipe's 33) — our detectors use shoulders/wrists/hips which all exist in COCO,
but `_inject_pose_params` uses eyes/mouth landmarks for brow/mouth lines and would
need per-schema mapping.

**Recommendation:** hold in reserve. Only reach for this if on-site profiling shows
the combined pipeline blowing the budget (watch for: render fps dips during Scene 10
urgent shots where Pose + Hands + Vosk all run).

## Depth camera (Orbbec Gemini 335) — the real z axis

Every z-approach gesture today (`forward_reach`, `push_out`, `throw`) uses hand-bbox
growth as a monocular depth proxy. It works but is scale-confounded (a hand moving
sideways toward the camera edge grows too) and fails when Hands loses the hand.

With the Gemini 335's aligned depth stream:
- z-gestures read **real wrist depth in mm** (e.g. throw release = 250mm approach
  within 400ms) — far more robust and tunable in physical units.
- Background subtraction by depth (visitor vs. passers-by behind them) becomes
  trivial, which matters for a public installation.
- MediaPipe keeps doing what it's good at (landmarks on the color stream); depth is
  sampled at landmark coordinates. No model changes.

Adapter code (not yet wired): `engines/depth/orbbec_camera.py`. See its docstring
for the integration plan. SDK: `pyorbbecsdk` (Orbbec SDK v2), UVC on USB3.

## Decision summary

| Action | When | Risk |
|---|---|---|
| Downscale Pose input, alternate Pose frames, symmetric Hands gating | anytime, cheap | low |
| Tasks API migration behind `vision_backend` profile flag | post-wiring, pre-launch calibration | medium |
| GestureRecognizer for Layer A | only if GRLib shadow logs disappoint | low |
| ONNX+DirectML pose offload | only if budget blows on-site | medium-high |
| Orbbec depth for z-gestures | when hardware arrives (adapter ready) | medium |
