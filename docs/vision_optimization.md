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

Every z-approach gesture originally used hand-bbox growth as a monocular depth
proxy. It works but is scale-confounded (a hand moving sideways toward the camera
edge grows too) and fails when Hands loses the hand.

Adapter: `engines/depth/orbbec_camera.py` (WIRED July 2026 — color+aligned depth,
`depth_mm_at` sampler). SDK: `pyorbbecsdk2` (import name `pyorbbecsdk`), UVC on USB3.

## Depth fusion layer (BUILT, July 2026) — `engines/depth/fusion.py`

The question was raised: *can we train a model that uses both depth and MediaPipe
to detect hands/pose better?* Short answer: **retraining the landmark models is the
wrong tool for this project; fusing real depth with MediaPipe's outputs at the
detector layer captures most of the benefit at ~zero CPU cost.** That fusion is now
built and wired. Details below, training analysis after.

`PoseDepth` (constructed per dispatch tick by `gesture_engine._dispatch` and the
tuner loop, injected as `context["_pose_depth"]`) gives every detector five fused
capabilities:

1. **Lift** — `landmark_mm(idx)`: real millimetre depth sampled at any Pose
   landmark (median of a 5px patch, lazily cached per frame).
2. **Veto** — `plausible(idx)`: MediaPipe reports positions for occluded/
   out-of-frame joints ("phantom landmarks"); sampling depth at those spots hits
   the BACKGROUND, metres behind the torso. Landmarks whose depth sits >
   `phantom_behind_mm` (400) behind the torso plane are vetoed. This stacks with
   the visibility gate via `trusted_landmark()` — used by `point_region`,
   `survey`, `throw`, `directional_draw`, `speed_bilateral`. Design rule:
   **depth vetoes, never rescues** — false positives stay strictly below the
   visibility-only baseline, and everything degrades to the old behaviour on a
   plain webcam.
3. **Meter** — `reach_mm(idx)`: how far a joint reaches toward the camera from
   the torso plane. `forward_point` uses it for hands aimed at the camera;
   `throw` (`require_growth`), `push_out` and `forward_reach` now use REAL depth
   deltas (`min_depth_delta_mm`) instead of bbox growth whenever depth flows —
   and when depth is flowing it is authoritative (the bbox proxy is suppressed).
4. **Metric velocity** — `metric_point()` converts (normalised x, y, depth) to
   millimetres via the G335 FOV; `speed_bilateral` measures hand speed in
   **mm/s** (`min_speed_mm_s`, default 800). The same physical shake now fires
   identically at 1m and 3m from the camera — normalised velocity thresholds
   silently favoured close-standing players.
5. **Player band gate** — `torso_depth_mm()` is the tracked person's distance;
   the gesture engine drops poses whose torso is outside
   `config.depth.player_min_mm..player_max_mm` (500–3200). A passer-by 4m behind
   the visitor can no longer steer detections — the cheapest, most reliable form
   of "person segmentation" for a public installation.

Config: `config.json` `"depth": { player_min_mm, player_max_mm, phantom_behind_mm }`.
The tuner shows a live `DEPTH torso NNNNmm reach L+NNN R+NNN` badge when the
Gemini is the capture device.

## Can we train a fused RGB-D hand/pose model? (analysis)

Ranked by feasibility for this project:

- **Retrain/fine-tune MediaPipe on RGB-D — not possible.** The Hands/Pose models
  ship as closed, frozen TFLite graphs; there is no upstream training pipeline for
  them, and their input is strictly 3-channel RGB.
- **Train a new RGB-D landmark model from scratch — not worth it here.** Serious
  RGB-D hand-pose models (research lineage: DeepPrior++, A2J, HandFoldingNet on
  ICVL/NYU/BigHand2.2M) need 100k+ annotated frames, GPU training, and would land
  on our CPU-only Ryzen at a fraction of MediaPipe's XNNPACK-optimised speed. We'd
  trade a solved problem (landmarks) for an unsolved one (our budget).
- **Train a small fused GESTURE classifier — feasible, and the right "phase 2" if
  rule-based fusion ever hits a ceiling.** Feature vector per frame: 33 pose
  landmarks (x, y, visibility) + per-landmark fused depth + torso-relative reach
  from `PoseDepth` — ~150 floats. A gradient-boosted tree or tiny MLP over a
  ~0.5s window classifies gesture/no-gesture per interaction. Trainable on the
  mini PC itself with sklearn from data recorded during tuner sessions (label =
  which gesture you were performing). The existing per-gesture
  `detector_authority` / `shadow_layer` mechanism means such a classifier can run
  in shadow mode against the rule-based layer on-site and be promoted per gesture
  by config flip — zero code risk. Do this only for gestures whose shadow logs
  show the rules underperforming.
- **MediaPipe GestureRecognizer + Model Maker** (RGB only) remains the low-risk
  Layer A upgrade — see the section above.

## Remaining Gemini 335 capabilities on the table

- **IR stream for dark venues.** The G335 has two IR imagers with active
  illumination; the left IR image is registered to depth. If exhibition lighting
  (projection wash, dim room) degrades RGB landmarks, MediaPipe can be fed the IR
  image replicated to 3 channels — landmarks come out in the same normalised
  space. Worth a 30-minute experiment on-site with `enable_stream(IR)`; wire as a
  `use_ir_for_landmarks` fallback flag only if RGB proves unreliable.
- ~~Depth-ROI hand rescue~~ — **superseded by the pose-guided version, BUILT
  July 2026** (see next section). The pose wrist is a better ROI anchor than a
  depth blob: it's already tracked, side-labelled, and works on plain webcams.

## Pose-hand fusion layer (BUILT, July 2026)

`engines/pose_hand_filter.py`, wired into the gesture engine's capture loop
(config `"pose_hand"`). MediaPipe Holistic's pose→hand coupling, à la carte,
without Holistic's CPU cost:

- **Phantom-hand veto** — a Hands detection matching no trusted pose wrist is
  dropped, but only when BOTH wrists are trusted (full skeleton). Passer-by
  hands and face/pattern false positives stop reaching detectors. With the
  depth player-band gating the pose, this extends passer-by protection to all
  hand gestures.
- **Handedness from pose** — matched hands take Left/Right from the pose side
  (landmark 15/16), replacing the Hands classifier's guess (unreliable when
  mirrored). Fixes cursor L/R dots and anything reading handedness.
- **Stale-track arbitration** — a matched hand lagging more than
  `arbitration_scale` × its own bbox size behind the pose wrist is a
  motion-blur ghost; dropped for the frame (the regime where `throw` already
  ignores Hands).
- **Pose-guided hand rescue** (`"rescue": true`, default off) — when Pose sees
  a wrist Hands missed, a second Hands inference (static, max 1 hand) runs on
  a crop centred on the pose wrist, sized from the forearm length, and the
  landmarks are remapped to frame space. Costs one 192×192 inference only on
  miss frames — flip on during on-site testing and watch the `pose_hand`
  counters in `debug_info()`.
- Pose now runs at a throttled cadence (`pose_every_n`, default every 2nd
  frame) even during hands-only holds so the skeleton is available for the
  filter; full rate whenever a pose-reading detector is armed.
- **Pose vetoes, never rescues** (rescue is explicit opt-in): with no fresh
  pose the Hands output passes through byte-for-byte. Contract pinned by
  `tests/test_pose_hand_filter.py`.
- **Per-pixel person mask** (depth threshold at torso ± 500mm) for compositing
  the player's silhouette into scenes — a render-side effect, not a detection
  need.

## Decision summary

| Action | When | Risk |
|---|---|---|
| Downscale Pose input, alternate Pose frames, symmetric Hands gating | anytime, cheap | low |
| Tasks API migration behind `vision_backend` profile flag | post-wiring, pre-launch calibration | medium |
| GestureRecognizer for Layer A | only if GRLib shadow logs disappoint | low |
| ONNX+DirectML pose offload | only if budget blows on-site | medium-high |
| ~~Orbbec depth for z-gestures~~ | **DONE July 2026** — fusion layer wired | — |
| ~~Pose-hand fusion (veto/handedness/arbitration)~~ | **DONE July 2026** — on by default, `"pose_hand"` config | — |
| Pose-guided hand rescue | built; flip `"pose_hand".rescue` on-site, watch fps | low |
| Learned fused gesture classifier (sklearn, shadow mode) | only if rule fusion hits a ceiling on-site | low |
| IR-stream landmark fallback | on-site experiment if lighting hurts RGB | low |
