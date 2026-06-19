"""
main.py — Black Heritage Reclaimed entry point.

Boot sequence:
  1. Resolve host profile (CLI --profile flag → BHR_HOST_PROFILE env var → hostname → auto)
  2. Load config.json + merge host profile
  3. Init Pygame + display (fullscreen/windowed per profile)
  4. Wire up EventBus and all engines
  5. Run calibration (Scene 1 hand-tracking liveness check)
  6. Enter the main loop: process camera frame → update engines → render
"""

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import cv2
import pygame

from engines.event_bus import EventBus
from engines.gesture_engine import GestureEngine
from engines.voice_engine import VoiceEngine
from engines.render_engine import RenderEngine
from engines.sequence_loader import load_sequence
from engines.shot_sequence_player import ShotSequencePlayer, PLAYER_RUNNING
from engines.narration_adapter import NarrationAdapter


ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.json"
PROFILES_DIR = ROOT / "config" / "host_profiles"
SCENES_ROOT = ROOT / "scenes"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def resolve_profile_name(cli_name: str | None) -> str:
    """Return the profile name to load, following the precedence chain."""
    if cli_name:
        return cli_name
    if env := os.environ.get("BHR_HOST_PROFILE"):
        return env
    hostname = socket.gethostname().lower()
    for path in PROFILES_DIR.glob("*.json"):
        if path.stem.lower() in hostname or hostname in path.stem.lower():
            return path.stem
    return "auto"


def load_profile(name: str) -> dict:
    path = PROFILES_DIR / f"{name}.json"
    if not path.exists():
        fallback = PROFILES_DIR / "auto.json"
        print(f"[main] Profile '{name}' not found, falling back to auto.", file=sys.stderr)
        path = fallback
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def open_camera(profile: dict) -> cv2.VideoCapture:
    """Open camera using DirectShow backend and device-name matching per host profile."""
    cam_cfg = profile.get("camera", {})
    backend = cv2.CAP_DSHOW if cam_cfg.get("backend", "dshow") == "dshow" else cv2.CAP_ANY
    w, h = cam_cfg.get("resolution", [1280, 720])
    fps = cam_cfg.get("fps", 30)

    name_match = cam_cfg.get("device_name_match")
    fallback_match = cam_cfg.get("fallback_match")
    fallback_index = cam_cfg.get("fallback_index", 0)

    if name_match or fallback_match:
        cap = _open_by_name(name_match, fallback_match, fallback_index, backend, w, h, fps)
    else:
        cap = cv2.VideoCapture(fallback_index, backend)
        _set_cap_props(cap, w, h, fps)

    if not cap or not cap.isOpened():
        raise RuntimeError(f"[main] Failed to open camera (profile: {profile.get('profile_name')})")
    return cap


def _open_by_name(name: str | None, fallback_name: str | None, fallback_index: int,
                  backend: int, w: int, h: int, fps: int) -> cv2.VideoCapture:
    """Try each camera index until we find one whose DirectShow friendly name matches."""
    for idx in range(10):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        friendly = _get_dshow_friendly_name(idx)
        if name and name.lower() in friendly.lower():
            _set_cap_props(cap, w, h, fps)
            print(f"[main] Opened camera '{friendly}' at index {idx}", file=sys.stderr)
            return cap
        if fallback_name and fallback_name.lower() in friendly.lower():
            _set_cap_props(cap, w, h, fps)
            print(f"[main] Opened camera '{friendly}' (fallback match) at index {idx}", file=sys.stderr)
            return cap
        cap.release()

    # Last resort: use the numeric fallback index
    cap = cv2.VideoCapture(fallback_index, backend)
    _set_cap_props(cap, w, h, fps)
    print(f"[main] No name-matched camera found; using index {fallback_index}", file=sys.stderr)
    return cap


def _get_dshow_friendly_name(index: int) -> str:
    """Best-effort friendly name lookup for a camera index via cv2 backend property."""
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        name = cap.getBackendName() if cap.isOpened() else ""
        cap.release()
        return name
    except Exception:
        return ""


def _set_cap_props(cap: cv2.VideoCapture, w: int, h: int, fps: int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)


def init_display(profile: dict) -> pygame.Surface:
    display_cfg = profile.get("display", {})
    w, h = display_cfg.get("resolution", [1920, 1080])
    flags = pygame.FULLSCREEN if display_cfg.get("fullscreen", False) else 0
    return pygame.display.set_mode((w, h), flags)


def detector_test(detector_name: str, params: dict):
    """
    Live detector test mode.  Opens the camera, runs MediaPipe Hands + Pose,
    shows a skeleton overlay, and prints FIRE / waiting state for the named detector.
    Press Q or Escape to quit.

    Usage:
        python main.py --detector-test touch_head
        python main.py --detector-test arms_crossed
        python main.py --detector-test push_out --detector-params '{"min_growth_pct": 50}'
    """
    import json
    import math

    import mediapipe as mp
    import cv2 as _cv2

    from engines.detectors import REGISTRY

    det_fn = REGISTRY.get(detector_name)
    if det_fn is None:
        print(f"[detector-test] Unknown detector '{detector_name}'. Available: {sorted(REGISTRY)}")
        return

    profile_name = resolve_profile_name(None)
    profile = load_profile(profile_name)
    cam_cfg = profile.get("camera", {})
    backend = _cv2.CAP_DSHOW if cam_cfg.get("backend", "dshow") == "dshow" else _cv2.CAP_ANY
    w, h = cam_cfg.get("resolution", [1280, 720])
    cap = _cv2.VideoCapture(cam_cfg.get("fallback_index", 0), backend)
    cap.set(_cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, h)

    mp_hands = mp.solutions.hands
    mp_pose  = mp.solutions.pose
    mp_draw  = mp.solutions.drawing_utils

    hands = mp_hands.Hands(max_num_hands=2,
                            min_detection_confidence=0.6,
                            min_tracking_confidence=0.5)
    # model_complexity=1 matches CLAUDE.md and production host profile (was 0 — fixed)
    pose  = mp_pose.Pose(model_complexity=1, enable_segmentation=False,
                         min_detection_confidence=0.5, min_tracking_confidence=0.5)

    context: dict = {}
    last_pose_lm   = None
    last_pose_time = 0.0
    pose_stale_s   = 0.5
    fire_count     = 0
    fire_flash_until = 0.0

    print(f"[detector-test] Running '{detector_name}' | params={params} | Q/Esc to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)
        pose_results = pose.process(rgb)

        now = time.monotonic()
        if pose_results.pose_landmarks:
            last_pose_lm   = pose_results.pose_landmarks.landmark
            last_pose_time = now

        pose_age = now - last_pose_time
        context["_pose_lm"] = last_pose_lm if pose_age < pose_stale_s else None
        pose_status = ("OK" if context["_pose_lm"] else
                       f"STALE({pose_age:.1f}s)" if last_pose_lm else "NONE")

        hand_lm_list = hand_results.multi_hand_landmarks or []
        fired = det_fn(hand_lm_list, params, context)
        if fired:
            fire_count += 1
            fire_flash_until = now + 0.8
            # Reset one-shot fired flags so it can fire again in test mode
            for key in ("forward_reach_fired", "push_fired", "unravel_fired"):
                context.pop(key, None)

        # Draw hand skeleton
        if hand_lm_list:
            for hl in hand_lm_list:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        # Draw pose skeleton (key landmarks only)
        if last_pose_lm:
            POSE_DRAW = [11, 12, 15, 16, 23, 24, 7, 8]
            for idx in POSE_DRAW:
                lm = last_pose_lm[idx]
                cx, cy = int((1 - lm.x) * w), int(lm.y * h)  # mirrored
                _cv2.circle(frame, (cx, cy), 5, (80, 200, 80), -1)

        # FIRE flash overlay
        if now < fire_flash_until:
            overlay = frame.copy()
            _cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 80), -1)
            _cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
            _cv2.putText(frame, f"FIRED! (#{fire_count})",
                         (w // 2 - 140, h // 2),
                         _cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 80), 4, _cv2.LINE_AA)

        # Status panel
        status_color = (0, 255, 80) if now < fire_flash_until else (200, 200, 200)
        _cv2.putText(frame, f"detector: {detector_name}",
                     (12, 32), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        _cv2.putText(frame, f"POSE: {pose_status}",
                     (12, 60), _cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                     (0, 220, 80) if pose_status == "OK" else (60, 60, 220), 2)
        _cv2.putText(frame, f"fires: {fire_count}",
                     (12, 88), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        _cv2.putText(frame, "Q/Esc: quit",
                     (12, h - 14), _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

        _cv2.imshow(f"BHR detector-test: {detector_name}", frame)
        key = _cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    hands.close()
    pose.close()
    cap.release()
    _cv2.destroyAllWindows()


def _run_dry_run(config: dict) -> None:
    """
    Console-only dry-run: walk all 78 shots through ShotSequencePlayer.

    No camera, no display, no audio. Every shot with a TODO interaction
    auto-advances via the 200ms fallback timeout.  Shows a summary table
    on exit.

    Usage:
        py -3.12 main.py --dry-run
    """
    shots = load_sequence(SCENES_ROOT, config)
    bus   = EventBus()

    # Shorten every HOLD timeout so the run completes in under a second
    DRY_TIMEOUT_S = 0.2
    for s in shots:
        s.fallback.update({"timeout_s": DRY_TIMEOUT_S, "reprompt_s": []})

    # Collect events for summary
    holds_entered: list[str]  = []
    cg_windows:    list[str]  = []
    vi_steps:      list[str]  = []

    bus.subscribe("shot_state_change",
                  lambda d: holds_entered.append(d["shot_id"]) if d["state"] == "HOLD" else None)
    bus.subscribe("cg_window_open",
                  lambda d: cg_windows.append(d["interaction"].get("id", "?")))
    bus.subscribe("vi_chain_step",
                  lambda d: vi_steps.append(d["step"].get("keyword", "?")))

    player = ShotSequencePlayer(shots, config, bus)
    player.start()

    max_iter = 100_000
    for _ in range(max_iter):
        if player._player_state != PLAYER_RUNNING:
            break
        player.update()

    interactive = [s for s in shots if s.kind == "interactive"]
    todo        = [s for s in shots if s.interaction_todo]

    print()
    print("=" * 62)
    print("BHR DRY-RUN COMPLETE")
    print(f"  {len(shots)} shots  ({len([s for s in shots if s.kind == 'playback'])} playback, "
          f"{len(interactive)} interactive)")
    print(f"  {len(todo)}/{len(interactive)} interactive shots still TODO interaction")
    print(f"  {len(holds_entered)} HOLD states entered")
    print(f"  {len(cg_windows)} CG windows opened: {cg_windows or '--'}")
    print(f"  {len(vi_steps)} VI chain steps armed: {vi_steps or '--'}")
    print(f"  Final state: {player._player_state}")
    print("=" * 62)

    if interactive:
        print("\nInteractive shots (all acts):")
        for s in interactive:
            status = "TODO" if s.interaction_todo else "wired"
            print(f"  shot {s.shot}  act {s.act}  [{status}]  timing={s.timing_profile}"
                  + ("  assets_ok" if not s.assets_pending else ""))


def main():
    parser = argparse.ArgumentParser(description="Black Heritage Reclaimed")
    parser.add_argument("--profile", metavar="NAME", help="Host profile name")
    parser.add_argument("--start-shot", metavar="SHOT", default=None,
                        help="Shot number to start at (e.g. 09). Skips all earlier shots.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Console-only: walk all 78 shots through ShotSequencePlayer "
                             "with no camera, display, or audio, then exit")
    parser.add_argument("--detector-test", metavar="DETECTOR",
                        help="Run live detector test for the named detector and exit")
    parser.add_argument("--detector-params", metavar="JSON", default="{}",
                        help="JSON params dict for --detector-test (default: {})")
    args = parser.parse_args()

    if args.dry_run:
        _run_dry_run(load_config())
        return

    if args.detector_test:
        import json as _json
        try:
            params = _json.loads(args.detector_params)
        except Exception:
            params = {}
        detector_test(args.detector_test, params)
        return

    profile_name = resolve_profile_name(args.profile)
    profile = load_profile(profile_name)
    config = load_config()
    config["_profile"] = profile

    print(f"[main] Using host profile: {profile.get('profile_name', profile_name)}", file=sys.stderr)

    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

    bus = EventBus()

    render = RenderEngine(config, bus)
    render.init_display()

    gesture = GestureEngine(config, bus)
    voice   = VoiceEngine(config, bus)

    shots             = load_sequence(SCENES_ROOT, config)
    player            = ShotSequencePlayer(shots, config, bus)
    narration_adapter = NarrationAdapter(config, bus, shots)

    # Start the continuous look-ahead frame cache: preloads every shot with art in
    # the background (forward from the current shot) so future shots are fully ready.
    render.attach_cache(shots)

    start_index = 0
    if args.start_shot:
        target = args.start_shot.zfill(2)
        ids = [s.shot for s in shots]
        if target in ids:
            start_index = ids.index(target)
            print(f"[main] Starting at shot {target} (index {start_index})", file=sys.stderr)
        else:
            print(f"[main] --start-shot {target!r} not found in sequence; starting from shot 01",
                  file=sys.stderr)
    player.start(start_index)

    voice.start()

    cap = open_camera(profile)
    # Camera capture + MediaPipe inference run on their own thread; the render loop
    # below only consumes the latest landmarks and never blocks on detection.
    gesture.start_capture(cap)
    render_fps = profile.get("performance", {}).get("render_fps_cap", 30)
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _shutdown(cap, gesture, voice)
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                _shutdown(cap, gesture, voice)
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                player._advance()

        gesture.update()
        player.update()
        narration_adapter.update()

        render.update(
            landmark_data=gesture._last_landmarks,
            handedness_data=gesture._last_handedness,
            pose_data=gesture._last_pose_lm,
            gesture_debug=gesture.debug_info(),
            scene_debug=None,
            voice_debug=voice.debug_info(),
            narration_debug=player.debug_info(),
        )

        clock.tick(render_fps)


def _shutdown(cap, gesture, voice):
    voice.stop()
    gesture.close()
    cap.release()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
