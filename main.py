"""
main.py — Black Heritage Reclaimed entry point.

Boot sequence:
  1. Load config.json (the "host" section carries the hardware binding —
     camera/mic/display/performance + on-site tuning; formerly config/host_profiles/)
  2. Init Pygame + display (fullscreen/windowed per config["host"]["display"])
  4. Wire up EventBus and all engines
  5. Run calibration (Scene 1 hand-tracking liveness check)
  6. Enter the main loop: process camera frame → update engines → render
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import pygame

from engines.event_bus import EventBus
from engines.gesture_engine import GestureEngine
from engines.voice_engine import VoiceEngine
from engines.render_engine import RenderEngine
from engines.sequence_loader import load_sequence
from engines.shot_sequence_player import ShotSequencePlayer, PLAYER_RUNNING, PLAYER_FINAL
from engines.narration_adapter import NarrationAdapter
from engines.audio_mixer import ShotAudioMixer
from engines.tutorial_engine import TutorialEngine


class MetaVoice:
    """Long-lived voice commands outside shot wiring: "skip" during the
    tutorial / prologue / epilogue, "ready" on the camera-setup screen.

    The voice engine expires windows and clears them all on every input_lock,
    so ensure() re-opens the wanted window whenever it vanished. Fired events
    arrive on the voice thread; they queue here and main's loop drains them.
    """

    def __init__(self, voice: VoiceEngine):
        self._voice = voice
        self._events: deque[str] = deque()
        self._wid = None
        self._desc = None
        voice.subscribe(self._on_event)

    def _on_event(self, ev) -> None:                # voice thread
        if str(ev.vi_id).startswith("meta_"):
            self._events.append(ev.vi_id)

    def ensure(self, desc) -> None:
        """desc: (vi_id, [keywords]) to keep open, or None for no meta window."""
        if desc == self._desc and (desc is None or self._voice.window_open(self._wid)):
            return
        if self._wid:
            self._voice.close_window(self._wid)
            self._wid = None
        self._desc = desc
        if desc is not None:
            vi_id, keywords = desc
            self._wid = self._voice.open_window({
                "id": vi_id, "keywords": keywords, "mode": "keyword",
                "tier": "reaction", "window_ms": 3_600_000,
            })

    def drain(self):
        while self._events:
            yield self._events.popleft()


# Frozen (PyInstaller) builds: __file__ points inside the bundled _internal dir,
# so resolve ROOT to the folder holding BHR.exe — config.json, scenes/,
# models/ and assets/ are deployed next to the exe (see scripts/build_exe.py).
if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).parent
    SCENES_ROOT = ROOT / "scenes"                    # staged by build_exe.py
else:
    ROOT = Path(__file__).parent
    # The .bhrx export is the single source of truth (Aug 2026): dev runs
    # default to the generated tree (scripts/export_experience.py output).
    SCENES_ROOT = ROOT / "export" / "generated"
CONFIG_PATH = ROOT / "config.json"

# Report any main-loop iteration slower than this, broken down by phase.
# A frozen picture with the audio running on is always a stall here.
try:
    LOOP_SLOW_MS = float(os.environ.get("BHR_SLOW_MS", "60")) * 2.5
except ValueError:
    LOOP_SLOW_MS = 150.0


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def host_profile(config: dict) -> dict:
    """The hardware-binding section of config.json (camera/mic/display/
    performance + on-site tuning). Formerly config/host_profiles/<name>.json."""
    return config.get("host", {})


def open_camera(profile: dict, config: dict | None = None):
    """Open the capture device.

    When config["use_orbbec_camera"] is true, the Orbbec Gemini 335 is opened
    first (color + aligned depth via engines.depth); if the SDK or camera is
    unavailable it falls back to the webcam path below, so the same build runs
    on the dev laptop without the depth camera attached.
    """
    if config and config.get("use_orbbec_camera"):
        from engines.depth.orbbec_camera import try_open_orbbec
        cam_cfg = profile.get("camera", {})
        w, h = cam_cfg.get("resolution", [1280, 720])
        cap = try_open_orbbec(resolution=(w, h), fps=cam_cfg.get("fps", 30))
        if cap is not None:
            print("[main] Using Orbbec Gemini 335 (color + depth)", file=sys.stderr)
            return cap
        print("[main] Orbbec unavailable — falling back to webcam", file=sys.stderr)

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
        raise RuntimeError("[main] Failed to open camera (check config.json \"host\" camera section)")
    return cap


def _enumerate_camera_names() -> list[str]:
    """DirectShow device friendly names, index-aligned with cv2 CAP_DSHOW.

    Needs the optional `pygrabber` package (pip install pygrabber) — OpenCV
    itself cannot report device names. Returns [] when unavailable."""
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore
        return FilterGraph().get_input_devices()
    except Exception:
        return []


def _open_by_name(name: str | None, fallback_name: str | None, fallback_index: int,
                  backend: int, w: int, h: int, fps: int) -> cv2.VideoCapture:
    """Resolve the camera index by DirectShow friendly name, then open it ONCE."""
    names = _enumerate_camera_names()
    idx = None
    if names:
        for want in (name, fallback_name):
            if not want:
                continue
            for i, friendly in enumerate(names):
                if want.lower() in friendly.lower():
                    idx = i
                    print(f"[main] Camera '{friendly}' matched {want!r} at index {i}",
                          file=sys.stderr)
                    break
            if idx is not None:
                break
        if idx is None:
            print(f"[main] No camera name matched {name!r}/{fallback_name!r} "
                  f"among {names} — using index {fallback_index}", file=sys.stderr)
    else:
        print("[main] Camera name matching unavailable (pip install pygrabber) "
              f"— using index {fallback_index}", file=sys.stderr)

    cap = cv2.VideoCapture(fallback_index if idx is None else idx, backend)
    _set_cap_props(cap, w, h, fps)
    return cap


def _set_cap_props(cap: cv2.VideoCapture, w: int, h: int, fps: int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)


def detector_test(detector_name: str, params: dict):
    """
    Live detector test mode.  Opens the camera, runs MediaPipe Pose,
    shows a skeleton overlay, and prints FIRE / waiting state for the named detector.
    Press Q or Escape to quit.

    Usage:
        python main.py --detector-test touch_head
        python main.py --detector-test arms_crossed
        python main.py --detector-test push_out --detector-params '{"window_ms": 400}'
    """
    import mediapipe as mp
    import cv2 as _cv2

    from engines.detectors import REGISTRY

    det_fn = REGISTRY.get(detector_name)
    if det_fn is None:
        print(f"[detector-test] Unknown detector '{detector_name}'. Available: {sorted(REGISTRY)}")
        return

    config = load_config()
    profile = host_profile(config)
    cam_cfg = profile.get("camera", {})
    w, h = cam_cfg.get("resolution", [1280, 720])
    # Same camera selection as the main runtime (Orbbec first when enabled).
    cap = open_camera(profile, config)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False,
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
        pose_results = pose.process(rgb)

        now = time.monotonic()
        if pose_results.pose_landmarks:
            last_pose_lm   = pose_results.pose_landmarks.landmark
            last_pose_time = now

        pose_age = now - last_pose_time
        context["_pose_lm"] = last_pose_lm if pose_age < pose_stale_s else None
        context["_depth_mm_at"] = getattr(cap, "depth_mm_at", None)
        pose_status = ("OK" if context["_pose_lm"] else
                       f"STALE({pose_age:.1f}s)" if last_pose_lm else "NONE")

        fired = det_fn([], params, context)
        if fired:
            fire_count += 1
            fire_flash_until = now + 0.8
            # Reset one-shot fired flags so it can fire again in test mode
            for key in ("forward_reach_fired", "push_fired", "unravel_fired"):
                context.pop(key, None)

        # Draw pose skeleton (key landmarks incl. the pose hand points 17-22)
        if last_pose_lm:
            POSE_DRAW = [11, 12, 15, 16, 23, 24, 7, 8, 17, 18, 19, 20, 21, 22]
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

    pose.close()
    cap.release()
    _cv2.destroyAllWindows()


def _run_prewarm(config: dict, scenes_root: Path) -> None:
    """Build every shot's framecache.npy pack at the display resolution, then
    return. Ship this + run once on a fresh machine (`BHR.exe --prewarm`) so
    the first real run has no cold-JPEG-decode stutter; packs persist on disk.

    Mirrors scripts/prewarm_frame_cache.py but resolves ROOT/resolution the way
    the app does, so it works inside the frozen exe (which has no scripts/)."""
    from engines.frame_cache import FrameCacheManager
    display_cfg = config.get("_profile", {}).get("display", {})
    res = display_cfg.get("resolution") or config.get("resolution", [1920, 1080])
    w, h = int(res[0]), int(res[1])
    frame_dirs = sorted(d for d in (list(scenes_root.glob("scenes/scene_*/frames"))
                                    + list(scenes_root.glob("act_*/shot_*/frames")))
                        if d.is_dir())
    if not frame_dirs:
        print(f"[prewarm] no scenes/scene_*/frames under {scenes_root}")
        return
    total = sum(1 for d in frame_dirs for _ in d.glob("*.jpg"))
    print(f"[prewarm] {len(frame_dirs)} shots ({total} frames, "
          f"~{total * w * h * 3 / 1e9:.1f} GB of packs) at {w}x{h}", flush=True)
    cache = FrameCacheManager((w, h))
    built = skipped = failed = 0
    t0 = time.monotonic()
    for i, d in enumerate(frame_dirs, 1):
        shot = d.parent.name
        paths = cache._ensure_paths(d)
        if not paths:
            continue
        pack = cache._pack_path(d)
        if pack.exists() and cache._pack_valid(pack, len(paths)):
            skipped += 1
            continue
        print(f"[prewarm] {i}/{len(frame_dirs)} building {shot} "
              f"({len(paths)} frames)...", flush=True)
        if cache._build_pack(pack, paths):
            built += 1
        else:
            failed += 1
            print(f"[prewarm] {shot}: pack build FAILED")
    print(f"[prewarm] done in {time.monotonic() - t0:.0f}s — {built} built, "
          f"{skipped} already valid, {failed} failed", flush=True)


def _run_dry_run(config: dict, scenes_root=None) -> None:
    """
    Console-only dry-run: walk all 78 shots through ShotSequencePlayer.

    No camera, no display, no audio. Every shot with a TODO interaction
    auto-advances via the 200ms fallback timeout.  Shows a summary table
    on exit.

    Usage:
        py -3.12 main.py --dry-run
        py -3.12 main.py --dry-run --scenes export/scenes_generated
    """
    shots = load_sequence(scenes_root or SCENES_ROOT, config)
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
    # interaction=None is the FSM's "disarm" signal — not a window opening.
    bus.subscribe("cg_window_open",
                  lambda d: cg_windows.append(d["interaction"].get("id", "?"))
                  if d.get("interaction") else None)
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
    parser.add_argument("--start-shot", metavar="SHOT", default=None,
                        help="Shot number to start at (e.g. 09). Skips all earlier shots. "
                             "On an exported tree with a shot_map.json, this is the "
                             "ORIGINAL master-timeline shot number (tracker 01-79) and "
                             "seeks into the exported shot that contains it.")
    parser.add_argument("--scenes", metavar="DIR", default=None,
                        help="Alternate scenes root (a folder containing sequence.json). "
                             "Default: export/generated (the .bhrx export), or the "
                             "staged scenes/ folder in a frozen build.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Console-only: walk all 78 shots through ShotSequencePlayer "
                             "with no camera, display, or audio, then exit")
    parser.add_argument("--prewarm", action="store_true",
                        help="Build every shot's frame-cache pack at the display "
                             "resolution, then exit (run once on a fresh machine "
                             "to remove first-run stutter; packs persist on disk)")
    parser.add_argument("--detector-test", metavar="DETECTOR",
                        help="Run live detector test for the named detector and exit")
    parser.add_argument("--detector-params", metavar="JSON", default="{}",
                        help="JSON params dict for --detector-test (default: {})")
    args = parser.parse_args()

    scenes_root = SCENES_ROOT
    if args.scenes:
        scenes_root = Path(args.scenes).resolve()
        if not (scenes_root / "sequence.json").exists():
            sys.exit(f"[main] --scenes {scenes_root} has no sequence.json — "
                     f"point it at a scenes root (e.g. export/scenes_generated)")
        print(f"[main] Using scenes root: {scenes_root}", file=sys.stderr)

    if args.dry_run:
        _run_dry_run(load_config(), scenes_root)
        return

    if args.detector_test:
        import json as _json
        try:
            params = _json.loads(args.detector_params)
        except Exception:
            params = {}
        detector_test(args.detector_test, params)
        return

    config = load_config()
    profile = host_profile(config)
    config["_profile"] = profile
    if not profile:
        print("[main] WARNING: config.json has no \"host\" section — "
              "camera/mic/display fall back to built-in defaults", file=sys.stderr)

    if args.prewarm:
        _run_prewarm(config, scenes_root)
        return

    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

    bus = EventBus()

    render = RenderEngine(config, bus)
    render.init_display()

    gesture = GestureEngine(config, bus)
    voice   = VoiceEngine(config, bus)

    shots             = load_sequence(scenes_root, config)
    player            = ShotSequencePlayer(shots, config, bus)
    narration_adapter = NarrationAdapter(config, bus, shots)
    audio_mixer       = ShotAudioMixer(config, bus,
                                       frame_provider=lambda: render.playback_frame)
    tutorial          = TutorialEngine(config, bus, gesture)

    # Start the continuous look-ahead frame cache: preloads every shot with art in
    # the background (forward from the current shot) so future shots are fully ready.
    render.attach_cache(shots)

    # Exported trees carry a shot_map.json: original master-timeline shot
    # numbers -> exported shot id + local frame (many original shots share one
    # exported stretch shot). Enables --start-shot by original number and
    # skip-prologue even though the prologue is baked into the first shot.
    shot_map = {}
    map_path = scenes_root / "shot_map.json"
    if map_path.exists():
        try:
            with open(map_path, encoding="utf-8") as f:
                shot_map = json.load(f).get("shots", {})
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[main] shot_map.json unreadable ({exc}) — ignored", file=sys.stderr)

    ids = [s.shot for s in shots]
    if shot_map:
        prologue_end = next(
            (e for _, e in sorted(shot_map.items())
             if e.get("act") != "act_00_prologue" and e.get("shot") in ids), None)
        if prologue_end:
            player.set_prologue_end(ids.index(prologue_end["shot"]),
                                    int(prologue_end.get("frame") or 1))

    start_index = 0
    start_frame = None
    if args.start_shot:
        target = args.start_shot.zfill(2)
        entry = shot_map.get(target)
        if entry and entry.get("shot") in ids:
            start_index = ids.index(entry["shot"])
            start_frame = int(entry.get("frame") or 1)
            print(f"[main] Starting at original shot {target} -> exported shot "
                  f"{entry['shot']} frame {start_frame}", file=sys.stderr)
        elif target in ids:
            start_index = ids.index(target)
            print(f"[main] Starting at shot {target} (index {start_index})", file=sys.stderr)
        else:
            print(f"[main] --start-shot {target!r} not found in sequence; starting from shot 01",
                  file=sys.stderr)
    player.start(start_index, start_frame)

    voice.start()

    cap = open_camera(profile, config)
    # Camera capture + MediaPipe inference run on their own thread; the render loop
    # below only consumes the latest landmarks and never blocks on detection.
    gesture.start_capture(cap)
    render_fps = profile.get("performance", {}).get("render_fps_cap", 30)
    clock = pygame.time.Clock()

    # Boot into the paused state — the experience waits on a PAUSED screen until
    # an attendant presses Space/P to begin.
    paused = True
    tutorial_was_active = False
    player.pause()
    render.pause()
    gesture.pause()
    pygame.mixer.pause()

    # Meta voice commands ("skip" / "ready") + camera-setup screen + end loop.
    meta_voice = MetaVoice(voice)
    camera_setup_active = bool(config.get("camera_setup", {}).get("enabled", False))
    end_loop_cfg = config.get("end_loop", {})
    final_since = None   # when the player entered FINAL_ADDRESS (for the loop)
    if camera_setup_active:
        print("[main] camera setup — frame your body, then ENTER or say 'ready'")
    print("[main] started PAUSED — press Space to begin")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _shutdown(cap, gesture, voice, render)
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                _shutdown(cap, gesture, voice, render)
                return
            if camera_setup_active and event.type == pygame.KEYDOWN and \
                    event.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_p):
                camera_setup_active = False
                print("[main] camera setup confirmed — press Space to begin")
                continue
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_p):
                # Toggle pause/play. Freezes frames, audio, the shot clock and
                # detection together; resume shifts every timer so nothing fires late.
                if tutorial.active:
                    # Space during the tutorial ends it; the transition check
                    # below resumes the experience.
                    tutorial.skip()
                elif not paused:
                    paused = True
                    player.pause()
                    render.pause()
                    gesture.pause()
                    pygame.mixer.pause()
                    print("[main] PAUSED")
                elif tutorial.begin():
                    # First start with tutorial enabled: the experience stays
                    # paused while the tutorial teaches the gesture vocabulary.
                    print("[main] tutorial running — experience stays paused")
                else:
                    paused = False
                    player.resume()
                    render.resume()
                    gesture.resume()
                    pygame.mixer.unpause()
                    print("[main] RESUMED")
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_F11, pygame.K_f):
                render.toggle_fullscreen()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                render.toggle_debug()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_k:
                # Skeleton-in-corner display option (independent of debug overlay).
                render.toggle_skeleton()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                # Captions on/off (pause-menu option; works any time).
                render.toggle_captions()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                if tutorial.active:
                    tutorial.skip()
                # Skip the prologue (act 0) or the epilogue — cut audio and jump.
                elif player.skip_prologue() or player.skip_epilogue():
                    pygame.mixer.stop()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT and not paused:
                player._advance()

            # Pause-menu volume slider: Up/Down keys or click/drag on the bar.
            if paused:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    render.adjust_volume(+0.05)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                    render.adjust_volume(-0.05)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    render.handle_volume_click(event.pos)
                if event.type == pygame.MOUSEMOTION and event.buttons[0]:
                    render.handle_volume_click(event.pos)

        # ── Meta voice commands: keep the right window open, act on fires ──
        if camera_setup_active:
            meta_desired = ("meta_ready", ["ready"])
        elif tutorial.active:
            meta_desired = ("meta_skip", ["skip"])
        else:
            shot = player.current_shot
            act = shot.act if shot else None
            if not paused and (player.in_prologue()
                               or act in ShotSequencePlayer.EPILOGUE_ACTS):
                meta_desired = ("meta_skip", ["skip"])
            else:
                meta_desired = None
        meta_voice.ensure(meta_desired)
        for vi_id in meta_voice.drain():
            if vi_id == "meta_ready" and camera_setup_active:
                camera_setup_active = False
                print("[main] camera setup confirmed by voice — press Space to begin")
            elif vi_id == "meta_skip":
                if tutorial.active:
                    print("[main] voice skip — tutorial")
                    tutorial.skip()
                elif player.skip_prologue() or player.skip_epilogue():
                    pygame.mixer.stop()   # cut any playing narration/beds

        # ── Camera-setup screen: live view + skeleton until confirmed ──
        if camera_setup_active:
            gesture.update()   # refresh the published skeleton; nothing is armed
            render.draw_camera_setup(gesture.latest_camera_frame(),
                                     gesture.fresh_pose_lm())
            clock.tick(render_fps)
            continue

        # ── End of experience: hold on FINAL_ADDRESS, then loop to the start ──
        if end_loop_cfg.get("enabled") and player.player_state == PLAYER_FINAL:
            if final_since is None:
                final_since = time.monotonic()
                print(f"[main] experience complete — looping back in "
                      f"{end_loop_cfg.get('delay_s', 5)}s")
            elif time.monotonic() - final_since >= float(end_loop_cfg.get("delay_s", 5)):
                final_since = None
                pygame.mixer.stop()
                player.start(0)
                player.pause()
                render.pause()
                gesture.pause()
                tutorial.done = False   # a fresh visitor gets the tutorial again
                paused = True
                print("[main] looped back to the beginning — PAUSED (press Space)")
        else:
            final_since = None

        # Tutorial finished (completed or skipped) while the experience was held
        # paused for it — release everything in one place.
        if paused and tutorial_was_active and not tutorial.active:
            paused = False
            player.resume()
            render.resume()
            gesture.resume()
            pygame.mixer.unpause()
            print("[main] tutorial done — RESUMED")
        tutorial_was_active = tutorial.active

        # Per-phase timing: a stall anywhere on the main thread freezes the
        # picture while the audio runs on. Only prints when a loop iteration is
        # actually slow, and then names the phase — so it can't be missed the
        # way a stall outside the instrumented calls was.
        _t = time.perf_counter
        _phase = {}
        _p0 = _t()

        if tutorial.active:
            # Detection runs, the shot player stays frozen.
            gesture.update()
            tutorial.update()
        elif not paused:
            gesture.update()
            _phase["gesture"] = _t()
            player.update()
            _phase["player"] = _t()
            narration_adapter.update()
            audio_mixer.update()
            _phase["audio"] = _t()

        _p1 = _t()
        render.update(
            pose_data=gesture.fresh_pose_lm(),
            gesture_debug=gesture.debug_info(),
            voice_debug=voice.debug_info(),
            narration_debug=player.debug_info(),
            tutorial_card=tutorial.card_info() if tutorial.active else None,
        )
        _p2 = _t()

        if (_p2 - _p0) * 1000.0 >= LOOP_SLOW_MS:
            parts, prev = [], _p0
            for name in ("gesture", "player", "audio"):
                if name in _phase:
                    parts.append(f"{name} {(_phase[name] - prev) * 1000:.0f}ms")
                    prev = _phase[name]
            parts.append(f"render {(_p2 - _p1) * 1000:.0f}ms")
            print(f"[perf] slow loop {(_p2 - _p0) * 1000:.0f}ms — "
                  + "  ".join(parts), flush=True)

        clock.tick(render_fps)


def _shutdown(cap, gesture, voice, render=None):
    voice.stop()
    gesture.close()
    cap.release()
    # Stop the frame-cache worker so a mid-build pack doesn't leave a
    # framecache.npy.building temp behind (also swept at next start).
    if render is not None and getattr(render, "_cache", None) is not None:
        render._cache.stop()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
