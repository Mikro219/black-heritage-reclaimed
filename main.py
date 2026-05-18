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
from engines.narration_engine import NarrationEngine
from engines.gesture_engine import GestureEngine
from engines.voice_engine import VoiceEngine
from engines.scene_manager import SceneManager
from engines.render_engine import RenderEngine


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


def main():
    parser = argparse.ArgumentParser(description="Black Heritage Reclaimed")
    parser.add_argument("--profile", metavar="NAME", help="Host profile name")
    args = parser.parse_args()

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

    narration = NarrationEngine(config, bus)
    gesture = GestureEngine(config, bus)
    voice = VoiceEngine(config, bus)
    scene_mgr = SceneManager(config, bus, str(SCENES_ROOT))

    bus.subscribe("scene_load", lambda d: narration.load_scene(
        d["metadata"],
        d["audio_dir"],
        on_finished=lambda: bus.emit("sequence_finished", {})
    ))
    bus.subscribe("narration_cg_completed", lambda d: narration.on_cg_completed(
        d.get("gesture_id", ""), choice=d.get("choice")
    ))
    bus.subscribe("narration_oi_detected", lambda d: narration.on_oi_detected(d.get("gesture_id", "")))
    bus.subscribe("narration_vi_reaction", lambda d: narration.on_vi_detected(d.get("voice_id", "")))
    bus.subscribe("scene_load", lambda _: narration.start())

    voice.start()
    scene_mgr.start()

    cap = open_camera(profile)
    render_fps = profile.get("performance", {}).get("render_fps_cap", 30)
    clock = pygame.time.Clock()

    calibration_passed = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _shutdown(cap, gesture, voice)
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                _shutdown(cap, gesture, voice)
                return

        ret, frame = cap.read()
        if not ret:
            continue

        gesture.process_frame(frame)

        if not calibration_passed:
            if gesture.hands_detected():
                calibration_passed = True
                scene_mgr.on_calibration_passed()

        scene_mgr.tick(hands_detected=gesture.hands_detected())
        narration.update()
        render.update(
            landmark_data=gesture._last_landmarks,
            handedness_data=gesture._last_handedness,
            gesture_debug=gesture.debug_info(),
            scene_debug=scene_mgr.debug_info(),
            voice_debug=voice.debug_info(),
            narration_debug=narration.debug_info(),
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
