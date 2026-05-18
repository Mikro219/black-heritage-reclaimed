"""
SceneManager — owns scene transitions and the top-level state machine.

States:
  BOOT → CALIBRATION → SCENE_PLAYBACK → FINAL_ADDRESS → ACKNOWLEDGMENTS → ATTRACT_LOOP

Does not control narration directly; receives `sequence_finished` from
NarrationEngine and decides when to transition.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional


SCENES = [
    "scene_01", "scene_02", "scene_03", "scene_04", "scene_05", "scene_06",
    "scene_07", "scene_08", "scene_09", "scene_10", "scene_11",
]

STATE_BOOT = "BOOT"
STATE_CALIBRATION = "CALIBRATION"
STATE_SCENE_PLAYBACK = "SCENE_PLAYBACK"
STATE_FINAL_ADDRESS = "FINAL_ADDRESS"
STATE_ACKNOWLEDGMENTS = "ACKNOWLEDGMENTS"
STATE_ATTRACT_LOOP = "ATTRACT_LOOP"

# Dev stubs used when no metadata.json exists — drawn in the debug overlay.
DEV_SCENES = [
    {
        "id": "scene_01", "title": "The Quilt Awakens",
        "description": (
            "Auntie Liza introduces herself and the long road ahead. "
            "A quilt hangs outside — its patterns are a secret map."
        ),
        "gestures": [
            "CG  Raise both hands above shoulders — signal freedom",
            "VI  Say 'Freedom' (alternative to raise)",
            "OI  Point at the Crossroads quilt block",
        ],
    },
    {
        "id": "scene_02", "title": "Crossroads",
        "description": (
            "Two paths diverge in the dark. Listen first, then choose "
            "your direction with conviction."
        ),
        "gestures": [
            "OI  Cup hand to ear — listen",
            "CG  Point left or right — choose path",
            "CG  Say 'Go' — commit to the path",
        ],
    },
    {
        "id": "scene_03", "title": "Flying Geese",
        "description": (
            "Birds fly north overhead. Track their direction — "
            "they're pointing the way and marking water below."
        ),
        "gestures": [
            "OI  Look up — both hands raised upward",
            "CG  Shade eyes then sweep arm left to right",
            "VI  Say 'North' (reaction)",
        ],
    },
    {
        "id": "scene_04", "title": "North Star",
        "description": (
            "The star that never moves. Learn to find it using the "
            "Drinking Gourd — and learn the song that is a map."
        ),
        "gestures": [
            "CG  Trace the Big Dipper (curved bowl + handle)",
            "CG  Draw a 5-point star with one finger",
            "CG  Hum 500ms OR say 'Follow the gourd'",
            "OI  Point arm at Polaris and hold",
        ],
    },
    {
        "id": "scene_05", "title": "Monkey Wrench",
        "description": (
            "Pack deliberately. Each tool earns its place. "
            "The most important thing you carry cannot be searched."
        ),
        "gestures": [
            "CG  Hover + grab: food bundle",
            "CG  Hover + grab: rope",
            "CG  Hover + grab: dark cloth",
            "CG  Hover + grab: lantern",
            "CG  Touch your temple — carry the knowledge",
        ],
    },
    {
        "id": "scene_06", "title": "Shoofly",
        "description": (
            "A door in the dark. The knock has to be right: "
            "short-short-long. Someone inside decided the risk was worth it."
        ),
        "gestures": [
            "OI  Scan left — check the road",
            "OI  Scan right — all clear",
            "CG  Three forward knocks: short-short-long",
            "VI  Whisper 'Shoofly' (alternative to knock)",
            "OI  Hold still, palms open",
            "OI  Reach forward and clasp",
        ],
    },
    {
        "id": "scene_07", "title": "Bear's Paw",
        "description": (
            "Off every road. Paw prints in the mud lead through "
            "the wilderness to water — if you read them right."
        ),
        "gestures": [
            "CG  Reach out and trace the trail (60% waypoints)",
            "CG  Point to the correct fork",
            "CG  Lower both hands — duck under canopy",
            "CG  Raise both hands — canopy lifts",
        ],
    },
    {
        "id": "scene_08", "title": "Bow Tie / Hourglass",
        "description": (
            "Change your silhouette. Hundreds of people passed "
            "checkpoints by becoming someone the patrol didn't see."
        ),
        "gestures": [
            "CG  Both hands to brow height — put on the hat",
            "CG  Arms crossed at center — put on the coat",
            "VI  Say 'Ready' (line swap — affects which line plays next)",
        ],
    },
    {
        "id": "scene_09", "title": "Wagon Wheel",
        "description": (
            "Two paths, both real. Ride hidden under the boards "
            "or walk alongside like you belong on this road."
        ),
        "gestures": [
            "CG  Point at compartment (Path A) or alongside (Path B)",
            "CG-A  Lift up, press down, hold still 2s — hide",
            "CG-B  Bilateral alternating walk — steady pace",
            "CG  Touch the wagon wheel — every turn is a mile",
        ],
    },
    {
        "id": "scene_10", "title": "Tumbling Blocks  [URGENT]",
        "description": (
            "Stop. Listen. Someone is on that road who should not be. "
            "Pack fast. Leave nothing. Go."
        ),
        "gestures": [
            "OI  Cup ear — flash + knock signal",
            "CG  Fast bilateral gather — 3+ bursts in 3s",
            "CG  Bilateral forward push at speed — run",
            "OI  Fast alternating arms — run harder",
        ],
    },
    {
        "id": "scene_11", "title": "The Conestogo",
        "description": (
            "The river. The other side is the Queen's Bush — "
            "land nobody wanted, kept by the people who needed it most."
        ),
        "gestures": [
            "CG  Both hands rotate outward 2+ cycles — unravel boat",
            "CG  Bilateral forward push (slow) — launch",
            "CG  Arcing paddle strokes, 4 counts — row",
            "CG  One arm extends forward and down — touch shore",
            "OI  Both arms wide — spread and feel it",
        ],
    },
]

_DEV_AUTO_ADVANCE_MS = 30_000  # 30s matches profile_standard auto_advance


class SceneManager:
    def __init__(self, config: dict, event_bus: "EventBus", scenes_root: str):
        self.config = config
        self.event_bus = event_bus
        self.scenes_root = scenes_root
        self._state = STATE_BOOT
        self._current_scene_id: Optional[str] = None
        self._current_metadata: Optional[dict] = None

        # Dev-mode state (active when no metadata.json is found)
        self._dev_mode = False
        self._dev_idx = 0
        self._dev_scene_start: float = 0.0
        self._dev_last_hands = False
        self._dev_advance_cooldown_until: float = 0.0

        # Storyboard index — loaded once, used to supply background frames in dev mode
        self._storyboard_index: Optional[dict] = self._load_storyboard_index()

        self.event_bus.subscribe("sequence_finished", self._on_sequence_finished)
        self.event_bus.subscribe("cg_detected", self._on_cg_detected)
        self.event_bus.subscribe("oi_detected", self._on_oi_detected)
        self.event_bus.subscribe("vi_detected", self._on_vi_detected)

    def start(self):
        self._state = STATE_CALIBRATION
        self.event_bus.emit("state_change", {"state": STATE_CALIBRATION})

    def on_calibration_passed(self):
        self._load_scene(self.config.get("start_scene", "scene_01"))

    def _load_scene(self, scene_id: str):
        metadata_path = self._find_metadata(scene_id)
        if not metadata_path:
            if not self._dev_mode:
                print(f"[SceneManager] No metadata.json found — entering dev preview mode.")
                self._dev_mode = True
                self._dev_idx = next(
                    (i for i, s in enumerate(DEV_SCENES) if s["id"] == scene_id), 0
                )
                self._dev_scene_start = time.monotonic()
                self._dev_advance_cooldown_until = time.monotonic() + 1.5
                self._dev_last_hands = True
                self._emit_dev_frames()
            # Release any input lock that was set by _transition_to before calling here
            self.event_bus.emit("input_lock", {"locked": False})
            return

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        self._current_scene_id = scene_id
        self._current_metadata = metadata
        self._state = STATE_SCENE_PLAYBACK

        self.event_bus.emit("input_lock", {"locked": False})
        self.event_bus.emit("scene_load", {
            "scene_id": scene_id,
            "metadata": metadata,
            "audio_dir": os.path.join(os.path.dirname(metadata_path), "audio"),
        })

        # If the scene has no animator frames yet, use storyboard pages as placeholder
        frames_dir = os.path.join(os.path.dirname(metadata_path), "frames")
        has_frames = os.path.isdir(frames_dir) and any(
            f.lower().endswith((".png", ".jpg", ".jpeg"))
            for f in os.listdir(frames_dir)
        )
        if not has_frames:
            storyboard_paths = self._get_storyboard_paths(scene_id)
            if storyboard_paths:
                self.event_bus.emit("dev_frames_load", {"frames_paths": storyboard_paths, "fps": 0.5})
                print(f"[SceneManager] No animator frames for {scene_id} — using storyboard ({len(storyboard_paths)} panels)")

    def tick(self, hands_detected: bool = False):
        """Call every frame. Rising-edge hand detection advances scene in dev mode."""
        if not self._dev_mode:
            return
        now = time.monotonic()

        # Rising-edge hands detection advances scene (with cooldown)
        if hands_detected and not self._dev_last_hands and now >= self._dev_advance_cooldown_until:
            self._dev_advance_scene()
        self._dev_last_hands = hands_detected

    def _dev_advance_scene(self):
        self._dev_idx = (self._dev_idx + 1) % len(DEV_SCENES)
        self._dev_scene_start = time.monotonic()
        self._dev_advance_cooldown_until = time.monotonic() + 1.5
        print(f"[SceneManager] Dev advance -> {DEV_SCENES[self._dev_idx]['title']}")
        self._emit_dev_frames()

    # ------------------------------------------------------------------
    # Storyboard index helpers (dev mode only)
    # ------------------------------------------------------------------

    def _load_storyboard_index(self) -> Optional[dict]:
        index_path = Path(self.scenes_root).parent / "assets" / "storyboard" / "index.json"
        if not index_path.exists():
            return None
        try:
            with open(index_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"[SceneManager] Could not load storyboard index: {exc}")
            return None

    def _get_storyboard_paths(self, scene_id: str) -> list:
        if not self._storyboard_index:
            return []
        repo_root = Path(self.scenes_root).parent
        paths = []
        for key in sorted(self._storyboard_index["pages"]):
            entry = self._storyboard_index["pages"][key]
            if entry.get("script_scene") == scene_id:
                p = repo_root / entry["path"]
                if p.exists():
                    paths.append(str(p))
        return paths

    def _emit_dev_frames(self):
        scene_id = DEV_SCENES[self._dev_idx]["id"]
        paths = self._get_storyboard_paths(scene_id)
        if paths:
            self.event_bus.emit("dev_frames_load", {"frames_paths": paths, "fps": 0.5})
        else:
            # Clear the display so it doesn't hold the previous scene's frames
            self.event_bus.emit("dev_frames_load", {"frames_paths": [], "fps": 0.5})

    def debug_info(self) -> Optional[dict]:
        if not self._dev_mode:
            return None
        scene = DEV_SCENES[self._dev_idx]
        return {
            "scene_idx": self._dev_idx,
            "scene_total": len(DEV_SCENES),
            "scene_id": scene["id"],
            "title": scene["title"],
            "description": scene["description"],
            "gestures": scene["gestures"],
        }

    def _find_metadata(self, scene_id: str) -> Optional[str]:
        for artist_dir in os.listdir(self.scenes_root):
            candidate = os.path.join(self.scenes_root, artist_dir, scene_id, "metadata.json")
            if os.path.exists(candidate):
                return candidate
        return None

    def _on_sequence_finished(self, data: dict):
        if self._state == STATE_SCENE_PLAYBACK:
            next_scene = (self._current_metadata or {}).get("transitions", {}).get("next_scene")
            if next_scene and next_scene != self._current_scene_id:
                self._transition_to(next_scene)
            elif self._current_scene_id == "scene_11":
                self._enter_final_address()
        elif self._state == STATE_FINAL_ADDRESS:
            self._state = STATE_ACKNOWLEDGMENTS
            self.event_bus.emit("state_change", {"state": STATE_ACKNOWLEDGMENTS})

    def _transition_to(self, scene_id: str):
        self.event_bus.emit("input_lock", {"locked": True})
        self.event_bus.emit("scene_transition_start", {"from": self._current_scene_id, "to": scene_id})
        self._load_scene(scene_id)

    def _enter_final_address(self):
        self._state = STATE_FINAL_ADDRESS
        self.event_bus.emit("input_lock", {"locked": True})
        self.event_bus.emit("state_change", {"state": STATE_FINAL_ADDRESS})

    def _on_cg_detected(self, data: dict):
        self.event_bus.emit("narration_cg_completed", data)

    def _on_oi_detected(self, data: dict):
        self.event_bus.emit("narration_oi_detected", data)

    def _on_vi_detected(self, data: dict):
        # Always route through narration_vi_reaction so NarrationEngine.on_vi_detected
        # can translate voice_id → gesture_id before deciding whether to advance.
        # Routing cg_alternative directly to narration_cg_completed would pass the
        # voice_id ("freedom") instead of the gesture_id ("raise_hands"), which
        # NarrationEngine.on_cg_completed would reject.
        self.event_bus.emit("narration_vi_reaction", data)

    @property
    def state(self) -> str:
        return self._state
