# Black Heritage Reclaimed

A gesture-controlled, audio-visual installation that retells a historical event through animated frame sequences, synchronized audio, and real-time computer vision.

> **Note:** This document is a living draft and will evolve alongside the project. Many content decisions (historical event, scene count, gestures) are still open — see `CLAUDE.md` for the full design spec.

---

## What it does

Users stand in front of a projected display and use hand gestures to navigate scenes of a historical narrative. A camera feeds a live hand-tracking pipeline (MediaPipe) that drives scene transitions and playback controls.

---

## Quick start

### Prerequisites

- Python 3.10+
- A USB webcam
- A projector or large display

### Install

```bash
git clone <repo-url>
cd BHR
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
python main.py
```

Press `Esc` to exit.

---

## Project layout

```
BHR/
├── main.py               # Entry point — init, game loop
├── gesture_engine.py     # Camera capture + MediaPipe gesture detection
├── scene_manager.py      # State machine (IDLE → TRANSITIONING → PLAYING)
├── audio_manager.py      # pygame.mixer channels for narration / ambience / SFX
├── config.json           # All tunable runtime parameters (no hardcoded constants)
├── requirements.txt
├── scenes/
│   ├── scene_01/
│   │   ├── frames/       # Drop PNG sequence here (frame_0001.png, ...)
│   │   └── metadata.json # FPS, audio paths, transition rules
│   └── scene_02/
│       ├── frames/
│       └── metadata.json
└── audio/
    ├── narration/        # Per-scene narration MP3s
    ├── ambience/         # Looping background audio
    └── sfx/              # One-shot sound effects
```

---

## Configuring the installation

All tuneable values live in `config.json`. Nothing is hardcoded in application logic.

| Key | Default | Description |
|-----|---------|-------------|
| `resolution` | `[1920, 1080]` | Display resolution |
| `camera_index` | `0` | USB camera device index |
| `min_detection_confidence` | `0.7` | MediaPipe detection threshold |
| `min_tracking_confidence` | `0.6` | MediaPipe tracking threshold |
| `gesture_cooldown_ms` | `800` | Minimum ms between accepted gestures |
| `gesture_confirmation_frames` | `4` | Frames a gesture must be held to fire |
| `projector_audio_offset_ms` | `40` | Audio delay to compensate for projector lag |
| `debug_overlay` | `false` | Show live camera feed window for calibration |
| `start_scene` | `"scene_01"` | First scene to load on launch |

---

## Gesture reference

| Gesture | Action |
|---------|--------|
| Swipe right | Advance to next scene |
| Swipe left | Return to previous scene |
| Open palm hold | Pause / resume playback |

Input is locked during scene transitions to prevent accidental double-triggers.

---

## Adding scenes

1. Create `scenes/scene_XX/frames/` and drop in a numbered PNG sequence (`frame_0001.png`, `frame_0002.png`, …).
2. Copy `scenes/scene_01/metadata.json` and update `scene_id`, `fps`, audio paths, and `transitions.next_scene`.
3. Add audio files to the appropriate `audio/` subdirectory.
4. Update `config.json → start_scene` if needed.

### Animator deliverable checklist

- [ ] Numbered PNGs: `frame_0001.png` … `frame_NNNN.png`
- [ ] Consistent resolution across all scenes
- [ ] RGBA (transparent) PNGs where layer compositing is required
- [ ] Completed `metadata.json` per scene

---

## On-site calibration

Set `"debug_overlay": true` in `config.json` before the event to open a live camera window. Adjust the following in `config.json` under exhibition lighting:

- `min_detection_confidence`
- `min_tracking_confidence`
- `gesture_cooldown_ms`
- `projector_audio_offset_ms`

Set `debug_overlay` back to `false` before the experience goes live.

---

## Hardware minimum spec

| Component | Minimum |
|-----------|---------|
| CPU / GPU | Intel Iris Xe integrated or dedicated GPU |
| RAM | 16 GB |
| Storage | SSD (HDD will bottleneck frame loading) |
| Camera USB | USB 3.0 |

An NVIDIA Jetson is recommended if projector glare causes tracking issues with a standard webcam.

---

## Open questions (tracked in `CLAUDE.md`)

- Historical event and scene count TBD
- Transition animations: dedicated frames vs. in-code crossfade?
- Idle / attract state when no user is present
- Accessibility fallback if gestures are not suitable for all users
