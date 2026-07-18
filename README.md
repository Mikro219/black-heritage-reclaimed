# Black Heritage Reclaimed

A gesture-controlled installation that retells a historical event through animated scenes. You stand in front of a projected display and use hand gestures to move through the story — no controllers, no touchscreen.

> For the full technical spec, see [`CLAUDE.md`](./CLAUDE.md).

---

## Who this guide is for

This README has two sections:

- 🎨 **Animators** — add your scenes to the project (no coding required)
- 💻 **Developers** — get the code running locally

If you're an animator, **the only section you need to read is the one with your name on it.** Skip everything else.

---

# 🎨 For Animators

Welcome! Your job is simple:

1. Drop your animation frames into your folder
2. Fill in a small information file
3. Send your work to the project using GitHub Desktop

You will not need to write or read any code. If something in this guide doesn't make sense, message the lead developer — don't guess.

---

## What you'll need

- A computer (Windows or Mac)
- A GitHub account — sign up free at [github.com](https://github.com/)
- **GitHub Desktop** installed — download from [desktop.github.com](https://desktop.github.com/)

That's it. No Python, no terminal, no command line.

---

## One-time setup

You only do this once.

### 1. Get added to the project

Send your GitHub username to the lead developer. They'll add you to the project so you can see it.

### 2. Install GitHub Desktop and sign in

Open GitHub Desktop after installing it and sign in with your GitHub account.

### 3. Download the project to your computer

In GitHub Desktop:

1. Click **File → Clone Repository**
2. Find **BHR** in the list
3. Choose where to save it on your computer (your Documents folder is fine)
4. Click **Clone**

You now have a copy of the whole project on your computer.

### 4. Switch to your branch

A **branch** is your own personal copy of the project where you can work without affecting anyone else. You have one branch named after you — for example, `art/wendy` or `art/natasha`.

In GitHub Desktop:

1. At the top of the window, click the **Current Branch** button
2. Find the branch with your name (e.g. `art/wendy`)
3. Click it

You'll know it worked because the **Current Branch** button now shows your name.

> **Important:** Always check that you're on your branch before you start working. If you see `main` at the top instead of your name, you're on the wrong branch — switch back to yours.

---

## Finding your folder

Inside the project there's a folder called `scenes/`. Inside it, every animator has their own folder named after them:

```
scenes/
├── natasha/
├── wendy/
├── elayna/
└── felicia/
```

**Only put your work inside your own folder.** Don't touch anyone else's.

Inside your folder you'll find two files waiting for you:

- `_READ_ME_FIRST.txt` — a quick reminder of the rules
- `_TEMPLATE_metadata.json` — a template you'll copy for each new scene

Don't delete either of these.

---

## Adding a new scene

Each scene you make becomes its own folder inside your folder. Here's the layout:

```
scenes/wendy/
├── _READ_ME_FIRST.txt
├── _TEMPLATE_metadata.json
└── scene_03/                  ← a new scene
    ├── frames/
    │   ├── frame_0001.png
    │   ├── frame_0002.png
    │   └── ...
    └── metadata.json
```

### Step 1 — Get your scene number

Before you make a new scene, ask the lead developer which scene number is yours. Scene numbers are shared across the whole project — only one person can own each number.

Example: the lead might tell you "you have scenes 03, 07, and 11."

### Step 2 — Make the scene folder

Inside your folder, create a new folder named like `scene_03` (use the number you were given, with a leading zero if it's under 10: `scene_01`, `scene_07`, `scene_11`).

Inside that, make a folder called `frames`.

### Step 3 — Drop in your frames

Save your animation frames into the `frames/` folder.

**Naming rules — please follow exactly:**

- PNG files only
- Named `frame_0001.png`, `frame_0002.png`, `frame_0003.png`, and so on
- Always 4 digits, with zeros at the front (so `frame_0042.png`, not `frame_42.png`)
- Same resolution for every frame in every scene (the lead developer will tell you the resolution before you start)

### Step 4 — Fill in the information file

Copy `_TEMPLATE_metadata.json` from your folder into your new scene folder, then **rename the copy** to `metadata.json` (no underscores, no `_TEMPLATE_`).

Open it in any text editor (TextEdit on Mac, Notepad on Windows — it's just a text file). It looks like this:

```json
{
  "scene_id": "scene_01",
  "artist": "natasha",
  "fps": 24,
  "resolution": [1920, 1080],
  "transitions": {
    "accepts_input_after_frame": 12,
    "next_scene": "scene_02"
  }
}
```

Change the values to match your scene:

| Field | What to put |
|-------|-------------|
| `scene_id` | The scene number you were given, like `"scene_03"`. Keep the quotes. |
| `artist` | Your folder name, like `"wendy"`. Keep the quotes. |
| `fps` | Frames per second your animation runs at. Default is 24. |
| `resolution` | Leave as the lead developer instructed (same for all scenes). |
| `accepts_input_after_frame` | After this frame, the viewer can swipe to the next scene. Usually 12. Ask the lead if unsure. |
| `next_scene` | The scene that comes after yours, like `"scene_04"`. Ask the lead. |

> **Watch out:** JSON is fussy. Don't remove the curly braces `{}`, the square brackets `[]`, the commas, or the quotes. Only change the values, not the field names.

---

## Sending your work to the project

When you've finished a scene (or want to save your progress), here's how to share it.

### 1. Open GitHub Desktop

You should see your new files listed on the left under **Changes**.

### 2. Double-check you're on your own branch

Look at the top of the window. The **Current Branch** button should show your branch name (`art/wendy`, etc.). If it shows `main`, **stop** — switch to your branch first.

### 3. Double-check the file list

Every file listed under **Changes** should be inside `scenes/<your_name>/`. If you see anything outside your folder, **don't commit it.** Message the lead developer.

### 4. Write a short summary

In the box at the bottom-left, type a short description of what you did. Examples:

- `Add scene_03 frames`
- `Update scene_03 metadata`
- `Fix frame numbering in scene_07`

### 5. Click **Commit to art/&lt;your_name&gt;**

The button at the bottom-left will say "Commit to" followed by your branch name. Click it.

### 6. Click **Push origin**

A button at the top will say **Push origin**. Click it. This uploads your work.

That's it. Your work is now saved on GitHub and the lead developer will see it.

---

## Animator checklist

Before you call a scene done, check:

- [ ] You're on your own branch (not `main`)
- [ ] Frames are in `scenes/<your_name>/scene_NN/frames/`
- [ ] Frames are named `frame_0001.png`, `frame_0002.png`, … (4 digits, leading zeros)
- [ ] All frames are the agreed project resolution
- [ ] `metadata.json` is filled in and saved next to the `frames/` folder
- [ ] You only changed files inside your own folder
- [ ] You committed and pushed in GitHub Desktop

---

## Common questions

**"GitHub Desktop is showing me files I never touched."**
Don't commit them. This usually means hidden system files (`.DS_Store` on Mac, `Thumbs.db` on Windows) snuck in. Message the lead developer — they'll add a rule to ignore them.

**"I made a mistake — how do I undo?"**
Don't panic and don't try to fix it yourself. Message the lead developer with what you did. Git keeps a full history; nothing is ever truly lost.

**"I see a notice about a 'merge conflict' or a 'pull'."**
Stop and message the lead developer before clicking anything.

**"Can I work on two scenes at once?"**
Yes — make as many scene folders as you have scene numbers for. Just keep each scene's frames in its own folder.

**"How big can my frames be?"**
Whatever resolution the lead developer agrees on for the project (same for all scenes). File size per frame should be reasonable — if a single PNG is over 10 MB, something's probably wrong with your export settings.

---

# 💻 For Developers

### Prerequisites

- Python 3.10 or higher — [python.org/downloads](https://www.python.org/downloads/)
- A USB webcam
- A projector or large display

### 1. Clone the repository

```bash
git clone <repo-url>
cd BHR
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
py -3.12 main.py
```

Press `Esc` to exit. `Space` pause/play · `S` skip prologue/tutorial/epilogue · `F` fullscreen · `D` debug overlay · `K` skeleton panel · `Enter` confirm camera setup.

Voice commands: say **"skip"** during the tutorial, prologue or epilogue to skip that part; say **"ready"** on the camera-setup screen to confirm. The camera-setup screen (`config.json "camera_setup"`) shows the live camera view with the tracked skeleton so the camera can be aimed before the experience starts; when the experience ends it pauses and loops back to the beginning (`config.json "end_loop"`).

---

## Project layout

```
main.py                  boot, key handling, main loop
config.json              every tunable value (see Configuration)
engines/
  shot_sequence_player   the runtime spine: shot FSMs, holds, branching
  sequence_loader        scenes/sequence.json + per-shot metadata -> Shot objects
  render_engine          frame playback, hand-icon cursors, pause menu, tutorial cards
  gesture_engine         camera + MediaPipe Pose thread (pose-only), detector dispatch, depth fusion
  voice_engine           Vosk keyword spotting + hum/whisper DSP
  audio_mixer            layered stem audio: looping music/ambience beds +
                         frame-anchored SFX from per-shot audio_events
  narration_adapter      plays a shot's AL-XX-YYY audio lines
  tutorial_engine        code-rendered calibration/tutorial at start
  event_bus, frame_cache plumbing
  depth/                 Orbbec Gemini 335 adapter + depth fusion (fusion.py)
  detectors/rules/       one file per gesture detector (+ shared hand_pose.py)
scripts/                 dev tools: gesture_tuner, voice_tuner, build_exe,
                         copy_frames, prepare_hand_icons, rasterize_storyboard,
                         export_experience (Experience Builder -> scenes tree),
                         capcut_audio (CapCut draft audio -> audio_events),
                         extract_comp_audio (comp scenes -> voice-line MP3s)
tools/
  experience_builder/    browser-based flow editor: open an MP4, carve it into
                         blocks, wire choices/merges, drop interaction windows,
                         preview, export (see its README)
tests/                   regression suite (see below)
scenes/                  sequence.json + act_NN_<name>/shot_NN/{frames,metadata.json}
assets/                  hand icons, storyboard, misc media
```

## Running the tests

Run the regression suite after **any** engine or detector change — it encodes
the playtest behaviour contract for every interaction, the depth-fusion rules,
the tutorial flow, render timing, and the shot-sequence wiring invariants:

```bash
py -3.12 -m unittest discover -s tests
```

~70 tests, ~5 seconds, no camera or display needed. If a test goes red, a
scene interaction regressed — fix the code, not the test (tests only change
when the intended behaviour changes).

---

## Configuration

All tunable values live in `config.json`. Nothing is hardcoded in the application.

| Key | Default | Description |
|-----|---------|-------------|
| `resolution` | `[1920, 1080]` | Display resolution |
| `camera_index` | `0` | USB camera device index |
| `min_detection_confidence` | `0.7` | Gesture detection threshold |
| `min_tracking_confidence` | `0.6` | Gesture tracking threshold |
| `gesture_cooldown_ms` | `800` | Minimum ms between accepted gestures |
| `gesture_confirmation_frames` | `4` | Frames a gesture must be held to fire |
| `projector_audio_offset_ms` | `40` | Audio delay for projector lag (deferred — audio not yet active) |
| `debug_overlay` | `false` | Show live camera feed for on-site calibration |
| `start_scene` | `"scene_01"` | First scene loaded on launch |

---

## On-site calibration

Set `"debug_overlay": true` in `config.json` before the event to open a live camera window. Under exhibition lighting, adjust the following until gestures feel reliable:

- `min_detection_confidence`
- `min_tracking_confidence`
- `gesture_cooldown_ms`

Set `debug_overlay` back to `false` before the experience goes live.

---

## Gesture reference

| Gesture | Action |
|---------|--------|
| Swipe right | Advance to next scene |
| Swipe left | Return to previous scene |
| Open palm hold | Pause / resume playback |

Input is locked during scene transitions to prevent accidental double-triggers.

---

## A note on audio

Music, ambience and SFX are live (July 2026): each shot's `metadata.json` can
carry an `audio_events` list — beds loop through interaction holds, SFX are
frame-anchored one-shots — played by `engines/audio_mixer.py` from the shared
`scenes/_audio/` pool. The timeline was imported from the CapCut master draft
with `py -3.12 scripts/capcut_audio.py assets/audio/draft_content.json --apply-scenes`.
Auntie Liza's voice lines exist as per-comp extractions in `assets/audio/voice_lines/`
(`py -3.12 scripts/extract_comp_audio.py`) but are not wired into shots yet.
Narration (Auntie Liza's `AL-XX-YYY` voice lines) is still pending recording;
it plays on its own channel and needs no audio_events changes when it lands.
Animators are not responsible for audio. See [`CLAUDE.md`](./CLAUDE.md) for the schema.

---

## Need help?

- **Animators** → message the lead developer for anything technical, scene-numbering, or "is this OK?"
- **Developers** → see [`CLAUDE.md`](./CLAUDE.md) for the full design spec
- **Project / narrative questions** → contact the project lead
