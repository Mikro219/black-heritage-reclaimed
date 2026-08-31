# BHR Experience Builder

A visual flow editor for authoring gesture-driven experiences from an MP4 —
open the master draft, carve it into blocks, wire them into a flow chart with
choices and merges, drop interaction windows on the timeline, and export a
BHR runtime scenes tree.

## Run it

Open `index.html` in **Chrome or Edge** (double-click works — no server, no
install). Everything runs locally; the MP4 never leaves your machine.

## Workflow

1. **Import media** — drop `BHR Draft 1.mp4` (or any MP4) on the Media tab.
   Drop an MP3/WAV on the Sounds tab and click its ⚡ bolt to make it the
   **global detect sound** (plays whenever any interaction fires).
2. **Build the flow** — drag a clip onto the canvas to create a playback
   block. Add **Choice** / **Merge** blocks from the canvas toolbar. Drag
   from a block's right-edge port to the next block to connect them. The
   green flag marks the start block (change it in the inspector).
3. **Trim blocks** — select a block; the bottom strip shows its slice of the
   source video (drag the brush edges to trim, the middle to move) and a
   block-local timeline with playhead. Space = play/pause, ←/→ = frame step.
4. **Interaction windows** — "Add window" at the playhead (or double-click a
   window bar to edit). Draw the region on the paused frame, pick a **BHR
   detector** (point at region, knock, raise hands, push, draw stroke, …) and
   tune its params. Drag the bar to move it, drag its right edge to resize
   its duration (drag to the end = "until end").
5. **Choices** — on a choice block, each window's "On trigger → go to block"
   creates a branch. The block plays its clip, pauses on the last frame, and
   waits; the timeout auto-advances (the runtime always falls into the FIRST
   branch on timeout).
6. **Audio lanes** — under the block timeline sit four lanes: **VO**,
   **MUSIC**, **AMB** and **SFX**. Drag a sound from the Sounds tab onto a
   lane to place it (the lane sets its role); drag a bar to move it, drag its
   right edge to resize, double-click for the clip editor (gain, fades,
   source offset, "continues previous block's bed", preview mute). Music and
   ambience are *beds*: in the runtime they loop through interaction holds
   and hand over seamlessly to the next shot when it has a matching
   "continues" clip. SFX are one-shots fired at their timeline position. VO
   is Auntie Liza's voice: it plays once on the runtime's dedicated voice
   channel and a "continues" VO clip never restarts a line. **Click a lane's
   label to mute/unmute that whole lane** (strikethrough = muted; applies to
   the block player and the preview flow — monitoring only, never exported).
   The Sounds tab supports multi-file import, a filter box, a ▶ play button
   per sound, and 📁 **link folder** — point it at `assets/audio/` once and
   every sound (stems + voice lines) re-links by name, recursively.
7. **Block player with sound** — the bottom timeline's ▶ plays the scrub
   video *and* the block's audio clips in sync from the playhead (respecting
   clip mutes and lane mutes). Scrubbing or pausing stops the audio.
8. **Preview flow** — plays the experience from Start. Clicking a window's
   region simulates the gesture: green flash + detect sound, branches follow
   your choice. Audio lanes play through WebAudio with the same bed-handover
   semantics as the runtime. Esc exits.
9. **Save / Export** — Ctrl+S saves the `.bhrx.json` project (autosaves a
   draft to the browser as you work; media files re-link on reopen). Export
   shows the command that generates the runtime tree:

   ```
   py -3.12 scripts/export_experience.py "My_Experience.bhrx.json"
   ```

   Output goes to `export/generated/` (`scenes/scene_NN/` dirs) — per-scene
   `metadata.json`, `sequence.json`, extracted frames (needs ffmpeg:
   `winget install ffmpeg`), and `detect.mp3` copied into each scene that
   references it. The runtime runs it directly (`py -3.12 main.py` — it is
   the default scenes root). `--no-frames` skips extraction for a quick
   metadata-only look.

## How blocks map to the runtime

| Editor | Exported shot |
|---|---|
| Playback, no windows | `kind: playback` |
| Playback, 1 window | playback + non-blocking OI with `oi_frame_window` (shot 58 pattern) |
| Playback, 2+ windows | interactive play-through FSM with per-window `oi` states (shot 24 pattern) |
| Choice (2 branches) | interactive region-fork FSM (point left/right, shot 09 pattern); branch chains gated with `play_if` |
| Merge | no shot — it just ends `play_if` gating |

**Voice windows** (detector "Voice keyword") export as real runtime wiring:

- on a **playback block** they become keyword VI states in the play-through
  FSM — saying the keyword during the window fires the detect sound + flash;
- on a **choice block**, give the voice window a "go to block" target that
  matches one of the two gesture branches and it becomes a *spoken branch
  pick* (`"voice"` on the waiting state + a `voice_<keyword>` transition —
  the shot 09 pattern). One keyword per choice; a voice window without a
  target is preview-only and the export warns.

Runtime constraints the editor inherits: forks are two-way (left/right),
one `play_if` per shot, one voice keyword per choice hold, and the runtime
records fork choices from gesture picks only (a voice pick leaves branch-gated
shots on the first-branch default — the export warns when this applies).

## Merging CapCut audio

`BHR_Experience.bhrx.json` is the single source of truth (the old
`scenes_to_builder.py` reverse importer was retired Aug 2026 along with the
hand-authored `scenes/` tree). The CapCut master-timeline audio
(music/ambience/SFX placements from `assets/audio/draft_content.json`, plus
Auntie Liza's VO slices) can be merged into the project as lane clips:

```
py -3.12 scripts/capcut_audio.py assets/audio/draft_content.json --to-builder BHR_Experience.bhrx.json
```

The project then **loads automatically** when you open `index.html`: the
importer regenerates `js/project_data.js` (a script-tag bundle of the
`.bhrx.json` — `file://` pages can't fetch JSON, but they can load scripts;
regenerate by hand with `py -3.12 scripts/bundle_builder_project.py`). A
fresh bundle wins over the browser's autosave draft (the draft is backed up
in localStorage); an unchanged bundle keeps your in-browser edits.

Media/sound *files* can't be bundled — browsers can't open local files by
path — so on first load re-link `BHR Draft 1.mp4` (click it in the Media
tab) and use 📁 link-folder on the Sounds tab pointed at `assets/audio/`
(recursive — covers `stems/` and `voice_lines/` in one go). The links
persist across sessions via stored file handles (Chrome may show a one-click
permission prompt per session).

> **Play button does nothing?** The clip isn't linked yet — a project file
> stores only the video's *name* (browsers can't reopen local files by
> themselves). Open the **Media** tab: a red clip needs one click to re-link.
> The timeline header also shows "NOT LINKED" in red while this is the case. Timing comes from
`scripts/copy_frames.py`'s shot → master-frame mapping (30fps). Notes printed
by the importer flag the places where editor and runtime semantics differ
(shot 37's wrong-way retry, the shot 19/20 draw chains, shot 57 duplicated per
fork-50 branch).

## Files

- `js/detectors.js` — detector list **hand-synced** from
  `engines/detectors/__init__.py`; `tests/test_experience_export.py` fails if
  it drifts from the runtime REGISTRY.
- `scripts/export_experience.py` — project → scenes tree converter.
- `tests/test_experience_export.py` — export invariants (same checks
  `tests/test_sequence.py` applies to the live tree).
