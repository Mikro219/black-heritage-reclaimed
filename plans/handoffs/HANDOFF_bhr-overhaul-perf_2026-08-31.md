# BHR full-day overhaul: .bhrx single source of truth, builder tooling, design picks, and the first-OI freeze killed end to end

**Date:** 2026-08-31
**Status:** COMPLETED (session closed cleanly; field-verified on the exhibition exe)
**Bead(s):** none
**Epic:** none
**Chain:** `standalone-dbfb991d` seq `1`
**Parent:** none — first in chain
**Prior chain:** none — first in chain

---

## Reference Documents

- `CLAUDE.md` — THE living technical spec (145KB; gitignored but present). Updated this session: new Aug-2026 changelog entry, host-section docs, file-structure tree, test counts.
- `State.md` — living status snapshot (gitignored, local-only per Mike). Carries a per-round session log of this entire day — richest secondary source.
- `docs/HOW_TO_RUN.md` — operator guide; `docs/packaging.md` — deploy guide.
- Memory: `~/.claude/projects/-mnt-d-Archived-Projects-BHR/memory/ask-when-reports-contradict.md` — Mike tests with the BUILT `dist/BHR/BHR.exe`; ask which binary/tree a report came from before assuming.

## The Goal

BHR (Black Heritage Reclaimed) is a gesture/voice museum installation for the Guelph Black Heritage Society, in exhibition-hardening phase. This session's arc: onboard the repo to clanker, clean and reorganize the codebase around `BHR_Experience.bhrx.json` as the single source of truth, fix Experience Builder UX bugs, implement Mike's Claude Design picks for the camera-setup and tutorial screens, and — the centerpiece — eliminate the recurring audio/visual desync ("freeze") around the first OI window, which took four diagnostic rounds against real exhibition logs to fully kill.

## Where We Are

- Everything is committed. HEAD = `6a26f16` "perf: kill the first-OI freeze end to end; WAV master audio; tutorial ping" (23 files, +1482/−464). Working tree clean. Branch `main`, no remote push done.
- Mike committed intermediate phases himself mid-session: `048092e` (desync/prewarm fixes), `3a8130b` (camera/tutorial/pause redesign), `072a e3e` (workflow/dir renames).
- **Suite: 303 tests** (started the day at 269 in a differently-shaped codebase; dipped to 252 when scenes/-dependent tests were retired, grew back with perf/design pins). On WSL: 5 errors are environmental only (`PortAudio library not found` — needs `sudo apt install libportaudio2`), 1 skip (ffprobe). Green otherwise.
- **Field state (Mike's exhibition/dev box, Windows, runs `dist/BHR/BHR.exe`)**: first-gesture freeze GONE; prologue-skip freeze GONE after the WAV bake rebuild. His words: "rebuilt and tested, skip is smooth now."
- `BHR_Experience.bhrx.json` is the single source of truth: hand-authored `scenes/` tree DELETED, `scripts/scenes_to_builder.py` DELETED, exporter output renamed `export/generated/scenes/scene_NN/` (was `export/scenes_generated/act_01_experience/shot_NN/`). `main.py` dev default scenes root = `export/generated`; frozen exe uses staged `scenes/` next to it.
- `config/host_profiles/` dissolved: laptop_dev content lives in `config.json` under `"host"`; `auto`/`mini_pc_prod` scrapped (git history has them); `--profile`/`BHR_HOST_PROFILE`/hostname resolution removed from main.py; tuners read/write `host.gesture_tuning`/`host.voice_tuning` in config.json (save aborts on read failure instead of writing a gutted config).
- `scripts/build_exe.py` re-exports the .bhrx before staging (`--skip-export` opts out), stages no `config/` dir, and `--keep-packs` preserves `dist/BHR/scenes` packs (staging skips `framecache.npy` via `SKIP_FILENAMES`).
- Frame pipeline (engines/frame_cache.py + render_engine.py FrameView): finished packs serve zero-copy (`get_frame_buffer` → `pygame.image.frombuffer(...).convert()`, measured 10.5→3.1 ms/frame at 1080p); packs mid-build serve INCREMENTALLY via a written-row bitmap; the playhead STEERS the builder via `warm_segment` hints; main-thread fallback decodes are written back into the building pack; the pack-page warm is PACED (60-frame burst then 45 fps).
- Master audio for the 5 baked shots is now exported as `audio.wav` (pcm_s16le) — mixer.music seek measured **0.1ms vs 316ms** for the mp3 scan. Runtime is extension-agnostic; back-compat mp3 fallback both ways in the exporter.
- Experience Builder: stuck-drag fixed (global missed-mouseup guard in `js/app.js`), zoom-button native-drag fixed (`user-select:none` on `#canvas-wrap` + preventDefault + dragstart kill in `js/graph.js`), Export button actually runs the exporter via new `scripts/builder_server.py` (localhost:8798, streaming), caption preview borderless to match runtime.
- Design picks implemented from Mike's Claude Design project (DesignSync, project `871c95af-2653-44b3-bb22-fec75eca2bb9`): camera setup = **1c North Star Arch** (active) with **1a Lantern Vignette kept as `draw_camera_setup_1a`** for revert; tutorial = **2a Quilt Card** (diamond progress row, serif type, centered figure panel, oval targets, no background hand icons, borderless captions). Pause menu got the serif/tracked type ladder. All design hexes mapped 1:1 to existing `palette.py` tokens — zero palette changes, contrast tests untouched.
- `R` hotkey restarts the experience any time (same reset as end_loop; handles tutorial-active edge without auto-resume). In pause-menu grid, README, HOW_TO_RUN.
- Tutorial success now plays `detect.mp3` (ch 2) with the green flash; `TutorialEngine(detect_sfx=...)` wired from main.py via scenes-tree glob.
- Clanker: BHR registered (symlink `~/projects/BHR` → `/mnt/d/Archived Projects/BHR`); session_start `ead982035e93e73b`, session_end `e53b1228eec6c207` logged with summary.
- Local WSL test env: scratchpad venv at `/tmp/claude-1000/.../scratchpad/bhr-venv` (numpy, pygame, Pillow, sounddevice, opencv-headless, mediapipe==0.10.14, vosk). System python has no pip; `python3.12-venv` was missing → bootstrapped via get-pip.
- CI workflow `.github/workflows/tests.yml` was seeded early-session but `.github/` is now GITIGNORED per Mike (along with `assets/`, `State.md`, `.claude/`) — it exists locally only and will never reach GitHub unless force-added.

## What We Tried (Chronological)

1. **Onboarding + baseline (early).** clanker init expected `~/projects/BHR` → symlinked instead of moving the 33GB repo. Suite initially ERRORED 14 modules — missing numpy/pygame in system python; built a scratchpad venv; final baseline 264/269 pass, 5 env-only PortAudio errors. Seeded CI + State.md.
2. **Cleanup pass (early).** Two Explore agents mapped debt. Deleted: 4 root `.bhrx` backups, `build/` (105MB), `Log/`, `scripts/camera_test.py`, `rename_frames.py`, `simplify_builder_project.py`, 4.6MB dead builder mockup HTML. Fixed: `GestureEngine.hands_detected()` (only dead function in the runtime), BOM on render_engine.py, stale narration_adapter comment, `projector_audio_offset_ms` removed from all 3 host profiles (read by no code), 2 unused imports. CLAUDE.md doc corrections ×4.
3. **.bhrx single source of truth (early-mid).** Mike: "these were all artifacts of the old workflow." Deleted scenes/ + scenes_to_builder.py + test_sequence.py + 3 real-tree test classes in test_meta_flow (SkipEpilogue/LoopRestart/CrossroadsSwitch); retargeted the 2 voice-marshalling tests in test_review_fixes to synthetic shots (they pin a threading contract, not tree content). Renamed export tree ON DISK (`mv`, preserving 9.4GB frames) + dist staging; loader got flat-`scenes`/`scene_` preference with legacy `act_*/shot_*` fallback so temp-tree fixtures still load. Suite 269→252.
4. **Config merge (early-mid).** laptop_dev.json → `config.json["host"]` (verified content-identical via json compare before deleting the dir). Prod values preserved in the `//host` comment block. Caveat surfaced to Mike: exhibition box now boots with laptop values until host.display/performance edited.
5. **Builder stuck-drag (mid).** Diagnosis: 14 window-level move/up dragger pairs across 5 JS modules all lose the mouseup when released off-window. ONE global capture-level guard in `js/app.js` synthesizes the missed mouseup (tracks pressed state; fires on `mousemove` with `e.buttons===0` or window blur). Verified all release handlers tolerate synthetic events.
6. **Zoom-button "drag" (mid).** Not a draggable element — text selection: pan mousedown never preventDefault'd, selected chrome text ("100%") becomes natively draggable; the drag ghost swallows mousemove/mouseup. Fixed 3-deep: `user-select:none` on `#canvas-wrap`, preventDefault on pan mousedown, dragstart kill on the wrap (library→canvas drops unaffected — their dragstart fires outside).
7. **Export button (mid).** Browsers can't spawn processes → `scripts/builder_server.py` (stdlib ThreadingHTTPServer, 127.0.0.1:8798, CORS for file:// null-origin incl. Private-Network header, GET /ping, POST /export streams exporter stdout chunked, one-export lock, saves posted project to the canonical .bhrx with `ensure_ascii=False`). Dialog detects the server → "Run export" + `--no-frames` checkbox + live log; falls back to showing the command. **Side catch while testing without ffmpeg:** exporter referenced RAW audio when a trimmed render already existed (reuse check nested under `if ffmpeg`) — my test export degraded ~70 events; fixed reuse-without-ffmpeg and re-exported to heal (45 trimmed refs back; warnings 73→28; remaining 23 renders need the ffmpeg box).
8. **Perf round 1 (mid).** Two audit agents (window-open path; frame pipeline). Mechanism found: segment jumps at window-open hit cold frames — mmap page faults + 3-copy conversion, or 25-45ms/frame PIL fallback; full-screen FX composite stacked 8-18ms on draw windows. Two fix agents landed: `warm_segment` (page pre-touch / bounded fallback pre-decode), JPEG `draft()` fallback (**caveat verified by my benchmark: draft() NO-OPS above 1/2 scale — at 1080p→720p the win is BILINEAR-only, 58→41ms; at equal resolution fallback is pure decode**), entry-sound prefetch on `prefetch_shot` (thread-safe never-replace caches), caption-card memo + eager font, FX dirty-rect. I added the copy-chain fix myself (frombuffer zero-copy, 10.5→3.1ms measured, integration-verified 0 bytes-path calls on a live pack).
9. **Design import (late).** `/design-login` → DesignSync read `Camera & Tutorial Redesigns.dc.html`. Implemented 1a + 2a; then Mike switched camera to **1c** (1a kept as method). Iterations from his screenshots: diamond spacing bug (rotated square spans 2×side; centers were side+8px → overlapped), background hand icons removed (drew behind the centered figure panel), Pointing target moved to right third (raw mirror x=1−x−w kept in sync), rects→ovals, caption border removed (+ builder preview parity).
10. **Perf round 2 (late).** Two more audits (steady-state per-frame; off-thread+startup). Fix agent: depth kept uint16 (was 3.7MB float32/frame on the capture thread), RGB→BGR→RGB round trip removed (`frame_is_rgb`), lazy Orbbec SDK import (PEP 562; pose_helpers import 0.91s, no SDK module loaded), Vosk model → voice thread, **Vosk gated to open windows** (was ~10-20% of a core all run, results discarded; Reset-on-reopen on the voice thread), PoseDepth single construction. Me: ring cache (pulse-bucketed), cursor rotation 5° cache, paused-screen composite cache (1.0ms/frame measured), star-trail pre-alpha + blits(), `_effective_timing` memo. Deliberately SKIPPED: gating Pose on armed windows (re-opens the warm_pose_hz cold-start class — audit and history agree).
11. **Freeze round 1 (late).** Mike: "first gesture (take a drink) freezes... even after restart." Found in-tree: scene_01 = 9,148 frames, NO pack (excluded from prewarm recommendation); cache served packs only when build COMPLETED → the whole 5-min shot ran on decode fallback every run. Fix: **incremental serving** — builder registers memmap + progress; readers copy written rows UNDER the lock (never hold the mapping — Windows refuses os.replace under an open mapping).
12. **Freeze round 2 (late).** Mike's exe logs: prologue skip lands at frame 5171 vs build cursor ~250 → sustained fallback; ALSO my "builder is the warm-up" no-op wrongly disabled warming after seeks; AND straight-through play races (decode ≈ playback rate). Fix: building state → `{mm, written bitmap, hint, count}`; `warm_segment` HINTS the builder (jumps cursor to the played region, wraps to fill); main-thread fallback decodes WRITE BACK into the pack (either thread can finish it). Deterministic order test: hint mid-build → `[0,4,5,1,2,3]`.
13. **Freeze round 3 (late).** New logs: freeze SMALLER and MOVED — `slow loop 192ms — render 192ms` at between_1 entry + 80-124ms gaps for ~5s. **Self-inflicted**: warm_segment's page pre-touch swept the whole 982-frame segment (~2.7GB) unthrottled, saturating disk against the main thread's own faults. Fix: `_warm_pack_pages` PACED — `_WARM_BURST_FRAMES=60` immediate, then `_WARM_RATE_FPS=45.0`, sleeps ≤50ms.
14. **Prewarm mystery (late).** Mike: "even though I ran the pre-warm script..." then "you may be looking in the wrong directory, it's in dist/BHR." His 128GB of 720p packs (scene_01's = 25GB) lived under `dist/BHR/scenes/`; dev runs read `export/generated` → never saw them. HARDLINKED all 18 packs across (same NTFS volume, zero disk); `_pack_valid` confirmed OK at 720p. `main.py --prewarm` gained `--shots`; staged .bat comment updated; docs recommend shot 01 first.
15. **Skip-after-restart stall (late).** Last `[perf]`: `music.load+seek audio.mp3 blocked 316ms` (mp3 decode-scan to 172s). Root fix: exporter bakes **WAV** (`pcm_s16le`, `"audio": "audio.wav"`); measured load 1.1ms + play(start=172.3) 0.1ms. Back-compat: no-ffmpeg export references existing wav/mp3 instead of skipping (also fixes the old "shot 19 audio skipped" class); failed bake falls back to mp3. Mike rebuilt: "skip is smooth now."
16. **Close (late).** Tutorial detect ping; commit `6a26f16`; clanker session_end.

## Key Decisions

- **.bhrx is THE source of truth; scenes/ deleted rather than kept as fixtures.** Rejected: retargeting tests at export/generated (gitignored 9.4GB artifact — tests can't depend on it). Tests that encoded old-tree specifics were deleted; contracts that were tree-independent were retargeted to synthetic shots.
- **laptop_dev chosen as the merged host content; mini_pc_prod scrapped** (Mike's explicit call). Prod values live in the `//host` comment + CLAUDE.md. Consequence accepted: exhibition box needs manual host edits (1920×1080, complexity 1, debug off, Brio/PowerConf name matches).
- **Loader keeps legacy `act_*/shot_*` fallback** so temp-tree test fixtures (test_audio_events, test_capcut_import) run unchanged — new names preferred, old accepted.
- **One global missed-mouseup guard** instead of patching 14 draggers — synthesizes the event all draggers already listen for.
- **Export button saves the posted project over BHR_Experience.bhrx.json** (source of truth stays in sync with what was exported). Rejected: temp-file export (would let .bhrx and export/generated drift).
- **1a kept as a parallel method (`draw_camera_setup_1a`)**, not commented out — "I may go back to 1a later" → revert is a rename.
- **Skeleton keeps the green-when-framed switch** even though designs 1a/1c show static amber — green-means-done is the piece's one learned signal (deliberate deviation, documented).
- **Skipped Pose-gating on armed windows** (perf round 2) — the warm_pose_hz history proves the cold-start hitch class it re-opens; audit flagged risky, I concurred.
- **Incremental pack readers copy UNDER the lock and never hold the mapping** — Windows `os.replace` fails under an open mapping; refcounting rejected as complexity.
- **WAV over threading for the seek stall.** Rejected: off-thread mixer.music calls (SDL_mixer thread-safety risk across pause/volume/stop paths). WAV kills the class at the root for ~250MB total.
- **Warm sweep paced at 1.5× playback, not unthrottled** — disk headroom for the render loop beats warming speed.
- **dist/ never deleted; export/ kept** (expensive to regenerate); `assets/_archive` left alone despite CLAUDE.md calling it deletable (irreversible client deliverables).
- **`.github/`, `assets/`, `State.md`, `.claude/` gitignored** (Mike's call) — CI is local-only now.

## Evidence & Data

**Measured benchmarks (WSL box unless noted):**

| Path | Before | After | Where |
|---|---:|---:|---|
| Warm frame convert (tobytes+fromstring+convert vs frombuffer+convert, 1080p) | 10.54 ms/f | 3.07 ms/f | FrameView zero-copy |
| Fallback decode 1080p→720p (noise JPEG) | 78.5 ms/f | 63.5 ms/f | draft mostly no-op at 2/3 scale |
| Fallback decode 1080p→720p (smooth frame) | 58.4 ms/f | 40.9 ms/f | BILINEAR is the win; draft size stayed (1920,1080) |
| Paused-screen redraw | est 6-10 ms/f | 1.00 ms/f | composite cache |
| Target ring (914×534 class) | 1.26 ms/f | 0.95 ms/f | alloc share removed (WSL underestimates mini-PC alloc cost) |
| mixer.music seek to 172.3s | 316 ms (mp3, field log) | 1.1ms load + **0.1ms** play (WAV) | seek_test.wav 200s/35MB |
| pose_helpers import | ~3 s (Orbbec SDK) | 0.91 s, SDK not in sys.modules | PEP 562 lazy |

**Field log timeline (Mike's exe, the freeze saga):**

| Round | Log signature | Diagnosis | Fix |
|---|---|---|---|
| 1 | first OI freeze "even after restart" | scene_01: 9,148 frames, NO pack; all-or-nothing pack serving | incremental serving (written rows) |
| 2 | skip→`re-sync drift 164.0s`, `music.load+seek 316ms`, `frame gap 385ms`, then 92/94ms gaps; OI at 5525: 81/145ms, pacing 20/s | seek lands 5171 vs cursor ~250; builder race; warm no-op after seek | hint-steered builder + shared decodes |
| 3 | `slow loop 192ms — render 192ms` @5739 + 81-124ms gaps ~5s (frames 5739→5891) | warm sweep = 982 frames ≈ 2.7GB unthrottled disk flood (self-inflicted) | paced warm (60-burst, 45fps, ≤50ms sleeps) |
| 4 | skip stall only (`316ms music.load+seek`) | mp3 decode-scan seek | WAV bake |
| 5 | — | "skip is smooth now" | done |

**Suite trajectory:** 269 (start) → 252 (scenes retirement: −10 test_sequence, −6 meta_flow classes, −1 round-trip; ±region-mirror rework) → 260s→263→269 were PRE-session history → 252 → 257 (+5 warm) → 270 (+13 render fixes) → 272 (+2 zero-copy) → 295 (+23 test_perf_fixes) → 299 (+4 incremental) → 300 (+hint/wrap) → 301 (+pacing) → **303** (+2 tutorial ping). Constant: 5 PortAudio env errors + 1 ffprobe skip on WSL.

**Commits this session (working tree was heavily uncommitted at start; Mike committed mid-session):**

| Hash | Summary |
|---|---|
| `072ae3e` | housekeeping: reorganize workflow and directory names (Mike) |
| `048092e` | fix: resolve audio/visual desync by pre-warming (Mike) |
| `3a8130b` | redesign: camera setup, tutorial, pause menu (Mike) |
| `6a26f16` | perf: kill the first-OI freeze end to end; WAV master audio; tutorial ping (this session's final batch, 23 files +1482/−464) |

**Disk/geometry facts:** repo 33GB (dist 20G→with packs more; export 9.4G; assets 3.6G). scene_01 = 9,148 frames = frames [1..~9148] master; first OI `oi_1` (mouth_proximity_tip "quilt_gesture", region {x:.224,y:.532,w:.533,h:.439}) at segment [5525,5738], intro [1,5524]; oi_2 = raise_hands + voice "freedom" at [6721,6997]; oi_3 = point_quilt_block at [7378,7558]. Packs: 2.76MB/frame at 720p (scene_01 pack 25,292,390,528 bytes), 6.22MB at 1080p. Full prewarm ≈124GB @720p / ≈280GB @1080p. Mike's dist packs: 18 files, 128GB, 720p, hardlinked into export/generated.

**Exporter/audio facts:** 5 baked master-audio shots = exported ids 01/05/08/11/19. `master_audio_offset_ms: 50` global in config. WAV bake: `-c:a pcm_s16le -ar 44100`. Known-open export bug (pre-existing): shot 02 FSM logs `clip not found: 'bhr_scene_09-A.mp3'` — choice_audio file not shipped by export.

**Builder server contract:** `GET /ping` → `{"ok":true,...}`; `POST /export` `{project, no_frames}` → chunked text ending `[exit N]`; port 8798; writes canonical .bhrx first.

## Code Analysis

- `FrameCacheManager._building: dict[Path, dict]` = `{"mm": np.memmap, "written": bytearray(n), "hint": int|None, "count": int}`. Readers: `get_frame_bytes` serves written rows via `bytes(b["mm"][i])` under `self._lock`; fallback decode contributes rows back (`np.frombuffer(data,...).reshape(h,w,3)`). Builder pops `_building` under lock BEFORE `del mm; os.replace(tmp, pack)`.
- `warm_segment(frames_dir, start_idx, end_idx)` — 0-based FrameView indices, non-blocking, never raises; live pack → paced `_warm_pack_pages` (touches `mm[i,:,0,0].sum()` — one byte/row < 4KB page); building → sets `hint`; no pack → bounded pre-decode dict (`_WARM_DECODE_MAX=25`, `_WARM_READY_CAP=48`).
- Constants: `_WARM_BURST_FRAMES=60`, `_WARM_RATE_FPS=45.0`; render `_CAPTION_CARD_MAX=8`, ring cache cap 32, rot-icon cache 128 (5° angle, 16-step alpha buckets); FrameView LRU `_cap=240` surfaces.
- `get_frame_buffer(frames_dir, i)` → `(mm[i], (w,h))` zero-copy view or None; caller must wrap+convert immediately (priority shot never evicted mid-play).
- `_pack_path(frames_dir) = frames_dir.parent / "framecache.npy"` — pack sits in the SHOT dir, sibling of frames/. `_pack_valid` checks count+dims only → resolution-bound.
- `VoiceEngine._process_chunk`: DSP always-on; recognizer gated on `not input_locked and windows non-empty` (pruned under `_lock` via `_prune_expired_locked`); `_vosk_idle` flag → `Reset()` on first fed chunk after reopen (voice thread — KaldiRecognizer not thread-safe).
- `OrbbecCapture.frame_is_rgb = True`; depth stored raw uint16 + `depth_scale`, scaled per-patch in `depth_at(..., scale=1.0)` (float input passes through — test contract).
- `engines/depth/__init__.py` PEP 562 `__getattr__` lazy-imports orbbec_camera; `ORBBEC_AVAILABLE` via `importlib.util.find_spec`.
- Exporter master-audio block (~line 1349): no-ffmpeg → reference existing audio.wav/mp3; bake failure → mp3 fallback → else pop "audio".
- `TutorialEngine(config, bus, gesture, detect_sfx=None)`; success emits `oi_flash` + `play_sfx {path, channel:2}`. main.py resolves via `scenes_root.glob("scenes/*/audio/detect.mp3")`.
- main.py restart: `K_r` (not during camera setup) → skip tutorial if active, `tutorial_was_active=False` (blocks auto-resume), `mixer.stop()`, `player.start(0)`, pause all, `tutorial.done=False`.
- Builder server: `Handler(SimpleHTTPRequestHandler)` with `directory=tools/experience_builder`; `_export_lock` non-blocking → 409.

## Files Changed

### Runtime engines
- `engines/frame_cache.py` — draft-fallback decode, warm_segment (paced pages / hint / bounded pre-decode), incremental build serving (bitmap+hint+count, write-back), `get_frame_buffer`
- `engines/render_engine.py` — zero-copy FrameView, warm hint on play_segment, entry-sound prefetch (shot_load + prefetch_shot), caption memo+borderless, FX dirty-rect, ring/rot-icon/paused/star caches, camera setup 1c (+1a kept), tutorial 2a card (diamonds/serif/ovals), pause serif+R key, `_serif_font`/`_tracked_label`
- `engines/gesture_engine.py` — RGB fast path, PoseDepth single-construct, dead fn removed
- `engines/voice_engine.py` — Vosk on voice thread + window gating
- `engines/depth/__init__.py`, `engines/depth/orbbec_camera.py` — lazy SDK, raw uint16 depth
- `engines/tutorial_engine.py` — 2a steps (icons removed, oval target at {x:.68,y:.45,w:.20,h:.30}, "glowing circle"), detect_sfx ping
- `engines/shot_sequence_player.py` — `_timing_cache`; `engines/sequence_loader.py` — `_shot_dir` new/legacy naming; `engines/palette.py` — camera-setup now themed (scope note)

### Scripts & tools
- `scripts/export_experience.py` — scenes/scene_NN naming, `export/generated` default, render-reuse-without-ffmpeg, WAV master bake
- `scripts/build_exe.py` — .bhrx export step, `--skip-export`, no config/ staging, .bat mentions shot 01 + `--shots`
- `scripts/builder_server.py` — NEW; `scripts/prewarm_frame_cache.py`, `main.py` — host-config resolution, `--shots`, R hotkey, detect_sfx wiring
- `scripts/gesture_tuner.py` / `voice_tuner.py` — config.json[host] persistence, RGB compat
- `tools/experience_builder/js/app.js` (mouseup guard, export run UI), `js/graph.js`, `js/timeline.js` (caption border), `index.html`, `css/builder.css`

### Tests (303 total)
- NEW: `tests/test_perf_fixes.py` (23); heavily extended: `test_frame_cache.py` (13), `test_render.py` (70), `test_tutorial.py` (+2 detect ping); retired: `test_sequence.py`, real-tree classes in `test_meta_flow.py`

### Deleted
- `scenes/` (tracked, in history), `scripts/scenes_to_builder.py`, `camera_test.py`, `rename_frames.py`, `simplify_builder_project.py`, `config/host_profiles/`, 4 `.bhrx` backups, `build/`, `Log/`, mockup HTML

### Local-only (gitignored)
- `State.md` (full session log), `.github/workflows/tests.yml`, `plans/handoffs/` (this file — plans/ not ignored, actually commit-able), hardlinked packs in `export/generated/scenes/*/framecache.npy`

## User Feedback & Preferences (REQUIRED — never omit)

- **"just ask me if it seems what I'm saying contradicts what you're seeing as it may help bridge assumptions"** — SAVED TO MEMORY (`ask-when-reports-contradict.md`). He tests with `dist/BHR/BHR.exe`, not source runs. Ask which binary/tree before assuming.
- "Plan and send out agents" / "fix those with agents that you verify" — likes agent-parallel work but expects the main agent to verify diffs personally.
- "I may go back to 1a later" — keep alternatives revertable, don't destroy.
- "these were all artifacts of the old workflow which are not needed anymore" — comfortable deleting aggressively once superseded (git history is the archive).
- Gitignore `.github`, `assets`, `State.md` — wants a lean tracked surface; CI on GitHub not currently a goal.
- Names things directly: "scenes_generated could just be generated", "act_01_experience can just be scenes", "scene_xx instead of shot_xx" — prefers plain naming.
- Visual taste: "use circles/ovals instead of rectangles", "diamonds are too close", "no borders" on subtitles, remove background hand icons — leans minimal/clean; iterates from screenshots.
- Commits himself mid-session with his own message style (lowercase, prefix like "fix:", "redesign:").
- Appreciates explanation: asked for the packs/builder debrief, closed with "I also learned that WAV files are better for seeking."
- End-of-day: "good job on this session, you're an absolute beast!"

## Where We're Going

1. **Next Windows rebuild** picks up the tutorial detect ping (already committed) — everything else is field-verified.
2. **Exhibition config decision:** set `config.json host.display.resolution` (1920×1080?) + `performance.mediapipe_complexity: 1`, `debug_overlay: false`, Brio/PowerConf `device_name_match` — THEN re-prewarm at that resolution (~280GB at 1080p; current 128GB of packs are 720p-bound).
3. **Open export bug** (pre-existing, in State.md): shot 02 `clip not found: 'bhr_scene_09-A.mp3'` — choice_audio file not shipped into the shot's audio/ dir; needs an export-side fix + re-export.
4. Optional: push `main` to remote (nothing pushed this session; git-lfs missing on WSL — install before pushing if LFS-tracked files matter).
5. Optional polish: caption alignment weak spots (scene 10 urgent 7/12, epilogue 19/28); frame-pack retention strategy for 1080p (~280GB) before opening night.

## Risks & Blockers

- **Pack/resolution coupling:** changing display resolution silently invalidates every pack (`_pack_valid` = count+dims). The 128GB investment is 720p-only.
- **Exhibition host config not yet set** — box currently boots with laptop values (windowed 720p, complexity 0, debug on).
- WSL can't run PyInstaller for Windows, ffmpeg missing, PortAudio missing → 5 env test errors are permanent here; builds/bakes happen on Mike's box.
- git-lfs configured but absent on WSL (post-commit hook warns; commit succeeded).
- The 23 remaining exporter warnings (re-timed clip renders) resolve only on an ffmpeg-equipped export.

## Open Questions

- Where should the exhibition land on pack retention at 1080p (~280GB vs the 1TB disk, delete-on-evict never implemented)?
- Should CI (.github, currently gitignored) ever go to GitHub, or stay local?
- Is the one-time ~316ms→now-0.1ms seek path worth revisiting for the drift re-sync case too (it already uses the same WAV path — believed covered, unverified in field).

## Appendix A — Raw Field Log Excerpts (primary evidence)

Round-2 log (skip path), verbatim key lines:

```
[ShotPlayer] skip prologue -> frame 5171 (in-shot)
[RenderEngine] play_segment [5171-5524]  loop=False
[RenderEngine] audio re-sync: picture 172.3s vs audio 8.3s (drift 164.0s)
[perf] music.load+seek audio.mp3 blocked the main thread for 316ms
[RenderEngine] shot audio re-synced to 172.3s (+0ms load latency)
[perf] bus 'play_segment' -> RenderEngine._on_play_segment took 326ms
[perf] frame gap 385ms (shot 01, frame 249)
[perf] pacing: 20 updates in the last second (target ~30, shot 01, frame 249)
[perf] frame gap 92ms (shot 01, frame 5195)
[perf] frame gap 94ms (shot 01, frame 5213)
```

(Note the `+0ms load latency` vs the 316ms warn — the old compensation measured
a different span than the stall; superseded by the WAV bake, never separately
fixed.)

Round-3 log (straight-through, rebuilt exe), verbatim key lines:

```
[ShotPlayer] FSM -> between_1  loop=False
[RenderEngine] play_segment [5739-6720]  loop=False  carry=8ms
[perf] pose inference 201ms (capture thread — does not block the picture)
[perf] frame gap 183ms (shot 01, frame 5739)
[perf] slow loop 192ms — gesture 0ms  player 0ms  audio 0ms  render 192ms
[perf] pacing: 20 updates in the last second (target ~30, shot 01, frame 5750)
[perf] frame gap 94ms (shot 01, frame 5761)  ... 81/83/124/81ms through frame 5891
[perf] pacing: 18 updates in the last second (target ~30, shot 01, frame 5810)
```

Diagnostic reading that cracked it: all four phase buckets 0 except render →
the stall was inside frame serving; 5739→5891 ≈ 5s matches 982 frames ×
2.76MB ≈ 2.7GB streamed at disk speed → the warm sweep was the flood.

## Appendix B — Perf Audit Findings NOT Acted On (cleared or deferred)

These were audited and explicitly cleared/deferred — do NOT re-investigate
from scratch:

| Item | Verdict | Reason |
|---|---|---|
| Gate Pose inference on armed windows (~20-40ms/frame of a core during playback) | DEFERRED — risky | Re-opens the warm_pose_hz cold-start hitch class (GIL probe history proved pose can't block the picture; gating creates cold detectors at window-open) |
| Detector rules shared-math caching | CLEARED | Only 1-2 detectors dispatch/frame; helpers called ≤2×; all histories time-window-pruned, no O(t) growth |
| audio_mixer per-tick anchor scan | CLEARED | Tens of events/shot; pre-sorted cursor would save µs |
| `_effective_timing` per-HOLD-tick | FIXED (memo) | ~µs but free win |
| event_bus per-handler perf_counter pairs | CLEARED | Emit volume is event-driven; not a steady-state cost |
| Voice DSP float64 upcast (32KB, 4×/s) | CLEARED | Irrelevant next to the Vosk gating win |
| `debug_info()` built per tick + voice lock cross-thread | NOTED, minor | ~0.05ms; split cursor_state() if jitter ever matters |
| Skeleton mini-panel / debug overlay per-frame renders | NOTED, minor | 0.2-1.5ms only when toggled on (diagnostic surfaces) |
| `smoothscale` in frame path | CLEARED | Packs are built at display resolution; no scaling in the frame path |
| `forward_point` named-region ring branch | DEAD in shipped tree | zero `target_region` matches in export/generated |

## Appendix C — Design Canvas Reference (for future design rounds)

Project: claude.ai/design `871c95af-2653-44b3-bb22-fec75eca2bb9`
("Camera and tutorial redesigns", owner Mike Peng). File read via DesignSync
`get_file` — `Camera & Tutorial Redesigns.dc.html`; `support.js` is only the
dc-canvas React runtime (sc-if/props plumbing), not spec.

Design→palette token mapping (all 1:1, no palette changes were needed):

| Design hex | palette.py token | RGB |
|---|---|---|
| #ffc954 | LANTERN | (255,201,84) |
| #b08434 / #b68235 | LANTERN_DIM | (176,132,52) |
| #0c0e18 | NIGHT | (12,14,24) |
| #070910 | NIGHT_DEEP | (7,9,16) |
| #18141e | CLOTH | (24,20,30) |
| #f0ece4 | LINEN | (240,236,228) |
| #a8a298 | LINEN_DIM | (168,162,152) |
| #848078 | LINEN_FAINT | (132,128,120) |
| #fffaeb | NORTH_STAR | (255,250,235) |
| rgba(255,240,214,.13-.18) | EDGE_RGBA | (255,240,214,34) |
| #3ce05a (success green text) | ≈ HAND_L, kept SUCCESS (0,240,96) | deliberate |

Variants NOT implemented (available in the canvas if Mike asks): 1b "Archive
Plate" (framed photo plate + side column), 2b "Editorial Spread" (giant serif
numeral + two-column), 2c "Constellation Path" (star-dot progress line).
Fonts in canvas: Cormorant Garamond + Lora → kiosk ladder
`garamond,georgia,palatinolinotype,book antiqua,timesnewroman` via
`RenderEngine._serif_font(size, italic, bold)` (cached);
`_tracked_label(font, [(text,color),...], tracking_px)` for letter-spaced
small caps (pygame has no tracking — per-glyph render).

Implementation geometry (1c North Star Arch): arch h=0.62·sh, w=h·(220/290),
top y=0.155·sh; mask via `pygame.draw.rect` border_top radii = w//2, bottom
radii = max(3, h//72), cached per size (`_arch_mask`); feed COVER-cropped;
star r=0.020·sh at y=arch_top−0.055·sh, halo = 6 stacked rings alpha 8+i·7.
Tutorial 2a: diamonds side=0.020·sh, centre spacing = 2·side+max(10,0.020·sh)
(the "too close" bug was spacing=side+8 — a rotated square spans 2·side);
figure box centred fw=0.20·sw fh=0.34·sh at y=0.50·sh with mini-panel nudge;
prompt max width 0.58·sw, leading ×1.35.

## Appendix D — Cleanup Inventory (chunk 1, for the record)

Deleted (all tracked → recoverable from git history unless noted):
`BHR_Experience.bhrx.backup.json` (63KB Jul18), `.pre_simplify.json` (54KB
Jul17), `.alignbak` (116KB Jul28, was untracked), `.capbak` (78KB Jul27,
untracked), `build/` (105MB PyInstaller, regenerable), `Log/OrbbecSDK.log.txt`,
`scripts/camera_test.py` (zero refs), `scripts/rename_frames.py` (artist-era),
`scripts/simplify_builder_project.py` (one-shot, already run),
`assets/docs/Experience Builder (standalone).html` (4.6MB mockup),
`engines/gesture_engine.hands_detected()` (the ONLY dead function found by a
full AST cross-reference of the runtime), `import numpy` (gesture_tuner),
`import shutil` (setup_voice), `projector_audio_offset_ms` (all 3 profiles;
read by no code), UTF-8 BOM on render_engine.py.

Kept deliberately: `export/` (9.4GB, expensive), `dist/` (packs),
`docs/BHR ASSET TRACKER.xlsx` (13.4MB production record),
`docs/vision_optimization.md` (CLAUDE.md links it), legacy detector modes
(`mode:"sweep"` test-pinned; `require_scan`/`reference:"hips"`/`target_x|y`
uncovered escape hatches), `tools/experience_builder/js/project_data.js`
(file:// zero-setup bundle), `assets/_archive` (irreversible deliverables),
unread config keys (`start_scene`, `keyboard_fallback_enabled`,
`feedback_layer`, `audio_mix`, 6 detection_thresholds — authored spec values).

## Appendix E — Config Host Section (as merged; prod values to set)

Current `config.json["host"]` = laptop_dev content: camera dshow/null-match/
1280×720/30fps; mic 16k mono; display windowed 1280×720; performance
complexity 0, fps cap 30, debug ON; voice_tuning {hum_rms .008, hum_min 400,
whisper_max −20}; gesture_tuning = 28 entries (unravel prox_frac 1.75, paddle
waist_y_offset −0.18, run_arms −0.09, three_knock wrist_y_offset −0.05...).

For the exhibition mini PC (from the `//host` comment + CLAUDE.md): fullscreen
true, resolution [1920,1080], mediapipe_complexity 1, debug_overlay false,
camera device_name_match "Logitech 4K Pro Webcam" (fallback "Logitech BRIO"),
microphone "PowerConf". Tuners now save into config.json["host"] directly
(`_save_tune_params` aborts on read failure rather than gutting the file).

## Appendix F — Test Names Added This Session (for grep)

- test_frame_cache: `TestWarmSegment` (fallback_decode_exact_size_rgb,
  warm_predecodes_and_frame_bytes_consumes, warm_is_safe_noop,
  warm_touches_live_pack_without_closing_it, warm_ready_dict_is_bounded,
  pack_page_warm_is_paced_past_the_burst), `TestIncrementalBuildServe`
  (serves_written_rows_without_decoding,
  fallback_past_cursor_contributes_to_the_pack,
  warm_segment_hints_the_builder_while_building,
  hint_reorders_the_build_and_wraps_to_fill,
  build_registers_progress_and_unregisters_on_finish), `TestEviction` (2).
- test_render: `TestSegmentFrameWarm` (3), `TestPrefetchEntrySounds` (4),
  `TestCaptionCardMemo` (3), `TestDrawFxDirtyRect` (3),
  `TestFrameViewZeroCopy` (2), plus figure/palette/camera pins updated for
  1c/2a (figure box centred + nudge, oval accents).
- test_perf_fixes (NEW, 23): `TestDepthAtScale` (5), `TestLazyOrbbecImport`
  (3), `TestCaptureRgbFastPath` (4), `TestPoseDepthSingleConstruction` (2),
  `TestVoskOffBootPath` (3), `TestVoskWindowGate` (6) — voice tests stub
  sounddevice so they run without PortAudio.
- test_tutorial: `TestDetectSound` (success_plays_detect_sfx_when_wired,
  no_sfx_event_without_a_path).
- test_experience_export: region round-trip reworked to
  `test_region_screen_to_raw_mirror` (importer half retired with
  scenes_to_builder); metadata paths now `scene_{id}`.

## Appendix G — Builder Server & Export Button Contract

`scripts/builder_server.py` — stdlib only; `py -3.12 scripts/builder_server.py
[--port N]` (default 8798, binds 127.0.0.1 only). Serves
`tools/experience_builder/` statically; `GET /ping` →
`{"ok": true, "project": "BHR_Experience.bhrx.json"}`; `POST /export` body
`{"project": <obj>, "no_frames": bool}` → writes the project to the canonical
.bhrx (`json.dumps(indent=2, ensure_ascii=False)` — byte-stable with the
builder's own saves), runs export_experience.py via Popen, streams stdout
chunked, final line `[exit N]`; concurrent export → 409. CORS: `ACAO: *`,
`Access-Control-Allow-Private-Network: true` on OPTIONS (file:// null-origin
works in Chrome/Edge). UI: `openExport()` pings (800ms AbortController);
server up → `#export-run-ui` with Run button + `--no-frames` checkbox +
`#export-log` streamed via `res.body.getReader()`; down → command fallback +
tip. Tested end-to-end via curl during the session, including a real
metadata-only export (`[exit 0]`, .bhrx round-tripped json-equal).

## Appendix H — Session Timeline (chunked map)

| Chunk | Topic | Key outputs |
|---|---|---|
| early | clanker onboard, baseline, review agents | symlink + registry; venv; 264/269 baseline; CI+State.md seeded |
| early | cleanup pass | 2 Explore agents; deletions + doc fixes (Appendix D) |
| early-mid | .bhrx SoT + renames | scenes/ retired; export/generated/scenes/scene_NN; loader fallback; suite 269→252; CLAUDE.md changelog entry |
| early-mid | config merge | host section; profiles dir deleted; tuners rewired; prewarm/main resolution |
| early-mid | README rewrite | full rewrite around pipeline diagram; dead animator workflow dropped |
| mid | builder UX | missed-mouseup guard; native-drag fix; export button + server |
| mid | restart hotkey + copy-chain | R key; get_frame_buffer zero-copy 10.5→3.1ms |
| mid | perf round 1 | 2 audits + 2 fix agents; warm_segment; draft; prefetch; FX dirty-rect; caption memo |
| mid | export & build | final export run; Windows-build limitation explained; --keep-packs Q&A |
| late | design picks | 1a+2a → 1c swap; diamonds/icons/ovals/borderless iterations from screenshots |
| late | perf round 2 | 2 audits + 1 fix agent + my render batch; Vosk gating; lazy SDK |
| late | freeze saga r1-r3 | incremental serve → hint steering + write-back → paced warm |
| late | prewarm mystery | dist/BHR packs found; hardlinks; --shots; docs/bat |
| late | WAV bake | skip 316ms → 0.1ms; field-confirmed smooth |
| close | tutorial ping, commit 6a26f16, clanker end, this handoff | — |

## Appendix I — Frame-Serving Mechanism (as debriefed to Mike)

Three tiers, best available wins, per frame:
1. **Finished pack** — mmapped `framecache.npy`, zero-copy wrap+convert
   (~3ms). Produced by prewarm OR by a completed runtime build; files never
   deleted by the runtime (eviction only unmaps).
2. **Pack mid-build** — the builder's `.building` memmap served row-by-row
   via the written bitmap (~2-3ms copy under lock). The playhead steers the
   build cursor (warm_segment hint; wrap-fill before finalize); main-thread
   fallback decodes write back into it; either thread can complete it, then
   rename → tier 1 for all future runs.
3. **Decode fallback** — main-thread PIL decode (30-45ms), now rare: only
   frames the playhead wins in the race, and each one feeds the pack.

Prewarm is therefore an optimization, not a requirement: first run of an
unpacked shot rides tier 2 near-realtime and finalizes the pack organically.
`warm_segment` on finished packs = paced page pre-touch (prewarmed file ≠
resident pages). `--keep-packs` rebuilds preserve dist packs (staging skips
framecache.npy). Packs live at `<shot_dir>/framecache.npy` (sibling of
frames/), validity = frame count + dimensions only → resolution-bound.

## Appendix J — Docs Touched This Session

- `CLAUDE.md`: new top changelog entry (Aug 2026 .bhrx SoT + renames); Host
  Section rewrite (replaced Host Profile Resolution); file-structure tree
  (export/generated); test-count lines; projector_audio_offset_ms ×3;
  pose-hand supersession note; tuner-save locations; kiosk-boot note.
- `README.md`: full rewrite (pipeline diagram, quickstart, keys incl. R,
  builder+server, layout, tests ~250→"~250", config/host, audio, packaging);
  later prewarm example updated to lead with shot 01.
- `docs/HOW_TO_RUN.md`: R key row; `docs/packaging.md`: host-section
  references, export/generated; `docs/START_SHOT_MAPPING.txt`: path rename.
- `tools/experience_builder/README.md`: server workflow, export path,
  scenes_to_builder retirement; `index.html` export dialog copy.
- `State.md` (local): 20+ session-log entries — the fine-grained record.

## Appendix K — Command Crib (Windows box)

```bat
:: Full rebuild preserving packs (re-exports .bhrx first; bakes WAVs w/ ffmpeg)
py -3.12 scripts\build_exe.py --keep-packs

:: Selective prewarm (recommended set; works in the frozen exe too)
BHR.exe --prewarm --shots 01,02,03,04,09,10,12
py -3.12 scripts\prewarm_frame_cache.py --shots 01,02,03,04,09,10,12

:: Run from source against the dev tree (default export\generated)
py -3.12 main.py
py -3.12 main.py --scenes dist\BHR\scenes     :: run source against dist tree

:: Builder with one-click export
py -3.12 scripts\builder_server.py            :: then open tools\experience_builder\index.html
```

WSL: suite via the scratchpad venv (see Quick Start); ffmpeg/PyInstaller/
PortAudio absent by design — bakes and builds happen on Windows.

## Appendix L — Process Notes That Worked (for future sessions)

- Mike's console logs were the convergence engine: every freeze round, the
  `[perf]` instrumentation (frame-gap watchdog 80ms, phase-bucket slow-loop
  line, pacing monitor, bus-handler timing) named the culprit. When chasing
  runtime behaviour, ask for the log excerpt FIRST.
- Verify agents' work personally: the frame-cache agent's draft() claim was
  overstated (no-ops above 1/2 scale) — caught only by my own benchmark.
  The render agent's work was solid but its warm sweep needed pacing later.
- Parallel agents on DISJOINT FILES with an explicit shared API contract
  (warm_segment signature specified identically in both prompts) worked
  cleanly twice.
- Renaming multi-GB trees in place (`mv` + loader fallback) preserved 9.4GB
  frames + 128GB packs across two naming migrations with zero re-generation.
- Hardlinks across same-volume trees = free pack duplication.
- Benchmarks were run for every headline claim (copy-chain, draft, WAV seek,
  paused cache, ring) — numbers in Evidence are measured, not estimated,
  except where marked "est".

## Quick Start for Next Session

```bash
# Restore context — read these first
cat "/mnt/d/Archived Projects/BHR/State.md"          # per-round session log (richest)
# CLAUDE.md loads automatically; Aug 2026 changelog entry describes this session's shape

# Key files
# engines/frame_cache.py        — incremental/steered/paced pack serving
# engines/render_engine.py      — FrameView zero-copy + all draw caches + 1c/2a screens
# scripts/export_experience.py  — WAV bake + render reuse (~line 1100-1400)
# scripts/builder_server.py     — export API

# Verify current state (WSL venv)
SP=/tmp/claude-1000/-mnt-d-Archived-Projects-BHR/cfb4a106-2b0a-45be-ae67-86ec54d7a189/scratchpad
cd "/mnt/d/Archived Projects/BHR" && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  $SP/bhr-venv/bin/python -m unittest discover -s tests
# expect: Ran 303, FAILED (errors=5 PortAudio-only, skipped=1)
# NOTE: venv lives in /tmp — may be gone; rebuild per State.md "Dev environment notes"

# Next action
# Fix the shot-02 choice_audio export bug: exporter must ship the authored
# choice_audio file (bhr_scene_09-A.mp3) into the choice shot's audio/ dir —
# see collect_sfx_names/on_enter_audio walk in scripts/export_experience.py,
# then metadata-only re-export + verify the pick/switch no longer logs
# "clip not found".
```

---

*Handoff generated with the Chunked mining pass (~690K context tokens, 100+
tool calls, 4 chronological segments). Self-validated: chain seq 1 standalone,
all identifiers current (no parent → no stale-refs), evidence includes 6+
tables with measured numbers, Quick Start's next action is concrete (shot-02
choice_audio export bug). Session closed via clanker (end id e53b1228eec6c207).*

## Session Closed
**Closed at:** 2026-08-31 18:35 EDT
**Commit:** (this handoff commit — see git log)
**Session status:** Handed off to next session
