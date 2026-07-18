# Packaging BHR as a Windows Executable & Releasing on GitHub

Companion to `scripts/build_exe.py`. Everything below runs from the BHR project root
on Windows with `py -3.12`.

## 1. Building the exe

One-time setup:

```powershell
py -3.12 -m pip install pyinstaller
```

Build:

```powershell
py -3.12 scripts/build_exe.py
```

This produces **`dist/BHR/`** — a self-contained, relocatable folder:

```
dist/BHR/
  BHR.exe          the app (console build: logs stay visible / capturable)
  _internal/       bundled Python + MediaPipe + Vosk + pygame + OpenCV
  config.json
  config/          host profiles
  scenes/          sequence.json + shot metadata + shot audio (frames NOT included by default)
  models/          vosk model (staged automatically if present locally)
```

Useful flags:

| Flag | Effect |
|---|---|
| `--scenes-root PATH` | Stage a different tree as the packaged `scenes/` (e.g. `export/scenes_generated` — the Builder-exported experience) |
| `--with-frames` | Also stage every shot's `frames/` folder (multi-GB — single-folder deploys) |
| `--link-frames` | Hardlink staged scene files instead of copying — zero extra disk on the same NTFS volume; they become real files when the folder is copied to another drive |
| `--with-assets` | Also stage `assets/` (storyboard pages, reference video). `assets/hand_icons/` is always staged regardless — the runtime cursors live there |
| `--zip --version v1.0.0` | Produce `dist/BHR-v1.0.0-win64.zip` for a release |
| `--skip-build` | Re-stage content / re-zip without rebuilding the exe |
| `--dry-run` | Print the PyInstaller command and exit |

The Builder-exported single-folder build used for playtests:

```powershell
py -3.12 scripts/build_exe.py --scenes-root export/scenes_generated --with-frames --link-frames
```

Disk note: at runtime the frame cache writes `framecache.npy` packs NEXT TO each
shot's frames — inside `dist/BHR/scenes/` — and eviction never deletes them
(CLAUDE.md open consideration). At 1920×1080 a full playthrough can accumulate
100+ GB of packs, so the target drive needs the staged folder PLUS pack
headroom (the 1TB mini PC is fine; a small SSD is not).

Design choices baked in (don't change casually):
- **one-dir, not one-file** — MediaPipe/Vosk make one-file exes slow to start
  (unpacks to temp every launch) and prone to antivirus quarantine.
- **console, not windowed** — the kiosk launcher captures stdout/stderr to the
  rolling log (`%LOCALAPPDATA%/BHR/logs/`, see CLAUDE.md auto-launch section).
- The app resolves `config.json`, `scenes/`, `models/` **relative to `BHR.exe`**
  (frozen-mode checks in `main.py` and `engines/voice_engine.py`), so the folder
  can live anywhere — `C:\BHR\` recommended on the mini PC (keep it out of
  OneDrive-synced paths).

## 2. Installing on a target machine (mini PC / exhibition)

1. Copy the `BHR/` folder (unzip the release) to `C:\BHR\`.
2. **Frames**: if the build was made without `--with-frames`, copy the full
   `scenes/` tree (with frames) over `C:\BHR\scenes\`, or copy `final_frames/`
   plus a Python install and run `copy_frames.py`. Frames are required for
   playback; shots without frames auto-skip.
3. **Vosk model** (if `models/` wasn't staged): download
   `vosk-model-small-en-us-0.15` from https://alphacephei.com/vosk/models,
   unzip to `C:\BHR\models\vosk-model-small-en-us-0.15\`.
4. Plug in the Brio webcam + PowerConf mic, then smoke test:
   ```powershell
   C:\BHR\BHR.exe --dry-run        # no camera/audio needed; walks all shots
   C:\BHR\BHR.exe --profile mini_pc_prod
   ```
5. First real launch: app starts **paused** — press **Space** to begin.
   Keys: Space/P pause·play · F11/F fullscreen · D debug overlay · S skip
   prologue · Up/Down volume (while paused) · → advance shot · Esc quit.
6. For unattended exhibition boot, follow the CLAUDE.md *Windows-Specific
   Considerations* section (auto-login, Startup-folder wrapper with
   `BHR_HOST_PROFILE=mini_pc_prod`, Defender exclusion for `C:\BHR`, High
   Performance power plan, USB selective suspend off).

Defender note: unsigned PyInstaller exes sometimes trigger SmartScreen
("Windows protected your PC") — click *More info → Run anyway*, and add
`C:\BHR` to the Defender exclusion list (also a frame-rate requirement, see
CLAUDE.md).

## 3. Publishing a GitHub release

Versioning: tag releases `vMAJOR.MINOR.PATCH` (e.g. `v1.0.0`; bump MINOR for
content/wiring additions, PATCH for fixes).

```powershell
# 1. Make sure the tree is committed and pushed
git status
git push

# 2. Build the release zip
py -3.12 scripts/build_exe.py --zip --version v1.0.0

# 3. Tag the commit the build came from
git tag v1.0.0
git push origin v1.0.0

# 4. Create the release and attach the zip (gh CLI)
gh release create v1.0.0 dist/BHR-v1.0.0-win64.zip `
  --title "BHR v1.0.0" `
  --notes "App bundle (no frames). Deploy frames separately per docs/packaging.md."
```

No `gh`? GitHub web UI: **Releases → Draft a new release → Choose a tag →
attach the zip → Publish**.

### What goes in the release (and what doesn't)

- **Attach:** `BHR-vX.Y.Z-win64.zip` (app + config + scene metadata + vosk
  model — typically well under 1 GB).
- **Do NOT attach frames.** GitHub caps each release asset at **2 GB**, and the
  full frame set exceeds that. Options, in order of preference:
  1. Deploy frames out-of-band (external drive / Drive share) — they change on
     a different cadence than code anyway.
  2. If you want them on the release page, split the frames into per-act zips
     under 2 GB each (`scenes/act_NN_*` each zip) and attach those.
  Git LFS is *not* recommended here — LFS bandwidth quotas make multi-GB frame
  pulls painful for collaborators.
- Release notes: mention which script/spec revision the shot wiring matches and
  any host-profile changes the mini PC needs.

### Suggested release checklist

- [ ] `py -3.12 main.py --dry-run` passes locally
- [ ] `py -3.12 scripts/build_exe.py --zip --version vX.Y.Z`
- [ ] `dist\BHR\BHR.exe --dry-run` passes (frozen smoke test)
- [ ] Real-camera spot check: one region-fork shot (09) + one OI shot (58)
- [ ] Tag pushed, release created, zip attached
- [ ] Install-tested on the mini PC from the downloaded zip, not the dev tree
