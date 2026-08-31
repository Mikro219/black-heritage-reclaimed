"""
build_exe.py — package BHR into a Windows executable with PyInstaller.

THE .bhrx PROJECT IS THE SINGLE SOURCE OF TRUTH: by default the build first
runs scripts/export_experience.py on BHR_Experience.bhrx.json (sound files
resolve from assets/) and stages the generated tree (export/generated)
as the packaged scenes/. Use --skip-export to reuse the existing export, or
--scenes-root to stage a different tree.

Run from the BHR project root:
    py -3.12 scripts/build_exe.py                     # export .bhrx + build dist/BHR/BHR.exe
    py -3.12 scripts/build_exe.py --keep-packs        # rebuild WITHOUT wiping dist/BHR
                                                      # (prewarmed framecache packs survive)
    py -3.12 scripts/build_exe.py --skip-export       # stage the existing export as-is
    py -3.12 scripts/build_exe.py --with-frames       # also stage scene frames (GBs)
    py -3.12 scripts/build_exe.py --with-assets       # also stage assets/ (storyboard, video)
    py -3.12 scripts/build_exe.py --zip --version v1.0.0   # produce BHR-v1.0.0-win64.zip
    py -3.12 scripts/build_exe.py --dry-run           # print the PyInstaller command and exit

Output layout (one-dir build — do NOT use one-file, MediaPipe/Vosk startup and
antivirus behaviour are much worse):

    dist/BHR/
      BHR.exe            <- console app (logs to stdout; kiosk launcher captures them)
      _internal/         <- Python runtime + libraries (PyInstaller-managed)
      config.json        <- includes the "host" hardware section (edit on target)
      scenes/            <- the exported .bhrx tree: sequence.json + metadata/audio
                            (+ frames/ only with --with-frames)
      models/            <- vosk model (staged if present locally)
      assets/            <- only with --with-assets

The app resolves everything relative to BHR.exe (see the frozen-mode checks in
main.py and engines/voice_engine.py), so the dist/BHR folder is fully relocatable:
copy it anywhere on the target machine and run BHR.exe.

Frames note: shot frames are multi-GB and usually deployed separately (copy the
scenes/ tree over the staged one, or run scripts/copy_frames.py on the target
machine against final_frames/). --with-frames stages them into dist for a
single-folder deployment.

See docs/packaging.md for install + GitHub release instructions.
"""

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
APP_NAME = "BHR"

# Packages whose data files (models, DLLs, graph binaries) must be bundled.
# pyorbbecsdk ships the Gemini 335 native DLLs; harmless if absent on target
# hardware (try_open_orbbec probes and falls back to the webcam).
COLLECT_ALL = ["mediapipe", "vosk", "sounddevice", "pyorbbecsdk"]


def pyinstaller_cmd(distpath: Path = DIST) -> list[str]:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(ROOT / "main.py"),
        "--name", APP_NAME,
        "--noconfirm",
        "--clean",
        # one-dir console build: fastest startup, logs visible/capturable
        "--onedir",
        "--console",
        "--distpath", str(distpath),
        "--workpath", str(BUILD),
        "--specpath", str(BUILD),
    ]
    for pkg in COLLECT_ALL:
        cmd += ["--collect-all", pkg]
    return cmd


def _chmod_retry(func, path, excinfo):
    """rmtree helper: clear read-only and retry once."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def swap_in_build(fresh_app: Path, app_dir: Path) -> None:
    """Move a fresh PyInstaller output into an existing dist/BHR, replacing
    only what PyInstaller produced (BHR.exe + _internal/ on PyInstaller 6;
    every top-level entry it emitted on older layouts) and touching NOTHING
    else — scenes/ with its multi-GB prewarmed framecache.npy packs, models/,
    config and assets all stay in place. Same-volume moves are renames, so
    this is fast regardless of build size.

    A running BHR.exe (or an antivirus scan) denies DELETION of the old files
    — but Windows does allow RENAMING a running exe, so locked entries are
    shoved aside as *.stale (the standard updater trick) and swept on the
    next successful swap."""
    app_dir.mkdir(parents=True, exist_ok=True)
    for old in app_dir.glob("*.stale"):
        try:
            shutil.rmtree(old, onexc=_chmod_retry) if old.is_dir() \
                else old.unlink()
        except OSError:
            pass   # still locked — swept on a later build
    for entry in fresh_app.iterdir():
        target = app_dir / entry.name
        try:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target, onexc=_chmod_retry)
            elif target.exists():
                try:
                    target.unlink()
                except PermissionError:
                    os.chmod(target, stat.S_IWRITE)
                    target.unlink()
        except PermissionError:
            stale = target.with_name(target.name + ".stale")
            if stale.exists():
                raise   # previous stale copy is ALSO still locked — give up
            target.rename(stale)
        shutil.move(str(entry), str(target))


def report_kept_packs(app_dir: Path) -> None:
    packs = list((app_dir / "scenes").rglob("framecache.npy")) \
        if (app_dir / "scenes").is_dir() else []
    if packs:
        gb = sum(p.stat().st_size for p in packs) / (1024 ** 3)
        print(f"  kept {len(packs)} prewarmed framecache pack(s), {gb:.1f} GB "
              f"— no re-prewarm needed")
    else:
        print("  no framecache packs found under dist scenes/ "
              "(nothing was prewarmed here yet)")


# Derived caches that must never ship: framecache.npy packs are multi-GB,
# machine-generated from frames/ on first run (same rationale as .gitignore).
SKIP_FILENAMES = {"framecache.npy"}
SKIP_SUFFIXES = {".building"}   # *.npy.building — half-written packs


def _copy_tree(src: Path, dst: Path, skip_dirnames: set[str] = frozenset(),
               link: bool = False) -> int:
    """Copy a tree, skipping skip_dirnames directories and derived cache files.
    With link=True, files are NTFS-hardlinked instead of copied (zero extra
    disk on the same volume; they materialize as real files when the dist
    folder is copied to another drive). Returns number of files staged."""
    count = 0
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if any(part in skip_dirnames for part in rel.parts):
            continue
        if path.name in SKIP_FILENAMES or path.suffix in SKIP_SUFFIXES:
            continue
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            if link:
                try:
                    os.link(path, target)
                except OSError:          # cross-volume / FS without hardlinks
                    shutil.copy2(path, target)
            else:
                shutil.copy2(path, target)
            count += 1
    return count


def stage_content(app_dir: Path, with_frames: bool, with_assets: bool,
                  scenes_root: Path, link_frames: bool) -> None:
    print("\n-- staging runtime content next to the exe --")

    shutil.copy2(ROOT / "config.json", app_dir / "config.json")
    print("  config.json")

    guide = ROOT / "docs" / "HOW_TO_RUN.md"
    if guide.exists():
        shutil.copy2(guide, app_dir / "HOW_TO_RUN.md")
        print("  HOW_TO_RUN.md  (non-technical operator guide)")

    start_shot_doc = ROOT / "docs" / "START_SHOT_MAPPING.txt"
    if start_shot_doc.exists():
        shutil.copy2(start_shot_doc, app_dir / "START_SHOT_MAPPING.txt")
        print("  START_SHOT_MAPPING.txt  (--start-shot numbering reference)")

    # One-click preload: build every shot's frame-cache pack ahead of time so
    # the first real run has no cold-decode stutter (BHR.exe --prewarm).
    bat = app_dir / "Prewarm cache (run once).bat"
    bat.write_text(
        "@echo off\r\n"
        "rem Builds the speed-up cache for every scene so the first show runs\r\n"
        "rem smoothly. Run this ONCE after copying the BHR folder. Takes a while\r\n"
        "rem and needs plenty of free disk. Safe to re-run.\r\n"
        'cd /d "%~dp0"\r\n'
        "BHR.exe --prewarm\r\n"
        "echo.\r\n"
        "echo Done. You can close this window.\r\n"
        "pause\r\n",
        encoding="ascii")
    print("  Prewarm cache (run once).bat")

    skip = set() if with_frames else {"frames"}
    n = _copy_tree(scenes_root, app_dir / "scenes", skip_dirnames=skip,
                   link=link_frames)
    label = ("frames HARDLINKED (no extra disk on this volume)" if
             (with_frames and link_frames) else
             "with frames" if with_frames else
             "metadata/audio only, frames SKIPPED")
    print(f"  scenes/  <- {scenes_root}  ({n} files, {label})")
    if not with_frames:
        print("           deploy frames separately: copy the full scenes tree over this one")

    models = ROOT / "models"
    if models.is_dir():
        n = _copy_tree(models, app_dir / "models")
        print(f"  models/  ({n} files)")
    else:
        print("  models/  NOT FOUND locally — voice engine will run without Vosk until")
        print("           you place models/vosk-model-small-en-us-0.15 next to the exe")

    # Hand-icon cursors are a runtime asset (render_engine loads them from
    # assets/hand_icons next to the exe) — always staged, they are tiny.
    icons = ROOT / "assets" / "hand_icons"
    if icons.is_dir():
        n = _copy_tree(icons, app_dir / "assets" / "hand_icons")
        print(f"  assets/hand_icons/  ({n} files)")

    if with_assets:
        assets = ROOT / "assets"
        if assets.is_dir():
            n = _copy_tree(assets, app_dir / "assets")
            print(f"  assets/  ({n} files)")
    else:
        print("  assets/  skipped beyond hand_icons (use --with-assets for the rest)")


def make_zip(app_dir: Path, version: str) -> Path:
    zip_base = DIST / f"{APP_NAME}-{version}-win64"
    print(f"\n-- zipping -> {zip_base}.zip (this can take a while) --")
    archive = shutil.make_archive(str(zip_base), "zip",
                                  root_dir=app_dir.parent, base_dir=app_dir.name)
    return Path(archive)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BHR.exe with PyInstaller.")
    parser.add_argument("--with-frames", action="store_true",
                        help="Stage scene frame folders into dist (multi-GB).")
    parser.add_argument("--link-frames", action="store_true",
                        help="Hardlink staged scene files instead of copying "
                             "(zero extra disk on the same NTFS volume).")
    parser.add_argument("--project", type=Path,
                        default=ROOT / "BHR_Experience.bhrx.json",
                        help="Experience Builder project to export and stage "
                             "(the single source of truth).")
    parser.add_argument("--skip-export", action="store_true",
                        help="Don't re-run export_experience.py; stage the "
                             "existing scenes root as-is.")
    parser.add_argument("--scenes-root", type=Path,
                        default=ROOT / "export" / "generated",
                        help="Scenes tree to stage as the packaged scenes/ "
                             "(default: the .bhrx export output).")
    parser.add_argument("--with-assets", action="store_true",
                        help="Stage assets/ (storyboard pages, reference video).")
    parser.add_argument("--zip", action="store_true",
                        help="Zip dist/BHR into dist/BHR-<version>-win64.zip.")
    parser.add_argument("--version", default="dev",
                        help="Version tag used in the zip name (e.g. v1.0.0).")
    parser.add_argument("--keep-packs", action="store_true",
                        help="Rebuild without wiping the existing dist/BHR "
                             "(PyInstaller's --noconfirm normally deletes it, "
                             "taking the prewarmed multi-GB framecache.npy "
                             "packs with it). Builds into a temp distpath and "
                             "swaps in only what PyInstaller produced.")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip PyInstaller; only re-stage content / re-zip.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the PyInstaller command and exit.")
    args = parser.parse_args()

    # With --keep-packs, PyInstaller must not own dist/ — its --noconfirm
    # wipes the whole dist/BHR output dir, which is exactly what destroys the
    # prewarmed packs on every rebuild.
    fresh_dist = DIST / "_fresh" if args.keep_packs else None
    cmd = pyinstaller_cmd(distpath=fresh_dist or DIST)
    if args.dry_run:
        print(" ".join(cmd))
        return 0

    app_dir = DIST / APP_NAME

    def finish_swap() -> bool:
        # A running BHR.exe blocks the swap: Windows allows renaming the exe
        # itself, but not the _internal/ directory (the loader holds handles
        # inside it). Detect it up front and name the culprit.
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {APP_NAME}.exe",
                 "/FO", "CSV", "/NH"], capture_output=True, text=True)
            if f"{APP_NAME}.exe" in (out.stdout or ""):
                print(f"ERROR: {APP_NAME}.exe is currently RUNNING — the old "
                      f"runtime cannot be replaced under it.")
                print(f"    {out.stdout.strip().splitlines()[0]}")
                print("The finished build is safe in dist/_fresh. Close it, then")
                print("finish WITHOUT rebuilding:")
                print("    py -3.12 scripts/build_exe.py --keep-packs --skip-build")
                return False
        except OSError:
            pass   # tasklist unavailable — fall through and let the swap try
        print(f"-- swapping the fresh runtime into {app_dir} "
              f"(scenes/ and packs untouched) --")
        try:
            swap_in_build(fresh_dist / APP_NAME, app_dir)
        except (PermissionError, OSError) as exc:
            print(f"ERROR: could not replace files in {app_dir}: {exc}")
            print("Something is still holding them — usually BHR.exe is running.")
            print("The finished build is safe in dist/_fresh. Close BHR.exe, then")
            print("finish WITHOUT rebuilding:")
            print("    py -3.12 scripts/build_exe.py --keep-packs --skip-build")
            return False
        shutil.rmtree(fresh_dist, ignore_errors=True)
        report_kept_packs(app_dir)
        return True

    if not args.skip_build:
        try:
            import PyInstaller  # noqa: F401
        except ImportError:
            print("PyInstaller is not installed. Run:")
            print(f"    {Path(sys.executable).name} -m pip install pyinstaller")
            return 1
        if fresh_dist is not None:
            shutil.rmtree(fresh_dist, ignore_errors=True)
        print("-- running PyInstaller (several minutes; MediaPipe/Vosk are large) --")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print("PyInstaller FAILED — see output above.")
            return result.returncode
        if fresh_dist is not None and not finish_swap():
            return 1
    elif fresh_dist is not None and (fresh_dist / APP_NAME).is_dir():
        # A previous --keep-packs build finished but its swap was blocked
        # (BHR.exe was running) — possibly PARTIALLY (a blocked swap can have
        # moved some entries already), so any leftover dir means unfinished
        # work. Finish it now instead of rebuilding.
        print("-- found a pending build in dist/_fresh --")
        if not finish_swap():
            return 1
    if not (app_dir / f"{APP_NAME}.exe").exists():
        print(f"ERROR: {app_dir / (APP_NAME + '.exe')} not found — build first.")
        return 1

    scenes_root = args.scenes_root if args.scenes_root.is_absolute() \
        else ROOT / args.scenes_root

    # The .bhrx project is the single source of truth: re-export it so the
    # staged scenes/ always matches the authored experience. The exporter is
    # incremental (frames re-extract only when counts mismatch; sounds resolve
    # from assets/), so an up-to-date export is quick.
    default_export = (ROOT / "export" / "generated").resolve()
    if args.skip_export:
        print("-- --skip-export: staging the existing scenes root as-is --")
    elif scenes_root.resolve() != default_export:
        print(f"-- custom --scenes-root, skipping the .bhrx export --")
    else:
        if not args.project.exists():
            print(f"ERROR: {args.project} not found — the .bhrx project is the "
                  f"build's source of truth (or pass --skip-export).")
            return 1
        print(f"-- exporting {args.project.name} -> {scenes_root} --")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "export_experience.py"),
             str(args.project), "--out", str(scenes_root)], cwd=ROOT)
        if result.returncode != 0:
            print("export_experience.py FAILED — see output above.")
            return result.returncode

    if not (scenes_root / "sequence.json").exists():
        print(f"ERROR: {scenes_root} has no sequence.json — not a scenes tree.")
        return 1
    stage_content(app_dir, args.with_frames, args.with_assets,
                  scenes_root, args.link_frames)

    if args.zip:
        archive = make_zip(app_dir, args.version)
        size_mb = archive.stat().st_size / (1024 * 1024)
        print(f"  {archive.name}  ({size_mb:.0f} MB)")

    print(f"\nDone. Run it:  {app_dir / (APP_NAME + '.exe')}")
    print("Smoke test without camera/audio:  BHR.exe --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
