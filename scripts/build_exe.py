"""
build_exe.py — package BHR into a Windows executable with PyInstaller.

Run from the BHR project root:
    py -3.12 scripts/build_exe.py                     # build dist/BHR/BHR.exe
    py -3.12 scripts/build_exe.py --with-frames       # also stage scene frames (GBs)
    py -3.12 scripts/build_exe.py --with-assets       # also stage assets/ (storyboard, video)
    py -3.12 scripts/build_exe.py --zip --version v1.0.0   # produce BHR-v1.0.0-win64.zip
    py -3.12 scripts/build_exe.py --dry-run           # print the PyInstaller command and exit

Output layout (one-dir build — do NOT use one-file, MediaPipe/Vosk startup and
antivirus behaviour are much worse):

    dist/BHR/
      BHR.exe            <- console app (logs to stdout; kiosk launcher captures them)
      _internal/         <- Python runtime + libraries (PyInstaller-managed)
      config.json
      config/host_profiles/*.json
      scenes/            <- sequence.json + per-shot metadata/audio
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


def pyinstaller_cmd() -> list[str]:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(ROOT / "main.py"),
        "--name", APP_NAME,
        "--noconfirm",
        "--clean",
        # one-dir console build: fastest startup, logs visible/capturable
        "--onedir",
        "--console",
        "--distpath", str(DIST),
        "--workpath", str(BUILD),
        "--specpath", str(BUILD),
    ]
    for pkg in COLLECT_ALL:
        cmd += ["--collect-all", pkg]
    return cmd


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

    n = _copy_tree(ROOT / "config", app_dir / "config")
    print(f"  config/  ({n} files)")

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
    parser.add_argument("--scenes-root", type=Path, default=ROOT / "scenes",
                        help="Scenes tree to stage as the packaged scenes/ "
                             "(e.g. export/scenes_generated).")
    parser.add_argument("--with-assets", action="store_true",
                        help="Stage assets/ (storyboard pages, reference video).")
    parser.add_argument("--zip", action="store_true",
                        help="Zip dist/BHR into dist/BHR-<version>-win64.zip.")
    parser.add_argument("--version", default="dev",
                        help="Version tag used in the zip name (e.g. v1.0.0).")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip PyInstaller; only re-stage content / re-zip.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the PyInstaller command and exit.")
    args = parser.parse_args()

    cmd = pyinstaller_cmd()
    if args.dry_run:
        print(" ".join(cmd))
        return 0

    if not args.skip_build:
        try:
            import PyInstaller  # noqa: F401
        except ImportError:
            print("PyInstaller is not installed. Run:")
            print(f"    {Path(sys.executable).name} -m pip install pyinstaller")
            return 1
        print("-- running PyInstaller (several minutes; MediaPipe/Vosk are large) --")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print("PyInstaller FAILED — see output above.")
            return result.returncode

    app_dir = DIST / APP_NAME
    if not (app_dir / f"{APP_NAME}.exe").exists():
        print(f"ERROR: {app_dir / (APP_NAME + '.exe')} not found — build first.")
        return 1

    scenes_root = args.scenes_root if args.scenes_root.is_absolute() \
        else ROOT / args.scenes_root
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
