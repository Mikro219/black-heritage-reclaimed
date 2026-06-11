"""
rename_frames.py — Rename artist frame PNGs to the standard NNNN.png format.

Artist naming patterns handled:
  frame_NNNN.png   (elayna, natasha)
  frame-NNNN.png   (variant)
  final-NNNN.png   (wendy, felicia)

All three are renamed to zero-padded 4-digit NNNN.png.
Files already in NNNN.png format are silently skipped.

After renaming, run  git add -A  to stage the changes (git detects renames
by content hash, so no information is lost).

Usage:
    py -3.12 scripts/rename_frames.py [--dry-run] [path ...]

    No paths  — scans every frames/ subdirectory under scenes/
    path ...  — scans only those directories (can be a scenes/artist/scene_NN
                or a specific frames/ dir)

    --dry-run   Print what would be renamed without touching the filesystem.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENES_ROOT = ROOT / "scenes"

# Matches frame_NNNN, frame-NNNN, final-NNNN  (case-insensitive)
_PATTERN = re.compile(r'^(?:frame[_-]|final-)(\d+)\.png$', re.IGNORECASE)
# Already in target format — exactly 4+ digits followed by .png
_TARGET = re.compile(r'^\d+\.png$')


def _find_frames_dirs(roots: list[Path]) -> list[Path]:
    """Yield all frames/ subdirectories reachable from each root."""
    dirs: list[Path] = []
    for root in roots:
        if root.name == "frames" and root.is_dir():
            dirs.append(root)
        else:
            dirs.extend(sorted(root.rglob("frames")))
    return dirs


def rename_frames(search_roots: list[Path], dry_run: bool) -> int:
    """Rename frames in all discovered frames/ dirs. Returns count of renames."""
    renamed = 0
    skipped = 0
    conflicts = 0

    for frames_dir in _find_frames_dirs(search_roots):
        pngs = sorted(frames_dir.glob("*.png"))
        for png in pngs:
            name = png.name

            if _TARGET.match(name):
                continue  # already correct

            m = _PATTERN.match(name)
            if not m:
                print(f"[skip] unrecognized pattern: {png.relative_to(ROOT)}")
                skipped += 1
                continue

            number = int(m.group(1))
            new_name = f"{number:04d}.png"
            new_path = frames_dir / new_name

            if new_path.exists():
                print(f"[conflict] {png.relative_to(ROOT)} → {new_name} already exists — skipping")
                conflicts += 1
                continue

            if dry_run:
                print(f"  {name} -> {new_name}   ({frames_dir.relative_to(ROOT)})")
            else:
                png.rename(new_path)
            renamed += 1

    label = "Would rename" if dry_run else "Renamed"
    print(f"\n{label}: {renamed}   skipped (unrecognized): {skipped}   conflicts: {conflicts}")
    if renamed and not dry_run:
        print("Run  git add -A  to stage the renames.")
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths", nargs="*",
        help="Directories to scan. Default: all of scenes/",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be renamed without touching the filesystem.",
    )
    args = parser.parse_args()

    roots = [Path(p).resolve() for p in args.paths] if args.paths else [SCENES_ROOT]
    missing = [p for p in roots if not p.exists()]
    if missing:
        for p in missing:
            print(f"[error] path not found: {p}", file=sys.stderr)
        sys.exit(1)

    rename_frames(roots, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
