"""
copy_frames.py — copy source frames from final_frames/ into shot frame folders.

Run from the BHR project root:
    py -3.12 scripts/copy_frames.py           # skip shots that already have frames
    py -3.12 scripts/copy_frames.py --force   # overwrite existing frames

Add new entries to SHOT_FRAMES as shots are mapped:
    ("act_folder", "shot_id", global_start, global_end)
"""

import argparse
import shutil
from pathlib import Path

SRC    = Path("final_frames")
SCENES = Path("scenes")

# ── Shot frame mapping ─────────────────────────────────────────────────────────
# (act_folder, shot_id, global_start_frame, global_end_frame)  -- both inclusive

SHOT_FRAMES = [
    # Act 00 — Prologue
    ("act_00_prologue",     "01",  1,     5170),

    # Act 01 — The Quilt Awakens
    ("act_01_quilt_awakens","02",  5171,  6018),
    ("act_01_quilt_awakens","03",  6019,  6403),
    ("act_01_quilt_awakens","04",  6404,  7052),
    ("act_01_quilt_awakens","05",  7053,  7856),

    # Act 02 — Crossroads
    ("act_02_crossroads",   "06",  7857,  8156),
    ("act_02_crossroads",   "07",  8157,  8574),
    ("act_02_crossroads",   "08",  8575,  9148),
    ("act_02_crossroads",   "09",  9149, 10770),

    # Act 03 — Flying Geese
    ("act_03_flying_geese", "10", 10771, 11038),
    ("act_03_flying_geese", "11", 11039, 11494),
    ("act_03_flying_geese", "12", 11495, 11877),
    ("act_03_flying_geese", "13", 11878, 13058),

    # Act 04 — North Star
    ("act_04_north_star",   "14", 13059, 13356),
    ("act_04_north_star",   "15", 13357, 13670),
    ("act_04_north_star",   "16", 13671, 14202),
    ("act_04_north_star",   "17", 14203, 14500),
    ("act_04_north_star",   "18", 14501, 14676),
    ("act_04_north_star",   "19", 14676, 16361),
    ("act_04_north_star",   "20", 16361, 18492),
    ("act_04_north_star",   "21", 18493, 18642),
]

# ──────────────────────────────────────────────────────────────────────────────

def copy_shot(act: str, shot: str, start: int, end: int, force: bool) -> None:
    dst = SCENES / act / f"shot_{shot}" / "frames"

    if dst.exists() and any(dst.iterdir()):
        if not force:
            print(f"  SKIP   {act}/shot_{shot}  ({end - start + 1} frames already present)")
            return
        print(f"  FORCE  {act}/shot_{shot}  overwriting …")
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)

    for local_idx, global_idx in enumerate(range(start, end + 1), start=1):
        src_file = SRC / f"{global_idx:05d}.jpg"
        if not src_file.exists():
            print(f"  WARN   missing source frame {global_idx:05d}.jpg  (shot_{shot} local {local_idx:04d})")
            continue
        shutil.copy2(src_file, dst / f"{local_idx:04d}.jpg")

    count = end - start + 1
    print(f"  OK     {act}/shot_{shot}  {count} frames  (0001 … {count:04d})")


def main():
    parser = argparse.ArgumentParser(description="Copy final_frames into shot frame folders.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite shots that already have frames.")
    args = parser.parse_args()

    if not SRC.is_dir():
        print(f"ERROR: source directory '{SRC}' not found. Run from the BHR project root.")
        return

    print(f"Source : {SRC.resolve()}")
    print(f"Scenes : {SCENES.resolve()}")
    print(f"Force  : {args.force}")
    print()

    for act, shot, start, end in SHOT_FRAMES:
        copy_shot(act, shot, start, end, args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
