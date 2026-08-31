"""
copy_frames.py — copy source frames from final_frames/ into shot frame folders.

Run from the BHR project root:
    py -3.12 scripts/copy_frames.py           # skip shots that already have frames
    py -3.12 scripts/copy_frames.py --force   # overwrite existing frames

Add new entries to SHOT_FRAMES as shots are mapped:
    ("act_folder", "shot_id", global_start, global_end)

NOTE: this module doubles as the canonical shot <-> master-frame map.
SHOT_FRAMES is imported by capcut_audio.py,
export_experience.py and build_captions.py (and consumed by tests) — treat
it as data with a CLI attached, not a one-off script.
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

    # Act 05 — Monkey Wrench
    ("act_05_monkey_wrench", "22", 18643, 18930),
    ("act_05_monkey_wrench", "23", 18930, 19680),
    ("act_05_monkey_wrench", "24", 19681, 21388),
    ("act_05_monkey_wrench", "25", 21389, 22558),

    # Act 06 — Shoofly
    ("act_06_shoofly", "28", 22559, 22808),
    ("act_06_shoofly", "29", 22809, 23858),
    ("act_06_shoofly", "30", 23859, 24450),
    ("act_06_shoofly", "31", 24451, 25034),
    ("act_06_shoofly", "32", 25035, 25673),
    ("act_06_shoofly", "33", 25674, 26278),

    # Act 07 — Bear's Paw
    ("act_07_bear_paw", "34", 26278, 26537),
    ("act_07_bear_paw", "35", 26538, 27455),
    ("act_07_bear_paw", "36", 27456, 28055),   # survey OI window global 27750-27914 -> local 295-459
    ("act_07_bear_paw", "37", 28056, 28844),   # fork FSM: intro 1-203, wrong 521-673, correct 674-789
    ("act_07_bear_paw", "38", 28844, 29258),

    # Act 08 — Bow Tie / Hourglass  (shots 39-42 skipped on purpose)
    ("act_08_bowtie_hourglass", "43", 29258, 29525),
    ("act_08_bowtie_hourglass", "44", 29526, 30059),
    ("act_08_bowtie_hourglass", "45", 30060, 32129),   # hat OI 649-821, coat OI 1097-1271

    # Act 09 — Wagon Wheel  (shots 46-48 skipped on purpose)
    ("act_09_wagon_wheel", "49", 32129, 32379),
    ("act_09_wagon_wheel", "50", 32380, 34998),   # fork FSM: intro 1-981, path A (L) 1269-1925, path B (R) 1926-2619

    # Act 10 — Tumbling Blocks  (shots 51-56 skipped on purpose)
    ("act_10_tumbling_blocks", "57", 34998, 35234),   # playback, no OI
    ("act_10_tumbling_blocks", "58", 35234, 35944),   # cup-ear OI local 211-395 (global 35444-35628)
    ("act_10_tumbling_blocks", "59", 35944, 36530),   # cup-ear OI local 97-275  (global 36040-36218)
    ("act_10_tumbling_blocks", "60", 36530, 37142),   # fast-gather OI local 163-329 (global 36692-36858)
    ("act_10_tumbling_blocks", "61", 37142, 38040),   # run-arms OI local 137-331 (global 37278-37472)

    # Act 11 — Conestogo (whole act as ONE shot; supersedes tracker shots 67-78)
    ("act_11_conestogo", "66", 38040, 43054),  # unravel/throw/push/paddle/spread OI chain

    # Act 12 — Epilogue / Final Address
    ("act_12_epilogue", "79", 43054, 45631),   # epilogue playback to end of footage
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
        src_file = SRC / f"{global_idx:05d}.png"
        if not src_file.exists():
            print(f"  WARN   missing source frame {global_idx:05d}.png  (shot_{shot} local {local_idx:04d})")
            continue
        shutil.copy2(src_file, dst / f"{local_idx:04d}.png")

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
