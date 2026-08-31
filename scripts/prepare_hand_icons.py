"""
prepare_hand_icons.py — one-shot asset prep for the on-screen hand cursors.

Reads the labelled reference sheets in `assets/source/Hand Assets/` (white background,
black text label in the bottom-left corner), strips the background and the
label, crops to the hand, and writes runtime-ready transparent PNGs to
`assets/hand_icons/`:

    fist_l.png  fist_r.png   open_l.png  open_r.png
    point_l.png point_r.png  knock_l.png knock_r.png

Re-run whenever the source sheets change:
    py -3.12 scripts/prepare_hand_icons.py
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "source" / "Hand Assets"
DST = ROOT / "assets" / "hand_icons"

# Source sheet number -> output icon name.
# Convention (mirrored display): a "<shape>_l" icon must LOOK like the mirror
# image of the visitor's left hand — thumb toward the body midline, i.e. on
# the RIGHT side of the image (and vice versa for _r). The delivered OPEN
# sheets 3/4 are chirality-swapped relative to their labels (audit Aug 2026:
# sheet 3's thumb sits on the image LEFT = a right hand), so they map crossed
# here on purpose. Fist/point/knock sheets are labelled correctly.
MAPPING = {
    "1": "fist_l",
    "2": "fist_r",
    "3": "open_r",
    "4": "open_l",
    "5": "point_l",
    "6": "point_r",
    "7": "knock_l",
    "8": "knock_r",
}

WHITE_THRESHOLD = 235   # r,g,b all >= this -> background
LABEL_BAND_FRAC = 0.14  # bottom strip holding the text label — cleared outright
OUT_SIZE = 512          # max dimension of the output icon


def process(src_path: Path, dst_path: Path) -> None:
    img = Image.open(src_path).convert("RGBA")
    w, h = img.size
    px = img.load()

    label_top = int(h * (1.0 - LABEL_BAND_FRAC))
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if y >= label_top:
                px[x, y] = (0, 0, 0, 0)         # label band — always cleared
            elif r >= WHITE_THRESHOLD and g >= WHITE_THRESHOLD and b >= WHITE_THRESHOLD:
                px[x, y] = (r, g, b, 0)         # white background -> transparent

    bbox = img.getbbox()   # bounding box of non-zero (non-transparent) pixels
    if bbox:
        img = img.crop(bbox)

    img.thumbnail((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    img.save(dst_path)
    print(f"  {src_path.name} -> {dst_path.relative_to(ROOT)}  {img.size[0]}x{img.size[1]}")


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"source folder not found: {SRC}")
    DST.mkdir(parents=True, exist_ok=True)
    print(f"[prepare_hand_icons] {SRC} -> {DST}")
    for num, name in MAPPING.items():
        src = SRC / f"{num}.png"
        if not src.exists():
            print(f"  MISSING: {src.name} (wanted for {name})")
            continue
        process(src, DST / f"{name}.png")
    print("[prepare_hand_icons] done")


if __name__ == "__main__":
    main()
