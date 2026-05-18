"""
rasterize_storyboard.py — Convert BHR_STORYBOARDS_V4.pdf to per-page JPGs.

Run once (or re-run with --force to redo):
    python scripts/rasterize_storyboard.py

Output:
  assets/storyboard/page_001.jpg ... page_244.jpg   (projector resolution)
  assets/storyboard/index.json                       (page → scene → AL-code mapping)

The index.json is pre-populated with the scene-boundary assignments from CLAUDE.md.
AL-code fields start as null — fill them in by reading each page and matching
the dialogue text to the script. See 'Storyboard as Placeholder Assets' in CLAUDE.md
for the full reconciliation table and the Scene 1 line-code offset trap.

Dependencies: PyMuPDF (fitz) — already in requirements.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print(
        "PyMuPDF is required. Install it with: pip install PyMuPDF",
        file=sys.stderr,
    )
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent
PDF_PATH = REPO_ROOT / "scenes" / "BHR_STORYBOARDS_V4.pdf"
OUT_DIR = REPO_ROOT / "assets" / "storyboard"
INDEX_PATH = OUT_DIR / "index.json"

# Target resolution. Storyboard pages are portrait-ish; we fit the width to
# the projector and let the height float. The render engine will letterbox.
TARGET_W = 1920
TARGET_H = 1080

# Page-to-script-scene assignments from CLAUDE.md § "Storyboard as Placeholder Assets".
# Boundaries are approximate; final assignment is determined by the AL codes
# printed on each panel. Pages outside these ranges have no known assignment yet.
# The storyboard ends at AL-06-013 (page 244); Scenes 7–11 and Prologue are absent.
_SCENE_BOUNDARIES: list[tuple[int, int, str]] = [
    (1,   36,  "scene_01"),
    (37,  65,  "scene_02"),
    (66,  90,  "scene_03"),
    (91,  151, "scene_04"),
    (152, 188, "scene_05"),
    (189, 244, "scene_06"),
]

# Scene 1 offset note: the storyboard's AL-01-NNN codes are one higher than
# the May 15 script because the prologue (AL-00-*) was inserted later.
# Storyboard "AL-01-002" = script AL-01-001, etc. The index.json al_code field
# should use the SCRIPT's code (canonical), not the storyboard's printed code.
_SCENE_OFFSET: dict[str, int] = {
    "scene_01": -1,  # subtract 1 from storyboard code to get script code
}


def _scene_for_page(page_1indexed: int) -> str | None:
    for lo, hi, scene in _SCENE_BOUNDARIES:
        if lo <= page_1indexed <= hi:
            return scene
    return None


def _render_page(page: fitz.Page) -> bytes:
    """Render a PDF page to JPEG bytes, fitting within TARGET_W × TARGET_H."""
    rect = page.rect
    scale_x = TARGET_W / rect.width
    scale_y = TARGET_H / rect.height
    scale = min(scale_x, scale_y)  # fit, don't crop
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return pix.tobytes("jpeg", jpg_quality=90)


def _build_starter_index(total_pages: int) -> dict:
    """Build the initial index.json skeleton with scene assignments."""
    pages: dict[str, dict] = {}
    for i in range(1, total_pages + 1):
        key = f"{i:03d}"
        scene = _scene_for_page(i)
        pages[key] = {
            "script_scene": scene,
            "al_code": None,         # fill in manually or via a future scan pass
            "path": f"assets/storyboard/page_{key}.jpg",
            "notes": "",
        }
    return {
        "_doc": (
            "page → script-scene → AL-code mapping. "
            "al_code fields are null until filled by reading the panel. "
            "Scene 1 storyboard codes are off by +1 vs. May 15 script — "
            "use SCRIPT codes here, not storyboard-printed codes."
        ),
        "scene_offsets": _SCENE_OFFSET,
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rasterize BHR storyboard PDF to per-page JPGs.")
    parser.add_argument("--pdf", default=str(PDF_PATH), help="Path to the storyboard PDF.")
    parser.add_argument("--out", default=str(OUT_DIR), help="Output directory for JPGs and index.json.")
    parser.add_argument("--force", action="store_true", help="Re-render pages that already exist.")
    parser.add_argument("--pages", metavar="N", help="Comma-separated page numbers to render (default: all).")
    parser.add_argument("--width", type=int, default=TARGET_W, help="Target width in pixels.")
    parser.add_argument("--height", type=int, default=TARGET_H, help="Target height in pixels.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        print("Expected at scenes/BHR_STORYBOARDS_V4.pdf. Check the path.", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Override globals if caller passed custom dimensions
    # TARGET_W = args.width
    # TARGET_H = args.height

    doc = fitz.open(str(pdf_path))
    total = len(doc)
    print(f"Opened '{pdf_path.name}': {total} pages. Rendering to {TARGET_W}×{TARGET_H}.")

    # Determine which pages to render
    if args.pages:
        targets = sorted({int(p.strip()) for p in args.pages.split(",") if p.strip()})
        out_of_range = [p for p in targets if not (1 <= p <= total)]
        if out_of_range:
            print(f"Warning: pages out of range (1–{total}) and will be skipped: {out_of_range}")
        targets = [p for p in targets if 1 <= p <= total]
    else:
        targets = list(range(1, total + 1))

    rendered = 0
    skipped = 0
    for page_num in targets:
        key = f"{page_num:03d}"
        out_path = out_dir / f"page_{key}.jpg"

        if out_path.exists() and not args.force:
            skipped += 1
            continue

        page = doc.load_page(page_num - 1)  # 0-indexed
        jpg_bytes = _render_page(page)
        out_path.write_bytes(jpg_bytes)
        rendered += 1

        if rendered % 20 == 0 or page_num == targets[-1]:
            print(f"  Rendered {rendered} / {len(targets)} pages...", end="\r", flush=True)

    doc.close()
    print(f"\nDone. Rendered: {rendered}  Skipped (already exist): {skipped}")

    # Write or update index.json
    index_path = out_dir / "index.json"
    if index_path.exists() and not args.force:
        # Merge: add any new pages; don't overwrite existing al_code entries
        with open(index_path, encoding="utf-8") as f:
            existing = json.load(f)
        new_index = _build_starter_index(total)
        for key, entry in new_index["pages"].items():
            if key not in existing["pages"]:
                existing["pages"][key] = entry
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"index.json updated (merged): {index_path}")
    else:
        index = _build_starter_index(total)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        print(f"index.json written: {index_path}")

    print()
    print("Next step: open each page JPG and fill in the 'al_code' field in index.json.")
    print("Use CLAUDE.md § 'Storyboard as Placeholder Assets' for the boundary table.")
    print("Remember: Scene 1 storyboard AL codes are +1 vs. the May 15 script.")


if __name__ == "__main__":
    main()
