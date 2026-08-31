"""build_captions.py — seed a starting caption track into a .bhrx project.

Parses Auntie Liza's voice script (docs/Black Heritage Reclaimed Script.pdf) for
the AL-XX-YYY lines, maps each script scene to its master-timeline span via
copy_frames.SHOT_FRAMES (act_NN == script scene NN), distributes that scene's
lines evenly across the span, and appends them as `captions` on whichever
Experience-Builder block contains each line's time (local at_s).

This is a COARSE starting point: line text is exact, but intra-scene timing is
even (no word-level sync), so the intended workflow is to run this once, then
drag / retime / reposition the captions in the Experience Builder's CAPTIONS
lane. Re-running replaces the auto-seeded set (entries tagged "auto": true);
hand-edited captions (no such tag) are preserved.

    py -3.12 scripts/build_captions.py                 # into BHR_Experience.bhrx.json
    py -3.12 scripts/build_captions.py --project other.bhrx.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import copy_frames  # noqa: E402

SCRIPT_PDF = ROOT / "docs" / "Black Heritage Reclaimed Script.pdf"
FPS = 30.0
DEFAULT_DUR = 4.5     # seconds a caption stays up (clamped to its slice)


def parse_script(pdf: Path) -> dict[int, list[str]]:
    """{scene_int: [line_text, ...] in code order} from the AL-XX-YYY lines.
    Address lines AL-11-021+ stay under scene 11."""
    import fitz
    doc = fitz.open(pdf)
    full = "\n".join(doc[p].get_text() for p in range(doc.page_count))
    # AL-<scene>-<line>[suffix]  "quoted text"  (text may wrap across newlines)
    pat = re.compile(r'AL-(\d\d)-(\d+)([a-zA-Z-]*)\s*"(.*?)"', re.S)
    by_scene: dict[int, list[tuple]] = {}
    for m in pat.finditer(full):
        scene = int(m.group(1))
        line = int(m.group(2))
        text = re.sub(r"\s+", " ", m.group(4)).strip()
        if text:
            by_scene.setdefault(scene, []).append((line, m.group(3), text))
    return {s: [t for _, _, t in sorted(v)] for s, v in by_scene.items()}


def act_spans() -> dict[int, tuple[float, float]]:
    """{scene_int: (start_s, end_s)} in master time from copy_frames.SHOT_FRAMES
    (act name prefix == scene number; act 12/13 fold into scene 11)."""
    spans: dict[int, list] = {}
    for act, _shot, a, b in copy_frames.SHOT_FRAMES:
        m = re.match(r"act_(\d+)", act)
        if not m:
            continue
        scene = min(11, int(m.group(1)))   # epilogue acts -> scene 11 (address)
        s = spans.setdefault(scene, [a, b])
        s[0] = min(s[0], a)
        s[1] = max(s[1], b)
    return {s: (a / FPS, b / FPS) for s, (a, b) in spans.items()}


def build(project_path: Path) -> None:
    project = json.loads(project_path.read_text(encoding="utf-8"))
    blocks = [b for b in project["blocks"] if b.get("range_s")]

    def block_for(t: float):
        for b in blocks:
            a, z = b["range_s"]
            if a <= t < z:
                return b
        return None

    scenes = parse_script(SCRIPT_PDF)
    spans = act_spans()

    # Clear previous auto seed; keep hand-authored captions.
    for b in project["blocks"]:
        if b.get("captions"):
            b["captions"] = [c for c in b["captions"] if not c.get("auto")]
            if not b["captions"]:
                b.pop("captions", None)

    n = 0
    for scene, lines in sorted(scenes.items()):
        span = spans.get(scene)
        if not span or not lines:
            continue
        start, end = span
        slice_s = max(0.5, (end - start) / len(lines))
        for i, text in enumerate(lines):
            t = start + i * slice_s
            b = block_for(t)
            if b is None:
                continue
            b.setdefault("captions", []).append({
                "id": f"capauto_{scene:02d}_{i:03d}",
                "at_s": round(t - b["range_s"][0], 2),
                "duration_s": round(min(DEFAULT_DUR, slice_s - 0.15), 2),
                "text": text,
                "auto": True,
            })
            n += 1

    for b in project["blocks"]:
        if b.get("captions"):
            b["captions"].sort(key=lambda c: c["at_s"])

    shutil.copy2(project_path, project_path.with_suffix(".json.capbak"))
    project_path.write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8")
    print(f"[build_captions] seeded {n} captions across "
          f"{sum(1 for b in project['blocks'] if b.get('captions'))} blocks "
          f"-> {project_path.name} (backup .capbak)")
    try:
        from bundle_builder_project import write_bundle
        write_bundle(project_path)
        print("[build_captions] builder bundle refreshed")
    except Exception as exc:
        print(f"[build_captions] WARNING could not refresh bundle: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", type=Path,
                    default=ROOT / "BHR_Experience.bhrx.json")
    args = ap.parse_args()
    if not args.project.exists():
        sys.exit(f"project not found: {args.project}")
    build(args.project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
