"""
prewarm_frame_cache.py — Build every shot's framecache.npy pack ahead of time.

    py -3.12 scripts/prewarm_frame_cache.py --scenes export/generated --shots 02,03,04,09,10,12
    py -3.12 scripts/prewarm_frame_cache.py --scenes export/generated   (ALL shots — mind the disk!)
    py -3.12 scripts/prewarm_frame_cache.py --resolution 1920x1080

Why: on the FIRST run of a freshly exported tree the runtime serves frames by
per-frame JPEG decode on the render thread while a background worker packs the
whole shot — a detection that jumps into a cold segment (e.g. the Crossroads
pick animation) stutters and lands late. The packs persist on disk, so one
prewarm pass after any frames export makes every run behave like a warm one.

DISK BUDGET: packs are raw RGB — width x height x 3 bytes per frame. The full
~45k-frame experience is ~124 GB at 1280x720 and ~280 GB at 1920x1080. Prefer
--shots with the choice/fork shots (where a pick jumps into cold frames); the
linear stretch shots build fine in the runtime's background window while the
previous shot plays.

Resolution must match the runtime's display resolution (config.json "host"
display.resolution, else config.json "resolution") — a mismatched pack is
rebuilt at runtime, which defeats the prewarm. This script resolves it the
same way main.py does; override with --resolution WxH if needed.

Already-valid packs are skipped, so re-running is cheap.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engines.frame_cache import FrameCacheManager  # noqa: E402

CONFIG_PATH = ROOT / "config.json"


def runtime_resolution() -> tuple[int, int]:
    """The display resolution the runtime will build packs for — mirrors
    main.py (config "host" display.resolution) and RenderEngine's fallback
    to the top-level config "resolution"."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)
    except OSError:
        config = {}

    res = (config.get("host", {}).get("display", {}).get("resolution")
           or config.get("resolution", [1920, 1080]))
    return int(res[0]), int(res[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--scenes", type=Path,
                        default=ROOT / "export" / "generated",
                        help="scenes root to prewarm (default: the .bhrx "
                             "export, export/generated)")
    parser.add_argument("--resolution", default=None, metavar="WxH",
                        help="override the pack resolution (default: the "
                             "runtime's display resolution)")
    parser.add_argument("--shots", default=None,
                        help="comma-separated shot ids to prewarm (e.g. "
                             "02,09,12); default: every shot — check the "
                             "disk budget first")
    args = parser.parse_args()

    if args.resolution:
        w, h = (int(v) for v in args.resolution.lower().split("x"))
    else:
        w, h = runtime_resolution()

    scenes = args.scenes.resolve()
    frame_dirs = sorted(d for d in scenes.glob("scenes/scene_*/frames")
                        if d.is_dir())
    if args.shots:
        wanted = {s.strip().zfill(2) for s in args.shots.split(",") if s.strip()}
        frame_dirs = [d for d in frame_dirs
                      if d.parent.name.removeprefix("scene_") in wanted]
    if not frame_dirs:
        sys.exit(f"[prewarm] no matching scenes/scene_*/frames dirs under {scenes}")

    total = sum(1 for d in frame_dirs for _ in d.glob("*.jpg"))
    print(f"[prewarm] {len(frame_dirs)} shots ({total} frames, "
          f"~{total * w * h * 3 / 1e9:.1f} GB of packs) under {scenes} "
          f"at {w}x{h}")
    cache = FrameCacheManager((w, h))
    built = skipped = failed = 0
    t0 = time.monotonic()
    for d in frame_dirs:
        shot = d.parent.name
        paths = cache._ensure_paths(d)
        if not paths:
            print(f"  {shot}: no frames — skipped")
            continue
        pack = cache._pack_path(d)
        if pack.exists() and cache._pack_valid(pack, len(paths)):
            skipped += 1
            continue
        if cache._build_pack(pack, paths):
            built += 1
        else:
            failed += 1
            print(f"  {shot}: pack build FAILED")
    dt = time.monotonic() - t0
    print(f"[prewarm] done in {dt:.0f}s — {built} built, {skipped} already "
          f"valid, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
