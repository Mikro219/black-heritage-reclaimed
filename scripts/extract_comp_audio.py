"""
extract_comp_audio — pull the audio track out of every final comped scene.

The comp renders in assets/video/comps/ (bhr_scene_NN.mp4) carry Auntie Liza's
voice lines baked into their audio track — the one layer NOT in the delivered
stem drop. This extracts each comp's audio as an MP3 into
assets/audio/voice_lines/ so the lines are ready to be wired into shots later
(wiring is a separate, deliberate step — this script only extracts).

    py -3.12 scripts/extract_comp_audio.py            # skips up-to-date files
    py -3.12 scripts/extract_comp_audio.py --force    # re-extract everything

Comps without an audio stream are reported and skipped. Requires ffmpeg
(winget install ffmpeg).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPS_DIR = ROOT / "assets" / "video" / "comps"
OUT_DIR = ROOT / "assets" / "audio" / "voice_lines"


def has_audio_stream(ffprobe: str, video: Path) -> bool:
    result = subprocess.run(
        [ffprobe, "-v", "quiet", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True)
    return bool(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract comp-scene audio tracks as MP3 voice-line files")
    parser.add_argument("--comps", type=Path, default=COMPS_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true",
                        help="re-extract even when the MP3 already exists")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        sys.exit("ffmpeg/ffprobe not on PATH (winget install ffmpeg)")
    if not args.comps.is_dir():
        sys.exit(f"comps dir not found: {args.comps}")

    args.out.mkdir(parents=True, exist_ok=True)
    done = skipped = silent = failed = 0
    for video in sorted(args.comps.glob("*.mp4")):
        out = args.out / f"{video.stem}.mp3"
        if out.exists() and not args.force \
                and out.stat().st_mtime >= video.stat().st_mtime:
            skipped += 1
            continue
        if not has_audio_stream(ffprobe, video):
            print(f"  [no audio] {video.name}")
            silent += 1
            continue
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(video), "-vn", "-c:a", "libmp3lame", "-q:a", "2",
             str(out)],
            capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [FAILED]   {video.name}: {result.stderr.strip()[:160]}")
            failed += 1
        else:
            print(f"  extracted  {out.name}")
            done += 1

    print(f"\n[extract_comp_audio] {done} extracted, {skipped} up-to-date, "
          f"{silent} without audio, {failed} failed -> {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
