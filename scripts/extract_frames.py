"""
extract_frames.py — Extract every frame from the BHR master video.

Usage:
    py -3.12 scripts/extract_frames.py
    py -3.12 scripts/extract_frames.py --video "assets/BHR Draft 1.mp4"
    py -3.12 scripts/extract_frames.py --format jpg --quality 2
    py -3.12 scripts/extract_frames.py --format png

Output:
    final_frames/00001.jpg (or .png) — one file per frame, zero-padded to 5 digits.

Notes:
    • ffmpeg must be on PATH. On the production machine install via:
          winget install ffmpeg
      then open a new terminal so PATH updates.
    • PNG is lossless but large (~3–5 GB per minute of footage at 1080p).
      JPEG at --quality 2 (ffmpeg scale: 1=best, 31=worst) is visually
      indistinguishable and ~10× smaller. Default is JPEG quality 2.
    • -fps_mode passthrough preserves the source frame count exactly —
      no interpolation, no frame dropping.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT      = Path(__file__).parent.parent
VIDEO_DEFAULT = ROOT / "assets" / "BHR Draft 1.mp4"
OUTPUT_DIR    = ROOT / "final_frames"


def find_ffmpeg() -> str:
    """Return the ffmpeg executable path, or exit with a helpful message."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    print(
        "[extract_frames] ffmpeg not found on PATH.\n"
        "Install it with:\n"
        "    winget install ffmpeg\n"
        "Then close and reopen this terminal so PATH updates.",
        file=sys.stderr,
    )
    sys.exit(1)


def get_frame_count(ffmpeg: str, video: Path) -> int:
    """Best-effort frame count via ffprobe (bundled with ffmpeg)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                str(video),
            ],
            capture_output=True, text=True, timeout=60,
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from BHR master video.")
    parser.add_argument("--video", default=str(VIDEO_DEFAULT),
                        help=f"Path to source video (default: {VIDEO_DEFAULT})")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg",
                        help="Output format: jpg (smaller, default) or png (lossless, large)")
    parser.add_argument("--quality", type=int, default=2,
                        help="JPEG quality 1–31 (1=best, default 2). Ignored for PNG.")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    video  = Path(args.video)
    outdir = Path(args.output)
    fmt    = args.format

    if not video.exists():
        print(f"[extract_frames] Video not found: {video}", file=sys.stderr)
        sys.exit(1)

    outdir.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg()

    # Warn about PNG size upfront
    if fmt == "png":
        print("[extract_frames] WARNING: PNG output can exceed 100 GB for a full-length video.")
        print("[extract_frames] Use --format jpg for a ~10x smaller file set with no visible quality loss.")
        print()

    total = get_frame_count(ffmpeg, video)
    if total:
        size_hint = f"~{total * (4 if fmt == 'png' else 0.4):.0f} MB estimated" if total else ""
        print(f"[extract_frames] Source: {video.name}  ({total} frames  {size_hint})")
    else:
        print(f"[extract_frames] Source: {video.name}")

    print(f"[extract_frames] Output: {outdir}  format={fmt.upper()}")
    print("[extract_frames] Starting — this will take several minutes...\n")

    pattern = str(outdir / f"%05d.{fmt}")

    cmd = [ffmpeg, "-i", str(video), "-fps_mode", "passthrough"]
    if fmt == "jpg":
        cmd += ["-q:v", str(args.quality)]
    cmd.append(pattern)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[extract_frames] ffmpeg exited with code {result.returncode}.", file=sys.stderr)
        sys.exit(result.returncode)

    written = len(list(outdir.glob(f"*.{fmt}")))
    print(f"\n[extract_frames] Done. {written} frames written to {outdir}")


if __name__ == "__main__":
    main()
