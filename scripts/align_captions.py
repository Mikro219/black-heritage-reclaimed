"""align_captions.py — retime the auto caption track to the master mp4's audio.

build_captions.py seeds captions with EVEN intra-scene timing (no relation to
when a line is actually spoken). This script fixes that: it transcribes the
master draft's audio with Vosk (word-level timestamps), aligns each script
scene's AL-line text against the recognized word stream for that scene's
master-time span (difflib over normalized words), and rewrites the auto-seeded
captions ("auto": true) at the moments the lines are actually spoken. Lines the
recognizer couldn't anchor (music over VO, mumbled edits) are interpolated
between their matched neighbours. Hand-authored captions are untouched.

    py -3.12 scripts/align_captions.py                        # full pipeline
    py -3.12 scripts/align_captions.py --words words.json     # reuse transcript
    py -3.12 scripts/align_captions.py --save-words words.json

Then re-export (`py -3.12 scripts/export_experience.py BHR_Experience.bhrx.json`)
so the runtime metadata picks up the new times.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_captions  # noqa: E402  (parse_script / act_spans / SCRIPT_PDF)

MASTER_MP4 = ROOT / "assets" / "video" / "BHR Draft 1.mp4"
VOSK_MODEL = ROOT / "models" / "vosk-model-small-en-us-0.15"

SPAN_MARGIN_S = 12.0     # hyp words considered for a scene: span +/- margin
LEAD_S = 0.15            # caption appears slightly before the first word
TAIL_S = 0.35            # ... and lingers slightly after the last
MIN_DUR_S = 1.2
MAX_DUR_S = 10.0
MIN_COVERAGE = 0.35      # fraction of a line's words that must match to anchor


def norm_word(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower())


# ----------------------------------------------------------------------
# Transcription
# ----------------------------------------------------------------------

def extract_wav(mp4: Path, out_wav: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
         "-ac", "1", "-ar", "16000", "-vn", str(out_wav)],
        check=True)


def transcribe(wav_path: Path) -> list[dict]:
    """[{word, start, end}, ...] for the whole file via Vosk word timestamps."""
    from vosk import KaldiRecognizer, Model, SetLogLevel
    SetLogLevel(-1)
    model = Model(str(VOSK_MODEL))
    wf = wave.open(str(wav_path), "rb")
    if wf.getframerate() != 16000 or wf.getnchannels() != 1:
        sys.exit("wav must be 16kHz mono (use extract_wav)")
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)
    words: list[dict] = []
    total = wf.getnframes()
    done = 0
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        done += 4000
        if rec.AcceptWaveform(data):
            words.extend(json.loads(rec.Result()).get("result") or [])
        if done % (16000 * 120) < 4000:   # progress every ~2 audio minutes
            print(f"  ... {done / 16000:6.0f}s / {total / 16000:.0f}s "
                  f"({len(words)} words)", flush=True)
    words.extend(json.loads(rec.FinalResult()).get("result") or [])
    wf.close()
    return words


# ----------------------------------------------------------------------
# Alignment
# ----------------------------------------------------------------------

def align_scene(lines: list[str], span: tuple[float, float],
                words: list[dict]) -> list[tuple[float, float] | None]:
    """Per line: (start_s, end_s) in master time, or None if unanchored.

    Matches the scene's concatenated line words against the recognized words
    inside the scene's master-time span (+/- margin), then reads each line's
    time range off its matched words.
    """
    a, b = span[0] - SPAN_MARGIN_S, span[1] + SPAN_MARGIN_S
    hyp = [w for w in words if a <= w["start"] <= b and norm_word(w["word"])]
    hyp_norm = [norm_word(w["word"]) for w in hyp]

    ref_norm: list[str] = []
    ref_line: list[int] = []          # ref token index -> line index
    for li, text in enumerate(lines):
        for w in text.split():
            nw = norm_word(w)
            if nw:
                ref_norm.append(nw)
                ref_line.append(li)

    sm = difflib.SequenceMatcher(a=ref_norm, b=hyp_norm, autojunk=False)
    per_line: dict[int, list[tuple[float, float]]] = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            li = ref_line[blk.a + k]
            h = hyp[blk.b + k]
            per_line.setdefault(li, []).append((h["start"], h["end"]))

    out: list[tuple[float, float] | None] = []
    for li, text in enumerate(lines):
        n_words = max(1, len([w for w in text.split() if norm_word(w)]))
        hits = per_line.get(li, [])
        need = 1 if n_words <= 2 else max(2, int(n_words * MIN_COVERAGE))
        if len(hits) < need:
            out.append(None)
            continue
        start = min(h[0] for h in hits)
        end = max(h[1] for h in hits)
        # A stray far-away match can stretch a line absurdly — reject ranges
        # wildly longer than the line could take to speak (~0.6s/word + slack).
        if end - start > n_words * 0.9 + 6.0:
            out.append(None)
            continue
        out.append((start, end))
    return out


def interpolate_gaps(times: list[tuple[float, float] | None],
                     lines: list[str],
                     span: tuple[float, float]) -> list[tuple[float, float]]:
    """Fill None entries by spreading them between the nearest anchored
    neighbours (span edges at the extremes), weighted by word count."""
    n = len(times)
    filled: list[tuple[float, float]] = list(times)  # type: ignore[arg-type]
    i = 0
    while i < n:
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < n and times[j] is None:
            j += 1
        lo = filled[i - 1][1] if i > 0 else span[0]
        hi = times[j][0] if j < n else span[1]
        if hi <= lo:
            hi = lo + 0.5 * (j - i)
        weights = [max(1, len(lines[k].split())) for k in range(i, j)]
        total = sum(weights)
        t = lo
        for k, wgt in zip(range(i, j), weights):
            slice_s = (hi - lo) * wgt / total
            filled[k] = (t, t + slice_s * 0.9)
            t += slice_s
        i = j
    # Enforce monotonic starts.
    for k in range(1, n):
        if filled[k][0] < filled[k - 1][0] + 0.2:
            s = filled[k - 1][0] + 0.2
            filled[k] = (s, max(filled[k][1], s + MIN_DUR_S))
    return filled


# ----------------------------------------------------------------------
# Project rewrite
# ----------------------------------------------------------------------

def rewrite_project(project_path: Path, words: list[dict]) -> None:
    project = json.loads(project_path.read_text(encoding="utf-8"))
    blocks = [b for b in project["blocks"] if b.get("range_s")]

    def block_for(t: float):
        for b in blocks:
            a, z = b["range_s"]
            if a <= t < z:
                return b
        return None

    scenes = build_captions.parse_script(build_captions.SCRIPT_PDF)
    spans = build_captions.act_spans()

    for b in project["blocks"]:
        if b.get("captions"):
            b["captions"] = [c for c in b["captions"] if not c.get("auto")]
            if not b["captions"]:
                b.pop("captions", None)

    n_total = n_anchored = 0
    for scene, lines in sorted(scenes.items()):
        span = spans.get(scene)
        if not span or not lines:
            continue
        raw = align_scene(lines, span, words)
        anchored = sum(1 for t in raw if t is not None)
        timed = interpolate_gaps(raw, lines, span)
        print(f"  scene {scene:02d}: {anchored}/{len(lines)} lines anchored "
              f"to speech, rest interpolated")
        n_total += len(lines)
        n_anchored += anchored
        for i, (text, (start, end)) in enumerate(zip(lines, timed)):
            t = max(span[0], start - LEAD_S)
            blk = block_for(t)
            if blk is None:
                continue
            block_end = blk["range_s"][1] - blk["range_s"][0]
            at = round(t - blk["range_s"][0], 2)
            dur = min(MAX_DUR_S, max(MIN_DUR_S, end + TAIL_S - t))
            dur = round(min(dur, max(0.5, block_end - at)), 2)
            blk.setdefault("captions", []).append({
                "id": f"capauto_{scene:02d}_{i:03d}",
                "at_s": at,
                "duration_s": dur,
                "text": text,
                "auto": True,
                "aligned": raw[i] is not None,
            })

    for b in project["blocks"]:
        if b.get("captions"):
            b["captions"].sort(key=lambda c: c["at_s"])

    shutil.copy2(project_path, project_path.with_suffix(".json.alignbak"))
    project_path.write_text(json.dumps(project, indent=2) + "\n",
                            encoding="utf-8")
    print(f"[align_captions] {n_anchored}/{n_total} lines speech-anchored -> "
          f"{project_path.name} (backup .alignbak)")
    try:
        from bundle_builder_project import write_bundle
        write_bundle(project_path)
        print("[align_captions] builder bundle refreshed")
    except Exception as exc:
        print(f"[align_captions] WARNING could not refresh bundle: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", type=Path,
                    default=ROOT / "BHR_Experience.bhrx.json")
    ap.add_argument("--wav", type=Path, help="pre-extracted 16k mono wav")
    ap.add_argument("--words", type=Path,
                    help="reuse a saved transcript JSON (skips Vosk)")
    ap.add_argument("--save-words", type=Path,
                    help="save the transcript JSON for reuse")
    args = ap.parse_args()
    if not args.project.exists():
        sys.exit(f"project not found: {args.project}")

    if args.words:
        words = json.loads(args.words.read_text(encoding="utf-8"))
        print(f"[align_captions] loaded {len(words)} words from {args.words}")
    else:
        wav = args.wav
        tmp = None
        if wav is None or not wav.exists():
            tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
            print(f"[align_captions] extracting audio from {MASTER_MP4.name}")
            extract_wav(MASTER_MP4, tmp)
            wav = tmp
        print("[align_captions] transcribing with Vosk (word timestamps)...")
        words = transcribe(wav)
        print(f"[align_captions] {len(words)} recognized words")
        if tmp:
            tmp.unlink(missing_ok=True)
    if args.save_words:
        args.save_words.write_text(json.dumps(words), encoding="utf-8")
        print(f"[align_captions] transcript saved -> {args.save_words}")

    rewrite_project(args.project, words)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
