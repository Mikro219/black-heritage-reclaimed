"""
capcut_audio — import the CapCut master-timeline audio into the BHR experience.

Source of truth: assets/audio/draft_content.json — the CapCut project the
master draft MP4 was exported from. Its audio tracks place the delivered stem
MP3s (assets/audio/stems/) on the same 30fps master timeline that copy_frames.py's
SHOT_FRAMES maps onto shots. This script cuts those placements against the
shot spans and emits per-shot ``audio_events`` (the layered-audio schema read
by sequence_loader / played by engines.audio_mixer.ShotAudioMixer).

What is imported: every placement of a standalone audio material
(``extract_music`` / ``music`` types) whose name matches a delivered file,
plus — since July 2026 — every ``video_original_sound`` placement (Auntie
Liza's VO, sliced from the scene renders). VO placements resolve to the
extracted comp audio in assets/audio/voice_lines/bhr_scene_NN.mp3 (scene
number parsed from the render's file name, plus explicit overrides such as
SC09 Option A/B -> bhr_scene_09-A/-B). A VO source only imports when the
local file's duration agrees with the draft's render within
VO_DUR_TOLERANCE_S — re-cut comps are skipped with a warning until their
offsets are re-aligned by hand. Sources not present locally are likewise
skipped with a warning (re-run the import when they land). ``--no-vo``
restores the stems-only import (for when real AL-line recordings replace
the comp VO). Still skipped: CapCut-internal compound clips.

Track → role mapping (verified against the draft):
    tracks 0–3  → sfx        (one-shots, frame-anchored)
    tracks 4–5  → ambience   (sustain beds — loop through HOLDs)
    tracks 6–7  → music      (sustain beds)

Runtime never slices audio, so placements that start mid-file (or are cut
short / tempo-shifted) are pre-rendered to trimmed OGGs with ffmpeg. Pieces of
one placement split across shots share a single render — that is what lets the
mixer hand a bed over between shots without restarting it.

Usage:
    py -3.12 scripts/capcut_audio.py assets/audio/draft_content.json               # report
    py -3.12 scripts/capcut_audio.py assets/audio/draft_content.json --apply-scenes
    py -3.12 scripts/capcut_audio.py assets/audio/draft_content.json --to-builder BHR_Experience.bhrx.json

    --audio-dir DIR   where the stem files live (default: assets/audio/stems)
    --no-trim         skip ffmpeg pre-rendering; events reference the original
                      files (offsets are lost — degraded but functional)
    --no-vo           stems only; skip VO (video_original_sound) placements

--apply-scenes writes ONLY the "audio_events" key into each live
scenes/<act>/shot_NN/metadata.json (idempotent re-runs) and copies the
referenced files into the shared pool scenes/_audio/.
--to-builder merges audio clips into an existing .bhrx.json project's blocks
(clips keep original file + source_offset_s; the exporter trims at export).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from copy_frames import SHOT_FRAMES  # noqa: E402

FPS = 30                       # master timeline fps (BHR Draft 1.mp4)
US = 1_000_000                 # CapCut times are microseconds

# CapCut audio-track index -> event role. Extra tracks beyond the table map to
# "sfx" (safest: one-shot, no looping).
ROLE_BY_TRACK = ["sfx", "sfx", "sfx", "sfx", "ambience", "ambience", "music", "music"]

AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}

# Trim thresholds: below these the original file is used as-is.
TRIM_OFFSET_S = 0.05
TRIM_TAIL_S   = 0.25

# Back-to-back placements of the same bed on the same track are one logical
# bed (CapCut-style manual looping). Merge when the gap is below this.
BED_MERGE_GAP_S = 0.75

# Roles that play as looping sustain beds; "sfx" and "vo" are one-shots.
BED_ROLES = ("music", "ambience")

# --- VO (video_original_sound) source resolution -----------------------------
# The draft slices VO from the scene renders; we resolve each render to the
# comp audio extracted by scripts/extract_comp_audio.py.
VO_DIR = ROOT / "assets" / "audio" / "voice_lines"

# norm_name(source file name) -> local audio file (absolute Path).
VO_OVERRIDES = {
    "sc09_v1-optiona.mov": VO_DIR / "bhr_scene_09-A.mp3",
    "sc09_v1-optionb.mov": VO_DIR / "bhr_scene_09-B.mp3",
    "img_5032.mov":        ROOT / "assets" / "audio" / "stems" / "IMG_5032.mov",
}

VO_SCENE_RE = re.compile(r"(?:sc[_ ]?|scene[_ ]?)(\d+)", re.I)

# A VO source only imports when the local file's duration matches the draft's
# render within this tolerance — beyond it the comp was re-cut and the draft's
# source offsets no longer line up (user decision July 2026: 0.5s).
VO_DUR_TOLERANCE_S = 0.5

_probe_cache: dict[str, float | None] = {}


def probe_duration(path: Path) -> float | None:
    """Duration in seconds via ffprobe (cached); None when unavailable."""
    key = str(path)
    if key in _probe_cache:
        return _probe_cache[key]
    dur = None
    ffprobe = shutil.which("ffprobe")
    if ffprobe and path.exists():
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True)
        try:
            dur = float(result.stdout.strip())
        except ValueError:
            dur = None
    _probe_cache[key] = dur
    return dur


def resolve_vo_source(source_name: str) -> Path | None:
    """Map a render file name (e.g. 'SC20_V1.mov', 'BHR_scene02.mov') to the
    local voice-line file, or None when unmapped/missing."""
    override = VO_OVERRIDES.get(norm_name(source_name))
    if override is not None:
        return override if override.exists() else None
    m = VO_SCENE_RE.search(source_name)
    if m is None:
        return None
    candidate = VO_DIR / f"bhr_scene_{int(m.group(1)):02d}.mp3"
    return candidate if candidate.exists() else None


def norm_name(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip().lower()


def shot_spans() -> list[tuple[str, str, float, float]]:
    """(act, shot, start_s, end_s) per shot on the master timeline."""
    return [(act, shot, (a - 1) / FPS, b / FPS) for act, shot, a, b in SHOT_FRAMES]


# ---------------------------------------------------------------------------
# Parse the CapCut draft
# ---------------------------------------------------------------------------

def index_audio_files(audio_dir: Path) -> dict[str, Path]:
    """name (NFC-lower, with and without extension) -> file path."""
    idx: dict[str, Path] = {}
    for p in sorted(audio_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            idx.setdefault(norm_name(p.name), p)
            idx.setdefault(norm_name(p.stem), p)
    return idx


def load_placements(draft_path: Path, audio_dir: Path, include_vo: bool = True):
    """Parse the draft into placement dicts + a skip/mismatch summary.

    placement: {name, path, role, track, t0, dur, src_off, speed, gain,
                fade_in_ms, fade_out_ms, natural_s}
    vo_skips: [{name, reason, t0, dur}] — VO placements that could not be
    imported (unmapped source / missing file / re-cut comp).
    """
    with open(draft_path, encoding="utf-8") as f:
        draft = json.load(f)

    mats  = draft.get("materials", {})
    auds  = {a["id"]: a for a in mats.get("audios", [])}
    fades = {fd["id"]: fd for fd in mats.get("audio_fades", [])}
    files = index_audio_files(audio_dir)

    placements: list[dict] = []
    stats = Counter()
    unmatched: Counter = Counter()
    vo_skips: list[dict] = []

    atracks = [t for t in draft.get("tracks", []) if t.get("type") == "audio"]
    for ti, track in enumerate(atracks):
        track_role = ROLE_BY_TRACK[ti] if ti < len(ROLE_BY_TRACK) else "sfx"
        for seg in track.get("segments", []):
            mat = auds.get(seg.get("material_id"))
            if mat is None:
                stats["no_material"] += 1
                continue
            tgt = seg.get("target_timerange", {})
            src = seg.get("source_timerange", {})

            if mat.get("type") == "video_original_sound":
                # Auntie Liza's VO, sliced from a scene render — resolve to
                # the extracted comp audio.
                stats["vo_render"] += 1
                if not include_vo:
                    continue
                src_name = Path(mat.get("path") or mat.get("name", "?")).name
                local = resolve_vo_source(src_name)
                reason = None
                natural = mat.get("duration", 0) / US
                if local is None:
                    reason = "no local voice-line file"
                else:
                    local_dur = probe_duration(local)
                    if local_dur is None:
                        reason = "ffprobe unavailable — cannot verify timing"
                    elif abs(local_dur - natural) > VO_DUR_TOLERANCE_S:
                        reason = (f"comp re-cut (local {local_dur:.2f}s vs "
                                  f"draft render {natural:.2f}s)")
                    else:
                        natural = local_dur   # trust the file we will slice
                if reason is not None:
                    stats["vo_skipped"] += 1
                    vo_skips.append({"name": src_name, "reason": reason,
                                     "t0": tgt.get("start", 0) / US,
                                     "dur": tgt.get("duration", 0) / US})
                    continue
                name, path, role = local.name, local, "vo"
                stats["vo_placed"] += 1
            else:
                path = files.get(norm_name(mat.get("name", "")))
                if path is None:
                    stats["unmatched"] += 1          # compound clips etc.
                    unmatched[mat.get("name", "?")] += 1
                    continue
                name, role = mat["name"], track_role
                natural = mat.get("duration", 0) / US

            fade_in_ms = fade_out_ms = 0
            for ref in seg.get("extra_material_refs", []):
                fd = fades.get(ref)
                if fd is not None:
                    fade_in_ms  = int(fd.get("fade_in_duration", 0) / 1000)
                    fade_out_ms = int(fd.get("fade_out_duration", 0) / 1000)

            placements.append({
                "name":       name,
                "path":       path,
                "role":       role,
                "track":      ti,
                "t0":         tgt.get("start", 0) / US,
                "dur":        tgt.get("duration", 0) / US,
                "src_off":    max(0.0, src.get("start", 0) / US),
                "speed":      float(seg.get("speed", 1.0) or 1.0),
                "gain":       float(seg.get("volume", 1.0)),
                "fade_in_ms":  fade_in_ms,
                "fade_out_ms": fade_out_ms,
                "natural_s":  natural,
            })
            stats["placed"] += 1

    return placements, stats, unmatched, vo_skips


def merge_bed_runs(placements: list[dict]) -> list[dict]:
    """Merge back-to-back same-file bed placements on the same track into one
    logical placement (CapCut manual looping). SFX and VO placements pass
    through — repeats there are intentional (paper crinkles; VO slices)."""
    beds = [p for p in placements if p["role"] in BED_ROLES]
    sfx  = [p for p in placements if p["role"] not in BED_ROLES]

    merged: list[dict] = []
    beds.sort(key=lambda p: (p["track"], norm_name(p["name"]), p["t0"]))
    for p in beds:
        prev = merged[-1] if merged else None
        if (prev is not None
                and prev["track"] == p["track"]
                and norm_name(prev["name"]) == norm_name(p["name"])
                and p["t0"] - (prev["t0"] + prev["dur"]) <= BED_MERGE_GAP_S):
            prev["dur"] = (p["t0"] + p["dur"]) - prev["t0"]
            prev["fade_out_ms"] = p["fade_out_ms"]
            prev["gain"] = max(prev["gain"], p["gain"])
        else:
            merged.append(dict(p))
    return sorted(merged + sfx, key=lambda p: p["t0"])


# ---------------------------------------------------------------------------
# Cut placements against spans (shots or builder blocks)
# ---------------------------------------------------------------------------

def cut_against_spans(placements: list[dict],
                      spans: list[tuple[str, float, float]]):
    """Split each placement across the given (span_id, start_s, end_s) list.

    Returns ({span_id: [piece, ...]}, dropped_count). A piece adds:
    at_s (span-relative), piece_dur, piece_src_off, continues.
    Spans are assumed sorted and non-overlapping on the master timeline
    (duplicated builder branch blocks are handled by the caller passing each
    block as its own span entry).
    """
    by_span: dict[str, list[dict]] = defaultdict(list)
    dropped = 0

    for p in placements:
        t0, t1 = p["t0"], p["t0"] + p["dur"]
        emitted_any = False
        covered = 0.0
        for span_id, s0, s1 in spans:
            a, b = max(t0, s0), min(t1, s1)
            if b - a <= 1e-6:
                continue
            piece = dict(p)
            piece["span"]          = span_id
            piece["at_s"]          = a - s0
            piece["piece_dur"]     = b - a
            piece["piece_src_off"] = p["src_off"] + (a - t0) * p["speed"]
            piece["continues"]     = emitted_any and (a > t0)
            by_span[span_id].append(piece)
            emitted_any = True
            covered += b - a
        if not emitted_any:
            dropped += 1

    for pieces in by_span.values():
        pieces.sort(key=lambda x: x["at_s"])
    return by_span, dropped


# ---------------------------------------------------------------------------
# Trim pre-rendering (shared with export_experience.py)
# ---------------------------------------------------------------------------

def needs_trim(p: dict) -> bool:
    if p["speed"] != 1.0:
        return True
    if p["src_off"] > TRIM_OFFSET_S:
        return True
    # pygame can only play audio containers — a video source (VO sliced from
    # a .mov render) must always be re-rendered.
    if p["path"].suffix.lower() not in AUDIO_EXTS:
        return True
    return p["dur"] < p["natural_s"] - TRIM_TAIL_S


def trim_name(src: Path, offset_s: float, dur_s: float, speed: float) -> str:
    tag = f"__t{int(round(offset_s * 1000))}_{int(round(dur_s * 1000))}"
    if speed != 1.0:
        tag += f"_x{speed:g}".replace(".", "p")
    return f"{src.stem}{tag}.ogg"


RESAMPLE_HZ = 48000


def render_trim(ffmpeg: str, src: Path, offset_s: float, dur_s: float,
                speed: float, out_path: Path) -> bool:
    """Pre-render a trimmed (and optionally speed-shifted) clip as OGG.

    ``dur_s`` is the OUTPUT (timeline) duration — a speeded clip reads
    ``dur_s * speed`` seconds of source. Speed shifts pitch WITH tempo
    (asetrate, CapCut's audible speed-change behaviour — the draft's
    stroke-beep ramp is a pitch ramp), not tempo-only atempo."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    speed = min(4.0, max(0.25, speed))
    read_s = max(0.05, dur_s) * speed
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{max(0.0, offset_s):.3f}", "-t", f"{read_s:.3f}",
           "-i", str(src)]
    if speed != 1.0:
        cmd += ["-filter:a",
                f"aresample={RESAMPLE_HZ},"
                f"asetrate={int(round(RESAMPLE_HZ * speed))},"
                f"aresample={RESAMPLE_HZ}"]
    cmd += ["-c:a", "libvorbis", "-q:a", "4", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[capcut_audio] WARNING ffmpeg trim failed for {src.name}: "
              f"{result.stderr.strip()[:200]}")
        return False
    return True


def pieces_to_events(pieces: list[dict], pool: Path,
                     ffmpeg: str | None, do_trim: bool,
                     rendered: dict[str, Path]) -> list[dict]:
    """Convert cut pieces to runtime audio_events, copying/pre-rendering the
    referenced files into the shared pool. ``rendered`` caches renders across
    shots (keyed by output name) so split beds share one file."""
    events = []
    for pc in pieces:
        src: Path = pc["path"]
        if do_trim and ffmpeg and needs_trim(pc):
            # One render per PLACEMENT (placement-level offset/duration), so
            # all pieces of a split bed reference the same file — that shared
            # name is what the mixer's continuity handover matches on.
            # dur is the OUTPUT duration; a speeded clip consumes speed×dur of
            # source, so the availability clamp converts to the output domain.
            dur = min(pc["dur"],
                      max(0.05, (pc["natural_s"] - pc["src_off"]) / pc["speed"])) \
                  if pc["natural_s"] > 0 else pc["dur"]
            name = trim_name(src, pc["src_off"], dur, pc["speed"])
            out = pool / name
            if name not in rendered:
                if out.exists() or render_trim(ffmpeg, src, pc["src_off"], dur,
                                               pc["speed"], out):
                    rendered[name] = out
            if name in rendered:
                file_name = name
            else:
                file_name = src.name
                _copy_into_pool(src, pool)
        else:
            file_name = src.name
            _copy_into_pool(src, pool)

        events.append({
            "file":            file_name,
            "role":            pc["role"],
            "at_s":            round(pc["at_s"], 3),
            "duration_s":      round(pc["piece_dur"], 3),
            "source_offset_s": round(pc["piece_src_off"], 3),
            "gain":            round(pc["gain"], 4),
            "fade_in_ms":      pc["fade_in_ms"],
            "fade_out_ms":     pc["fade_out_ms"],
            "sustain":         pc["role"] in BED_ROLES,
            "continues":       pc["continues"],
        })
    return events


def _copy_into_pool(src: Path, pool: Path) -> None:
    dst = pool / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    pool.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def report(by_shot, placements, stats, unmatched, dropped,
           vo_skips=()) -> None:
    W = 66
    print("=" * W)
    print("CapCut audio import — report")
    print("=" * W)
    print(f"  placements matched: {stats['placed']}  "
          f"(after bed merging: {len(placements)})")
    print(f"  VO placements imported: {stats['vo_placed']} of "
          f"{stats['vo_render']}   "
          f"unmatched materials skipped: {stats['unmatched']}")
    if unmatched:
        names = ", ".join(sorted(unmatched)[:6])
        print(f"    unmatched: {names}{' …' if len(unmatched) > 6 else ''}")
    if vo_skips:
        spans = shot_spans()
        print(f"  VO placements skipped: {len(vo_skips)}")
        by_source: dict[str, dict] = {}
        for sk in vo_skips:
            shots = [shot for _, shot, s0, s1 in spans
                     if min(sk["t0"] + sk["dur"], s1) - max(sk["t0"], s0) > 1e-6]
            rec = by_source.setdefault(sk["name"], {"reason": sk["reason"],
                                                    "n": 0, "shots": set()})
            rec["n"] += 1
            rec["shots"].update(shots)
        for name, rec in sorted(by_source.items()):
            shots = ", ".join(sorted(rec["shots"])) or "none"
            print(f"    WARNING {name}: {rec['reason']} — "
                  f"{rec['n']} slice(s), shots {shots}")
    print(f"  placements dropped (no shot covers them): {dropped}")
    print("-" * W)
    total = 0
    for act, shot, _, _ in SHOT_FRAMES:
        pieces = by_shot.get(shot, [])
        if not pieces:
            continue
        roles = Counter(p["role"] for p in pieces)
        total += len(pieces)
        parts = "  ".join(f"{r}:{n}" for r, n in sorted(roles.items()))
        print(f"  shot {shot} ({act[:24]:24s})  {len(pieces):3d} events   {parts}")
    print("-" * W)
    print(f"  {total} events across {len(by_shot)} shots")
    print("=" * W)


def apply_scenes(by_shot, scenes_root: Path, ffmpeg: str | None,
                 do_trim: bool) -> None:
    pool = scenes_root / "_audio"
    rendered: dict[str, Path] = {}
    written = 0
    for act, shot, _, _ in SHOT_FRAMES:
        shot_dir = scenes_root / act / f"shot_{shot}"
        meta_path = shot_dir / "metadata.json"
        if not shot_dir.is_dir():
            if by_shot.get(shot):
                print(f"[capcut_audio] WARNING no shot dir for shot {shot} "
                      f"({act}) — {len(by_shot[shot])} events skipped")
            continue
        if meta_path.exists():
            with open(meta_path, encoding="utf-8-sig") as f:  # some files carry a BOM
                meta = json.load(f)
        else:
            # Plain playback shots often have no metadata.json (they run off
            # sequence.json defaults) — create a minimal one to carry the events.
            if not by_shot.get(shot):
                continue
            meta = {"shot": shot}
        pieces = by_shot.get(shot, [])
        if pieces:
            meta["audio_events"] = pieces_to_events(pieces, pool, ffmpeg,
                                                    do_trim, rendered)
        else:
            meta.pop("audio_events", None)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        written += 1
    n_pool = len(list(pool.glob("*"))) if pool.is_dir() else 0
    print(f"[capcut_audio] wrote audio_events into {written} shot metadata files; "
          f"pool {pool} now holds {n_pool} files")


def to_builder(placements: list[dict], project_path: Path) -> None:
    with open(project_path, encoding="utf-8") as f:
        project = json.load(f)

    # Every block with a media range is its own span (branch blocks may
    # duplicate master ranges — each gets its own copy of the audio).
    spans = []
    for b in project.get("blocks", []):
        rng = b.get("range_s")
        if b.get("media") and rng:
            spans.append((b["id"], float(rng[0]), float(rng[1])))
    spans.sort(key=lambda s: s[1])

    by_block, dropped = cut_against_spans(placements, spans)

    # Register sounds (dedupe by name), replace blocks' audio lists.
    sounds = {norm_name(s["name"]): s for s in project.get("sounds", [])}
    next_id = len(project.get("sounds", [])) + 1
    clip_n = 0
    for b in project.get("blocks", []):
        pieces = by_block.get(b["id"], [])
        clips = []
        for pc in pieces:
            key = norm_name(pc["path"].name)
            if key not in sounds:
                sounds[key] = {"id": f"s{next_id}", "name": pc["path"].name}
                next_id += 1
            clip_n += 1
            clip = {
                "id":              f"a{clip_n}",
                "sound":           sounds[key]["id"],
                "at_s":            round(pc["at_s"], 3),
                "duration_s":      round(pc["piece_dur"], 3),
                "source_offset_s": round(pc["piece_src_off"], 3),
                "gain":            round(pc["gain"], 4),
                "fade_in_ms":      pc["fade_in_ms"],
                "fade_out_ms":     pc["fade_out_ms"],
                "sustain":         pc["role"] in BED_ROLES,
                "continues":       pc["continues"],
                "role":            pc["role"],
            }
            # CapCut clip speed shifts pitch with tempo (the draw-stroke beep
            # ramp); carry it so the export can reproduce the shift.
            if abs(pc["speed"] - 1.0) > 1e-3:
                clip["speed"] = round(pc["speed"], 4)
            clips.append(clip)
        if clips:
            b["audio"] = clips
        else:
            b.pop("audio", None)
    project["sounds"] = sorted(sounds.values(), key=lambda s: s["name"].lower())

    with open(project_path, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2)
        f.write("\n")
    print(f"[capcut_audio] merged {clip_n} audio clips into "
          f"{sum(1 for b in project['blocks'] if b.get('audio'))} blocks of "
          f"{project_path.name} ({dropped} placements fell outside all blocks); "
          f"{len(project['sounds'])} sounds registered")

    # Refresh the builder's auto-loaded bundle so the merged project shows up
    # the next time index.html is opened (file:// pages cannot fetch JSON).
    try:
        from bundle_builder_project import write_bundle
        write_bundle(project_path)
    except Exception as exc:                          # bundling is best-effort
        print(f"[capcut_audio] WARNING could not refresh builder bundle: {exc}")


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import CapCut master-timeline audio as BHR audio_events")
    parser.add_argument("draft", type=Path, help="CapCut draft_content.json")
    parser.add_argument("--audio-dir", type=Path,
                        default=ROOT / "assets" / "audio" / "stems")
    parser.add_argument("--apply-scenes", action="store_true",
                        help="write audio_events into the live scenes/ tree")
    parser.add_argument("--scenes-root", type=Path, default=ROOT / "scenes",
                        help="scenes tree for --apply-scenes (default: scenes/)")
    parser.add_argument("--to-builder", type=Path, default=None,
                        help="merge audio clips into a .bhrx.json project")
    parser.add_argument("--no-trim", action="store_true",
                        help="skip ffmpeg pre-rendering of offset/cut clips")
    parser.add_argument("--no-vo", action="store_true",
                        help="stems only — skip video_original_sound (VO) "
                             "placements (use once real AL-line recordings "
                             "replace the comp VO)")
    args = parser.parse_args()

    if not args.draft.exists():
        sys.exit(f"draft not found: {args.draft}")
    if not args.audio_dir.is_dir():
        sys.exit(f"audio dir not found: {args.audio_dir}")

    raw, stats, unmatched, vo_skips = load_placements(
        args.draft, args.audio_dir, include_vo=not args.no_vo)
    placements = merge_bed_runs(raw)

    spans = [(shot, s0, s1) for _, shot, s0, s1 in shot_spans()]
    by_shot, dropped = cut_against_spans(placements, spans)

    report(by_shot, placements, stats, unmatched, dropped, vo_skips)

    ffmpeg = shutil.which("ffmpeg")
    if not args.no_trim and ffmpeg is None and (args.apply_scenes):
        print("[capcut_audio] WARNING ffmpeg not found — trimmed clips fall "
              "back to whole source files (install: winget install ffmpeg)")

    if args.apply_scenes:
        apply_scenes(by_shot, args.scenes_root, ffmpeg, not args.no_trim)
    if args.to_builder is not None:
        if not args.to_builder.exists():
            sys.exit(f"builder project not found: {args.to_builder}")
        to_builder(placements, args.to_builder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
