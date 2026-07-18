"""
export_experience.py — Convert an Experience Builder project (.bhrx.json) into
a BHR runtime scenes tree.

    py -3.12 scripts/export_experience.py "My_Experience.bhrx.json"
    py -3.12 scripts/export_experience.py project.bhrx.json --no-frames
    py -3.12 scripts/export_experience.py project.bhrx.json --video "assets/video/BHR Draft 1.mp4"
    py -3.12 scripts/export_experience.py project.bhrx.json --out export/my_scenes

Output (default export/scenes_generated/ — the live scenes/ tree is NEVER touched):

    export/scenes_generated/
      sequence.json                    ordered shot list (loader-compatible)
      act_01_experience/
        act.json                       {"fps": <project fps>}
        shot_NN/metadata.json          playback / OI / choice-fork wiring
        shot_NN/frames/00001.jpg ...   (unless --no-frames)
        shot_NN/audio/detect.mp3       (when the global detect sound is found)

Mapping (mirrors the hand-authored patterns in scenes/):
  playback block, no windows   -> kind playback                        (plain)
  playback block, 1 window     -> kind playback + OI oi_frame_window   (shot 58 pattern)
  playback block, 2+ windows   -> kind interactive, play-through FSM
                                  with per-window `oi` states           (shot 24 pattern)
  choice block (2 branches)    -> kind interactive, region-fork FSM
                                  (waiting -> point_left/point_right)   (shot 09 pattern);
                                  branch chains gated with play_if
                                  {"shot": <fork>, "branch": "left"|"right"}
  merge block                  -> not a shot; branch gating ends there

Constraints inherited from the runtime:
  * Forks are two-way (left/right). Extra branches are dropped with a warning.
  * One play_if per shot: branching nests only through a merge/convergence.
  * `voice` windows need the voice engine's per-shot wiring — exported as a
    warning, not as a detector.

Frame extraction uses ffmpeg (winget install ffmpeg) with -vf fps=<project fps>
so frame indices match the metadata exactly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACT_ID = "01"
ACT_DIRNAME = f"act_{ACT_ID}_experience"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capcut_audio import trim_name, render_trim, TRIM_OFFSET_S, TRIM_TAIL_S  # noqa: E402

warnings: list[str] = []


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"  [warn] {msg}")


# ---------------------------------------------------------------------------
# Project loading / lookups
# ---------------------------------------------------------------------------

def load_project(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        project = json.load(f)
    if not isinstance(project.get("blocks"), list):
        sys.exit(f"[export] {path} is not an Experience Builder project (no blocks array)")
    return project


def block_map(project: dict) -> dict:
    return {b["id"]: b for b in project["blocks"]}


def flow_next(project: dict, block_id: str) -> str | None:
    for e in project.get("edges", []):
        if e.get("from") == block_id:
            return e.get("to")
    return None


def indegree(project: dict) -> dict:
    """Structural fan-in per block. Timeout edges and voice-alias branches are
    excluded: both are alternate triggers onto an existing path, not paths of
    their own, so they must not make their target look like a convergence."""
    deg: dict = {}
    for e in project.get("edges", []):
        deg[e.get("to")] = deg.get(e.get("to"), 0) + 1
    for b in project["blocks"]:
        if b.get("type") == "choice":
            fork_branches, _voice = split_choice_branches(b)
            for br in fork_branches[:2]:
                deg[br["to"]] = deg.get(br["to"], 0) + 1
    return deg


def enabled_windows(block: dict) -> list[dict]:
    return [w for w in (block.get("windows") or []) if not w.get("disabled")]


# ---------------------------------------------------------------------------
# Linearization: graph -> ordered shot list with play_if gating
# ---------------------------------------------------------------------------

def linearize(project: dict) -> list[dict]:
    """DFS from the start block. Returns [{"block": ..., "play_if_tag": ...}].

    play_if_tag is {"fork_block": <id>, "branch": "left"|"right"} carried along
    a branch chain until a merge block or a convergence node (indegree > 1).
    Merge blocks emit no shot.
    """
    blocks = block_map(project)
    deg = indegree(project)
    ordered: list[dict] = []
    emitted: set = set()

    def walk(node_id, tag, stop_at_boundary):
        """Emit a chain. With stop_at_boundary, stop WITHOUT emitting when a
        merge block or a fan-in (indegree > 1) node is reached, and return it —
        that's where a branch chain rejoins the main flow. skip_stop lets a
        choice resume AT its own convergence node without instantly re-stopping.
        """
        skip_stop = False
        while node_id:
            node = blocks.get(node_id)
            if node is None:
                warn(f"edge points at missing block {node_id!r}")
                return None
            boundary = node.get("type") == "merge" or deg.get(node_id, 0) > 1
            if stop_at_boundary and boundary and not skip_stop:
                return node_id
            skip_stop = False
            if node_id in emitted:
                return None
            if deg.get(node_id, 0) > 1:
                tag = None                      # convergence: no more gating
            if node.get("type") == "merge":
                emitted.add(node_id)
                node_id = flow_next(project, node_id)
                tag = None
                skip_stop = False
                continue

            emitted.add(node_id)
            ordered.append({"block": node, "play_if_tag": tag})

            if node.get("type") == "choice":
                # Voice branches are spoken aliases for a fork side (wired in
                # choice_metadata) — only GESTURE branches define the fork's
                # left/right chains here.
                branches, voice_branches = split_choice_branches(node)
                fork_targets = {br["to"] for br in branches[:2]}
                for vbr in voice_branches:
                    if vbr.get("to") not in fork_targets:
                        warn(f"choice {node.get('name')!r}: voice branch "
                             f"{vbr.get('label')!r} target is not a fork side — "
                             f"its chain is not emitted")
                if len(branches) > 2:
                    warn(f"choice {node.get('name')!r} has {len(branches)} gesture "
                         f"branches; the runtime forks two ways — extra branches dropped")
                    branches = branches[:2]
                tt_to = (node.get("timeout") or {}).get("to")
                if tt_to and branches and tt_to != branches[0]["to"]:
                    warn(f"choice {node.get('name')!r}: the runtime's timeout "
                         f"auto-advance always falls into the FIRST branch — the "
                         f"timeout target set in the editor can't override that")
                convergences = []
                for i, br in enumerate(branches):
                    c = walk(br["to"], {"fork_block": node_id,
                                        "branch": "left" if i == 0 else "right"},
                             stop_at_boundary=True)
                    if c is not None:
                        convergences.append(c)
                # both branch chains stop at the same rejoin point; continue there
                node_id = convergences[0] if convergences else None
                tag = None
                skip_stop = True
                continue
            node_id = flow_next(project, node_id)
        return None

    start = project.get("start")
    if not start or start not in blocks:
        sys.exit("[export] project has no valid start block")
    walk(start, None, stop_at_boundary=False)

    unreached = [b for b in project["blocks"]
                 if b["id"] not in emitted and b.get("type") != "merge"]
    for b in unreached:
        warn(f"block {b.get('name')!r} unreachable from start — not exported")
    return ordered


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------

def region_rect(win: dict) -> dict | None:
    r = win.get("region")
    if not r or r.get("w", 0) <= 0 or r.get("h", 0) <= 0:
        return None
    return {"x": round(r["x"], 3), "y": round(r["y"], 3),
            "w": round(r["w"], 3), "h": round(r["h"], 3)}


def slug(text: str, fallback: str) -> str:
    s = "".join(c if c.isalnum() else "_" for c in (text or "").lower()).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s or fallback


def oi_dict(win: dict, idx: int) -> dict:
    params = dict(win.get("params") or {})
    rect = region_rect(win)
    if rect:
        params["region_rect"] = rect
    return {
        "id": slug(win.get("label"), f"oi_{idx}"),
        "type": win["detector"],
        "params": params,
        "sfx": "detect.mp3",
        "feedback": "green_flash",
    }


def window_frames(win: dict, n_frames: int, fps: int) -> tuple[int, int]:
    """Window's [first, last] frame, 1-based, clamped to the shot."""
    a = int(round((win.get("appears_s") or 0) * fps)) + 1
    if win.get("duration_s") is None:
        b = n_frames
    else:
        b = int(round(((win.get("appears_s") or 0) + win["duration_s"]) * fps))
    a = max(1, min(n_frames, a))
    b = max(a, min(n_frames, b))
    return a, b


def gesture_windows(block: dict) -> list[dict]:
    """Enabled windows in appearance order — gesture AND voice. Voice windows
    become keyword OI states (playback) or spoken branch picks (choice)."""
    return sorted(enabled_windows(block), key=lambda w: w.get("appears_s") or 0)


def is_voice(win: dict) -> bool:
    return win.get("detector") == "voice"


def voice_oi_dict(win: dict, idx: int) -> dict:
    """FSM `oi` payload for a voice window: a `keywords` list makes the runtime
    open a VI window (reaction tier) instead of arming a gesture detector."""
    kw = (win.get("params") or {}).get("keyword") or "go"
    return {
        "id": slug(win.get("label"), f"vi_{idx}"),
        "keywords": [kw],
        "mode": (win.get("params") or {}).get("mode", "keyword"),
        "sfx": "detect.mp3",
        "feedback": "green_flash",
    }


def playback_metadata(block: dict, shot_id: str, n_frames: int, fps: int) -> dict:
    meta = {
        "shot": shot_id,
        "kind": "playback",
        "_generated_from": {"block": block["id"], "name": block.get("name")},
    }
    wins = gesture_windows(block)
    if not wins:
        return meta

    if len(wins) == 1 and not is_voice(wins[0]):
        a, b = window_frames(wins[0], n_frames, fps)
        inter = oi_dict(wins[0], 1)
        inter["tier"] = "OI"
        inter["oi_frame_window"] = [a, b]
        meta["interaction"] = inter
        return meta

    # 2+ windows (or any voice window): play-through FSM with oi states
    # (shot 24 pattern; voice windows become keyword VI states)
    meta["kind"] = "interactive"
    segments: dict = {}
    states: dict = {}
    order: list[str] = []

    spans = []
    cursor = 1
    for i, w in enumerate(wins):
        a, b = window_frames(w, n_frames, fps)
        if a <= cursor:
            a = cursor  # clip overlaps in appearance order
        if b < a:
            warn(f"block {block.get('name')!r}: window {w.get('label')!r} "
                 f"is squeezed out by an earlier window — skipped")
            continue
        spans.append((a, b, w, i))
        cursor = b + 1

    if not spans:
        meta["kind"] = "playback"
        return meta

    cursor = 1
    if spans[0][0] > 1:
        segments["intro"] = [1, spans[0][0] - 1]   # played by PLAY_INTRO
        cursor = spans[0][0]
    for a, b, w, i in spans:
        if a > cursor:
            gap = f"between_{i}"
            segments[gap] = [cursor, a - 1]
            states[gap] = {"segment": gap, "loop": False}
            order.append(gap)
        name = f"oi_{i + 1}"
        segments[name] = [a, b]
        payload = voice_oi_dict(w, i + 1) if is_voice(w) else oi_dict(w, i + 1)
        states[name] = {"segment": name, "loop": False, "oi": payload}
        order.append(name)
        cursor = b + 1
    if cursor <= n_frames:
        segments["outro"] = [cursor, n_frames]
        states["outro"] = {"segment": "outro", "loop": False}
        order.append("outro")

    transitions = [{"from": a, "on": "segment_end", "to": b}
                   for a, b in zip(order, order[1:])]
    transitions.append({"from": order[-1], "on": "segment_end", "to": "__advance__"})

    meta["segments"] = segments
    meta["interaction"] = {
        "tier": "OI",
        "interaction_fsm": {
            "initial": order[0],
            "states": states,
            "transitions": transitions,
            "fallback": {"timeout_s": 120, "on_timeout": "auto_advance"},
        },
    }
    return meta


def split_choice_branches(block: dict) -> tuple[list[dict], list[dict]]:
    """(fork_branches, voice_branches): the first two branches whose window is
    a gesture define the runtime fork's left/right; voice-window branches are
    spoken aliases for a side (matched by target block)."""
    wins = {w["id"]: w for w in (block.get("windows") or [])}
    branches = [br for br in (block.get("branches") or []) if br.get("to")]
    fork, voice = [], []
    for br in branches:
        w = wins.get(br.get("window"))
        (voice if (w and is_voice(w)) else fork).append(br)
    return fork, voice


def choice_metadata(block: dict, shot_id: str, n_frames: int, fps: int) -> dict:
    fork_branches, voice_branches = split_choice_branches(block)
    wins = gesture_windows(block)
    win_by_id = {w["id"]: w for w in wins}
    hold_ms = 600
    for w in wins:
        if isinstance((w.get("params") or {}).get("hold_ms"), (int, float)):
            hold_ms = int(w["params"]["hold_ms"])
            break
    non_point = [w for w in wins
                 if not is_voice(w) and w.get("detector") not in
                 ("point_region", "directional_point", "point_target_held", "forward_point")]
    if non_point:
        warn(f"choice {block.get('name')!r}: fork exported as the runtime's "
             f"point-left/right region model; window detector(s) "
             f"{sorted({w['detector'] for w in non_point})} are advisory only")

    intro_end = max(1, n_frames - 1)
    segments = {"intro": [1, intro_end], "idle_loop": [n_frames, n_frames]}
    waiting = {"segment": "idle_loop", "loop": True,
               "directions": ["left", "right"]}
    fsm = {
        "gesture_type": "region",
        "initial": "waiting",
        "states": {
            "waiting":       waiting,
            "confirm_left":  {"segment": "idle_loop", "loop": False,
                              "on_enter_sfx": "detect.mp3"},
            "confirm_right": {"segment": "idle_loop", "loop": False,
                              "on_enter_sfx": "detect.mp3"},
        },
        "transitions": [
            {"from": "waiting",       "on": "point_left",  "to": "confirm_left"},
            {"from": "waiting",       "on": "point_right", "to": "confirm_right"},
            {"from": "confirm_left",  "on": "segment_end", "to": "__advance__"},
            {"from": "confirm_right", "on": "segment_end", "to": "__advance__"},
        ],
        "fallback": {
            "timeout_s": int((block.get("timeout") or {}).get("seconds") or 30),
            "on_timeout": "auto_advance",
        },
    }

    # Spoken branch picks: a voice window whose branch targets the same block
    # as a fork side becomes "say the keyword to pick that side". The runtime
    # holds ONE voice keyword per state, so only the first is wired.
    side_by_target = {}
    for idx, br in enumerate(fork_branches[:2]):
        side_by_target[br["to"]] = "left" if idx == 0 else "right"
    voice_wired = False
    for br in voice_branches:
        w = win_by_id.get(br.get("window"))
        kw = ((w or {}).get("params") or {}).get("keyword") or "go"
        side = side_by_target.get(br.get("to"))
        if side is None:
            warn(f"choice {block.get('name')!r}: voice branch {br.get('label')!r} "
                 f"targets a block that is not one of the two fork sides — skipped")
            continue
        if voice_wired:
            warn(f"choice {block.get('name')!r}: the runtime supports one voice "
                 f"keyword per hold — extra voice branch {br.get('label')!r} skipped")
            continue
        waiting["voice"] = kw
        fsm["transitions"].insert(2, {"from": "waiting", "on": f"voice_{kw}",
                                      "to": f"confirm_{side}"})
        voice_wired = True
        warn(f"choice {block.get('name')!r}: '{kw}' picks the {side} side by "
             f"voice — note the runtime records fork choices from GESTURE picks "
             f"only, so branch-gated shots after a voice pick fall back to the "
             f"first branch")
    for w in wins:
        if is_voice(w) and not any(br.get("window") == w["id"] for br in voice_branches):
            warn(f"choice {block.get('name')!r}: voice window {w.get('label')!r} "
                 f"has no 'go to block' target — give it one to make it a spoken "
                 f"branch pick (left out of the export)")
    return {
        "shot": shot_id,
        "kind": "interactive",
        "hold_ms": hold_ms,
        "_generated_from": {"block": block["id"], "name": block.get("name"),
                            "branch_labels": [br.get("label") for br in fork_branches[:2]]},
        "segments": segments,
        "interaction": {"tier": "CG", "hold_ms": hold_ms, "interaction_fsm": fsm},
        "fallback": {"timeout_s": int((block.get("timeout") or {}).get("seconds") or 30),
                     "reprompt_s": [8, 16], "on_timeout": "auto_advance"},
    }


# ---------------------------------------------------------------------------
# Frames (ffmpeg) + detect sound
# ---------------------------------------------------------------------------

def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def resolve_video(project: dict, media_id: str | None,
                  project_path: Path, override: Path | None) -> Path | None:
    if override is not None:
        return override if override.exists() else None
    media = next((m for m in project.get("media", []) if m["id"] == media_id), None)
    if media is None:
        return None
    name = media["name"]
    for candidate in (project_path.parent / name,
                      ROOT / "assets" / "video" / name,
                      ROOT / "assets" / "video" / "comps" / name,
                      ROOT / "assets" / name, Path(name)):
        if candidate.exists():
            return candidate
    return None


def extract_frames(ffmpeg: str, video: Path, start_s: float, n_frames: int,
                   fps: int, out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, start_s):.3f}",
        "-i", str(video),
        "-vf", f"fps={fps}",
        "-frames:v", str(n_frames),
        "-q:v", "2",
        str(out_dir / "%05d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        warn(f"ffmpeg failed for {out_dir.parent.name}: {result.stderr.strip()[:300]}")
        return False
    return True


def resolve_detect_sound(project: dict, project_path: Path,
                         override: Path | None) -> Path | None:
    if override is not None:
        return override if override.exists() else None
    sid = (project.get("settings") or {}).get("global_detect_sound")
    sound = next((s for s in project.get("sounds", []) if s["id"] == sid), None)
    if sound is None:
        return None
    name = sound["name"]
    for candidate in (project_path.parent / name,
                      ROOT / "assets" / "audio" / "stems" / name,
                      ROOT / "assets" / name, Path(name)):
        if candidate.exists():
            return candidate
    # Fall back to any copy in the live scenes tree (the runtime ships one per
    # interactive shot, so the first hit is as good as any).
    hits = sorted((ROOT / "scenes").glob(f"*/*/{name}")) + \
           sorted((ROOT / "scenes").glob(f"*/*/audio/{name}"))
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Block audio -> audio_events (layered stem audio, July 2026)
# ---------------------------------------------------------------------------

def resolve_sound_file(name: str, project_path: Path) -> Path | None:
    """Find a sound file by name: next to the project, in assets/audio/stems/
    (the delivered stem drop), or anywhere under assets/audio/."""
    for candidate in (project_path.parent / name,
                      ROOT / "assets" / "audio" / "stems" / name,
                      ROOT / "assets" / name):
        if candidate.exists():
            return candidate
    audio_dir = ROOT / "assets" / "audio"
    if audio_dir.is_dir():
        hits = sorted(audio_dir.rglob(name))
        if hits:
            return hits[0]
    return None


_duration_cache: dict[str, float | None] = {}


def probe_duration(path: Path) -> float | None:
    """Natural duration via ffprobe (ships with ffmpeg). None if unavailable."""
    key = str(path)
    if key in _duration_cache:
        return _duration_cache[key]
    ffprobe = shutil.which("ffprobe")
    dur = None
    if ffprobe is not None:
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True)
        try:
            dur = float(result.stdout.strip())
        except ValueError:
            dur = None
    _duration_cache[key] = dur
    return dur


def block_audio_events(block: dict, sounds_by_id: dict, project_path: Path,
                       pool: Path, ffmpeg: str | None,
                       rendered: dict[str, Path]) -> list[dict]:
    """Convert a block's builder audio clips into runtime audio_events, copying
    or pre-rendering the referenced files into the tree's _audio/ pool.

    Builder clips reference the ORIGINAL file + source_offset_s (the browser
    preview seeks natively); the runtime plays files whole, so offset/cut clips
    are trimmed here with ffmpeg — split beds keep one shared render so the
    mixer's continuity handover can match on the file name."""
    events = []
    for clip in block.get("audio", []):
        sound = sounds_by_id.get(clip.get("sound"))
        if sound is None:
            warn(f"block {block.get('name')!r}: audio clip references unknown "
                 f"sound id {clip.get('sound')!r} — skipped")
            continue
        src = resolve_sound_file(sound["name"], project_path)
        if src is None:
            warn(f"block {block.get('name')!r}: sound file {sound['name']!r} "
                 f"not found — clip skipped")
            continue

        offset = float(clip.get("source_offset_s") or 0.0)
        dur    = clip.get("duration_s")
        natural = probe_duration(src)
        cut = (dur is not None and natural is not None
               and float(dur) < natural - TRIM_TAIL_S)
        file_name = src.name
        if ffmpeg and (offset > TRIM_OFFSET_S or cut):
            render_dur = float(dur) if dur is not None else \
                (max(0.05, natural - offset) if natural else 1.0)
            name = trim_name(src, offset, render_dur, 1.0)
            out = pool / name
            if name not in rendered:
                if out.exists() or render_trim(ffmpeg, src, offset, render_dur,
                                               1.0, out):
                    rendered[name] = out
            if name in rendered:
                file_name = name
            else:
                _pool_copy(src, pool)
        elif offset > TRIM_OFFSET_S:
            warn(f"block {block.get('name')!r}: {sound['name']!r} has a source "
                 f"offset but ffmpeg is missing — plays from the file start")
            _pool_copy(src, pool)
        else:
            _pool_copy(src, pool)

        role = clip.get("role", "sfx")
        events.append({
            "file":            file_name,
            "role":            role,
            "at_s":            round(float(clip.get("at_s") or 0.0), 3),
            "duration_s":      (None if dur is None else round(float(dur), 3)),
            "source_offset_s": round(offset, 3),
            "gain":            round(float(clip.get("gain", 1.0)), 4),
            "fade_in_ms":      int(clip.get("fade_in_ms") or 0),
            "fade_out_ms":     int(clip.get("fade_out_ms") or 0),
            "sustain":         bool(clip.get("sustain",
                                             role in ("music", "ambience"))),
            "continues":       bool(clip.get("continues", False)),
        })
    events.sort(key=lambda e: e["at_s"])
    return events


def _pool_copy(src: Path, pool: Path) -> None:
    dst = pool / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    pool.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export(project_path: Path, out_root: Path, do_frames: bool,
           video_override: Path | None, sound_override: Path | None) -> int:
    project = load_project(project_path)
    fps = int(project.get("fps") or 30)
    ordered = linearize(project)
    if not ordered:
        sys.exit("[export] nothing reachable from the start block")

    pad = max(2, len(str(len(ordered))))
    shot_ids = {entry["block"]["id"]: str(i + 1).zfill(pad)
                for i, entry in enumerate(ordered)}

    act_dir = out_root / ACT_DIRNAME
    act_dir.mkdir(parents=True, exist_ok=True)
    with open(act_dir / "act.json", "w", encoding="utf-8") as f:
        json.dump({"fps": fps, "_generated_from": project.get("name")}, f, indent=2)

    detect_sound = resolve_detect_sound(project, project_path, sound_override)
    if detect_sound is None:
        warn("global detect sound file not found next to the project, in assets/, "
             "or via --detect-sound — shots reference detect.mp3 but none was copied")

    ffmpeg = find_ffmpeg() if do_frames else None
    if do_frames and ffmpeg is None:
        warn("ffmpeg not on PATH (winget install ffmpeg) — writing metadata only")
        do_frames = False

    # Layered stem audio: block audio clips -> per-shot audio_events + _audio pool.
    audio_ffmpeg = find_ffmpeg()
    sounds_by_id = {s["id"]: s for s in project.get("sounds", [])}
    audio_pool   = out_root / "_audio"
    rendered: dict[str, Path] = {}

    seq_shots = []
    for entry in ordered:
        block = entry["block"]
        shot_id = shot_ids[block["id"]]
        rng = block.get("range_s") or [0, 0]
        n_frames = max(1, int(round((rng[1] - rng[0]) * fps)))

        if block.get("type") == "choice":
            meta = choice_metadata(block, shot_id, n_frames, fps)
        else:
            meta = playback_metadata(block, shot_id, n_frames, fps)

        tag = entry["play_if_tag"]
        if tag:
            fork_shot = shot_ids.get(tag["fork_block"])
            if fork_shot:
                meta["play_if"] = {"shot": fork_shot, "branch": tag["branch"]}

        if block.get("audio") and not block.get("master_audio"):
            events = block_audio_events(block, sounds_by_id, project_path,
                                        audio_pool, audio_ffmpeg, rendered)
            if events:
                meta["audio_events"] = events

        shot_dir = act_dir / f"shot_{shot_id}"
        shot_dir.mkdir(parents=True, exist_ok=True)

        # master_audio: slice the source video's own baked mix to the
        # runtime's whole-file audio.mp3 convention (played on channel 0,
        # suppressed only when audio_events exist — they don't here).
        if block.get("master_audio") and not meta.get("audio_events"):
            video = resolve_video(project, block.get("media"), project_path,
                                  video_override)
            if video is None or audio_ffmpeg is None:
                warn(f"shot {shot_id} ({block.get('name')!r}): master_audio "
                     f"needs the source video and ffmpeg — audio skipped")
            else:
                meta["audio"] = "audio.mp3"
                mp3 = shot_dir / "audio.mp3"
                if not mp3.exists():
                    dur = max(0.05, rng[1] - rng[0])
                    result = subprocess.run(
                        [audio_ffmpeg, "-hide_banner", "-loglevel", "error",
                         "-y", "-ss", f"{rng[0]:.3f}", "-t", f"{dur:.3f}",
                         "-i", str(video), "-vn", "-c:a", "libmp3lame",
                         "-q:a", "3", str(mp3)],
                        capture_output=True, text=True)
                    if result.returncode != 0:
                        warn(f"shot {shot_id}: master-mix audio extraction "
                             f"failed: {result.stderr.strip()[:160]}")
                        meta.pop("audio", None)

        with open(shot_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        needs_sfx = "detect.mp3" in json.dumps(meta)
        if detect_sound is not None and needs_sfx:
            audio_dir = shot_dir / "audio"
            audio_dir.mkdir(exist_ok=True)
            shutil.copyfile(detect_sound, audio_dir / "detect.mp3")

        if do_frames and block.get("type") != "merge":
            video = resolve_video(project, block.get("media"), project_path, video_override)
            if video is None:
                warn(f"shot {shot_id} ({block.get('name')!r}): source video not found "
                     f"— pass --video; frames skipped")
            else:
                print(f"  shot {shot_id}: extracting {n_frames} frames "
                      f"from {video.name} @ {rng[0]:.2f}s ...")
                extract_frames(ffmpeg, video, rng[0], n_frames, fps, shot_dir / "frames")

        seq_entry = {"shot": shot_id, "act": ACT_ID, "kind": meta["kind"],
                     "reuse_of": None, "audio_lines": []}
        seq_shots.append(seq_entry)

    sequence = {
        "_comment": f"Generated by scripts/export_experience.py from "
                    f"{project_path.name} ({project.get('name')}). Do not hand-edit — "
                    f"re-export from the Experience Builder instead.",
        "fps": fps,
        "shots": seq_shots,
    }
    with open(out_root / "sequence.json", "w", encoding="utf-8") as f:
        json.dump(sequence, f, indent=2)

    print(f"\n[export] {len(seq_shots)} shots -> {out_root}")
    if warnings:
        print(f"[export] {len(warnings)} warning(s) above")
    print("[export] to run it: point the runtime's scenes root at the generated tree "
          "(swap scenes/ manually — this script never touches the live tree).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("project", type=Path, help="path to the .bhrx.json project file")
    parser.add_argument("--out", type=Path, default=ROOT / "export" / "scenes_generated",
                        help="output scenes root (default export/scenes_generated)")
    parser.add_argument("--no-frames", action="store_true",
                        help="write metadata/sequence only; skip ffmpeg extraction")
    parser.add_argument("--video", type=Path, default=None,
                        help="override the source video path for every block")
    parser.add_argument("--detect-sound", type=Path, default=None,
                        help="override the global detect sound file")
    args = parser.parse_args()

    if not args.project.exists():
        sys.exit(f"[export] project file not found: {args.project}")
    return export(args.project.resolve(), args.out.resolve(),
                  not args.no_frames, args.video, args.detect_sound)


if __name__ == "__main__":
    sys.exit(main())
