"""
sequence_loader — loads and validates the shot sequence for the shot-driven runtime.

Public API:
    load_sequence(scenes_root, config) -> list[Shot]
    validate(shots) -> None   (prints a summary table; returns nothing)

Config inheritance (most specific value wins for each key):
    global defaults (config.json + sequence.json fps)
    ← act.json       (fps, timing_profile; loaded from scenes/act_<id>_*/act.json)
    ← metadata.json  (per-shot overrides; loaded from scenes/act_<id>_*/shot_<id>/metadata.json)
    ← sequence.json  (audio_lines, fallback; the per-shot entry in the sequence file)

Frame path resolution:
    Target layout:  scenes/act_<act>_<name>/shot_<shot>/frames/
    When reuse_of is set, frames are loaded from the source shot's directory (in the
    source shot's act). The audio directory is always the shot's own audio/, never the
    source's.
    If the target directory does not exist, assets_pending is set True.  This is
    expected for all shots until Mike executes the act/shot reorganisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json


_DEFAULT_FPS = 24
_DEFAULT_TIMING_PROFILE = "standard"
_DEFAULT_FALLBACK: dict = {
    "timeout_s": 30,
    "reprompt_s": [8, 16],
    "on_timeout": "auto_advance",
}


@dataclass
class Shot:
    """
    A single shot in the experience sequence.

    Config inheritance: global defaults ← act.json ← metadata.json ← sequence.json entry.

    Fields ``segments`` and ``interaction`` are None when the corresponding
    sequence.json value is the placeholder string "TODO".  The player must
    treat None as "not yet wired" and fall back to auto-advance behaviour.

    ``frames_dir`` and ``audio_dir`` are None until the act/shot directory
    tree is built.  ``assets_pending`` is True whenever frames_dir is absent
    or empty.
    """

    # Identity
    shot: str                   # zero-padded "01"–"78"
    act: str                    # "00"–"13"
    kind: str                   # "playback" | "interactive"

    # Content
    audio_lines: list           # AL-XX-YYY codes; empty list if not yet mapped
    tracker_type: str
    tracker_notes: str

    # Reuse (normalised to zero-padded string, or None)
    reuse_of: Optional[str]

    # Interactive config — None when not yet filled by Mike
    segments: Optional[dict]    # {intro: [a,b], idle_loop: [a,b], resolution: [a,b]}
    interaction: Optional[dict] # full interaction spec dict
    fallback: dict              # timeout_s, reprompt_s, on_timeout

    # Timing (resolved via inheritance chain)
    fps: int
    timing_profile: str         # "standard" | "urgent"

    # Resolved paths — None until act/shot tree is built
    frames_dir: Optional[Path]  # frames/ dir to load (follows reuse_of to source shot)
    audio_dir: Optional[Path]   # audio/ dir for AL-XX-YYY wav files
    audio_file: Optional[Path]  # single audio file (shot_NN/audio.mp3 convention)

    # Status flags
    assets_pending: bool        # True if frames_dir is None or contains no image files

    # Validation flags (computed by loader, inspected by validate())
    segments_todo: bool         # interactive and segments not yet filled
    interaction_todo: bool      # interactive and interaction not yet filled
    reuse_self: bool            # reuse_of == this shot's number (tracker artefact)


# ---------------------------------------------------------------------------
# Public: load
# ---------------------------------------------------------------------------

def load_sequence(scenes_root: Path, config: dict) -> list[Shot]:
    """
    Parse scenes/sequence.json and return an ordered list of Shot objects.

    Acts are located by globbing scenes_root for directories that start with
    act_<id>_ (e.g. act_02_crossroads).  Missing act or shot directories are
    handled gracefully — affected shots get assets_pending=True.

    Args:
        scenes_root: Path to the scenes/ directory (contains sequence.json).
        config:      Global config dict (config.json), used for fps fallback.
    """
    seq_path = scenes_root / "sequence.json"
    with open(seq_path, encoding="utf-8") as f:
        seq_data = json.load(f)

    global_fps: int = seq_data.get("fps", config.get("fps", _DEFAULT_FPS))
    raw_shots: list[dict] = seq_data.get("shots", [])

    # Pre-build act-dir cache: act_id -> Path (or None)
    act_dir_cache: dict[str, Optional[Path]] = {}

    # Pre-build shot -> act lookup for reuse_of cross-act resolution
    shot_act: dict[str, str] = {s["shot"]: s["act"] for s in raw_shots}
    shot_set: set[str] = set(shot_act.keys())

    shots: list[Shot] = []
    for raw in raw_shots:
        shot_id = raw["shot"]
        act_id  = raw["act"]

        # Resolve act directory (cached)
        if act_id not in act_dir_cache:
            act_dir_cache[act_id] = _find_act_dir(scenes_root, act_id)
        act_dir = act_dir_cache[act_id]

        # Load act-level and shot-level overrides
        act_cfg   = _load_act_config(act_dir)
        shot_meta = _load_shot_metadata(act_dir, shot_id)

        # Effective fps: global ← act ← shot_meta
        fps = shot_meta.get("fps", act_cfg.get("fps", global_fps))

        # Effective timing profile: global ← act ← shot_meta
        timing_profile = shot_meta.get(
            "timing_profile",
            act_cfg.get("timing_profile", _DEFAULT_TIMING_PROFILE),
        )

        # Effective fallback: global ← act ← shot_meta ← sequence entry
        fallback = dict(_DEFAULT_FALLBACK)
        fallback.update(act_cfg.get("fallback", {}))
        fallback.update(shot_meta.get("fallback", {}))
        if raw.get("fallback"):
            fallback.update(raw["fallback"])

        # Segments: metadata.json ← sequence.json (most specific wins)
        raw_segments = raw.get("segments") or shot_meta.get("segments")
        if raw_segments in (None, "TODO"):
            segments = None
        elif isinstance(raw_segments, dict) and any(
            v == "TODO" for v in raw_segments.values()
        ):
            segments = None
        else:
            segments = raw_segments

        # Interaction: metadata.json ← sequence.json (most specific wins)
        raw_interaction = raw.get("interaction") or shot_meta.get("interaction")
        interaction = None if raw_interaction in (None, "TODO") else raw_interaction

        # Kind: metadata.json ← sequence.json
        kind = raw.get("kind") or shot_meta.get("kind", "playback")
        is_interactive = kind == "interactive"
        segments_todo   = is_interactive and (segments is None)
        interaction_todo = is_interactive and (interaction is None)

        # reuse_of: normalise to zero-padded string
        reuse_of_raw = raw.get("reuse_of")
        if reuse_of_raw is None:
            reuse_of = None
        else:
            reuse_of = str(reuse_of_raw).zfill(2)
        reuse_self = reuse_of == shot_id

        # Resolve frames directory
        # If reuse_of is set (and not self-referential), look in the source shot's dir.
        if reuse_of and not reuse_self and reuse_of in shot_set:
            src_act  = shot_act[reuse_of]
            if src_act not in act_dir_cache:
                act_dir_cache[src_act] = _find_act_dir(scenes_root, src_act)
            src_act_dir = act_dir_cache[src_act]
            frames_dir = _resolve_frames_dir(src_act_dir, reuse_of)
        else:
            frames_dir = _resolve_frames_dir(act_dir, shot_id)

        audio_dir = _resolve_audio_dir(act_dir, shot_id)
        audio_file = _resolve_audio_file(act_dir, shot_id,
                                         raw.get("audio") or shot_meta.get("audio"))

        # assets_pending: frames dir absent or empty
        if frames_dir is None:
            assets_pending = True
        else:
            image_exts = {".png", ".jpg", ".jpeg"}
            has_frames = any(
                f.suffix.lower() in image_exts for f in frames_dir.iterdir()
            )
            assets_pending = not has_frames

        shots.append(Shot(
            shot=shot_id,
            act=act_id,
            kind=kind,
            audio_lines=raw.get("audio_lines") or [],
            tracker_type=raw.get("tracker_type", ""),
            tracker_notes=raw.get("tracker_notes", ""),
            reuse_of=reuse_of,
            segments=segments,
            interaction=interaction,
            fallback=fallback,
            fps=fps,
            timing_profile=timing_profile,
            frames_dir=frames_dir,
            audio_dir=audio_dir,
            audio_file=audio_file,
            assets_pending=assets_pending,
            segments_todo=segments_todo,
            interaction_todo=interaction_todo,
            reuse_self=reuse_self,
        ))

    return shots


# ---------------------------------------------------------------------------
# Public: validate
# ---------------------------------------------------------------------------

def validate(shots: list[Shot]) -> None:
    """
    Print a summary validation table for the loaded shot sequence.

    Reports:
      1. Counts (total, playback, interactive, assets_pending)
      2. Interactive shots still carrying TODO segments or interaction
      3. Self-referential reuse_of entries (tracker artefacts)
      4. Act-label coverage vs expected range 00–13
    """
    total       = len(shots)
    n_playback  = sum(1 for s in shots if s.kind == "playback")
    n_inter     = sum(1 for s in shots if s.kind == "interactive")
    n_pending   = sum(1 for s in shots if s.assets_pending)

    seg_todo    = [s for s in shots if s.segments_todo]
    int_todo    = [s for s in shots if s.interaction_todo]
    self_refs   = [s for s in shots if s.reuse_self]

    expected_acts = {f"{i:02d}" for i in range(14)}   # "00"–"13"
    actual_acts   = {s.act for s in shots}
    act_gaps      = sorted(expected_acts - actual_acts)

    W = 64
    print("=" * W)
    print("BHR Sequence Validation")
    print(f"  {total} shots total: {n_playback} playback, {n_inter} interactive")
    print("=" * W)

    # 1. Assets pending
    _section("1. Assets pending")
    if n_pending == total:
        print(f"  {n_pending}/{total} shots -- act/shot directory tree not yet built (expected)")
    elif n_pending == 0:
        print(f"  0/{total} -- all shots have frames (ok)")
    else:
        print(f"  {n_pending}/{total} shots missing frames:")
        for s in shots:
            if s.assets_pending:
                print(f"    shot {s.shot}  act {s.act}  {s.tracker_type}")

    # 2. Interactive TODOs
    _section("2. Interactive shots with TODO content")
    if not seg_todo and not int_todo:
        print(f"  All {n_inter} interactive shots have segments and interaction filled (ok)")
    else:
        print(f"  TODO segments:    {len(seg_todo)}/{n_inter} interactive shots")
        print(f"  TODO interaction: {len(int_todo)}/{n_inter} interactive shots")
        if seg_todo:
            print("  Shots needing segments + interaction:")
            for s in seg_todo:
                notes = s.tracker_notes or s.tracker_type or "--"
                print(f"    shot {s.shot}  act {s.act}  reuse_of={s.reuse_of or '--'}  [{notes}]")

    # 3. Self-referential reuse_of
    _section("3. Self-referential reuse_of (tracker artefacts)")
    if not self_refs:
        print("  None (ok)")
    else:
        print(f"  {len(self_refs)} shots -- needs Mike's review:")
        for s in self_refs:
            print(f"    shot {s.shot}  act {s.act}  reuse_of='{s.reuse_of}'  [{s.tracker_type}]")

    # 4. Act coverage
    _section("4. Act coverage (expected 00-13)")
    if not act_gaps:
        print(f"  All 14 acts (00-13) represented (ok)")
    else:
        print(f"  GAPS — acts with no shots: {act_gaps}")

    print("=" * W)


def _section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_act_dir(scenes_root: Path, act_id: str) -> Optional[Path]:
    """Return the first directory matching scenes_root/act_<id>_*, or None."""
    prefix = f"act_{act_id}_"
    try:
        for entry in scenes_root.iterdir():
            if entry.is_dir() and entry.name.startswith(prefix):
                return entry
    except OSError:
        pass
    return None


def _load_act_config(act_dir: Optional[Path]) -> dict:
    if act_dir is None:
        return {}
    act_json = act_dir / "act.json"
    if not act_json.exists():
        return {}
    try:
        with open(act_json, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_shot_metadata(act_dir: Optional[Path], shot_id: str) -> dict:
    if act_dir is None:
        return {}
    meta = act_dir / f"shot_{shot_id}" / "metadata.json"
    if not meta.exists():
        return {}
    try:
        with open(meta, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _resolve_frames_dir(act_dir: Optional[Path], shot_id: str) -> Optional[Path]:
    if act_dir is None:
        return None
    d = act_dir / f"shot_{shot_id}" / "frames"
    return d if d.is_dir() else None


def _resolve_audio_dir(act_dir: Optional[Path], shot_id: str) -> Optional[Path]:
    if act_dir is None:
        return None
    d = act_dir / f"shot_{shot_id}" / "audio"
    return d if d.is_dir() else None


def _resolve_audio_file(act_dir: Optional[Path], shot_id: str,
                        filename: Optional[str]) -> Optional[Path]:
    if act_dir is None or not filename:
        return None
    p = act_dir / f"shot_{shot_id}" / filename
    return p if p.exists() else None
