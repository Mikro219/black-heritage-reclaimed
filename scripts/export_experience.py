"""
export_experience.py — Convert an Experience Builder project (.bhrx.json) into
a BHR runtime scenes tree.

    py -3.12 scripts/export_experience.py "My_Experience.bhrx.json"
    py -3.12 scripts/export_experience.py project.bhrx.json --no-frames
    py -3.12 scripts/export_experience.py project.bhrx.json --video "assets/video/BHR Draft 1.mp4"
    py -3.12 scripts/export_experience.py project.bhrx.json --out export/my_scenes

Output (default export/generated/):

    export/generated/
      sequence.json                    ordered shot list (loader-compatible)
      scenes/
        act.json                       {"fps": <project fps>}
        scene_NN/metadata.json         playback / OI / choice-fork wiring
        scene_NN/frames/00001.jpg ...  (unless --no-frames)
        scene_NN/audio/detect.mp3      (when the global detect sound is found)

Mapping (the hand-authored shot patterns these were derived from):
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
ACT_DIRNAME = "scenes"          # flat single-act layout (Aug 2026)

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
                emitted_branches = [br for br in branches
                                    if not br.get("retry")]
                tt_to = (node.get("timeout") or {}).get("to")
                if tt_to and emitted_branches and tt_to != emitted_branches[0]["to"]:
                    warn(f"choice {node.get('name')!r}: the runtime's timeout "
                         f"auto-advance always falls into the FIRST emitted "
                         f"branch — the timeout target set in the editor can't "
                         f"override that")
                convergences = []
                for i, br in enumerate(branches):
                    if br.get("retry"):
                        # Wrong-way retry: the chain is folded into the choice
                        # shot (frames via hold_segments, audio via the export
                        # fold) — its blocks emit no shots of their own.
                        for fb in branch_chain(project, blocks, deg, br["to"]):
                            emitted.add(fb["id"])
                        continue
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
    # A gesture window may declare a spoken alternative (params.voice_alternative
    # = keyword string): the runtime opens a parallel VI window whose keyword
    # fires the same reaction as the gesture (Scene 1 raise-hands / "freedom").
    voice_alt = params.pop("voice_alternative", None)
    rect = region_rect(win)
    if rect:
        if win["detector"] == "directional_draw":
            # Pure display: positions/sizes the on-screen stroke indicator.
            # Stays in the Builder's SCREEN space (the camera isn't involved).
            params["indicator_rect"] = rect
        else:
            # Detection region. The Builder authors regions in SCREEN space
            # (drawn on the video frame); detectors test the RAW (unmirrored)
            # camera coords — flip x so the visitor points where the box IS,
            # not at its mirror image. Display paths mirror back, so the
            # indicator ring still lands on the authored box.
            params["region_rect"] = {**rect,
                                     "x": round(1.0 - rect["x"] - rect["w"], 3)}
    out = {
        "id": slug(win.get("label"), f"oi_{idx}"),
        "type": win["detector"],
        "params": params,
    }
    # Draw strokes complete silently and without the green flash — the comet
    # indicator / star trail / the stroke animation playing are the feedback;
    # a detect ping + flash per stroke was noise.
    if win["detector"] != "directional_draw":
        out["feedback"] = "green_flash"
        out["sfx"] = "detect.mp3"
    if voice_alt:
        kw = str(voice_alt).strip().lower()
        warn_if_unknown_keyword(kw, f"window {win.get('label')!r} voice_alternative")
        out["voice_alternative"] = {
            "id": slug(kw, f"voice_alt_{idx}"),
            "keywords": [kw],
            "mode": "keyword",
            "tier": "cg_alternative",
        }
    return out


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


def is_draw(win: dict) -> bool:
    return win.get("detector") == "directional_draw"


# A draw window BLOCKS: its span loops until the stroke is detected. This is
# how long the runtime waits before giving up and moving on (per stroke) —
# override per window with params.block_timeout_s in the Builder.
DRAW_BLOCK_TIMEOUT_S = 45.0


_grammar_cache: set | None = None


def warn_if_unknown_keyword(kw: str, where: str) -> None:
    """An authored keyword that is not in config.json's Vosk grammar can NEVER
    fire (the recognizer only emits grammar entries) — surface it at export
    time instead of failing silently at the exhibition."""
    global _grammar_cache
    if _grammar_cache is None:
        try:
            with open(ROOT / "config.json", encoding="utf-8") as f:
                grammar = json.load(f).get("voice", {}).get("grammar") or []
            _grammar_cache = {str(g).strip().lower() for g in grammar}
        except OSError:
            _grammar_cache = set()
    if _grammar_cache and kw not in _grammar_cache:
        warn(f"{where}: keyword {kw!r} is not in config.json voice.grammar — "
             f"Vosk will never emit it; add it to the grammar")


def voice_oi_dict(win: dict, idx: int) -> dict:
    """FSM `oi` payload for a voice window: a `keywords` list makes the runtime
    open a VI window (reaction tier) instead of arming a gesture detector).
    params.keyword may be a single string, a list, or a comma-separated
    string — any of these authors multiple acceptable keywords (e.g. "gourd"
    OR "follow" for the North Star hum-or-gourd VI)."""
    raw = (win.get("params") or {}).get("keyword") or "go"
    if isinstance(raw, str):
        kws = [k.strip() for k in raw.split(",") if k.strip()]
    elif isinstance(raw, list):
        kws = [str(k).strip() for k in raw if str(k).strip()]
    else:
        kws = [str(raw).strip()]
    if not kws:
        kws = ["go"]
    for kw in kws:
        warn_if_unknown_keyword(kw.lower(), f"voice window {win.get('label')!r}")
    return {
        "id": slug(win.get("label"), f"vi_{idx}"),
        "keywords": kws,
        "mode": (win.get("params") or {}).get("mode", "keyword"),
        "sfx": "detect.mp3",
        "feedback": "green_flash",
    }


def block_captions(block: dict) -> list:
    """A block's authored captions -> shot metadata (screen-space, display-only).
    Each: {at_s, duration_s, text, rect?}. The runtime draws them against the
    shot playhead; rect (x,y,w,h fractions) places the subtitle, else a default
    bottom band."""
    out = []
    for c in block.get("captions") or []:
        text = str(c.get("text", "")).strip()
        if not text:
            continue
        entry = {"at_s": round(float(c.get("at_s", 0.0) or 0.0), 3),
                 "duration_s": round(float(c.get("duration_s", 0.0) or 0.0), 3),
                 "text": text}
        rect = c.get("rect")
        if isinstance(rect, dict) and all(k in rect for k in ("x", "y", "w", "h")):
            entry["rect"] = {k: round(float(rect[k]), 4)
                             for k in ("x", "y", "w", "h")}
        out.append(entry)
    out.sort(key=lambda e: e["at_s"])
    return out


def playback_metadata(block: dict, shot_id: str, n_frames: int, fps: int) -> dict:
    meta = {
        "shot": shot_id,
        "kind": "playback",
        "_generated_from": {"block": block["id"], "name": block.get("name")},
    }
    wins = gesture_windows(block)
    if not wins:
        return meta

    # Draw windows block (loop until the stroke lands), which only the FSM
    # model can express — a lone draw window skips the oi_frame_window path.
    if len(wins) == 1 and not is_voice(wins[0]) and not is_draw(wins[0]):
        a, b = window_frames(wins[0], n_frames, fps)
        inter = oi_dict(wins[0], 1)
        inter["tier"] = "OI"
        inter["oi_frame_window"] = [a, b]
        meta["interaction"] = inter
        return meta

    # 2+ windows (or any voice/draw window): play-through FSM with oi states
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
        payload = voice_oi_dict(w, i + 1) if is_voice(w) else oi_dict(w, i + 1)
        if is_draw(w):
            # FREEZE-AND-HOLD stroke (the live shot-20 idle_N/play_to_N
            # model): freeze on the window's FIRST frame with the detector
            # armed; the window's own animation plays only after the stroke
            # lands (oi_<id>) or the give-up window expires (oi_done) — the
            # video stays continuous either way. window_ms doubles as the
            # armed-detector duration AND the oi_done deadline.
            wait_s = float((w.get("params") or {}).get(
                "block_timeout_s", DRAW_BLOCK_TIMEOUT_S))
            payload.get("params", {}).pop("block_timeout_s", None)
            hold_seg = f"{name}_hold"
            segments[hold_seg] = [a, a]
            segments[name] = [a, b]
            states[name] = {"segment": hold_seg, "loop": True, "oi": payload,
                            "window_ms": int(wait_s * 1000)}
            states[f"{name}_play"] = {"segment": name, "loop": False}
            order += [name, f"{name}_play"]
        else:
            segments[name] = [a, b]
            states[name] = {"segment": name, "loop": False, "oi": payload}
            order.append(name)
        cursor = b + 1
    if cursor <= n_frames:
        segments["outro"] = [cursor, n_frames]
        states["outro"] = {"segment": "outro", "loop": False}
        order.append("outro")

    # Blocking (loop) states never fire segment_end — they leave on the
    # stroke's oi_<id> detection or the oi_done window expiry.
    def _exit_events(state_id: str) -> list[str]:
        st = states[state_id]
        if st.get("loop") and st.get("oi"):
            return [f"oi_{st['oi']['id']}", "oi_done"]
        return ["segment_end"]

    transitions = [{"from": frm, "on": ev, "to": to}
                   for frm, to in zip(order, order[1:] + ["__advance__"])
                   for ev in _exit_events(frm)]

    # The runtime's HOLD timeout reads the TOP-LEVEL fallback (shot.fallback),
    # not the one nested in interaction_fsm. Without it the standard profile's
    # 30s auto_advance cuts any play-through FSM longer than 30s mid-shot, so
    # size the timeout to the full FSM content (everything after the intro)
    # plus every blocking window's worst-case wait, plus slack — it should
    # only ever fire if playback stalls.
    hold_s = (n_frames - spans[0][0] + 1) / max(1, fps)
    block_s = sum(st.get("window_ms", 0) for st in states.values()
                  if st.get("loop")) / 1000.0
    timeout_s = max(120, int(hold_s + block_s) + 60)
    meta["segments"] = segments
    meta["interaction"] = {
        "tier": "OI",
        "interaction_fsm": {
            "initial": order[0],
            "states": states,
            "transitions": transitions,
            "fallback": {"timeout_s": timeout_s, "on_timeout": "auto_advance"},
        },
    }
    meta["fallback"] = {"timeout_s": timeout_s, "reprompt_s": [],
                        "on_timeout": "auto_advance"}
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


# Pick/switch sound for voice-confirmed choice holds (same file for both —
# the UIAlert already used by the crossroads cluster's confirm clip).
CHOICE_HOLD_SFX = "UIAlert-Hybrid_Apple_Pay_+_T-Elevenlabs.mp3"

# Selected/switch animation segments a choice block may carry in its optional
# "hold_segments" field (block-local frames, e.g. authored from live shot 09).
_HOLD_ANIM_KEYS = ("left_selected", "right_selected", "left_to_right",
                   "right_to_left", "left_switch_hold", "right_switch_hold")


def choice_audio_spec(block: dict, sounds_by_id: dict) -> dict | None:
    """The block's authored pick/switch sound, as an `on_enter_audio` payload.

    Authored in the Builder's inspector (Choice Block -> Pick / switch audio):
    {sound, delay_s, source_offset_s, duration_s, gain}. When present it
    plays IN ADDITION to the state's built-in CHOICE_HOLD_SFX click — the
    click answers the gesture instantly, the authored clip (typically a
    delayed dialogue line) follows it.

    delay_s / source_offset_s / duration_s are honoured at RUNTIME (the render
    engine slices the decoded PCM), so they can be retuned in the Builder and
    re-exported as metadata only — no audio re-render."""
    spec = block.get("choice_audio")
    if not isinstance(spec, dict):
        return None
    snd = sounds_by_id.get(spec.get("sound"))
    if not snd:
        if spec.get("sound"):
            warn(f"choice {block.get('name')!r}: pick/switch sound "
                 f"{spec.get('sound')!r} is not in the project — ignored")
        return None
    out = {"file": snd["name"]}
    for key, default in (("delay_s", 0.0), ("source_offset_s", 0.0),
                         ("duration_s", 0.0), ("gain", 1.0)):
        try:
            val = float(spec.get(key, default) or default)
        except (TypeError, ValueError):
            val = default
        if val != default:
            out[key] = round(val, 3)
    return out


def hold_choice_fsm(block: dict, n_frames: int, confirm_kw: str,
                    segments: dict) -> dict:
    """The shot-09 hold model: point a side -> `<side>_selected` (pick SFX,
    waits for the confirm keyword); point the other side -> switch animation
    (switch SFX, input locked) -> `<side>_switch_hold` (can flip again); only
    `voice_<kw>` advances — the confirm animation is the branch block itself.
    Fork-choice recording still happens on every gesture pick, so
    `__advance__` branch-gates correctly.

    With the block's "hold_segments" (clamped to the block), the selected and
    switch states play their real animation frames; without it they hold on
    the idle frame (sound-only)."""
    hs = {k: [max(1, min(n_frames, int(v[0]))), max(1, min(n_frames, int(v[1])))]
          for k, v in (block.get("hold_segments") or {}).items()
          if isinstance(v, (list, tuple)) and len(v) == 2}
    animated = all(k in hs for k in _HOLD_ANIM_KEYS)
    if hs and not animated:
        warn(f"choice {block.get('name')!r}: hold_segments is missing "
             f"{sorted(set(_HOLD_ANIM_KEYS) - set(hs))} — falling back to "
             f"sound-only holds on the idle frame")

    if "intro" in hs or "idle_loop" in hs:
        segments["intro"]     = hs.get("intro", segments["intro"])
        segments["idle_loop"] = hs.get("idle_loop", segments["idle_loop"])
    if animated:
        for k in _HOLD_ANIM_KEYS:
            segments[k] = hs[k]

    def seg(name):
        return name if animated else "idle_loop"

    states = {
        "waiting": {"segment": "idle_loop", "loop": True,
                    "directions": ["left", "right"]},
        "left_selected":  {"segment": seg("left_selected"), "loop": not animated,
                           "on_enter_sfx": CHOICE_HOLD_SFX, "voice": confirm_kw},
        "right_selected": {"segment": seg("right_selected"), "loop": not animated,
                           "on_enter_sfx": CHOICE_HOLD_SFX, "voice": confirm_kw},
        "left_switch_hold":  {"segment": seg("left_switch_hold"), "loop": True,
                              "voice": confirm_kw},
        "right_switch_hold": {"segment": seg("right_switch_hold"), "loop": True,
                              "voice": confirm_kw},
    }
    transitions = [
        {"from": "waiting", "on": "point_left",  "to": "left_selected"},
        {"from": "waiting", "on": "point_right", "to": "right_selected"},
    ]
    if animated:
        # Switch plays its transition animation (input locked), then holds.
        states["left_to_right"] = {"segment": "left_to_right", "loop": False,
                                   "on_enter_sfx": CHOICE_HOLD_SFX}
        states["right_to_left"] = {"segment": "right_to_left", "loop": False,
                                   "on_enter_sfx": CHOICE_HOLD_SFX}
        transitions += [
            {"from": "left_selected",     "on": "point_right", "to": "left_to_right"},
            {"from": "right_selected",    "on": "point_left",  "to": "right_to_left"},
            {"from": "left_to_right",     "on": "segment_end", "to": "right_switch_hold"},
            {"from": "right_to_left",     "on": "segment_end", "to": "left_switch_hold"},
            {"from": "right_switch_hold", "on": "point_left",  "to": "right_to_left"},
            {"from": "left_switch_hold",  "on": "point_right", "to": "left_to_right"},
        ]
    else:
        # Sound-only: flips land directly on the other side's switch hold.
        states["left_switch_hold"]["on_enter_sfx"] = CHOICE_HOLD_SFX
        states["right_switch_hold"]["on_enter_sfx"] = CHOICE_HOLD_SFX
        transitions += [
            {"from": "left_selected",     "on": "point_right", "to": "right_switch_hold"},
            {"from": "right_selected",    "on": "point_left",  "to": "left_switch_hold"},
            {"from": "right_switch_hold", "on": "point_left",  "to": "left_switch_hold"},
            {"from": "left_switch_hold",  "on": "point_right", "to": "right_switch_hold"},
        ]
    for st in ("left_selected", "right_selected",
               "left_switch_hold", "right_switch_hold"):
        transitions.append({"from": st, "on": f"voice_{confirm_kw}",
                            "to": "__advance__"})
    return {"states": states, "transitions": transitions}


# Wrong-pick sound for retry choices exported WITHOUT an authored wrong_path
# animation (the sound-only bounce).
RETRY_WRONG_SFX = "UIAlert-wrong_button_sound,_-Elevenlabs.mp3"


def retry_choice_fsm(block: dict, n_frames: int, retry_side: str,
                     segments: dict) -> dict:
    """The live shot-37 wrong-way model: picking the retry side plays a
    redirect and RETURNS to waiting — only the other (correct) side advances.
    The correct side's confirm animation stays a separate play_if-gated branch
    shot; the retry side's chain is folded into this shot (frames via the
    block's extended range + "hold_segments", audio via the export fold).

    With hold_segments["wrong_path"] the redirect plays its real animation
    frames (its folded audio clips carry the SFX/VO); without it the wrong
    pick is a sound-only bounce on the idle frame (RETRY_WRONG_SFX)."""
    hs = {k: [max(1, min(n_frames, int(v[0]))), max(1, min(n_frames, int(v[1])))]
          for k, v in (block.get("hold_segments") or {}).items()
          if isinstance(v, (list, tuple)) and len(v) == 2}
    if "intro" in hs or "idle_loop" in hs:
        segments["intro"]     = hs.get("intro", segments["intro"])
        segments["idle_loop"] = hs.get("idle_loop", segments["idle_loop"])
    animated = "wrong_path" in hs
    if animated:
        segments["wrong_path"] = hs["wrong_path"]
    correct = "right" if retry_side == "left" else "left"

    states = {
        "waiting": {"segment": "idle_loop", "loop": True,
                    "directions": ["left", "right"]},
        "wrong_path": {"segment": "wrong_path" if animated else "idle_loop",
                       "loop": False},
        f"confirm_{correct}": {"segment": "idle_loop", "loop": False},
    }
    if not animated:
        states["wrong_path"]["on_enter_sfx"] = RETRY_WRONG_SFX
    transitions = [
        {"from": "waiting", "on": f"point_{retry_side}", "to": "wrong_path"},
        {"from": "waiting", "on": f"point_{correct}", "to": f"confirm_{correct}"},
        {"from": "wrong_path", "on": "segment_end", "to": "waiting"},
        {"from": f"confirm_{correct}", "on": "segment_end", "to": "__advance__"},
    ]
    return {"states": states, "transitions": transitions}


def retry_branch_side(fork_branches: list[dict]) -> str | None:
    """"left"/"right" of the branch marked retry (the wrong-way path), or None."""
    for i, br in enumerate(fork_branches[:2]):
        if br.get("retry"):
            return "left" if i == 0 else "right"
    return None


def branch_chain(project: dict, blocks: dict, deg: dict,
                 start_id: str | None) -> list[dict]:
    """The blocks of a branch chain, following edges until a merge block or a
    fan-in (indegree > 1) convergence — the same boundary rule linearize uses."""
    out: list[dict] = []
    node_id = start_id
    while node_id:
        node = blocks.get(node_id)
        if node is None or node.get("type") == "merge" or deg.get(node_id, 0) > 1:
            break
        out.append(node)
        node_id = flow_next(project, node_id)
    return out


def choice_metadata(block: dict, shot_id: str, n_frames: int, fps: int,
                    sounds_by_id: dict | None = None) -> dict:
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

    # A voice window referenced by NO branch is the CONFIRM keyword: picks hold
    # until the visitor says it (the shot 09 "say go" pattern).
    all_branch_windows = {br.get("window") for br in (block.get("branches") or [])}
    confirm_kw = next(
        (((w.get("params") or {}).get("keyword") or "go")
         for w in wins if is_voice(w) and w["id"] not in all_branch_windows),
        None)

    intro_end = max(1, n_frames - 1)
    segments = {"intro": [1, intro_end], "idle_loop": [n_frames, n_frames]}

    fallback = {
        "timeout_s": int((block.get("timeout") or {}).get("seconds") or 30),
        "on_timeout": "auto_advance",
    }
    retry_side = retry_branch_side(fork_branches)
    if confirm_kw and retry_side:
        warn(f"choice {block.get('name')!r}: a retry branch and a confirm "
             f"keyword can't combine — retry ignored, hold model exported")
        retry_side = None

    if confirm_kw:
        print(f"  choice {block.get('name')!r}: '{confirm_kw}' confirms the "
              f"picked side (voice-confirmed hold model)")
        body = hold_choice_fsm(block, n_frames, confirm_kw, segments)
    elif retry_side:
        correct = "right" if retry_side == "left" else "left"
        print(f"  choice {block.get('name')!r}: '{retry_side}' is the "
              f"wrong-way retry — only '{correct}' advances")
        body = retry_choice_fsm(block, n_frames, retry_side, segments)
    else:
        # Flat immediate-confirm: the pick advances silently into its gated
        # branch shot (which carries its own audio) — no detect ping.
        body = {
            "states": {
                "waiting":       {"segment": "idle_loop", "loop": True,
                                  "directions": ["left", "right"]},
                "confirm_left":  {"segment": "idle_loop", "loop": False},
                "confirm_right": {"segment": "idle_loop", "loop": False},
            },
            "transitions": [
                {"from": "waiting",       "on": "point_left",  "to": "confirm_left"},
                {"from": "waiting",       "on": "point_right", "to": "confirm_right"},
                {"from": "confirm_left",  "on": "segment_end", "to": "__advance__"},
                {"from": "confirm_right", "on": "segment_end", "to": "__advance__"},
            ],
        }
    # Authored pick/switch sound — stamped ONLY on states the visitor's action
    # enters (an incoming point_*/voice_* transition). Excluded:
    #   waiting     — the idle loop, never a detection
    #   wrong_path  — a wrong-way redirect says "not that way", not "picked".
    #                 It keeps its buzzer (sound-only model) or its own folded
    #                 redirect audio (animated model).
    #   segment_end-only states — the animated hold model's `*_switch_hold`
    #                 rest states are the TAIL of the switch animation whose
    #                 entering state (left_to_right/right_to_left) already
    #                 played the clip; stamping them too made every switch
    #                 play the sound twice (the live shot-09 switch-hold
    #                 states were made silent for exactly this reason).
    pick_audio = choice_audio_spec(block, sounds_by_id or {})
    if pick_audio:
        entered_by_action = {t["to"] for t in body["transitions"]
                             if t.get("on") != "segment_end"}
        stamped = []
        for sid, st in body["states"].items():
            if sid in ("waiting", "wrong_path") or sid not in entered_by_action:
                continue
            # LAYERED with the built-in click, not replacing it: the state's
            # CHOICE_HOLD_SFX answers the gesture instantly and the authored
            # clip (typically a delayed dialogue line) follows it — the field
            # ask is "change-path sound, then the dialogue right after".
            # The runtime fires on_enter_sfx and on_enter_audio independently.
            st["on_enter_audio"] = dict(pick_audio)
            stamped.append(sid)
        print(f"  choice {block.get('name')!r}: pick/switch audio "
              f"{pick_audio['file']!r} on {len(stamped)} state(s)"
              + (f" (delay {pick_audio['delay_s']}s)" if pick_audio.get("delay_s") else ""))

    waiting = body["states"]["waiting"]
    fsm = {
        "gesture_type": "region",
        "initial": "waiting",
        "states": body["states"],
        "transitions": body["transitions"],
        "fallback": fallback,
    }

    # Spoken branch picks: a voice window whose branch targets the same block
    # as a fork side becomes "say the keyword to pick that side". The runtime
    # holds ONE voice keyword per state, so only the first is wired — and only
    # in the flat model (a voice-confirmed hold already owns the keyword slot).
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
        if confirm_kw:
            warn(f"choice {block.get('name')!r}: voice branch {br.get('label')!r} "
                 f"skipped — the hold model's confirm keyword "
                 f"'{confirm_kw}' owns the voice slot")
            continue
        if retry_side:
            warn(f"choice {block.get('name')!r}: voice branch {br.get('label')!r} "
                 f"skipped — the retry model wires gesture picks only")
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
                   fps: int, out_dir: Path, force: bool = False) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Re-exports reuse frames: skip when the dir already holds exactly the
    # expected count; any mismatch (block range changed) clears stale files
    # first so the count stays authoritative for duration estimates.
    # --force-frames re-extracts regardless — needed when the SOURCE VIDEO was
    # replaced in place (same timeline, new content), which the count can't see.
    existing = list(out_dir.glob("*.jpg"))
    if not force and len(existing) == n_frames:
        return True
    for f in existing:
        f.unlink()
    # Invalidate the runtime's frame pack: its validity check is frame count +
    # dimensions only, so a content swap would keep serving the OLD art.
    pack = out_dir.parent / "framecache.npy"
    if pack.exists():
        try:
            pack.unlink()
        except OSError as exc:
            warn(f"could not remove stale {pack}: {exc}")
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
    # Fall back to any copy shipped by a previous export (one per interactive
    # scene, so the first hit is as good as any).
    prev = ROOT / "export" / "generated"
    hits = sorted(prev.glob(f"*/*/{name}")) + \
           sorted(prev.glob(f"*/*/audio/{name}"))
    return hits[0] if hits else None


def collect_sfx_names(meta: dict) -> set[str]:
    """Every SFX file name the shot metadata references (FSM on_enter_sfx and
    OI/voice feedback sfx) — each must be shipped into the shot's audio/ dir."""
    names: set[str] = set()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("sfx", "on_enter_sfx") and isinstance(v, str) and v:
                    names.add(v)
                elif k == "on_enter_audio" and isinstance(v, dict) and v.get("file"):
                    names.add(v["file"])
                else:
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(meta)
    return names


def resolve_sfx_file(name: str, project_path: Path,
                     detect_sound: Path | None) -> Path | None:
    """Locate an on_enter_sfx/feedback sound: the global detect sound for
    detect.mp3, else next to the project -> assets/audio/stems/ -> anywhere in
    a previous export."""
    if name == "detect.mp3":
        return detect_sound
    for candidate in (project_path.parent / name,
                      ROOT / "assets" / "audio" / "stems" / name,
                      ROOT / "assets" / "audio" / "voice_lines" / name):
        if candidate.exists():
            return candidate
    hits = sorted((ROOT / "export" / "generated").rglob(name))
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


def _is_bed(clip: dict) -> bool:
    role = clip.get("role", "sfx")
    return (role in ("music", "ambience")
            and bool(clip.get("sustain", True)))


def plan_bed_chains(project: dict) -> dict:
    """Chain-link sustain beds across blocks so every piece shares ONE render.

    The runtime mixer's bed handover matches on the FILE NAME: a shot entered
    with a `continues` bed event adopts the still-playing bed only if the
    names are equal. Per-clip trims (each block's own offset + duration) would
    give every piece of a chain a different render name — the bed would
    restart at every block boundary. Instead, pieces linked by `continues`
    through a direct block edge/branch inherit the chain head's plan: one
    render cut at the head's source offset and running to the end of the
    file (the loop then wraps the whole remaining file, not a block-length
    slice; the mixer never reads duration_s for beds).

    Returns {(block_id, clip_id): plan}; chain members share the plan dict
    ({"offset": head_source_offset_s}).
    """
    preds: dict[str, set] = {}

    def link(a, b):
        if a and b:
            preds.setdefault(b, set()).add(a)

    for e in project.get("edges", []):
        link(e.get("from"), e.get("to"))
    for b in project.get("blocks", []):
        for br in (b.get("branches") or []):
            link(b["id"], br.get("to"))
        link(b["id"], (b.get("timeout") or {}).get("to"))

    # process blocks after their predecessors (Kahn; cycles fall out as-is)
    blocks = {b["id"]: b for b in project.get("blocks", [])}
    remaining = dict(blocks)
    order: list = []
    while remaining:
        ready = [bid for bid in remaining
                 if not (preds.get(bid, set()) & remaining.keys())]
        if not ready:
            ready = list(remaining)      # cycle — take the rest unordered
        for bid in ready:
            order.append(bid)
            del remaining[bid]

    plans: dict = {}
    by_block_sound: dict = {}
    for bid in order:
        for clip in blocks[bid].get("audio") or []:
            if not _is_bed(clip):
                continue
            plan = None
            if clip.get("continues"):
                # a handover only ever happens shot -> next shot, so only
                # DIRECT predecessors can hand this bed over
                for pred in preds.get(bid, set()):
                    plan = by_block_sound.get((pred, clip.get("sound")))
                    if plan is not None:
                        break
            if plan is None:
                plan = {"offset": float(clip.get("source_offset_s") or 0.0)}
            plans[(bid, clip.get("id"))] = plan
            by_block_sound[(bid, clip.get("sound"))] = plan
    return plans


def _match_prev_render(pool: Path, src: Path, offset_s: float,
                       speed: float) -> Path | None:
    """A previous export's render of this source at this offset+speed, found
    when the exact trim_name can't be recomputed (ffprobe missing, so the
    duration half of the name tag is unknown). A render's duration is
    deterministic per (source, offset, speed) — it always ran to the end of
    the file — so a prefix match identifies the exact file."""
    prefix = f"{src.stem}__t{int(round(offset_s * 1000))}_"
    speed_tag = "" if speed == 1.0 else f"_x{speed:g}".replace(".", "p")
    try:
        candidates = sorted(pool.iterdir())
    except OSError:
        return None
    for p in candidates:
        if p.suffix != ".ogg" or not p.name.startswith(prefix):
            continue
        rest = p.name[len(prefix):-len(".ogg")]
        dur_part, sep, tail = rest.partition("_")
        if dur_part.isdigit() and (sep + tail if tail else "") == speed_tag:
            return p
    return None


def block_audio_events(block: dict, sounds_by_id: dict, project_path: Path,
                       pool: Path, ffmpeg: str | None,
                       rendered: dict[str, Path],
                       bed_plan: dict | None = None) -> list[dict]:
    """Convert a block's builder audio clips into runtime audio_events, copying
    or pre-rendering the referenced files into the tree's _audio/ pool.

    Builder clips reference the ORIGINAL file + source_offset_s (the browser
    preview seeks natively); the runtime plays files whole, so offset/cut clips
    are trimmed here with ffmpeg — beds chained by `continues` keep one shared
    render (see plan_bed_chains) so the mixer's continuity handover can match
    on the file name."""
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
        speed  = float(clip.get("speed") or 1.0)
        natural = probe_duration(src)
        plan = (bed_plan or {}).get((block.get("id"), clip.get("id")))
        if plan is not None:
            # chained sustain bed: shared render, head offset -> end of file.
            # Beds ignore speed — the continuity handover shares one render
            # whose seek offsets are in source time.
            if abs(speed - 1.0) > 1e-3:
                warn(f"block {block.get('name')!r}: chained bed "
                     f"{sound['name']!r} declares speed {speed:g} — ignored")
                speed = 1.0
            render_offset = plan["offset"]
            render_dur = max(0.05, natural - render_offset) if natural else 1.0
            need_render = render_offset > TRIM_OFFSET_S
        else:
            render_offset = offset
            # dur is the clip's TIMELINE duration; a speeded clip consumes
            # speed×dur seconds of source.
            cut = (dur is not None and natural is not None
                   and float(dur) * speed < natural - TRIM_TAIL_S)
            if dur is not None and natural is None and not cut:
                # ffprobe missing: the clip can't be compared against the
                # source's natural length — but a render cut by a previous
                # (ffprobe-equipped) export proves it ends short of the file.
                # Without this an ffprobe-less export silently shipped the
                # RAW full-length file over an existing trimmed render.
                cut = (pool / trim_name(src, offset, float(dur), speed)).exists()
            render_dur = float(dur) if dur is not None else \
                (max(0.05, (natural - offset) / speed) if natural else 1.0)
            need_render = (offset > TRIM_OFFSET_S or cut
                           or abs(speed - 1.0) > 1e-3)
        file_name = src.name
        if need_render:
            name = trim_name(src, render_offset, render_dur, speed)
            out = pool / name
            if name not in rendered and not out.exists() and natural is None:
                # render_dur above fell back to a placeholder (ffprobe
                # missing), so the computed name can't match the real render.
                # Adopt a previous export's render of the same source/offset/
                # speed instead — bed chains especially must keep referencing
                # the ONE shared render or the mixer's handover key breaks.
                prev = _match_prev_render(pool, src, render_offset, speed)
                if prev is not None:
                    name, out = prev.name, prev
            if name not in rendered:
                # A render left by a previous export is reused as-is — no
                # ffmpeg needed for a metadata-only re-export.
                if out.exists() or (ffmpeg and render_trim(
                        ffmpeg, src, render_offset, render_dur, speed, out)):
                    rendered[name] = out
            if name in rendered:
                file_name = name
            else:
                if not ffmpeg:
                    warn(f"block {block.get('name')!r}: {sound['name']!r} "
                         f"needs a pre-render (offset/speed) but ffmpeg is "
                         f"missing — plays untrimmed at normal speed")
                _pool_copy(src, pool)
        else:
            _pool_copy(src, pool)

        role = clip.get("role", "sfx")
        events.append({
            "file":            file_name,
            "role":            role,
            "at_s":            round(float(clip.get("at_s") or 0.0), 3),
            "duration_s":      (None if dur is None else round(float(dur), 3)),
            # offset into the file the runtime actually plays: chained bed
            # pieces reference the shared render, so their offset is relative
            # to the chain head's cut point
            "source_offset_s": round(offset - plan["offset"], 3)
                               if plan is not None else round(offset, 3),
            "gain":            round(float(clip.get("gain", 1.0)), 4),
            "fade_in_ms":      int(clip.get("fade_in_ms") or 0),
            "fade_out_ms":     int(clip.get("fade_out_ms") or 0),
            "sustain":         bool(clip.get("sustain",
                                             role in ("music", "ambience"))),
            "continues":       bool(clip.get("continues", False)),
        })
    events.sort(key=lambda e: e["at_s"])
    return events


def fold_retry_audio(project: dict, block: dict, blocks: dict, deg: dict,
                     sounds_by_id: dict, project_path: Path, pool: Path,
                     ffmpeg: str | None, rendered: dict[str, Path],
                     bed_plan: dict) -> list[dict]:
    """Audio of a choice block's folded retry chain, rebased into the choice
    shot's timeline (the chain's blocks emit no shots — see retry_choice_fsm).

    Sustain beds are dropped: the redirect never crosses a shot boundary, so
    the choice shot's own bed keeps looping through it. SFX/VO keep their
    master-timeline anchors (at_s += branch_start - choice_start)."""
    fork_branches, _ = split_choice_branches(block)
    rng = block.get("range_s") or [0, 0]
    out: list[dict] = []
    for br in fork_branches[:2]:
        if not br.get("retry"):
            continue
        for fb in branch_chain(project, blocks, deg, br.get("to")):
            if not fb.get("audio") or fb.get("master_audio"):
                continue
            fb_rng = fb.get("range_s") or rng
            if fb_rng[1] > rng[1] + 0.5:
                warn(f"choice {block.get('name')!r}: folded retry block "
                     f"{fb.get('name')!r} runs past the choice block's range — "
                     f"extend the choice range or late audio anchors never fire")
            base = float(fb_rng[0]) - float(rng[0])
            for ev in block_audio_events(fb, sounds_by_id, project_path,
                                         pool, ffmpeg, rendered, bed_plan):
                if ev["sustain"] and ev["role"] in ("music", "ambience"):
                    continue
                ev["at_s"] = round(ev["at_s"] + base, 3)
                ev["continues"] = False
                out.append(ev)
    return out


def _pool_copy(src: Path, pool: Path) -> None:
    dst = pool / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    pool.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_shot_map(out_root: Path, ordered: list, shot_ids: dict, fps: int) -> None:
    """shot_map.json: ORIGINAL master-timeline shot numbers (copy_frames.SHOT_FRAMES,
    the asset tracker's 01–79) -> exported shot id + local frame. The export merges
    many original shots into one stretch block, so this map is what lets the runtime
    start at any original shot (--start-shot) and skip the prologue.

    An original shot whose start time falls inside no exported block (a choice
    pick/switch animation region cut from the export) snaps to the next block's
    first frame and is marked "snapped"."""
    try:
        from copy_frames import SHOT_FRAMES
    except ImportError:
        warn("copy_frames.SHOT_FRAMES not importable — shot_map.json not written")
        return
    master_fps = 30.0   # the master draft's rate; SHOT_FRAMES frames are 1-based

    with_range = [(e["block"], shot_ids[e["block"]["id"]]) for e in ordered
                  if e["block"].get("range_s")]
    entries: dict[str, dict] = {}
    for act_dir, shot_no, f0, _f1 in SHOT_FRAMES:
        t = (int(f0) - 1) / master_fps
        # Adjacent blocks overlap by ~one master frame; a shot starting exactly
        # at a block boundary must map to the LATER block's frame 1, not the
        # last frame of the earlier one — prefer the latest containing start.
        containing = [(b, sid) for b, sid in with_range
                      if b["range_s"][0] - 1e-3 <= t < b["range_s"][1]]
        hit = (max(containing, key=lambda x: x[0]["range_s"][0])
               if containing else None)
        snapped = False
        if hit is None:
            following = [(b, sid) for b, sid in with_range if b["range_s"][0] >= t]
            if not following:
                continue   # past the end of the exported flow
            hit = min(following, key=lambda x: x[0]["range_s"][0])
            snapped = True
        block, sid = hit
        a, z = block["range_s"]
        n_frames = max(1, int(round((z - a) * fps)))
        frame = 1 if snapped else max(1, min(n_frames, int(round((t - a) * fps)) + 1))
        entry = {"master_s": round(t, 3), "shot": sid, "frame": frame,
                 "act": act_dir, "block": block.get("name")}
        if snapped:
            entry["snapped"] = True
        entries[str(shot_no)] = entry

    shot_map = {
        "_comment": "Original shot number (copy_frames.SHOT_FRAMES) -> exported "
                    "shot id + local frame. Read by main.py for --start-shot and "
                    "skip-prologue. Regenerated on every export.",
        "fps": fps,
        "shots": entries,
    }
    with open(out_root / "shot_map.json", "w", encoding="utf-8") as f:
        json.dump(shot_map, f, indent=1)
    print(f"[export] shot_map.json: {len(entries)} original shots mapped")


def export(project_path: Path, out_root: Path, do_frames: bool,
           video_override: Path | None, sound_override: Path | None,
           force_frames: bool = False) -> int:
    project = load_project(project_path)
    fps = int(project.get("fps") or 30)
    blocks = block_map(project)
    deg = indegree(project)
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
    bed_plan = plan_bed_chains(project)

    seq_shots = []
    for entry in ordered:
        block = entry["block"]
        shot_id = shot_ids[block["id"]]
        rng = block.get("range_s") or [0, 0]
        n_frames = max(1, int(round((rng[1] - rng[0]) * fps)))

        if block.get("type") == "choice":
            meta = choice_metadata(block, shot_id, n_frames, fps, sounds_by_id)
        else:
            meta = playback_metadata(block, shot_id, n_frames, fps)

        tag = entry["play_if_tag"]
        if tag:
            fork_shot = shot_ids.get(tag["fork_block"])
            if fork_shot:
                meta["play_if"] = {"shot": fork_shot, "branch": tag["branch"]}

        if block.get("audio") and not block.get("master_audio"):
            events = block_audio_events(block, sounds_by_id, project_path,
                                        audio_pool, audio_ffmpeg, rendered,
                                        bed_plan)
            if events:
                meta["audio_events"] = events

        if block.get("type") == "choice":
            folded = fold_retry_audio(project, block, blocks, deg,
                                      sounds_by_id, project_path, audio_pool,
                                      audio_ffmpeg, rendered, bed_plan)
            if folded:
                merged = (meta.get("audio_events") or []) + folded
                merged.sort(key=lambda e: e["at_s"])
                meta["audio_events"] = merged

        caps = block_captions(block)
        if caps:
            meta["captions"] = caps

        shot_dir = act_dir / f"scene_{shot_id}"
        shot_dir.mkdir(parents=True, exist_ok=True)

        # master_audio: slice the source video's own baked mix to the
        # runtime's whole-file audio.mp3 convention (played on channel 0,
        # suppressed only when audio_events exist — they don't here).
        if block.get("master_audio") and not meta.get("audio_events"):
            video = resolve_video(project, block.get("media"), project_path,
                                  video_override)
            if video is None or audio_ffmpeg is None:
                # Can't bake — but a previous export's file still plays.
                if (shot_dir / "audio.wav").exists():
                    meta["audio"] = "audio.wav"
                elif (shot_dir / "audio.mp3").exists():
                    meta["audio"] = "audio.mp3"
                else:
                    warn(f"shot {shot_id} ({block.get('name')!r}): "
                         f"master_audio needs the source video and ffmpeg "
                         f"— audio skipped")
            else:
                # WAV, not mp3 (Aug 2026): mixer.music seeking into an mp3
                # decode-scans to the target (~316ms main-thread stall at a
                # prologue skip); WAV seek is a constant-time file offset
                # (measured 0.1ms). ~50MB per baked shot — nothing next to
                # the frame packs.
                meta["audio"] = "audio.wav"
                wav = shot_dir / "audio.wav"
                if not wav.exists():
                    dur = max(0.05, rng[1] - rng[0])
                    result = subprocess.run(
                        [audio_ffmpeg, "-hide_banner", "-loglevel", "error",
                         "-y", "-ss", f"{rng[0]:.3f}", "-t", f"{dur:.3f}",
                         "-i", str(video), "-vn", "-c:a", "pcm_s16le",
                         "-ar", "44100", str(wav)],
                        capture_output=True, text=True)
                    if result.returncode != 0:
                        warn(f"shot {shot_id}: master-mix audio extraction "
                             f"failed: {result.stderr.strip()[:160]}")
                        # A previous export's mp3 still plays (with the
                        # mp3-seek stall) — better than silence.
                        if (shot_dir / "audio.mp3").exists():
                            meta["audio"] = "audio.mp3"
                        else:
                            meta.pop("audio", None)

        with open(shot_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        for sfx_name in sorted(collect_sfx_names(meta)):
            src = resolve_sfx_file(sfx_name, project_path, detect_sound)
            if src is None:
                if sfx_name != "detect.mp3":   # missing detect already warned
                    warn(f"shot {shot_id}: sfx {sfx_name!r} not found next to "
                         f"the project, in assets/audio/stems/, or in a "
                         f"previous export — not copied")
                continue
            audio_dir = shot_dir / "audio"
            audio_dir.mkdir(exist_ok=True)
            shutil.copyfile(src, audio_dir / sfx_name)

        if do_frames and block.get("type") != "merge":
            video = resolve_video(project, block.get("media"), project_path, video_override)
            if video is None:
                warn(f"shot {shot_id} ({block.get('name')!r}): source video not found "
                     f"— pass --video; frames skipped")
            else:
                print(f"  shot {shot_id}: extracting {n_frames} frames "
                      f"from {video.name} @ {rng[0]:.2f}s ...")
                extract_frames(ffmpeg, video, rng[0], n_frames, fps,
                               shot_dir / "frames", force=force_frames)

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

    write_shot_map(out_root, ordered, shot_ids, fps)

    print(f"\n[export] {len(seq_shots)} shots -> {out_root}")
    if warnings:
        print(f"[export] {len(warnings)} warning(s) above")
    print("[export] to run it: py -3.12 main.py (export/generated is the "
          "default scenes root).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("project", type=Path, help="path to the .bhrx.json project file")
    parser.add_argument("--out", type=Path, default=ROOT / "export" / "generated",
                        help="output scenes root (default export/generated)")
    parser.add_argument("--no-frames", action="store_true",
                        help="write metadata/sequence only; skip ffmpeg extraction")
    parser.add_argument("--force-frames", action="store_true",
                        help="re-extract every shot's frames even when the "
                             "count already matches (use after replacing the "
                             "source video in place); also clears stale "
                             "framecache.npy packs")
    parser.add_argument("--video", type=Path, default=None,
                        help="override the source video path for every block")
    parser.add_argument("--detect-sound", type=Path, default=None,
                        help="override the global detect sound file")
    args = parser.parse_args()

    if not args.project.exists():
        sys.exit(f"[export] project file not found: {args.project}")
    return export(args.project.resolve(), args.out.resolve(),
                  not args.no_frames, args.video, args.detect_sound,
                  force_frames=args.force_frames)


if __name__ == "__main__":
    sys.exit(main())
