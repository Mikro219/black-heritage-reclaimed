"""
scenes_to_builder.py — Convert the live scenes/ tree (the current BHR
experience) into an Experience Builder project (.bhrx.json).

    py -3.12 scripts/scenes_to_builder.py
    py -3.12 scripts/scenes_to_builder.py --out "My BHR.bhrx.json"

Source of truth:
  * scripts/copy_frames.py SHOT_FRAMES — each shot's [global_start, global_end]
    frame range in the master video (final_frames/ was extracted 1:1 from
    assets/video/"BHR Draft 1.mp4", 30fps, 25:21 = 45631 frames).
  * scenes/<act>/shot_NN/metadata.json — interactions, forks, OI windows.

Mapping into editor blocks:
  plain playback shot                  -> playback block, no windows
  OI with oi_frame_window              -> playback block + one window
  play-through FSM with `oi` states    -> playback block + one window per state
  region-fork FSM (shots 09/37/50)     -> choice block trimmed to the hold
                                          frame + branch playback blocks cut
                                          from the shot's own confirm/path
                                          segments; a wrong-way retry (shot 37
                                          left) marks its branch "retry": true
                                          and authors hold_segments so the
                                          exporter rebuilds the redirect loop
  draw-chain FSM (shots 19/20)         -> playback block + one directional_draw
                                          window per stroke (the blocking
                                          stroke-chain semantics stay in
                                          scenes/ — windows here are for
                                          editing/preview reference)

The result opens in tools/experience_builder/ — re-link "BHR Draft 1.mp4"
(click it in the Media tab) and detect.mp3 on first open.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from copy_frames import SHOT_FRAMES  # noqa: E402

FPS = 30
VIDEO_NAME = "BHR Draft 1.mp4"

# canvas layout
COL_W, ROW_Y = 250, {"up": 150, "mid": 340, "down": 530}


def act_title(act_folder: str) -> str:
    return act_folder.split("_", 2)[-1].replace("_", " ").title()


def load_meta(act: str, shot: str) -> dict:
    p = ROOT / "scenes" / act / f"shot_{shot}" / "metadata.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8-sig"))


def t(global_frame_1based: int) -> float:
    """Global frame number -> video time (seconds)."""
    return round((global_frame_1based - 1) / FPS, 3)


def rng(g_start: int, local_a: int, local_b: int) -> list[float]:
    """Local [a, b] frame range of a shot -> video [start_s, end_s]."""
    return [t(g_start + local_a - 1), round((g_start + local_b - 1) / FPS, 3)]


def win_id(shot: str, suffix: str) -> str:
    return f"w_{shot}_{suffix}"


def window_from_params(shot: str, suffix: str, label: str, detector: str,
                       params: dict, appears_s: float,
                       duration_s: float | None) -> dict:
    params = dict(params or {})
    region = None
    # Live region_rects are RAW (unmirrored) camera space; Builder regions are
    # SCREEN space — mirror x on import (the exporter mirrors back).
    # indicator_rect (directional_draw placement) is already screen space.
    rect = params.pop("region_rect", None)
    if isinstance(rect, dict):
        w_ = rect.get("w", 0.2)
        region = {"shape": "rect", "x": round(1.0 - rect.get("x", 0.4) - w_, 3),
                  "y": rect.get("y", 0.3), "w": w_, "h": rect.get("h", 0.3)}
    ind = params.pop("indicator_rect", None)
    if region is None and isinstance(ind, dict):
        region = {"shape": "rect", "x": ind.get("x", 0.4), "y": ind.get("y", 0.3),
                  "w": ind.get("w", 0.2), "h": ind.get("h", 0.3)}
    return {"id": win_id(shot, suffix), "label": label, "detector": detector,
            "params": params, "region": region,
            "appears_s": round(appears_s, 2),
            "duration_s": None if duration_s is None else round(duration_s, 2)}


def side_region(side: str) -> dict:
    x = 0.05 if side == "left" else 0.65
    return {"shape": "rect", "x": x, "y": 0.25, "w": 0.30, "h": 0.50}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", type=Path,
                        default=ROOT / "BHR_Experience.bhrx.json")
    args = parser.parse_args()

    mapped = [(act, shot, a, b) for act, shot, a, b in SHOT_FRAMES]
    total_frames = mapped[-1][3]

    blocks: list[dict] = []
    edges: list[dict] = []
    notes: list[str] = []

    col = 0

    def place(lane: str = "mid", advance: bool = True):
        nonlocal col
        pos = [col * COL_W + 60, ROW_Y[lane]]
        if advance:
            col += 1
        return pos

    # id of the block that should receive the "next shot" edge, or None when
    # the fork wiring already connected everything up to the next shot.
    tail: str | None = None

    def link(to_id: str):
        nonlocal tail
        if tail is not None:
            edges.append({"from": tail, "to": to_id})
        tail = to_id

    i = 0
    while i < len(mapped):
        act, shot, g_a, g_b = mapped[i]
        meta = load_meta(act, shot)
        inter = meta.get("interaction") or {}
        fsm = inter.get("interaction_fsm") or {}
        segments = meta.get("segments") or {}
        bid = f"b_{shot}"
        name = f"{shot} · {act_title(act)}"
        n_local = g_b - g_a + 1

        # ── region fork: choice block + branch blocks from the shot's own
        #    confirm/path segments ─────────────────────────────────────────
        if fsm.get("gesture_type") == "region":
            waiting = fsm["states"]["waiting"]
            hold_end = segments[waiting["segment"]][1]
            timeout_s = int((meta.get("fallback") or {}).get("timeout_s", 30))

            # direction -> first state it leads to from waiting
            first_to = {tr["on"]: tr["to"] for tr in fsm["transitions"]
                        if tr["from"] == "waiting"}

            def leads_to_advance(state: str, depth: int = 0) -> bool:
                if depth > 12:
                    return False
                for tr in fsm["transitions"]:
                    if tr["from"] == state and tr["on"] == "segment_end":
                        return (tr["to"] == "__advance__"
                                or leads_to_advance(tr["to"], depth + 1))
                return False

            def resolution_state(direction: str) -> str:
                """The segment that plays once this choice locks in: follow the
                FSM from waiting, preferring the exit that reaches __advance__
                (skips hold flourishes like shot 09's left/right switch)."""
                state = first_to[f"point_{direction}"]
                seen = set()
                while state not in seen:
                    seen.add(state)
                    exits = [tr for tr in fsm["transitions"]
                             if tr["from"] == state and tr["on"] != "segment_end"]
                    if not exits:
                        return state          # only segment_end exits: resolution
                    advancing = [tr for tr in exits if leads_to_advance(tr["to"])]
                    state = (advancing[0] if advancing else exits[0])["to"]
                return state

            choice = {
                "id": bid, "type": "choice", "name": name, "media": "m1",
                "range_s": rng(g_a, 1, hold_end), "hold": "pause_end",
                "pos": place(), "windows": [], "branches": [],
                "timeout": {"seconds": timeout_s, "to": None},
            }
            blocks.append(choice)
            link(bid)
            tail = None   # fork wiring takes over

            branch_infos = []
            for side in ("left", "right"):
                res = resolution_state(side)
                seg = fsm["states"][res]["segment"]
                a_l, b_l = segments[seg]
                sub_id = f"b_{shot}_{side}"
                sub = {"id": sub_id, "type": "playback",
                       "name": f"{shot}{side[0].upper()} · {res}",
                       "media": "m1", "range_s": rng(g_a, a_l, b_l),
                       "pos": place("up" if side == "left" else "down",
                                    advance=(side == "right")),
                       "windows": []}
                blocks.append(sub)
                choice["windows"].append({**window_from_params(
                    shot, side, "Go " + side.capitalize(), "point_region",
                    dict(inter.get("params") or {},
                         hold_ms=inter.get("hold_ms", 600)),
                    0.0, None), "region": side_region(side)})
                choice["branches"].append({"id": f"br_{shot}_{side}",
                                           "window": win_id(shot, side),
                                           "to": sub_id,
                                           "label": "Go " + side.capitalize()})
                # does this resolution loop back to waiting (37 wrong-way)?
                loops = any(tr["from"] == res and tr["to"] == "waiting"
                            for tr in fsm["transitions"])
                if loops:
                    # Wrong-way retry: mark the branch so the exporter folds
                    # its clip back into the choice shot and rebuilds the
                    # redirect loop; extend the choice to cover the redirect
                    # frames and author the exporter's hold_segments.
                    choice["branches"][-1]["retry"] = True
                    choice["range_s"] = rng(g_a, 1, max(hold_end, b_l))
                    choice["hold_segments"] = {
                        "intro": list(segments.get("intro", [1, hold_end])),
                        "idle_loop": list(segments[waiting["segment"]]),
                        "wrong_path": [a_l, b_l],
                    }
                    notes.append(f"shot {shot}: '{side}' is the wrong-way "
                                 f"retry — branch marked retry; the redirect "
                                 f"folds into the choice block on export")
                branch_infos.append((sub_id, side, loops))

            # voice confirm (shot 09) — reaction window for reference
            if any(tr["on"].startswith("voice_") for tr in fsm["transitions"]):
                kw = next(tr["on"][6:] for tr in fsm["transitions"]
                          if tr["on"].startswith("voice_"))
                choice["windows"].append(window_from_params(
                    shot, "voice", f'Say "{kw}"', "voice", {"keyword": kw},
                    0.0, None))
                notes.append(f"shot {shot}: voice confirm '{kw}' imported as a "
                             f"reaction window on the choice block")
            choice["timeout"]["to"] = branch_infos[0][0]

            # The fork's play_if window: every later shot gated on THIS fork,
            # up to the last one. Ungated shots inside that window play on
            # BOTH paths (e.g. shot 57 between fork 50 and gated 58/59) — in
            # the editor's graph they are duplicated per branch so each path
            # stays a straight chain.
            gate_of = {}
            for j2 in range(i + 1, len(mapped)):
                pi = load_meta(mapped[j2][0], mapped[j2][1]).get("play_if")
                if pi and pi.get("shot") == shot:
                    gate_of[j2] = pi["branch"]
            window_end = max(gate_of) if gate_of else i   # inclusive

            next_tails = []
            for sub_id, side, loops in branch_infos:
                lane = "up" if side == "left" else "down"
                side_tail = sub_id
                side_col = col
                for j2 in range(i + 1, window_end + 1):
                    branch = gate_of.get(j2)
                    if branch is not None and branch != side:
                        continue
                    act2, shot2, ga2, gb2 = mapped[j2]
                    blk = simple_block(act2, shot2, ga2, gb2,
                                       [side_col * COL_W + 60, ROW_Y[lane]], notes)
                    side_col += 1
                    if branch is None:
                        # shared shot: one copy per branch
                        blk["id"] = f"b_{shot2}_{side[0]}"
                        blk["name"] += f" ({side[0].upper()})"
                        for w in blk.get("windows") or []:
                            w["id"] += f"_{side[0]}"
                        if side == "left":
                            notes.append(f"shot {shot2}: plays on both of fork "
                                         f"{shot}'s paths — duplicated per branch")
                    blocks.append(blk)
                    edges.append({"from": side_tail, "to": blk["id"]})
                    side_tail = blk["id"]
                next_tails.append((side_tail, side_col))

            # all surviving side tails converge on the shot after the window
            i = window_end + 1
            if next_tails:
                col = max(c for _, c in next_tails)
            if i < len(mapped):
                act2, shot2, ga2, gb2 = mapped[i]
                blk = simple_block(act2, shot2, ga2, gb2, place(), notes)
                blocks.append(blk)
                for st, _ in next_tails:
                    edges.append({"from": st, "to": blk["id"]})
                tail = blk["id"]
                i += 1
            continue

        # ── everything else ─────────────────────────────────────────────
        blk = simple_block(act, shot, g_a, g_b, place(), notes)
        blocks.append(blk)
        link(blk["id"])
        i += 1

    project = {
        "version": 1,
        "name": "BHR Experience",
        "fps": FPS,
        "media": [{"id": "m1", "name": VIDEO_NAME,
                   "duration_s": round(total_frames / FPS, 2)}],
        "sounds": [{"id": "s1", "name": "detect.mp3"}],
        "settings": {"global_detect_sound": "s1"},
        "start": blocks[0]["id"] if blocks else None,
        "blocks": blocks,
        "edges": edges,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2)

    n_win = sum(len(b.get("windows") or []) for b in blocks)
    n_choice = sum(1 for b in blocks if b["type"] == "choice")
    print(f"[import] {len(blocks)} blocks ({n_choice} choices), {len(edges)} edges, "
          f"{n_win} interaction windows -> {args.out.name}")
    for n in notes:
        print(f"  [note] {n}")
    print("[import] open tools/experience_builder/index.html, then Open the "
          "project and re-link the MP4 (click it in the Media tab).")
    return 0


def simple_block(act: str, shot: str, g_a: int, g_b: int, pos, notes) -> dict:
    """Playback shot (plain, single-OI, multi-OI FSM, or draw chain) -> block."""
    meta = load_meta(act, shot)
    inter = meta.get("interaction") or {}
    fsm = inter.get("interaction_fsm") or {}
    segments = meta.get("segments") or {}
    n_local = g_b - g_a + 1
    blk = {
        "id": f"b_{shot}", "type": "playback",
        "name": f"{shot} · {act_title(act)}",
        "media": "m1", "range_s": rng(g_a, 1, n_local),
        "pos": pos, "windows": [],
    }

    if inter and not fsm:
        # single OI, frame-gated
        a, b = inter.get("oi_frame_window") or [1, n_local]
        blk["windows"].append(window_from_params(
            shot, "oi", inter.get("id") or inter.get("type"),
            inter["type"], inter.get("params"),
            (a - 1) / FPS, (b - a + 1) / FPS))
        return blk

    if fsm and fsm.get("gesture_type") == "draw":
        blk["name"] += " (draw chain)"
        notes.append(f"shot {shot}: draw stroke-chain imported as reference "
                     f"windows — the blocking chain semantics live in scenes/")
        k = 0
        for sid, st in fsm.get("states", {}).items():
            dirs = st.get("directions")
            if not dirs:
                # play-through states may still carry an oi (shot 20 oi_2)
                oi = st.get("oi") or {}
                if oi.get("type"):
                    a, b = segments[st["segment"]]
                    blk["windows"].append(window_from_params(
                        shot, f"oi{k}", oi.get("id") or oi["type"], oi["type"],
                        oi.get("params"), (a - 1) / FPS, (b - a + 1) / FPS))
                    k += 1
                continue
            a = segments[st["segment"]][0]
            blk["windows"].append(window_from_params(
                shot, f"s{k}", f"Draw {dirs[0]}", "directional_draw",
                {"direction": dirs[0]}, (a - 1) / FPS, 2.0))
            k += 1
        return blk

    if fsm:
        # play-through FSM with oi states (shots 24 / 45 / 66)
        k = 0
        for sid, st in fsm.get("states", {}).items():
            oi = st.get("oi") or {}
            if not oi.get("type"):
                continue
            a, b = segments[st["segment"]]
            blk["windows"].append(window_from_params(
                shot, f"oi{k}", oi.get("id") or oi["type"], oi["type"],
                oi.get("params"), (a - 1) / FPS, (b - a + 1) / FPS))
            k += 1
        return blk

    return blk


if __name__ == "__main__":
    sys.exit(main())
