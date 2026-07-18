"""
simplify_builder_project — collapse choice-free stretches of a .bhrx.json
project into single big playback blocks (July 2026 simplification pass).

What it does:
* Finds the choice CLUSTERS: each choice block plus its branch chains up to
  (not including) the block where the paths reconverge. Clusters keep their
  structure untouched — only their audio clips are cleared (branch audio is
  authored by hand later).
* DRAW blocks — any block with a ``directional_draw`` interaction window
  (the stroke-chain shots) — are barriers like the clusters: they stay their
  own blocks and keep their imported audio clips untouched (per-stroke
  SFX/VO must stay frame-synced to the strokes).
* Every maximal run of edge-connected playback blocks OUTSIDE the clusters
  and draw blocks becomes ONE merged block: range spans first start → last
  end (runs split at any master-timeline gap > MAX_GAP_S), interaction
  windows are re-based onto the merged timeline and kept, and the block is
  flagged ``master_audio: true`` — it plays the source video's own baked mix
  (the master draft's full mix) instead of lane clips, in the builder
  player, the preview flow, and the export (which slices the video's audio
  to the runtime's whole-file audio.mp3 convention).
* The sound library and the global detect sound are kept — choice audio is
  added by hand from it later.

The original file is backed up next to itself as
``<name>.pre_simplify.json`` and the builder bundle
(tools/experience_builder/js/project_data.js) is regenerated.

Usage:
    py -3.12 scripts/simplify_builder_project.py                      # BHR_Experience.bhrx.json
    py -3.12 scripts/simplify_builder_project.py path/to/project.json
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_PROJECT = ROOT / "BHR_Experience.bhrx.json"

# Consecutive blocks whose master-timeline ranges differ by more than this
# are NOT merged (a hidden jump would desync the master mix).
MAX_GAP_S = 0.5


def build_graph(project: dict):
    """(out_edges, distinct_preds) including choice branches + timeouts."""
    out: dict[str, list[str]] = {}
    preds: dict[str, set[str]] = {}

    def link(a: str, b: str | None):
        if not b:
            return
        out.setdefault(a, []).append(b)
        preds.setdefault(b, set()).add(a)

    for e in project.get("edges", []):
        link(e["from"], e["to"])
    for b in project.get("blocks", []):
        if b.get("type") == "choice":
            for br in b.get("branches", []):
                link(b["id"], br.get("to"))
            t = b.get("timeout")
            if t:
                link(b["id"], t.get("to"))
    return out, preds


def find_clusters(project: dict, out, preds):
    """{block_id} of every choice block + its branch chains up to (not
    including) the reconvergence block (first block with >1 distinct pred)."""
    blocks = {b["id"]: b for b in project["blocks"]}
    members: set[str] = set()
    for b in project["blocks"]:
        if b.get("type") != "choice":
            continue
        members.add(b["id"])
        heads = {br.get("to") for br in b.get("branches", [])}
        t = b.get("timeout")
        if t:
            heads.add(t.get("to"))
        for head in filter(None, heads):
            cur = head
            while cur and cur in blocks and len(preds.get(cur, set())) < 2 \
                    and cur not in members:
                members.add(cur)
                nxt = out.get(cur, [])
                cur = nxt[0] if len(nxt) == 1 else None
    return members


def is_draw_block(b: dict) -> bool:
    """Stroke-chain blocks keep their own identity + frame-synced audio."""
    return any(w.get("detector") == "directional_draw"
               for w in b.get("windows", []))


def trunk_runs(project: dict, out, clusters):
    """Maximal edge-connected runs of non-cluster playback blocks, split at
    master-timeline gaps > MAX_GAP_S. Draw blocks are barriers (kept out of
    every run). Returns lists of block dicts."""
    runs: list[list[dict]] = []
    run: list[dict] = []
    for b in project["blocks"]:
        if b["id"] in clusters or b.get("type") != "playback" \
                or not b.get("range_s") or is_draw_block(b):
            if run:
                runs.append(run)
                run = []
            continue
        if run:
            prev = run[-1]
            connected = b["id"] in out.get(prev["id"], [])
            same_media = b.get("media") == prev.get("media")
            gap = abs(b["range_s"][0] - prev["range_s"][1])
            if not (connected and same_media and gap <= MAX_GAP_S):
                runs.append(run)
                run = []
        run.append(b)
    if run:
        runs.append(run)
    return runs


def short_name(name: str) -> tuple[str, str]:
    """'02 · Quilt Awakens' -> ('02', 'Quilt Awakens')."""
    parts = [p.strip() for p in name.split("·", 1)]
    if len(parts) == 2:
        return parts[0], parts[1]
    return name, name


def merge_run(run: list[dict], idx: int) -> dict:
    first, last = run[0], run[-1]
    n0, t0 = short_name(first.get("name", first["id"]))
    n1, t1 = short_name(last.get("name", last["id"]))
    title = t0 if t0 == t1 else f"{t0} → {t1}"
    merged = {
        "id": f"b_stretch_{idx}",
        "type": "playback",
        "name": f"{n0}–{n1} · {title}",
        "media": first.get("media"),
        "range_s": [first["range_s"][0], last["range_s"][1]],
        "master_audio": True,
        "windows": [],
    }
    # canvas geometry: blocks render at block.pos = [x, y] — a block without
    # one crashes the graph layer (empty canvas). Keep the first member's.
    merged["pos"] = list(first.get("pos") or [80 + 260 * idx, 160])
    base = first["range_s"][0]
    for b in run:
        off = b["range_s"][0] - base
        for w in b.get("windows", []):
            w2 = json.loads(json.dumps(w))
            w2["appears_s"] = round((w2.get("appears_s") or 0) + off, 2)
            merged["windows"].append(w2)
    return merged


def simplify(project: dict) -> tuple[dict, list[str]]:
    out, preds = build_graph(project)
    clusters = find_clusters(project, out, preds)
    runs = trunk_runs(project, out, clusters)

    id_map: dict[str, str] = {}          # old block id -> merged id
    merged_blocks: dict[str, dict] = {}  # merged id -> block (in order)
    notes: list[str] = []

    for i, run in enumerate(r for r in runs if len(r) >= 1):
        if len(run) == 1:
            # single trunk block between clusters — still switch it to the
            # master mix for consistency, no merge needed
            b = run[0]
            b["master_audio"] = True
            b.pop("audio", None)
            notes.append(f"kept {b['name']} (single block) — master mix")
            continue
        m = merge_run(run, i + 1)
        merged_blocks[m["id"]] = m
        for b in run:
            id_map[b["id"]] = m["id"]
        notes.append(
            f"merged {len(run)} blocks {run[0]['name']!r} .. {run[-1]['name']!r}"
            f" -> {m['name']!r} ({m['range_s'][0]:.2f}-{m['range_s'][1]:.2f}s,"
            f" {len(m['windows'])} windows)")

    # rebuild the block list: merged blocks appear where their first member was
    new_blocks: list[dict] = []
    emitted: set[str] = set()
    for b in project["blocks"]:
        mid = id_map.get(b["id"])
        if mid is None:
            if b["id"] in clusters and b.get("audio"):
                b.pop("audio", None)     # choice audio is authored by hand
                notes.append(f"cleared audio on choice block {b['name']!r}")
            elif is_draw_block(b):
                notes.append(f"kept draw block {b['name']!r} untouched "
                             f"({len(b.get('audio', []))} audio clips)")
            new_blocks.append(b)
        elif mid not in emitted:
            emitted.add(mid)
            new_blocks.append(merged_blocks[mid])
    project["blocks"] = new_blocks

    remap = lambda bid: id_map.get(bid, bid)  # noqa: E731

    # edges: drop intra-merge edges, remap endpoints, dedupe
    seen = set()
    new_edges = []
    for e in project.get("edges", []):
        a, z = remap(e["from"]), remap(e["to"])
        if a == z or (a, z) in seen:
            continue
        seen.add((a, z))
        new_edges.append({"from": a, "to": z})
    project["edges"] = new_edges

    for b in project["blocks"]:
        if b.get("type") == "choice":
            for br in b.get("branches", []):
                if br.get("to"):
                    br["to"] = remap(br["to"])
            t = b.get("timeout")
            if t and t.get("to"):
                t["to"] = remap(t["to"])
    if project.get("start"):
        project["start"] = remap(project["start"])

    # Defensive: every block MUST have a canvas position — graph.js reads
    # block.pos[0] unconditionally and one missing pos blanks the whole
    # canvas. Lay any stragglers out along a line.
    for i, b in enumerate(project["blocks"]):
        if not (isinstance(b.get("pos"), list) and len(b["pos"]) == 2):
            b["pos"] = [80 + 250 * i, 210]
            notes.append(f"assigned canvas position to {b.get('name', b['id'])!r}")

    return project, notes


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROJECT
    if not path.exists():
        sys.exit(f"project not found: {path}")

    backup = path.with_suffix(".pre_simplify.json")
    shutil.copyfile(path, backup)

    with open(path, encoding="utf-8") as f:
        project = json.load(f)

    n_before = len(project["blocks"])
    project, notes = simplify(project)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2)
        f.write("\n")

    print(f"[simplify] {n_before} -> {len(project['blocks'])} blocks "
          f"(backup: {backup.name})")
    for n in notes:
        print(f"  - {n}")

    try:
        from bundle_builder_project import write_bundle
        write_bundle(path)
    except Exception as exc:
        print(f"[simplify] WARNING could not refresh builder bundle: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
