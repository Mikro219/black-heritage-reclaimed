/* Node-graph canvas: blocks (playback / choice / merge), bezier edges,
 * pan/zoom/fit, drag-to-move, drag-from-port-to-connect, drop-clip-to-create.
 *
 * Coordinates: block.pos is in graph space; the #graph layer carries
 * translate(pan) scale(zoom). Edge endpoints are measured from the rendered
 * block DOM (ports), so they stay correct whatever the block's height. */
(function () {
  const $ = (id) => document.getElementById(id);
  const wrap = $("canvas-wrap");
  const graph = $("graph");
  const blocksLayer = $("blocks-layer");
  const svg = $("edges-svg");

  const view = { x: 60, y: 40, zoom: 1 };
  EB.view = view;

  function applyView() {
    graph.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`;
    const pct = Math.round(view.zoom * 100) + "%";
    $("zoom-val").textContent = pct;
    $("tb-zoom-val").textContent = pct;
  }

  /* ── pan & zoom ── */
  let panning = null;
  wrap.addEventListener("mousedown", (e) => {
    if (e.target === wrap || e.target === svg || e.target === blocksLayer || e.target === graph) {
      e.preventDefault();   // no text-selection drag while panning
      panning = { sx: e.clientX, sy: e.clientY, ox: view.x, oy: view.y };
      if (!e.shiftKey) EB.select(null);
    }
  });
  // Nothing inside the canvas is meant to native-drag (library items live
  // outside it) — a drag ghost here would swallow mousemove/mouseup and
  // wedge the pan, so kill any that still starts (selected text, images).
  wrap.addEventListener("dragstart", (e) => e.preventDefault());
  window.addEventListener("mousemove", (e) => {
    if (!panning) return;
    view.x = panning.ox + (e.clientX - panning.sx);
    view.y = panning.oy + (e.clientY - panning.sy);
    applyView();
  });
  window.addEventListener("mouseup", () => { panning = null; });

  wrap.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = wrap.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const oldZoom = view.zoom;
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    view.zoom = Math.min(2.5, Math.max(0.25, view.zoom * factor));
    // zoom around cursor
    view.x = mx - (mx - view.x) * (view.zoom / oldZoom);
    view.y = my - (my - view.y) * (view.zoom / oldZoom);
    applyView();
  }, { passive: false });

  $("zoom-in").addEventListener("click", () => { view.zoom = Math.min(2.5, view.zoom * 1.2); applyView(); });
  $("zoom-out").addEventListener("click", () => { view.zoom = Math.max(0.25, view.zoom / 1.2); applyView(); });
  $("zoom-fit").addEventListener("click", fit);

  function fit() {
    const blocks = EB.project.blocks;
    if (!blocks.length) { view.x = 60; view.y = 40; view.zoom = 1; return applyView(); }
    let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
    for (const b of blocks) {
      minX = Math.min(minX, b.pos[0]); minY = Math.min(minY, b.pos[1]);
      maxX = Math.max(maxX, b.pos[0] + 220); maxY = Math.max(maxY, b.pos[1] + 190);
    }
    const rect = wrap.getBoundingClientRect();
    const pad = 60;
    const zx = (rect.width - pad * 2) / Math.max(1, maxX - minX);
    const zy = (rect.height - pad * 2) / Math.max(1, maxY - minY);
    view.zoom = Math.min(1.25, Math.max(0.25, Math.min(zx, zy)));
    view.x = pad - minX * view.zoom + (rect.width - pad * 2 - (maxX - minX) * view.zoom) / 2;
    view.y = pad - minY * view.zoom + (rect.height - pad * 2 - (maxY - minY) * view.zoom) / 2;
    applyView();
  }
  EB.fitView = fit;

  function graphPoint(clientX, clientY) {
    const rect = wrap.getBoundingClientRect();
    return [(clientX - rect.left - view.x) / view.zoom,
            (clientY - rect.top - view.y) / view.zoom];
  }

  /* ── create blocks ── */
  function addBlock(type, pos, mediaId) {
    const p = EB.project;
    const id = EB.uid("b");
    const media = mediaId ? EB.getMedia(mediaId) : (p.media[0] || null);
    const block = { id, type, name: "", pos: [Math.round(pos[0]), Math.round(pos[1])] };
    if (type === "playback") {
      block.name = media ? media.name.replace(/\.[^.]+$/, "") : "Playback";
      block.media = media ? media.id : null;
      block.range_s = media ? [0, Math.min(8, media.duration_s || 8)] : [0, 5];
      block.windows = [];
    } else if (type === "choice") {
      block.name = "Choice";
      block.media = media ? media.id : null;
      block.range_s = media ? [0, Math.min(8, media.duration_s || 8)] : [0, 5];
      block.hold = "pause_end";
      block.windows = [];
      block.branches = [];
      block.timeout = { seconds: 30, to: null };
    } else if (type === "merge") {
      block.name = "Merge";
    }
    EB.change("add " + type, () => {
      p.blocks.push(block);
      if (!p.start) p.start = id;
    });
    EB.select({ kind: "block", id });
    return block;
  }
  EB.addBlock = addBlock;

  $("add-choice").addEventListener("click", () => {
    const rect = wrap.getBoundingClientRect();
    addBlock("choice", graphPoint(rect.left + rect.width / 2 - 100, rect.top + rect.height / 2 - 80));
  });
  $("add-merge").addEventListener("click", () => {
    const rect = wrap.getBoundingClientRect();
    addBlock("merge", graphPoint(rect.left + rect.width / 2 - 100, rect.top + rect.height / 2 - 40));
  });

  /* drop a clip from the library */
  wrap.addEventListener("dragover", (e) => {
    if ([...e.dataTransfer.types].includes("application/x-bhr-media") ||
        [...e.dataTransfer.types].includes("Files")) {
      e.preventDefault();
      wrap.classList.add("drop-target");
    }
  });
  wrap.addEventListener("dragleave", (e) => {
    if (e.target === wrap) wrap.classList.remove("drop-target");
  });
  wrap.addEventListener("drop", (e) => {
    e.preventDefault();
    wrap.classList.remove("drop-target");
    const mediaId = e.dataTransfer.getData("application/x-bhr-media");
    const pt = graphPoint(e.clientX, e.clientY);
    if (mediaId) {
      addBlock("playback", [pt[0] - 98, pt[1] - 40], mediaId);
    }
  });

  /* ── connecting ── */
  let connecting = null;  // { fromBlock, branchId|null, timeout:bool, ghost }

  function startConnect(e, blockId, branchId, isTimeout) {
    e.stopPropagation(); e.preventDefault();
    const ghost = document.createElementNS("http://www.w3.org/2000/svg", "path");
    ghost.setAttribute("class", "edge-ghost");
    svg.appendChild(ghost);
    connecting = { fromBlock: blockId, branchId: branchId || null, timeout: !!isTimeout, ghost };
    updateGhost(e);
  }

  function updateGhost(e) {
    if (!connecting) return;
    const from = portAnchor(connecting.fromBlock, connecting.branchId, connecting.timeout);
    if (!from) return;
    const to = graphPoint(e.clientX, e.clientY);
    connecting.ghost.setAttribute("d", bezier(from, to));
  }

  window.addEventListener("mousemove", (e) => updateGhost(e));
  window.addEventListener("mouseup", (e) => {
    if (!connecting) return;
    const targetEl = document.elementFromPoint(e.clientX, e.clientY);
    const blockEl = targetEl && targetEl.closest(".block");
    connecting.ghost.remove();
    const c = connecting; connecting = null;
    if (!blockEl) return;
    const toId = blockEl.dataset.id;
    if (toId === c.fromBlock) return;
    const from = EB.getBlock(c.fromBlock);
    if (!from) return;
    EB.change("connect", () => {
      if (c.branchId) {
        const br = (from.branches || []).find(b => b.id === c.branchId);
        if (br) br.to = toId;
      } else if (c.timeout) {
        from.timeout = from.timeout || { seconds: 30, to: null };
        from.timeout.to = toId;
      } else {
        // single-out flow edge: replace any existing
        EB.project.edges = EB.project.edges.filter(ed => ed.from !== c.fromBlock);
        EB.project.edges.push({ from: c.fromBlock, to: toId });
      }
    });
  });

  /* ── rendering ── */
  function render() {
    const p = EB.project;
    $("canvas-empty").style.display = p.blocks.length ? "none" : "flex";
    blocksLayer.innerHTML = "";

    for (const b of p.blocks) {
      blocksLayer.appendChild(renderBlock(b));
    }
    requestAnimationFrame(renderEdges);
  }

  function renderBlock(b) {
    const el = document.createElement("div");
    el.className = "block " + b.type;
    el.dataset.id = b.id;
    el.style.left = b.pos[0] + "px";
    el.style.top = b.pos[1] + "px";
    const sel = EB.runtime.selection;
    if (sel && sel.kind === "block" && sel.id === b.id) el.classList.add("selected");

    const media = b.media ? EB.getMedia(b.media) : null;
    const len = EB.blockLen(b);
    const icon = b.type === "playback" ? "movie" : b.type === "choice" ? "call_split" : "call_merge";
    const isStart = EB.project.start === b.id;

    let html = `<div class="b-head"><span class="msr">${icon}</span><span class="b-name"></span>`;
    if (b.type === "playback" && isEndBlock(b)) html += `<span class="b-flag" style="background:rgba(52,211,153,.2);color:var(--green);">END</span>`;
    html += `</div>`;

    if (b.type !== "merge") {
      const nWin = (b.windows || []).length;
      html += `<div class="b-thumb">
        <div class="b-thumb-empty"><span class="msr">image</span></div>
        <span class="b-dur">${EB.fmtTime(len, false)}</span>
        ${nWin ? `<span class="b-winbadge"><span class="msr" style="font-size:12px;">touch_app</span>${nWin}</span>` : ""}
      </div>
      <div class="b-sub">${media ? escapeHtml(media.name) : "no clip — drop one from Media"}</div>`;
    } else {
      html += `<div class="b-sub">${countIn(b.id)} paths in</div>`;
    }

    if (b.type === "choice") {
      html += `<div class="b-branches">`;
      for (const br of (b.branches || [])) {
        const target = br.to ? EB.getBlock(br.to) : null;
        html += `<div class="b-branch" data-branch="${br.id}">
          <span class="msr">subdirectory_arrow_right</span>
          <span class="br-label">${escapeHtml(br.label || "branch")}</span>
          <span class="br-target">${target ? "→ " + escapeHtml(target.name) : "unlinked"}</span>
          <span class="port br-out" data-branch-port="${br.id}" title="Drag to the branch's target block"></span>
        </div>`;
      }
      const tt = b.timeout || {};
      const ttTarget = tt.to ? EB.getBlock(tt.to) : null;
      html += `<div class="b-branch timeout" data-branch="__timeout__">
          <span class="msr">timer</span>
          <span class="br-label">timeout ${tt.seconds != null ? tt.seconds + "s" : ""}</span>
          <span class="br-target">${ttTarget ? "→ " + escapeHtml(ttTarget.name) : "auto"}</span>
          <span class="port br-out" data-timeout-port="1" title="Drag to the timeout fallback block"></span>
        </div>`;
      html += `</div>`;
    }

    el.innerHTML = html;
    el.querySelector(".b-name").textContent = b.name || "(unnamed)";

    if (isStart) {
      const flag = document.createElement("div");
      flag.className = "start-flag"; flag.title = "Start block";
      flag.innerHTML = '<span class="msr fill">flag</span>';
      el.appendChild(flag);
    }

    // thumbnail
    if (b.type !== "merge" && b.media) {
      const key = b.media + "@" + Math.round((b.range_s ? b.range_s[0] : 0) * 10) / 10;
      const cached = EB.runtime.thumbs[key];
      const box = el.querySelector(".b-thumb");
      if (cached) {
        box.insertAdjacentHTML("afterbegin", `<img src="${cached}" alt="">`);
      } else {
        EB.thumbnail(b.media, b.range_s ? b.range_s[0] : 0);  // thumb-ready re-renders
      }
    }

    // ports
    if (b.type === "playback" || b.type === "merge") {
      const out = document.createElement("span");
      out.className = "port out";
      out.style.top = "50%";
      out.title = "Drag to the next block";
      out.addEventListener("mousedown", (e) => startConnect(e, b.id, null, false));
      el.appendChild(out);
    }
    const inp = document.createElement("span");
    inp.className = "port in";
    el.appendChild(inp);

    // branch ports
    el.querySelectorAll("[data-branch-port]").forEach(port => {
      port.addEventListener("mousedown", (e) => startConnect(e, b.id, port.dataset.branchPort, false));
    });
    el.querySelectorAll("[data-timeout-port]").forEach(port => {
      port.addEventListener("mousedown", (e) => startConnect(e, b.id, null, true));
    });

    // select + drag
    el.addEventListener("mousedown", (e) => {
      if (e.target.classList.contains("port")) return;
      e.stopPropagation();
      EB.select({ kind: "block", id: b.id });
      const start = { sx: e.clientX, sy: e.clientY, ox: b.pos[0], oy: b.pos[1], moved: false };
      const onMove = (ev) => {
        const dx = (ev.clientX - start.sx) / view.zoom;
        const dy = (ev.clientY - start.sy) / view.zoom;
        if (Math.abs(dx) + Math.abs(dy) > 2) start.moved = true;
        if (!start.moved) return;
        if (!start.began) { EB.beginChange(); start.began = true; }
        b.pos[0] = Math.round(start.ox + dx);
        b.pos[1] = Math.round(start.oy + dy);
        el.style.left = b.pos[0] + "px";
        el.style.top = b.pos[1] + "px";
        renderEdges();
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        if (start.began) EB.commit("move block");
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    el.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      if (b.type !== "merge") EB.emit("open-timeline", b.id);
    });

    return el;
  }

  function isEndBlock(b) {
    if (b.type === "choice") return false;
    return !EB.project.edges.some(e => e.from === b.id);
  }

  function countIn(id) {
    let n = EB.project.edges.filter(e => e.to === id).length;
    for (const b of EB.project.blocks) {
      if (b.type === "choice") {
        n += (b.branches || []).filter(br => br.to === id).length;
        if (b.timeout && b.timeout.to === id) n++;
      }
    }
    return n;
  }

  /* ── edges ── */
  function portAnchor(blockId, branchId, isTimeout) {
    const el = blocksLayer.querySelector(`.block[data-id="${blockId}"]`);
    const b = EB.getBlock(blockId);
    if (!el || !b) return null;
    // Port circles are 14px wide with their right edge 8px outside the block
    // (CSS right:-8px), so the port CENTER sits at block width + 1.
    if (branchId || isTimeout) {
      const sel = isTimeout ? '[data-timeout-port]' : `[data-branch-port="${branchId}"]`;
      const port = el.querySelector(sel);
      if (!port) return null;
      const row = port.closest(".b-branch");
      return [b.pos[0] + el.offsetWidth + 1,
              b.pos[1] + row.offsetTop + row.offsetHeight / 2];
    }
    return [b.pos[0] + el.offsetWidth + 1, b.pos[1] + el.offsetHeight / 2];
  }

  function inAnchor(blockId) {
    const el = blocksLayer.querySelector(`.block[data-id="${blockId}"]`);
    const b = EB.getBlock(blockId);
    if (!el || !b) return null;
    // .port.in is left:-8px, 14px wide -> center at -1
    return [b.pos[0] - 1, b.pos[1] + el.offsetHeight / 2];
  }

  function bezier(a, z) {
    const dx = Math.max(40, Math.abs(z[0] - a[0]) * 0.45);
    return `M ${a[0]} ${a[1]} C ${a[0] + dx} ${a[1]}, ${z[0] - dx} ${z[1]}, ${z[0]} ${z[1]}`;
  }

  function renderEdges() {
    // keep ghost if present
    const ghost = svg.querySelector(".edge-ghost");
    svg.innerHTML = "";
    if (ghost) svg.appendChild(ghost);

    const sel = EB.runtime.selection;
    const mk = (from, to, cls, selKey) => {
      if (!from || !to) return;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("class", "edge " + cls +
        (sel && sel.kind === "edge" && sel.id === selKey ? " selected" : ""));
      path.setAttribute("d", bezier(from, to));
      path.addEventListener("mousedown", (e) => {
        e.stopPropagation();
        EB.select({ kind: "edge", id: selKey });
      });
      svg.appendChild(path);
    };

    for (const e of EB.project.edges) {
      mk(portAnchor(e.from), inAnchor(e.to), "flow", "flow:" + e.from + ">" + e.to);
    }
    for (const b of EB.project.blocks) {
      if (b.type !== "choice") continue;
      for (const br of (b.branches || [])) {
        if (br.to) mk(portAnchor(b.id, br.id), inAnchor(br.to), "branch", "branch:" + b.id + ":" + br.id);
      }
      if (b.timeout && b.timeout.to) {
        mk(portAnchor(b.id, null, true), inAnchor(b.timeout.to), "timeout", "timeout:" + b.id);
      }
    }
  }

  /* delete current selection (called from app.js on Del) */
  EB.deleteSelection = function () {
    const sel = EB.runtime.selection;
    if (!sel) return;
    if (sel.kind === "block") {
      EB.change("delete block", () => EB.deleteBlock(sel.id));
      EB.select(null);
    } else if (sel.kind === "edge") {
      const [kind, a, b2] = sel.id.split(":");
      EB.change("delete edge", () => {
        if (kind === "flow") {
          const [from, to] = a.split(">");
          EB.project.edges = EB.project.edges.filter(e => !(e.from === from && e.to === to));
        } else if (kind === "branch") {
          const blk = EB.getBlock(a);
          const br = blk && (blk.branches || []).find(x => x.id === b2);
          if (br) br.to = null;
        } else if (kind === "timeout") {
          const blk = EB.getBlock(a);
          if (blk && blk.timeout) blk.timeout.to = null;
        }
      });
      EB.select(null);
    } else if (sel.kind === "window") {
      const f = EB.findWindow(sel.id);
      if (f) {
        EB.change("delete window", () => {
          f.block.windows = f.block.windows.filter(w => w.id !== sel.id);
          if (f.block.type === "choice") {
            f.block.branches = (f.block.branches || []).filter(br => br.window !== sel.id);
          }
        });
        EB.select({ kind: "block", id: f.block.id });
      }
    }
  };

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  EB.escapeHtml = escapeHtml;

  /* the flow edge is one-out for playback/merge; ghost + edge classes above */
  EB.on("project-changed", render);
  EB.on("selection-changed", render);
  EB.on("thumb-ready", render);
  EB.on("assets-changed", render);
  document.addEventListener("DOMContentLoaded", () => { applyView(); render(); });
})();
