/* Captions lane + editor (subtitles, July 2026).
 *
 * A CAPTIONS lane under the audio lanes shows the selected block's `captions`
 * as bars: drag to move `at_s`, drag the right edge to resize `duration_s`,
 * double-click for the editor. "+ Caption" adds one at the start. The editor
 * carries the text and an on-frame position box you can DRAW, MOVE and RESIZE
 * (screen-space {x,y,w,h}); "Bottom band" clears the box so the runtime uses
 * its default lower-third placement. Exported to each shot's `captions`
 * metadata and drawn by RenderEngine._draw_captions against the shot playhead.
 */
(function () {
  const $ = (id) => document.getElementById(id);
  let blockId = null;

  function block() { return blockId ? EB.getBlock(blockId) : null; }

  EB.on("tl-rendered", (id) => { blockId = id; renderLane(); });
  EB.on("project-changed", () => { if (blockId && !EB.getBlock(blockId)) blockId = null; });

  function capTimes(b, c) {
    const len = EB.blockLen(b);
    const start = Math.max(0, Math.min(len, c.at_s || 0));
    const end = Math.min(len, start + (c.duration_s || 2));
    return [start, Math.max(start + 0.02, end)];
  }

  // [{s,e}] for every caption except `c`, so drags can clamp against them.
  function otherIntervals(b, c) {
    return (b.captions || [])
      .filter(x => x !== c)
      .map(x => { const [s, e] = capTimes(b, x); return { s, e }; });
  }

  function renderLane() {
    const lane = $("cap-lane");
    if (!lane) return;
    const track = lane.querySelector(".clane-track");
    track.innerHTML = "";
    const b = block();
    if (!b) return;
    const len = EB.blockLen(b);
    const W = EB.tlContentW ? EB.tlContentW() : track.clientWidth;
    if (len <= 0 || W <= 0) return;
    for (const c of (b.captions || [])) track.appendChild(capEl(b, c, len, W));
  }

  function capEl(b, c, len, W) {
    const [s, e] = capTimes(b, c);
    const el = document.createElement("div");
    el.className = "tl-cap" + (c.rect ? " placed" : "");
    el.style.left = (s / len) * W + "px";
    el.style.width = Math.max(16, ((e - s) / len) * W) + "px";
    el.innerHTML = `<span class="c-name"></span><span class="c-resize"></span>`;
    el.querySelector(".c-name").textContent = c.text || "(empty)";
    el.title = `"${c.text || ""}" · ${EB.fmtTime(s)} → ${EB.fmtTime(e)}` +
      (c.rect ? " · positioned" : " · bottom band") +
      "\nDrag to move, edge to resize, double-click to edit";

    el.addEventListener("mousedown", (ev) => {
      if (ev.target.classList.contains("c-resize")) return;
      ev.stopPropagation();
      const Wd = EB.tlContentW ? EB.tlContentW() : W;
      const dur = Math.min(c.duration_s || 2, len);
      // Classify the other captions as left/right of THIS one (by its start at
      // drag begin) so the clamp is stable: the bar sticks to the nearest
      // neighbour on each side and can never overlap it.
      let leftLim = 0, rightLim = len;
      for (const o of otherIntervals(b, c)) {
        if (o.e <= s + 1e-4) leftLim = Math.max(leftLim, o.e);
        else if (o.s >= s + dur - 1e-4) rightLim = Math.min(rightLim, o.s);
      }
      const start = { sx: ev.clientX, s0: s, began: false };
      const onMove = (mv) => {
        if (Math.abs(mv.clientX - start.sx) < 3 && !start.began) return;
        if (!start.began) { EB.beginChange(); start.began = true; }
        const dt = ((mv.clientX - start.sx) / Wd) * len;
        let ns = start.s0 + dt;
        ns = Math.max(leftLim, Math.min(rightLim - dur, ns));
        ns = Math.max(0, Math.min(len - dur, ns));
        c.at_s = Math.round(ns * 100) / 100;
        renderLane();
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        if (start.began) EB.commit("move caption");
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    el.querySelector(".c-resize").addEventListener("mousedown", (ev) => {
      ev.stopPropagation();
      const Wd = EB.tlContentW ? EB.tlContentW() : W;
      const at = c.at_s || 0;
      // Can't grow past the next caption's start.
      let rightLim = len;
      for (const o of otherIntervals(b, c)) if (o.s >= at + 1e-4) rightLim = Math.min(rightLim, o.s);
      const start = { sx: ev.clientX, e0: e, began: false };
      const onMove = (mv) => {
        if (Math.abs(mv.clientX - start.sx) < 3 && !start.began) return;
        if (!start.began) { EB.beginChange(); start.began = true; }
        const dt = ((mv.clientX - start.sx) / Wd) * len;
        const ne = Math.max(at + 0.3, Math.min(rightLim, start.e0 + dt));
        c.duration_s = Math.round((ne - at) * 100) / 100;
        renderLane();
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        if (start.began) EB.commit("resize caption");
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    el.addEventListener("dblclick", (ev) => { ev.stopPropagation(); openModal(b, c); });

    el.addEventListener("contextmenu", (ev) => {
      ev.preventDefault(); ev.stopPropagation();
      EB.contextMenu(ev.clientX, ev.clientY, [
        { label: "Edit…", icon: "edit", onClick: () => openModal(b, c) },
        { sep: true },
        { label: "Delete caption", icon: "delete", danger: true, onClick: () => {
          EB.change("delete caption", () => {
            b.captions = (b.captions || []).filter(x => x.id !== c.id);
            if (!b.captions.length) delete b.captions;
          });
          renderLane();
        } },
      ]);
    });
    return el;
  }

  /* ── editor modal ── */
  let mctx = null;       // { block, cap, rect }
  let rect = null;       // working {x,y,w,h} or null

  function openModal(b, c) {
    mctx = { block: b, cap: c };
    rect = c.rect ? { ...c.rect } : null;
    $("cap-text").value = c.text || "";
    $("cap-at").value = EB.fmtTime(c.at_s || 0);
    $("cap-dur").value = String(c.duration_s || 2);
    $("cap-sub").textContent = b.name || "";
    drawFrame(b, c.at_s || 0);
    updateBox();
    $("cap-modal").classList.add("open");
    setTimeout(() => $("cap-text").focus(), 50);
  }
  function closeModal() { $("cap-modal").classList.remove("open"); mctx = null; rect = null; }

  function drawFrame(b, atLocal) {
    const canvas = $("cap-frame-canvas");
    const c2d = canvas.getContext("2d");
    canvas.width = 960; canvas.height = 540;
    c2d.fillStyle = "#05070a";
    c2d.fillRect(0, 0, canvas.width, canvas.height);
    $("cap-frame-label").textContent = "frame @ " + EB.fmtTime(atLocal);
    const url = b.media ? EB.runtime.mediaURLs[b.media] : null;
    if (!url) return;
    const v = document.createElement("video");
    v.muted = true; v.preload = "auto";
    v.onloadedmetadata = () => {
      v.currentTime = Math.min((b.range_s ? b.range_s[0] : 0) + atLocal,
                               Math.max(0, v.duration - 0.05));
    };
    v.onseeked = () => {
      try {
        const scale = Math.min(canvas.width / v.videoWidth, canvas.height / v.videoHeight);
        const dw = v.videoWidth * scale, dh = v.videoHeight * scale;
        c2d.drawImage(v, (canvas.width - dw) / 2, (canvas.height - dh) / 2, dw, dh);
      } catch (e) { /* leave black */ }
      v.removeAttribute("src"); v.load();
    };
    v.src = url;
  }

  const stage = () => $("cap-stage");
  function frac(e) {
    const r = stage().getBoundingClientRect();
    return [Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
            Math.max(0, Math.min(1, (e.clientY - r.top) / r.height))];
  }

  function updateBox() {
    const el = $("cap-region");
    if (!rect) { el.style.display = "none"; $("cap-pos-label").textContent = "Bottom band (default)"; return; }
    el.style.display = "block";
    el.style.left = rect.x * 100 + "%";
    el.style.top = rect.y * 100 + "%";
    el.style.width = rect.w * 100 + "%";
    el.style.height = rect.h * 100 + "%";
    $("cap-region-text").textContent = $("cap-text").value || "caption";
    $("cap-pos-label").textContent =
      `x ${rect.x.toFixed(2)} y ${rect.y.toFixed(2)} · ${rect.w.toFixed(2)}×${rect.h.toFixed(2)}`;
  }

  function wire() {
    // draw-create a box on empty stage; move by dragging body; resize by handle
    stage().addEventListener("mousedown", (e) => {
      if (!mctx) return;
      if (e.target.id === "cap-region" || e.target.classList.contains("cap-rz")) return;
      e.preventDefault();
      const [x0, y0] = frac(e);
      const onMove = (mv) => {
        const [x1, y1] = frac(mv);
        rect = { x: Math.min(x0, x1), y: Math.min(y0, y1),
                 w: Math.abs(x1 - x0), h: Math.abs(y1 - y0) };
        updateBox();
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        if (rect && (rect.w < 0.03 || rect.h < 0.03)) { rect = null; updateBox(); }
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    $("cap-region").addEventListener("mousedown", (e) => {
      if (!rect || e.target.classList.contains("cap-rz")) return;
      e.preventDefault(); e.stopPropagation();
      const [px, py] = frac(e), r0 = { ...rect };
      const onMove = (mv) => {
        const [x1, y1] = frac(mv);
        rect.x = Math.max(0, Math.min(1 - r0.w, r0.x + (x1 - px)));
        rect.y = Math.max(0, Math.min(1 - r0.h, r0.y + (y1 - py)));
        updateBox();
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    $("cap-region").querySelector(".cap-rz").addEventListener("mousedown", (e) => {
      if (!rect) return;
      e.preventDefault(); e.stopPropagation();
      const onMove = (mv) => {
        const [x1, y1] = frac(mv);
        rect.w = Math.max(0.05, Math.min(1 - rect.x, x1 - rect.x));
        rect.h = Math.max(0.04, Math.min(1 - rect.y, y1 - rect.y));
        updateBox();
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    $("cap-text").addEventListener("input", updateBox);
    $("cap-default").addEventListener("click", () => { rect = null; updateBox(); });
    $("cap-close").addEventListener("click", closeModal);
    $("cap-cancel").addEventListener("click", closeModal);
    $("cap-modal").addEventListener("mousedown", (e) => { if (e.target === $("cap-modal")) closeModal(); });
    $("cap-save").addEventListener("click", save);
    $("cap-delete").addEventListener("click", del);
    $("cap-add").addEventListener("click", addCaption);
  }

  function save() {
    if (!mctx) return closeModal();
    const c = mctx.cap;
    EB.beginChange();
    c.text = $("cap-text").value.trim();
    const at = EB.parseTime($("cap-at").value);
    if (at != null) c.at_s = Math.round(at * 100) / 100;
    const dur = parseFloat($("cap-dur").value);
    c.duration_s = isFinite(dur) && dur > 0 ? Math.round(dur * 100) / 100 : 2;
    if (rect) c.rect = { x: +rect.x.toFixed(4), y: +rect.y.toFixed(4),
                         w: +rect.w.toFixed(4), h: +rect.h.toFixed(4) };
    else delete c.rect;
    EB.commit("edit caption");
    renderLane();
    closeModal();
  }

  function del() {
    if (!mctx) return closeModal();
    const { block: b, cap: c } = mctx;
    EB.change("delete caption", () => {
      b.captions = (b.captions || []).filter(x => x.id !== c.id);
      if (!b.captions.length) delete b.captions;
    });
    renderLane();
    closeModal();
  }

  function addCaption() {
    const b = block();
    if (!b) return;
    let cap;
    EB.change("add caption", () => {
      b.captions = b.captions || [];
      cap = { id: EB.uid("cap"), at_s: 0, duration_s: 3, text: "New caption" };
      b.captions.push(cap);
    });
    renderLane();
    openModal(b, cap);
  }

  document.addEventListener("DOMContentLoaded", wire);
})();
