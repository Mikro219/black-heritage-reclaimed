/* Add / Edit Interaction Window modal (mockup 1B).
 *
 * The paused frame at the window's appears-at time is drawn on a canvas; the
 * author drags to draw the region. Regions are stored normalized (0..1) in
 * SCREEN space (what the player sees — the frame is shown unmirrored here
 * because the video IS the player view). */
(function () {
  const $ = (id) => document.getElementById(id);

  let ctx = null;        // { blockId, winId|null, at }  (winId null = adding)
  let shape = "rect";
  let region = null;     // { shape, x, y, w, h } normalized
  let drawing = null;

  /* ── open / close ── */
  EB.on("add-window", ({ blockId, at }) => open(blockId, null, at));
  EB.on("edit-window", ({ blockId, winId }) => open(blockId, winId, null));

  function open(blockId, winId, at) {
    const b = EB.getBlock(blockId);
    if (!b) return;
    const w = winId ? (b.windows || []).find(x => x.id === winId) : null;
    ctx = { blockId, winId, at: w ? (w.appears_s || 0) : (at || 0) };

    $("ix-title").textContent = w ? "Edit Interaction Window" : "Add Interaction Window";
    $("ix-save-label").textContent = w ? "Save changes" : "Add interaction";
    $("ix-delete").style.display = w ? "flex" : "none";
    $("ix-label").value = w ? (w.label || "") : "";
    $("ix-appears").value = EB.fmtTime(ctx.at);
    $("ix-duration").value = w && w.duration_s != null ? EB.fmtTime(w.duration_s) : "";
    shape = (w && w.region && w.region.shape) || "rect";
    region = w && w.region ? { ...w.region } : null;
    updateShapeButtons();

    fillDetectorSelect(w ? w.detector : "point_target_held");
    renderParams(w ? { ...(w.params || {}) } : null);
    fillGoto(b, w);
    drawFrame(b, ctx.at);
    updateRegionBox();

    $("ix-modal").classList.add("open");
    setTimeout(() => $("ix-label").focus(), 50);
  }

  function close() {
    $("ix-modal").classList.remove("open");
    ctx = null; region = null; drawing = null;
  }

  /* ── frame rendering ── */
  function drawFrame(b, atLocal) {
    const canvas = $("ix-frame-canvas");
    const c2d = canvas.getContext("2d");
    canvas.width = 960; canvas.height = 540;
    c2d.fillStyle = "#05070a";
    c2d.fillRect(0, 0, canvas.width, canvas.height);
    $("ix-frame-label").textContent = "frame @ " + EB.fmtTime(atLocal);

    const url = b.media ? EB.runtime.mediaURLs[b.media] : null;
    if (!url) {
      c2d.fillStyle = "#6b7684";
      c2d.font = "16px IBM Plex Sans, sans-serif";
      c2d.textAlign = "center";
      c2d.fillText("No clip linked — region can still be drawn", canvas.width / 2, canvas.height / 2);
      return;
    }
    const v = document.createElement("video");
    v.muted = true; v.preload = "auto";
    v.onloadedmetadata = () => {
      v.currentTime = Math.min((b.range_s ? b.range_s[0] : 0) + atLocal, Math.max(0, v.duration - 0.05));
    };
    v.onseeked = () => {
      try {
        // letterbox into 16:9 canvas
        const vw = v.videoWidth, vh = v.videoHeight;
        const scale = Math.min(canvas.width / vw, canvas.height / vh);
        const dw = vw * scale, dh = vh * scale;
        c2d.drawImage(v, (canvas.width - dw) / 2, (canvas.height - dh) / 2, dw, dh);
      } catch (e) { /* leave black */ }
      v.removeAttribute("src"); v.load();
    };
    v.src = url;
  }

  /* ── region drawing ── */
  const stage = () => $("ix-stage");

  function stageFrac(e) {
    const r = stage().getBoundingClientRect();
    return [Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
            Math.max(0, Math.min(1, (e.clientY - r.top) / r.height))];
  }

  document.addEventListener("DOMContentLoaded", () => {
    stage().addEventListener("mousedown", (e) => {
      if (!ctx) return;
      e.preventDefault();
      const [x0, y0] = stageFrac(e);
      drawing = { x0, y0 };
      const onMove = (mv) => {
        const [x1, y1] = stageFrac(mv);
        region = {
          shape,
          x: Math.min(x0, x1), y: Math.min(y0, y1),
          w: Math.abs(x1 - x0), h: Math.abs(y1 - y0),
        };
        updateRegionBox();
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        drawing = null;
        if (region && (region.w < 0.01 || region.h < 0.01)) { region = null; updateRegionBox(); }
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    /* shape buttons */
    document.querySelectorAll(".shape-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        shape = btn.dataset.shape;
        if (region) region.shape = shape;
        updateShapeButtons();
        updateRegionBox();
      });
    });

    $("ix-close").addEventListener("click", close);
    $("ix-cancel").addEventListener("click", close);
    $("ix-modal").addEventListener("mousedown", (e) => { if (e.target === $("ix-modal")) close(); });
    $("ix-save").addEventListener("click", save);
    $("ix-delete").addEventListener("click", del);
    $("ix-detector").addEventListener("change", () => renderParams(null));
  });

  function updateShapeButtons() {
    document.querySelectorAll(".shape-btn").forEach(b =>
      b.classList.toggle("active", b.dataset.shape === shape));
  }

  function updateRegionBox() {
    const el = $("ix-region");
    if (!region) { el.style.display = "none"; return; }
    el.style.display = "block";
    el.className = region.shape === "circle" ? "circle" : "";
    el.id = "ix-region";
    el.style.left = region.x * 100 + "%";
    el.style.top = region.y * 100 + "%";
    el.style.width = region.w * 100 + "%";
    el.style.height = region.h * 100 + "%";
    $("ix-region-label").textContent = $("ix-label").value || "region";
  }

  /* ── detector select + params ── */
  function fillDetectorSelect(selected) {
    const sel = $("ix-detector");
    sel.innerHTML = "";
    for (const d of EB.DETECTORS) {
      const opt = document.createElement("option");
      opt.value = d.type;
      opt.textContent = `${d.label}  (${d.type})`;
      if (d.type === selected) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.onchange = () => renderParams(null);
    updateHint();
  }

  function updateHint() {
    const d = EB.detectorByType($("ix-detector").value);
    $("ix-detector-hint").textContent = d ? d.hint : "";
    const voice = !!(d && d.voice);
    // Voice windows need a keyword, not a region — the drawn region (optional)
    // is only the preview click target.
    $("ix-shape-field").style.display = voice ? "none" : "block";
    const hint = $("ix-stage-hint");
    hint.innerHTML = voice
      ? '<span class="msr" style="color:var(--violet);">mic</span>Voice window — fires when the player says the keyword (set it under Parameters). No region needed; draw one only if you want a preview click target. On a choice block, set "go to block" to make it a spoken branch pick.'
      : '<span class="msr">info</span>Drag on the frame to draw the region the player interacts with. Player view is mirrored — draw where the player sees it.';
  }

  function renderParams(existing) {
    updateHint();
    const d = EB.detectorByType($("ix-detector").value);
    const box = $("ix-params");
    box.innerHTML = "";
    const params = { ...(d ? d.params : {}), ...(existing || {}) };
    const keys = Object.keys(params);
    $("ix-params-field").style.display = keys.length ? "block" : "none";
    for (const k of keys) {
      const row = document.createElement("div");
      row.className = "p-row";
      row.innerHTML = `<span class="p-key" title="${k}">${k}</span><input class="p-val" data-key="${k}">`;
      row.querySelector("input").value = JSON.stringify(params[k]).replace(/^"|"$/g, "");
      box.appendChild(row);
    }
  }

  function collectParams() {
    const out = {};
    $("ix-params").querySelectorAll(".p-val").forEach(inp => {
      const k = inp.dataset.key;
      const raw = inp.value.trim();
      if (raw === "") return;
      if (raw === "true") out[k] = true;
      else if (raw === "false") out[k] = false;
      else if (!isNaN(Number(raw))) out[k] = Number(raw);
      else out[k] = raw;
    });
    return out;
  }

  /* ── goto (branch target) ── */
  function fillGoto(b, w) {
    const sel = $("ix-goto");
    sel.innerHTML = "";
    const isChoice = b.type === "choice";
    $("ix-goto-field").style.display = isChoice ? "block" : "none";
    if (!isChoice) return;
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "— reaction only (no branch) —";
    sel.appendChild(none);
    for (const other of EB.project.blocks) {
      if (other.id === b.id) continue;
      const opt = document.createElement("option");
      opt.value = other.id;
      opt.textContent = other.name || other.id;
      sel.appendChild(opt);
    }
    if (w) {
      const br = (b.branches || []).find(x => x.window === w.id);
      sel.value = br && br.to ? br.to : "";
    }
  }

  /* ── save / delete ── */
  function save() {
    if (!ctx) return;
    const b = EB.getBlock(ctx.blockId);
    if (!b) return close();
    const label = $("ix-label").value.trim();
    const detector = $("ix-detector").value;
    const appears = EB.parseTime($("ix-appears").value) ?? 0;
    const durRaw = $("ix-duration").value.trim();
    const duration = durRaw === "" ? null : EB.parseTime(durRaw);
    const params = collectParams();
    const gotoId = $("ix-goto-field").style.display !== "none" ? $("ix-goto").value : "";

    const d = EB.detectorByType(detector);
    if (d && d.region === "required" && !region) {
      EB.toast(`${detector} needs a region — drag one on the frame`, true);
      return;
    }

    EB.change(ctx.winId ? "edit window" : "add window", () => {
      b.windows = b.windows || [];
      let w;
      if (ctx.winId) {
        w = b.windows.find(x => x.id === ctx.winId);
        if (!w) return;
      } else {
        w = { id: EB.uid("w") };
        b.windows.push(w);
      }
      w.label = label || (d ? d.label : detector);
      w.detector = detector;
      w.params = params;
      w.region = region ? { ...region } : null;
      w.appears_s = Math.round(appears * 100) / 100;
      w.duration_s = duration == null ? null : Math.round(duration * 100) / 100;

      if (b.type === "choice") {
        b.branches = b.branches || [];
        let br = b.branches.find(x => x.window === w.id);
        if (gotoId) {
          if (!br) { br = { id: EB.uid("br"), window: w.id }; b.branches.push(br); }
          br.to = gotoId;
          br.label = w.label;
        } else if (br) {
          b.branches = b.branches.filter(x => x !== br);
        }
      }
      ctx._saved = w.id;
    });
    EB.select({ kind: "window", id: ctx._saved });
    close();
  }

  function del() {
    if (!ctx || !ctx.winId) return;
    const b = EB.getBlock(ctx.blockId);
    if (b) {
      EB.change("delete window", () => {
        b.windows = (b.windows || []).filter(w => w.id !== ctx.winId);
        if (b.type === "choice") {
          b.branches = (b.branches || []).filter(br => br.window !== ctx.winId);
        }
      });
      EB.select({ kind: "block", id: b.id });
    }
    close();
  }

  // live label → region label
  document.addEventListener("input", (e) => {
    if (e.target && e.target.id === "ix-label") updateRegionBox();
  });

  EB.closeInteractionModal = close;
})();
