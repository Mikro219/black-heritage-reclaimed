/* Bottom timeline strip for the selected playback/choice block.
 *
 * Row 1 (#tl-source): the FULL source video with a draggable range brush —
 *   drag the edges to trim the block's slice, drag the middle to move it.
 * Row 2 (#tl-track): block-local time 0..len with ruler, playhead, and the
 *   block's interaction windows as draggable / right-edge-resizable bars.
 *
 * A single shared <video> (EB.tlVideo) is the scrub engine; the interaction
 * modal and block thumbnails reuse the same object URLs. */
(function () {
  const $ = (id) => document.getElementById(id);

  // Scrub preview — a picture-in-picture panel in the canvas corner, above
  // the zoom controls.
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.title = "Scrub preview of the selected block";
  Object.assign(video.style, {
    position: "absolute", right: "14px", bottom: "58px",
    width: "min(340px, 30vw)", aspectRatio: "16 / 9",
    borderRadius: "10px", border: "1px solid var(--border-2)", zIndex: 11,
    background: "#000", display: "none",
    boxShadow: "0 18px 44px -16px rgba(0,0,0,.85)",
  });
  EB.tlVideo = video;

  // Subtitle overlay laid exactly over the scrub preview, so playing the block
  // back shows how each caption reads against the video — a timing check that
  // mirrors the runtime's on-frame placement (bottom band, or the authored
  // screen-space rect). The preview box is 16/9 like the 1920×1080 source, so
  // rect fractions map straight onto it.
  const capOverlay = document.createElement("div");
  Object.assign(capOverlay.style, {
    position: "absolute", right: "14px", bottom: "58px",
    width: "min(340px, 30vw)", aspectRatio: "16 / 9",
    borderRadius: "10px", overflow: "hidden", zIndex: 12,
    pointerEvents: "none", display: "none",
  });
  capOverlay.id = "tl-cap-overlay";
  EB.tlCapOverlay = capOverlay;

  let blockId = null;       // block being edited
  let playing = false;
  let rafId = null;

  // Horizontal zoom: the shared scroller's content is `viewport × zoom` wide.
  // Every time-aligned row (ruler, window track, audio lanes, caption lane)
  // positions its bars against this content width via EB.tlContentW().
  let zoom = 1;
  const ZMIN = 1, ZMAX = 24;
  function hscroll() { return $("tl-hscroll"); }
  function contentWidth() {
    const sc = hscroll();
    const vw = sc ? sc.clientWidth : 0;
    return Math.max(0, Math.round(vw * zoom));
  }
  EB.tlContentW = contentWidth;

  document.addEventListener("DOMContentLoaded", () => {
    $("canvas-wrap").appendChild(video);
    $("canvas-wrap").appendChild(capOverlay);
    wire();
  });

  /* ── subtitle preview overlay ── */
  function activeCaption(b, t) {
    for (const c of (b.captions || [])) {
      const at = c.at_s || 0;
      const dur = Math.max(0.1, c.duration_s || 2);
      if (at <= t && t < at + dur && (c.text || "").trim()) return c;
    }
    return null;
  }

  function renderCaptionOverlay(t) {
    const b = block();
    if (!b || video.style.display === "none") { capOverlay.style.display = "none"; return; }
    const c = activeCaption(b, t);
    if (!c) { capOverlay.style.display = "none"; return; }
    capOverlay.style.display = "block";
    capOverlay.innerHTML = "";
    const card = document.createElement("div");
    card.textContent = c.text;
    const h = capOverlay.clientHeight || 190;
    Object.assign(card.style, {
      position: "absolute", boxSizing: "border-box",
      padding: ".28em .5em", borderRadius: "4px",
      background: "rgba(10,8,8,.72)", color: "#f8f5ee",
      font: `600 ${Math.max(9, Math.round(h * 0.062))}px 'IBM Plex Sans', 'Segoe UI', sans-serif`,
      lineHeight: "1.25", textAlign: "center", whiteSpace: "normal",
      textShadow: "0 1px 2px rgba(0,0,0,.7)",
      boxShadow: "0 1px 3px rgba(0,0,0,.5)",
    });
    const r = c.rect;
    if (r && ["x", "y", "w", "h"].every(k => k in r)) {
      card.style.left = (r.x + r.w / 2) * 100 + "%";
      card.style.top = (r.y + r.h / 2) * 100 + "%";
      card.style.transform = "translate(-50%,-50%)";
      card.style.maxWidth = Math.max(12, r.w * 100) + "%";
    } else {
      card.style.left = "50%";
      card.style.bottom = "7%";
      card.style.transform = "translateX(-50%)";
      card.style.maxWidth = "88%";
    }
    capOverlay.appendChild(card);
  }

  function block() { return blockId ? EB.getBlock(blockId) : null; }

  function mediaURL(b) { return b && b.media ? EB.runtime.mediaURLs[b.media] || null : null; }

  /* ── open / close ── */
  function openFor(id) {
    const b = EB.getBlock(id);
    if (!b || b.type === "merge" || !b.range_s) { close(); return; }
    const sameBlock = blockId === id;
    blockId = id;
    $("tl-empty").style.display = "none";
    $("tl-body").classList.add("active");
    if (!sameBlock && hscroll()) hscroll().scrollLeft = 0;
    const url = mediaURL(b);
    if (url && video.src !== url) {
      video.src = url;
      // re-apply the block seek once the file's metadata arrives
      video.addEventListener("loadedmetadata", () => seekLocal(localTime()), { once: true });
    }
    video.style.display = url ? "block" : "none";
    if (!sameBlock) seekLocal(0);
    renderAll();
  }

  function close() {
    blockId = null;
    stop();
    $("tl-empty").style.display = "flex";
    $("tl-body").classList.remove("active");
    video.style.display = "none";
    capOverlay.style.display = "none";
  }

  EB.on("selection-changed", () => {
    const b = EB.selectedBlock();
    if (b && b.type !== "merge") openFor(b.id);
    else close();
  });
  EB.on("open-timeline", (id) => openFor(id));
  EB.on("project-changed", () => {
    if (blockId && !EB.getBlock(blockId)) return close();
    if (blockId) renderAll();
  });
  EB.on("assets-changed", () => { if (blockId) openFor(blockId); });
  EB.on("lane-mutes-changed", () => {
    // re-schedule the live mix so a lane mute applies immediately
    if (playing && EB.pvAudio) EB.pvAudio.playBlockAt(block(), localTime());
  });

  /* ── time mapping ── */
  function localTime() {
    const b = block();
    if (!b) return 0;
    return Math.max(0, Math.min(EB.blockLen(b), (video.currentTime || 0) - b.range_s[0]));
  }

  function seekLocal(t) {
    const b = block();
    if (!b) return;
    const clamped = Math.max(0, Math.min(EB.blockLen(b), t));
    if (mediaURL(b)) video.currentTime = b.range_s[0] + clamped;
    updatePlayhead(clamped);
    updateTimecode(clamped);
  }
  EB.tlSeekLocal = seekLocal;
  EB.tlLocalTime = localTime;

  /* ── transport ── */
  function play() {
    const b = block();
    if (!b) return;
    if (!mediaURL(b)) {
      EB.toast(b.media
        ? "Video file not linked — open the Media tab and click the red clip to re-link it"
        : "This block has no clip — drag one from the Media tab onto the canvas", true);
      return;
    }
    if (video.readyState < 1) {
      // metadata still loading (first play on a large file) — start when ready
      video.addEventListener("loadedmetadata", () => { seekLocal(localTime()); play(); }, { once: true });
      return;
    }
    if (localTime() >= EB.blockLen(b) - 0.02) seekLocal(0);
    // master_audio blocks play the source video's own baked mix; everything
    // else stays muted and gets the lane clips instead
    video.muted = !b.master_audio;
    video.play().catch((e) => EB.toast("Playback failed: " + e.message, true));
    playing = true;
    // schedule the block's audio clips against the local playhead
    if (EB.pvAudio) EB.pvAudio.playBlockAt(b, localTime());
    $("tl-play").innerHTML = '<span class="msr">pause</span>';
    tick();
  }
  function stop() {
    video.pause();
    video.muted = true;
    playing = false;
    if (EB.pvAudio) EB.pvAudio.stopBlock();
    cancelAnimationFrame(rafId);
    const btn = $("tl-play");
    if (btn) btn.innerHTML = '<span class="msr">play_arrow</span>';
  }
  function tick() {
    if (!playing) return;
    const b = block();
    if (!b) return stop();
    const t = localTime();
    updatePlayhead(t);
    updateTimecode(t);
    if ((video.currentTime || 0) >= b.range_s[1] - 0.015) {
      stop();
      seekLocal(EB.blockLen(b));
    } else {
      rafId = requestAnimationFrame(tick);
    }
  }
  EB.tlTogglePlay = function () { playing ? stop() : play(); };
  EB.tlStepFrame = function (dir) {
    stop();
    const fps = EB.project.fps || 30;
    seekLocal(localTime() + dir / fps);
  };

  /* ── rendering ── */
  function renderAll() {
    const b = block();
    if (!b) return;
    // Size the shared scroll content to the current zoom before laying out any
    // time-aligned rows (they read EB.tlContentW()).
    const inner = $("tl-hscroll-inner");
    if (inner) inner.style.width = contentWidth() + "px";
    const media = b.media ? EB.getMedia(b.media) : null;
    $("tl-kind-icon").textContent = b.type === "choice" ? "call_split" : "movie";
    $("tl-kind-icon").style.color = b.type === "choice" ? "var(--amber)" : "var(--sky)";
    $("tl-block-name").textContent = b.name;
    const clipEl = $("tl-clip-name");
    if (!media) {
      clipEl.textContent = "no clip";
      clipEl.style.color = "";
    } else if (!mediaURL(b)) {
      clipEl.textContent = media.name + " — NOT LINKED (click it in the Media tab)";
      clipEl.style.color = "var(--red)";
    } else {
      clipEl.textContent = media.name;
      clipEl.style.color = "";
    }
    renderSource();
    renderRuler();
    renderWindows();
    updatePlayhead(localTime());
    updateTimecode(localTime());
    EB.emit("tl-rendered", b.id);
  }

  /* source brush */
  function renderSource() {
    const b = block();
    const media = b && b.media ? EB.getMedia(b.media) : null;
    const box = $("tl-source");
    const range = $("tl-src-range");
    const others = $("tl-src-others");
    others.innerHTML = "";
    if (!media || !media.duration_s) {
      range.style.display = "none";
      $("tl-src-label").textContent = "no source clip";
      return;
    }
    range.style.display = "block";
    const W = box.clientWidth;
    const px = (t) => (t / media.duration_s) * W;
    range.style.left = px(b.range_s[0]) + "px";
    range.style.width = Math.max(6, px(b.range_s[1]) - px(b.range_s[0])) + "px";
    $("tl-src-label").textContent =
      `${EB.fmtTime(b.range_s[0], false)} – ${EB.fmtTime(b.range_s[1], false)}  of  ${EB.fmtTime(media.duration_s, false)}`;

    // other blocks using the same media, for context
    for (const ob of EB.project.blocks) {
      if (ob.id === b.id || ob.media !== b.media || !ob.range_s) continue;
      const d = document.createElement("div");
      d.className = "other-range";
      d.style.left = px(ob.range_s[0]) + "px";
      d.style.width = Math.max(3, px(ob.range_s[1]) - px(ob.range_s[0])) + "px";
      d.title = ob.name;
      others.appendChild(d);
    }
  }

  /* ruler + track */
  function renderRuler() {
    const b = block();
    const ruler = $("tl-ruler");
    ruler.innerHTML = "";
    if (!b) return;
    const len = EB.blockLen(b);
    const W = contentWidth();
    if (len <= 0 || W <= 0) return;
    const targetPx = 80;
    const steps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60];
    let step = steps.find(s => (s / len) * W >= targetPx) || 60;
    for (let t = 0; t <= len + 1e-6; t += step) {
      const x = (t / len) * W;
      const tick = document.createElement("div");
      tick.className = "tick";
      tick.style.left = x + "px";
      ruler.appendChild(tick);
      const lbl = document.createElement("div");
      lbl.className = "tick-label";
      lbl.style.left = x + "px";
      lbl.textContent = EB.fmtTime(t, false);
      ruler.appendChild(lbl);
    }
  }

  function winTimes(b, w) {
    const len = EB.blockLen(b);
    const start = Math.max(0, Math.min(len, w.appears_s || 0));
    const end = w.duration_s == null ? len : Math.min(len, start + w.duration_s);
    return [start, Math.max(start + 0.05, end)];
  }

  function renderWindows() {
    const b = block();
    const track = $("tl-track");
    track.querySelectorAll(".tl-win").forEach(el => el.remove());
    if (!b) return;
    const len = EB.blockLen(b);
    const W = contentWidth();
    if (len <= 0 || W <= 0) return;
    const sel = EB.runtime.selection;

    for (const w of (b.windows || [])) {
      const [s, e] = winTimes(b, w);
      const el = document.createElement("div");
      const isBranch = b.type === "choice" && (b.branches || []).some(br => br.window === w.id);
      el.className = "tl-win" + (isBranch ? " branch-win" : "") +
        (sel && sel.kind === "window" && sel.id === w.id ? " selected" : "");
      el.style.left = (s / len) * W + "px";
      el.style.width = Math.max(24, ((e - s) / len) * W) + "px";
      const det = EB.detectorByType(w.detector);
      el.innerHTML = `<span class="msr">${det ? det.icon : "touch_app"}</span>
        <span class="w-name"></span><span class="w-resize"></span>`;
      el.querySelector(".w-name").textContent = w.label || w.detector;
      el.title = `${w.label || ""} · ${w.detector} · ${EB.fmtTime(s)} → ${w.duration_s == null ? "end" : EB.fmtTime(e)}`;

      // drag to move appears_s
      el.addEventListener("mousedown", (ev) => {
        if (ev.target.classList.contains("w-resize")) return;
        ev.stopPropagation();
        EB.select({ kind: "window", id: w.id });
        const start = { sx: ev.clientX, s0: s, began: false };
        const onMove = (mv) => {
          const dt = ((mv.clientX - start.sx) / W) * len;
          if (Math.abs(mv.clientX - start.sx) < 3 && !start.began) return;
          if (!start.began) { EB.beginChange(); start.began = true; }
          const dur = w.duration_s == null ? null : w.duration_s;
          let ns = Math.max(0, Math.min(len - 0.1, start.s0 + dt));
          if (dur != null) ns = Math.min(ns, len - Math.min(dur, len));
          w.appears_s = Math.round(ns * 100) / 100;
          renderWindows(); updateTimecode(localTime());
        };
        const onUp = () => {
          window.removeEventListener("mousemove", onMove);
          window.removeEventListener("mouseup", onUp);
          if (start.began) EB.commit("move window");
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
      });

      // right-edge resize = duration
      el.querySelector(".w-resize").addEventListener("mousedown", (ev) => {
        ev.stopPropagation();
        EB.select({ kind: "window", id: w.id });
        const start = { sx: ev.clientX, e0: e, began: false };
        const onMove = (mv) => {
          const dt = ((mv.clientX - start.sx) / W) * len;
          if (Math.abs(mv.clientX - start.sx) < 3 && !start.began) return;
          if (!start.began) { EB.beginChange(); start.began = true; }
          const ne = Math.max((w.appears_s || 0) + 0.1, Math.min(len, start.e0 + dt));
          // snap to "until end" when dragged to the tail
          w.duration_s = (len - ne) < 0.06 ? null : Math.round((ne - (w.appears_s || 0)) * 100) / 100;
          renderWindows();
        };
        const onUp = () => {
          window.removeEventListener("mousemove", onMove);
          window.removeEventListener("mouseup", onUp);
          if (start.began) EB.commit("resize window");
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
      });

      el.addEventListener("dblclick", (ev) => {
        ev.stopPropagation();
        EB.emit("edit-window", { blockId: b.id, winId: w.id });
      });

      el.addEventListener("contextmenu", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        EB.select({ kind: "window", id: w.id });
        EB.contextMenu(ev.clientX, ev.clientY, [
          { label: "Edit…", icon: "edit",
            onClick: () => EB.emit("edit-window", { blockId: b.id, winId: w.id }) },
          { sep: true },
          { label: "Delete window", icon: "delete", danger: true,
            onClick: () => EB.deleteSelection() },
        ]);
      });

      track.appendChild(el);
    }
  }

  function updatePlayhead(t) {
    const b = block();
    if (!b) return;
    const len = EB.blockLen(b);
    const W = contentWidth();
    $("tl-playhead").style.left = (len > 0 ? (t / len) * W : 0) + "px";
    keepPlayheadInView();
    renderCaptionOverlay(t);
  }

  // Auto-scroll so the playhead stays visible while zoomed and playing.
  function keepPlayheadInView() {
    const sc = hscroll();
    if (!sc || zoom <= 1) return;
    const x = parseFloat($("tl-playhead").style.left) || 0;
    const pad = 40;
    if (x < sc.scrollLeft + pad) sc.scrollLeft = Math.max(0, x - pad);
    else if (x > sc.scrollLeft + sc.clientWidth - pad)
      sc.scrollLeft = x - sc.clientWidth + pad;
  }

  /* ── zoom ── */
  function updateZoomLabel() {
    const el = $("tl-zoom-val");
    if (el) el.textContent = Math.round(zoom * 100) + "%";
  }
  // Set zoom, keeping the time under `anchorClientX` (or the viewport centre)
  // fixed on screen.
  function setZoom(z, anchorClientX) {
    const sc = hscroll();
    z = Math.max(ZMIN, Math.min(ZMAX, z));
    if (!sc) { zoom = z; updateZoomLabel(); return; }
    const rect = sc.getBoundingClientRect();
    const ax = (anchorClientX == null ? rect.left + rect.width / 2 : anchorClientX) - rect.left;
    const before = contentWidth() || 1;
    const frac = (sc.scrollLeft + ax) / before;
    zoom = z;
    if (block()) renderAll();
    const after = contentWidth();
    sc.scrollLeft = Math.max(0, frac * after - ax);
    updateZoomLabel();
  }
  EB.tlZoom = () => zoom;

  function updateTimecode(t) {
    const b = block();
    if (!b) return;
    $("tl-timecode").innerHTML =
      `${EB.fmtTime(t)} <span class="tc-total">/ ${EB.fmtTime(EB.blockLen(b))}</span>`;
  }

  /* ── wiring ── */
  function wire() {
    // drag the strip's top edge to resize it (persisted across sessions)
    const LS_H = "bhr_tl_height";
    const app = $("app");
    const savedH = parseInt(localStorage.getItem(LS_H), 10);
    if (savedH) app.style.setProperty("--tl-h", savedH + "px");
    const resizer = $("tl-resizer");
    if (resizer) resizer.addEventListener("mousedown", (e) => {
      e.preventDefault();
      resizer.classList.add("dragging");
      const startY = e.clientY;
      const startH = $("timeline").getBoundingClientRect().height;
      const onMove = (mv) => {
        const h = Math.max(150, Math.min(window.innerHeight * 0.7,
                                         startH + (startY - mv.clientY)));
        app.style.setProperty("--tl-h", Math.round(h) + "px");
      };
      const onUp = (mv) => {
        resizer.classList.remove("dragging");
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        try {
          localStorage.setItem(LS_H,
            String(Math.round($("timeline").getBoundingClientRect().height)));
        } catch (err) { /* private mode */ }
        if (blockId) renderAll();   // lane/track heights changed
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    $("tl-play").addEventListener("click", () => EB.tlTogglePlay());
    $("tl-prev-frame").addEventListener("click", () => EB.tlStepFrame(-1));
    $("tl-next-frame").addEventListener("click", () => EB.tlStepFrame(1));

    // zoom controls
    $("tl-zoom-in").addEventListener("click", () => setZoom(zoom * 1.5));
    $("tl-zoom-out").addEventListener("click", () => setZoom(zoom / 1.5));
    $("tl-zoom-fit").addEventListener("click", () => setZoom(1));
    updateZoomLabel();
    const sc = $("tl-hscroll");
    if (sc) sc.addEventListener("wheel", (e) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        setZoom(zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15), e.clientX);
      } else if (zoom > 1 && e.deltaY && !e.shiftKey) {
        // when zoomed, a plain vertical wheel pans the time axis left/right
        e.preventDefault();
        sc.scrollLeft += e.deltaY;
      }
    }, { passive: false });
    $("tl-add-window").addEventListener("click", () => {
      const b = block();
      if (b) EB.emit("add-window", { blockId: b.id, at: localTime() });
    });

    // scrub on ruler/track click-drag
    for (const id of ["tl-ruler", "tl-track"]) {
      const el = $(id);
      el.addEventListener("mousedown", (e) => {
        if (e.target.closest(".tl-win")) return;
        stop();
        const scrub = (ev) => {
          const b = block();
          if (!b) return;
          const r = el.getBoundingClientRect();
          const frac = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
          seekLocal(frac * EB.blockLen(b));
        };
        scrub(e);
        const onUp = () => {
          window.removeEventListener("mousemove", scrub);
          window.removeEventListener("mouseup", onUp);
        };
        window.addEventListener("mousemove", scrub);
        window.addEventListener("mouseup", onUp);
      });
    }

    // source brush: move / trim
    const range = $("tl-src-range");
    const box = $("tl-source");

    function brushDrag(e, mode) {
      e.stopPropagation();
      const b = block();
      const media = b && b.media ? EB.getMedia(b.media) : null;
      if (!media || !media.duration_s) return;
      const W = box.clientWidth;
      const perPx = media.duration_s / W;
      const start = { sx: e.clientX, r0: b.range_s.slice(), began: false };
      const minLen = 0.2;
      const onMove = (mv) => {
        const dt = (mv.clientX - start.sx) * perPx;
        if (Math.abs(mv.clientX - start.sx) < 2 && !start.began) return;
        if (!start.began) { EB.beginChange(); start.began = true; }
        let [a, z] = start.r0;
        if (mode === "move") {
          const lenR = z - a;
          a = Math.max(0, Math.min(media.duration_s - lenR, a + dt));
          z = a + lenR;
        } else if (mode === "l") {
          a = Math.max(0, Math.min(z - minLen, a + dt));
        } else {
          z = Math.max(a + minLen, Math.min(media.duration_s, z + dt));
        }
        b.range_s = [Math.round(a * 100) / 100, Math.round(z * 100) / 100];
        renderSource(); renderRuler(); renderWindows();
        updateTimecode(localTime());
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        if (start.began) {
          // keep windows inside the new range
          const lenNew = EB.blockLen(b);
          for (const w of (b.windows || [])) {
            w.appears_s = Math.max(0, Math.min(lenNew - 0.1, w.appears_s || 0));
            if (w.duration_s != null) w.duration_s = Math.min(w.duration_s, lenNew - w.appears_s);
          }
          EB.commit("trim block");
          seekLocal(0);
        }
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    }

    range.addEventListener("mousedown", (e) => {
      if (e.target.classList.contains("brush-handle")) return;
      brushDrag(e, "move");
    });
    range.querySelector(".brush-handle.l").addEventListener("mousedown", (e) => brushDrag(e, "l"));
    range.querySelector(".brush-handle.r").addEventListener("mousedown", (e) => brushDrag(e, "r"));

    // NOTE: clicking the empty source bar deliberately does NOTHING — an
    // earlier build moved the block's range to the click point, which made it
    // far too easy to silently tear a block out of the sequence. Ranges move
    // only by dragging the brush itself (or its trim handles).

    window.addEventListener("resize", () => { if (blockId) renderAll(); });
  }
})();
