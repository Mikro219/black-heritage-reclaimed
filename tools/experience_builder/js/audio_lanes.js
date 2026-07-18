/* Audio lanes on the block timeline (layered stem audio, July 2026).
 *
 * Four lanes — VO / MUSIC / AMB / SFX — under the interaction-window track
 * show the selected block's `audio` clips as colored bars: drag to move
 * `at_s`, drag the right edge to resize `duration_s`, double-click for the
 * clip editor modal (gain, fades, role, source offset, continues, preview
 * mute). New clips arrive by dragging a sound from the library onto a lane;
 * the lane sets the clip's role. Exported clips become the runtime's
 * per-shot `audio_events` (music/ambience sustain through HOLDs; sfx are
 * one-shots; vo plays once on the dedicated voice channel and a `continues`
 * vo clip never restarts the line).
 */
(function () {
  const $ = (id) => document.getElementById(id);
  const ROLES = ["vo", "music", "ambience", "sfx"];
  const BED_ROLES = ["music", "ambience"];

  /* Whole-lane mutes — click a lane label to silence that role in the block
   * player and the preview flow. Monitoring state only (localStorage, not
   * part of the project). */
  const LS_MUTES = "bhr_lane_mutes";
  let laneMutes;
  try { laneMutes = new Set(JSON.parse(localStorage.getItem(LS_MUTES) || "[]")); }
  catch (e) { laneMutes = new Set(); }
  EB.laneMuted = (role) => laneMutes.has(role);

  function toggleLaneMute(role) {
    laneMutes.has(role) ? laneMutes.delete(role) : laneMutes.add(role);
    try { localStorage.setItem(LS_MUTES, JSON.stringify([...laneMutes])); }
    catch (e) { /* private mode */ }
    applyLaneMuteClasses();
    EB.emit("lane-mutes-changed");
  }

  function applyLaneMuteClasses() {
    for (const lane of document.querySelectorAll(".tl-alane")) {
      lane.classList.toggle("muted", laneMutes.has(lane.dataset.role));
    }
  }

  let blockId = null;

  function block() { return blockId ? EB.getBlock(blockId) : null; }

  EB.on("tl-rendered", (id) => { blockId = id; renderLanes(); });
  EB.on("project-changed", () => { if (blockId && !EB.getBlock(blockId)) blockId = null; });
  EB.on("edit-audio-clip", ({ blockId: bid, clipId }) => {
    const b = EB.getBlock(bid);
    const c = b && (b.audio || []).find(x => x.id === clipId);
    if (b && c) { blockId = bid; openModal(b, c); }
  });

  function clipTimes(b, c) {
    const len = EB.blockLen(b);
    const start = Math.max(0, Math.min(len, c.at_s || 0));
    const end = c.duration_s == null ? len : Math.min(len, start + c.duration_s);
    return [start, Math.max(start + 0.02, end)];
  }

  function renderLanes() {
    const b = block();
    for (const lane of document.querySelectorAll(".tl-alane")) {
      const track = lane.querySelector(".alane-track");
      track.innerHTML = "";
      if (!b) continue;
      const role = lane.dataset.role;
      const len = EB.blockLen(b);
      const W = track.clientWidth;
      if (len <= 0 || W <= 0) continue;
      for (const c of (b.audio || [])) {
        if ((c.role || "sfx") !== role) continue;
        track.appendChild(clipEl(b, c, len, W));
      }
    }
  }

  function clipEl(b, c, len, W) {
    const [s, e] = clipTimes(b, c);
    const snd = EB.getSound(c.sound);
    const el = document.createElement("div");
    el.className = `tl-aclip ${c.role || "sfx"}` + (c.muted ? " muted" : "");
    el.style.left = (s / len) * W + "px";
    el.style.width = Math.max(14, ((e - s) / len) * W) + "px";
    el.innerHTML = `${c.continues ? '<span class="a-cont">→</span>' : ""}` +
      `<span class="a-name"></span><span class="a-resize"></span>`;
    el.querySelector(".a-name").textContent = snd ? snd.name : "(missing sound)";
    el.title = `${snd ? snd.name : c.sound} · ${c.role} · gain ${c.gain != null ? c.gain : 1}` +
      ` · ${EB.fmtTime(s)} → ${c.duration_s == null ? "end" : EB.fmtTime(e)}` +
      (c.continues ? " · continues previous block's bed" : "") +
      "\nDrag to move, edge to resize, double-click to edit";

    // drag = move at_s
    el.addEventListener("mousedown", (ev) => {
      if (ev.target.classList.contains("a-resize")) return;
      ev.stopPropagation();
      const track = el.parentElement;
      const start = { sx: ev.clientX, s0: s, began: false };
      const onMove = (mv) => {
        const dt = ((mv.clientX - start.sx) / W) * len;
        if (Math.abs(mv.clientX - start.sx) < 3 && !start.began) return;
        if (!start.began) { EB.beginChange(); start.began = true; }
        let ns = Math.max(0, Math.min(len - 0.05, start.s0 + dt));
        if (c.duration_s != null) ns = Math.min(ns, Math.max(0, len - Math.min(c.duration_s, len)));
        c.at_s = Math.round(ns * 100) / 100;
        renderLanes();
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        if (start.began) EB.commit("move audio clip");
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    // right edge = resize duration
    el.querySelector(".a-resize").addEventListener("mousedown", (ev) => {
      ev.stopPropagation();
      const start = { sx: ev.clientX, e0: e, began: false };
      const onMove = (mv) => {
        const dt = ((mv.clientX - start.sx) / W) * len;
        if (Math.abs(mv.clientX - start.sx) < 3 && !start.began) return;
        if (!start.began) { EB.beginChange(); start.began = true; }
        const ne = Math.max((c.at_s || 0) + 0.05, Math.min(len, start.e0 + dt));
        c.duration_s = (len - ne) < 0.06 ? null
          : Math.round((ne - (c.at_s || 0)) * 100) / 100;
        renderLanes();
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        if (start.began) EB.commit("resize audio clip");
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    el.addEventListener("dblclick", (ev) => { ev.stopPropagation(); openModal(b, c); });
    return el;
  }

  /* ── drop a library sound onto a lane ── */
  document.addEventListener("DOMContentLoaded", () => {
    // lane label = mute toggle for the whole role
    for (const label of document.querySelectorAll(".tl-alane .alane-label")) {
      const role = label.closest(".tl-alane").dataset.role;
      label.title = `Click to mute/unmute every ${role.toUpperCase()} clip ` +
        "(block player + preview flow)";
      label.style.cursor = "pointer";
      label.addEventListener("click", () => toggleLaneMute(role));
    }
    applyLaneMuteClasses();

    for (const lane of document.querySelectorAll(".tl-alane .alane-track")) {
      lane.addEventListener("dragover", (e) => {
        if ([...e.dataTransfer.types].includes("application/x-bhr-sound")) {
          e.preventDefault();
          lane.classList.add("dragover");
        }
      });
      lane.addEventListener("dragleave", () => lane.classList.remove("dragover"));
      lane.addEventListener("drop", (e) => {
        lane.classList.remove("dragover");
        const sid = e.dataTransfer.getData("application/x-bhr-sound");
        const b = block();
        if (!sid || !b) return;
        e.preventDefault();
        const role = lane.closest(".tl-alane").dataset.role;
        const r = lane.getBoundingClientRect();
        const at = Math.max(0, ((e.clientX - r.left) / r.width) * EB.blockLen(b));
        EB.change("add audio clip", () => {
          b.audio = b.audio || [];
          b.audio.push({
            id: EB.uid("a"), sound: sid, role,
            at_s: Math.round(at * 100) / 100,
            duration_s: null,
            source_offset_s: 0, gain: 1.0,
            fade_in_ms: 0, fade_out_ms: BED_ROLES.includes(role) ? 600 : 0,
            sustain: BED_ROLES.includes(role), continues: false,
          });
        });
        renderLanes();
      });
    }
    wireModal();
  });

  /* ── clip editor modal ── */
  let modalCtx = null;   // { block, clip, before }

  function openModal(b, c) {
    modalCtx = { block: b, clip: c, before: JSON.stringify(c) };
    const snd = EB.getSound(c.sound);
    $("ac-title").textContent = snd ? snd.name : "Audio clip";
    $("ac-sub").textContent = b.name || "";
    $("ac-role").value = c.role || "sfx";
    $("ac-gain").value = c.gain != null ? c.gain : 1;
    $("ac-gain-val").textContent = (+$("ac-gain").value).toFixed(2);
    $("ac-at").value = EB.fmtTime(c.at_s || 0);
    $("ac-dur").value = c.duration_s == null ? "until end" : String(c.duration_s);
    $("ac-offset").value = String(c.source_offset_s || 0);
    $("ac-fadein").value = String(c.fade_in_ms || 0);
    $("ac-fadeout").value = String(c.fade_out_ms || 0);
    $("ac-continues").checked = !!c.continues;
    $("ac-muted").checked = !!c.muted;
    $("ac-modal").classList.add("open");
  }

  function closeModal() {
    $("ac-modal").classList.remove("open");
    modalCtx = null;
  }

  function wireModal() {
    $("ac-gain").addEventListener("input", () => {
      $("ac-gain-val").textContent = (+$("ac-gain").value).toFixed(2);
    });
    $("ac-close").addEventListener("click", closeModal);
    $("ac-save").addEventListener("click", () => {
      if (!modalCtx) return closeModal();
      const c = modalCtx.clip;
      EB.beginChange();
      c.role = $("ac-role").value;
      c.sustain = BED_ROLES.includes(c.role);
      c.gain = Math.round(+$("ac-gain").value * 100) / 100;
      const at = EB.parseTime($("ac-at").value);
      if (at != null) c.at_s = Math.round(at * 100) / 100;
      const durRaw = $("ac-dur").value.trim();
      c.duration_s = (/^(until end|end)?$/i.test(durRaw)) ? null
        : Math.max(0.05, parseFloat(durRaw) || 0.05);
      c.source_offset_s = Math.max(0, parseFloat($("ac-offset").value) || 0);
      c.fade_in_ms = Math.max(0, parseInt($("ac-fadein").value, 10) || 0);
      c.fade_out_ms = Math.max(0, parseInt($("ac-fadeout").value, 10) || 0);
      c.continues = $("ac-continues").checked;
      c.muted = $("ac-muted").checked || undefined;
      if (!c.muted) delete c.muted;
      EB.commit("edit audio clip");
      renderLanes();
      closeModal();
    });
    $("ac-delete").addEventListener("click", () => {
      if (!modalCtx) return closeModal();
      const { block: b, clip: c } = modalCtx;
      EB.change("delete audio clip", () => {
        b.audio = (b.audio || []).filter(x => x.id !== c.id);
        if (!b.audio.length) delete b.audio;
      });
      renderLanes();
      closeModal();
    });
    $("ac-modal").addEventListener("mousedown", (e) => {
      if (e.target === $("ac-modal")) closeModal();
    });
  }
})();
