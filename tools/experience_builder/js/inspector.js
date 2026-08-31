/* Right-hand inspector for the current selection (mockup 1A right column). */
(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => EB.escapeHtml(s == null ? "" : s);

  function render() {
    const box = $("inspector");
    const b = EB.selectedBlock();
    if (!b) {
      box.innerHTML = `<div class="insp-empty">Select a block to edit its clip range, interaction windows and branches.<br><br>
        <b>Shortcuts</b><br>
        Space — play/pause timeline<br>
        ←/→ — step one frame<br>
        Del — delete selection<br>
        Ctrl+Z / Ctrl+Y — undo / redo<br>
        Ctrl+S — save project</div>`;
      return;
    }

    const media = b.media ? EB.getMedia(b.media) : null;
    const kindLabel = b.type === "playback" ? "Playback Block" : b.type === "choice" ? "Choice Block" : "Merge";
    const icon = b.type === "playback" ? "movie" : b.type === "choice" ? "call_split" : "call_merge";
    const isStart = EB.project.start === b.id;
    const sel = EB.runtime.selection;

    let html = `
      <div class="insp-head ${b.type}">
        <span class="msr">${icon}</span>
        <span class="ih-kind">${kindLabel}</span>
        <button class="ih-del" id="insp-del" title="Delete block"><span class="msr">delete</span></button>
      </div>
      <div class="insp-sec">
        <div class="insp-field">
          <label>Block name</label>
          <input type="text" id="insp-name" value="${esc(b.name)}" spellcheck="false">
        </div>`;
    if (!isStart) {
      html += `<button class="tb-btn" id="insp-set-start" style="width:100%;justify-content:center;">
        <span class="msr" style="font-size:16px;color:var(--green);">flag</span>Set as start block</button>`;
    } else {
      html += `<div class="insp-static"><span class="msr" style="color:var(--green);">flag</span>This is the start block</div>`;
    }
    html += `</div>`;

    if (b.type !== "merge") {
      html += `
      <div class="insp-sec">
        <div class="insp-sec-head"><span>Source clip</span></div>
        <div class="insp-static">
          <span class="msr">movie</span>
          <span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${media ? esc(media.name) : "none — drop a clip from Media onto the canvas"}</span>
        </div>
        <div style="margin-top:6px;font-family:var(--mono);font-size:11px;color:var(--text-mute);">
          ${b.range_s ? `${EB.fmtTime(b.range_s[0])} → ${EB.fmtTime(b.range_s[1])} · ${EB.fmtTime(EB.blockLen(b))}` : ""}
          ${b.type === "choice" ? " · pauses at end" : ""}
        </div>
      </div>`;
    }

    if (b.type === "choice") {
      html += `<div class="insp-sec">
        <div class="insp-sec-head"><span>Branches · ${(b.branches || []).length}</span></div>`;
      if (!(b.branches || []).length) {
        html += `<div style="font-size:11.5px;color:var(--text-mute);line-height:1.5;">No branches yet — add an interaction window and pick its "go to block" target.</div>`;
      }
      for (const br of (b.branches || [])) {
        html += `<div class="insp-list-item branch" data-branch="${br.id}">
          <span class="msr kind">subdirectory_arrow_right</span>
          <span class="w-label">${esc(br.label || "branch")}</span>
          <span class="br-arrow msr">arrow_forward</span>
          <select class="br-to" data-branch="${br.id}">
            <option value="">unlinked</option>
            ${EB.project.blocks.filter(x => x.id !== b.id).map(x =>
              `<option value="${x.id}" ${br.to === x.id ? "selected" : ""}>${esc(x.name)}</option>`).join("")}
          </select>
        </div>`;
      }
      const tt = b.timeout || { seconds: 30, to: null };
      html += `
        <div class="insp-row" style="margin-top:8px;">
          <div class="insp-field">
            <label>Timeout (s)</label>
            <input type="number" id="insp-timeout-s" min="1" step="1" value="${tt.seconds != null ? tt.seconds : 30}">
          </div>
          <div class="insp-field">
            <label>On timeout → </label>
            <select id="insp-timeout-to">
              <option value="">auto (first branch)</option>
              ${EB.project.blocks.filter(x => x.id !== b.id).map(x =>
                `<option value="${x.id}" ${tt.to === x.id ? "selected" : ""}>${esc(x.name)}</option>`).join("")}
            </select>
          </div>
        </div>
      </div>`;

      /* Pick / switch audio — fires when a side is detected and on every
         switch. Replaces the built-in UI click; a wrong-way retry keeps its
         own buzzer. Offset/duration/delay are applied at runtime, so retuning
         them is a metadata-only re-export. */
      const ca = b.choice_audio || {};
      const sounds = (EB.project.sounds || []).slice()
        .sort((x, y) => (x.name || "").localeCompare(y.name || ""));
      html += `<div class="insp-sec">
        <div class="insp-sec-head"><span>Pick / switch audio</span></div>
        <div class="insp-field">
          <label>Sound</label>
          <select id="insp-ca-sound">
            <option value="">— default UI click —</option>
            ${sounds.map(s => `<option value="${s.id}" ${ca.sound === s.id ? "selected" : ""}>${esc(s.name)}</option>`).join("")}
          </select>
        </div>
        <div class="insp-row" style="margin-top:6px;">
          <div class="insp-field">
            <label title="Wait this long after the gesture lands before playing">Delay (s)</label>
            <input type="number" id="insp-ca-delay" min="0" step="0.05" value="${ca.delay_s != null ? ca.delay_s : 0}">
          </div>
          <div class="insp-field">
            <label title="Start this far into the source file">Offset (s)</label>
            <input type="number" id="insp-ca-offset" min="0" step="0.05" value="${ca.source_offset_s != null ? ca.source_offset_s : 0}">
          </div>
        </div>
        <div class="insp-row" style="margin-top:6px;">
          <div class="insp-field">
            <label title="0 = play to the end of the file">Duration (s)</label>
            <input type="number" id="insp-ca-dur" min="0" step="0.05" value="${ca.duration_s != null ? ca.duration_s : 0}">
          </div>
          <div class="insp-field">
            <label>Gain</label>
            <input type="number" id="insp-ca-gain" min="0" max="2" step="0.05" value="${ca.gain != null ? ca.gain : 1}">
          </div>
        </div>
        <div style="margin-top:6px;font-size:11px;color:var(--text-mute);line-height:1.5;">
          Fires on detection and on every switch. Duration 0 plays to the end of the file.
          A wrong-way retry branch keeps its own buzzer.
        </div>
      </div>`;
    }

    if (b.type !== "merge") {
      html += `<div class="insp-sec">
        <div class="insp-sec-head">
          <span>Interaction windows · ${(b.windows || []).length}</span>
          <button class="add-btn" id="insp-add-win"><span class="msr">add</span>Add</button>
        </div>`;
      for (const w of (b.windows || [])) {
        const det = EB.detectorByType(w.detector);
        const selected = sel && sel.kind === "window" && sel.id === w.id;
        const end = w.duration_s == null ? "end" : EB.fmtTime((w.appears_s || 0) + w.duration_s);
        html += `<div class="insp-list-item${selected ? " selected" : ""}" data-win="${w.id}" style="${selected ? "border-color:var(--green);" : ""}">
          <span class="msr kind">${det ? det.icon : "touch_app"}</span>
          <span class="w-label" title="${esc(w.detector)}">${esc(w.label || w.detector)}</span>
          <span class="w-time">${EB.fmtTime(w.appears_s || 0, false)}→${end}</span>
          <button class="w-eye${w.disabled ? " off" : ""}" data-eye="${w.id}" title="${w.disabled ? "Disabled — click to enable" : "Enabled — click to disable"}">
            <span class="msr">${w.disabled ? "visibility_off" : "visibility"}</span>
          </button>
        </div>`;
      }
      html += `</div>`;
    }

    if (b.type !== "merge") {
      const roleIcon = { vo: "record_voice_over", music: "music_note", ambience: "waves", sfx: "graphic_eq" };
      html += `<div class="insp-sec">
        <div class="insp-sec-head"><span>Audio clips · ${(b.audio || []).length}</span></div>`;
      if (b.master_audio) {
        html += `<div style="font-size:11px;color:var(--green);line-height:1.5;">
          <span class="msr" style="font-size:13px;vertical-align:-2px;">volume_up</span>
          MASTER MIX — this block plays the source video's own baked audio
          (builder player, preview and export). Lane clips are ignored.</div>`;
      } else if (!(b.audio || []).length) {
        html += `<div style="font-size:11px;color:var(--text-mute);line-height:1.5;">
          Drag a sound from the Sounds tab onto a timeline audio lane (VO / MUSIC / AMB / SFX).
          Music and ambience beds keep looping through interaction holds in the runtime.</div>`;
      }
      for (const c of (b.audio || [])) {
        const snd = EB.getSound(c.sound);
        const end = c.duration_s == null ? "end" : EB.fmtTime((c.at_s || 0) + c.duration_s);
        html += `<div class="insp-list-item" data-aclip="${c.id}"${c.muted ? ' style="opacity:.45;"' : ""}>
          <span class="msr kind" style="color:var(--sky);">${roleIcon[c.role] || "graphic_eq"}</span>
          <span class="w-label" title="${esc(c.role || "sfx")}">${esc(snd ? snd.name : "(missing)")}${c.continues ? " →" : ""}</span>
          <span class="w-time">${EB.fmtTime(c.at_s || 0, false)}→${end}</span>
        </div>`;
      }
      html += `</div>`;
    }

    if (b.type === "merge") {
      html += `<div class="insp-sec"><div style="font-size:11.5px;color:var(--text-mute);line-height:1.6;">
        A merge is a join point: connect several branches into it, then drag from its output port
        to wherever the flow continues. It plays nothing itself.</div></div>`;
    }

    box.innerHTML = html;

    /* ── wiring ── */
    $("insp-del").addEventListener("click", () => {
      EB.change("delete block", () => EB.deleteBlock(b.id));
      EB.select(null);
    });

    const nameInput = $("insp-name");
    nameInput.addEventListener("change", () => {
      EB.change("rename block", () => {
        b.name = nameInput.value.trim() || b.name;
        // branch labels follow window labels, not block names — nothing else to sync
      });
    });

    const setStart = $("insp-set-start");
    if (setStart) setStart.addEventListener("click", () => {
      EB.change("set start", () => { EB.project.start = b.id; });
    });

    const addWin = $("insp-add-win");
    if (addWin) addWin.addEventListener("click", () => {
      EB.emit("add-window", { blockId: b.id, at: EB.tlLocalTime ? EB.tlLocalTime() : 0 });
    });

    box.querySelectorAll("[data-win]").forEach(item => {
      item.addEventListener("click", (e) => {
        if (e.target.closest("[data-eye]")) return;
        EB.emit("edit-window", { blockId: b.id, winId: item.dataset.win });
      });
    });
    box.querySelectorAll("[data-aclip]").forEach(item => {
      item.addEventListener("click", () => {
        // the lane module owns the clip editor — route through its dblclick path
        EB.emit("edit-audio-clip", { blockId: b.id, clipId: item.dataset.aclip });
      });
    });
    box.querySelectorAll("[data-eye]").forEach(btn => {
      btn.addEventListener("click", () => {
        const w = (b.windows || []).find(x => x.id === btn.dataset.eye);
        if (w) EB.change("toggle window", () => { w.disabled = !w.disabled; });
      });
    });
    box.querySelectorAll("select.br-to").forEach(selEl => {
      selEl.addEventListener("change", () => {
        const br = (b.branches || []).find(x => x.id === selEl.dataset.branch);
        if (br) EB.change("branch target", () => { br.to = selEl.value || null; });
      });
    });
    const ttS = $("insp-timeout-s");
    if (ttS) ttS.addEventListener("change", () => {
      EB.change("timeout", () => {
        b.timeout = b.timeout || {};
        b.timeout.seconds = Math.max(1, parseInt(ttS.value, 10) || 30);
      });
    });
    const ttTo = $("insp-timeout-to");
    if (ttTo) ttTo.addEventListener("change", () => {
      EB.change("timeout target", () => {
        b.timeout = b.timeout || { seconds: 30 };
        b.timeout.to = ttTo.value || null;
      });
    });

    /* Pick / switch audio */
    const caSound = $("insp-ca-sound");
    if (caSound) {
      const num = (id, min, max) => {
        const el = $(id);
        const v = parseFloat(el && el.value);
        if (!isFinite(v)) return min;
        return Math.max(min, max == null ? v : Math.min(max, v));
      };
      const commit = (label) => EB.change(label, () => {
        if (!caSound.value) { delete b.choice_audio; return; }
        b.choice_audio = {
          sound: caSound.value,
          delay_s: num("insp-ca-delay", 0),
          source_offset_s: num("insp-ca-offset", 0),
          duration_s: num("insp-ca-dur", 0),
          gain: num("insp-ca-gain", 0, 2),
        };
      });
      caSound.addEventListener("change", () => commit("pick audio"));
      ["insp-ca-delay", "insp-ca-offset", "insp-ca-dur", "insp-ca-gain"]
        .forEach(id => {
          const el = $(id);
          if (el) el.addEventListener("change", () => commit("pick audio timing"));
        });
    }
  }

  EB.on("selection-changed", render);
  EB.on("project-changed", render);
  document.addEventListener("DOMContentLoaded", render);
})();
