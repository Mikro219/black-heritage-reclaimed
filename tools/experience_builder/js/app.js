/* Boot, toolbar, keyboard shortcuts, project save/open, export dialog. */

/* Missed-mouseup guard. Every dragger in the builder (canvas pan, block
   drag, edge connect, timeline/caption/region bars) arms on mousedown and
   releases on a window "mouseup" — but that event never arrives when the
   button is released outside the window (off-window release, native menu,
   alt-tab), so the drag sticks to the cursor. Track presses at capture
   level and, when a move arrives with no button held (or focus is lost
   mid-press), synthesize the mouseup every armed dragger is waiting for. */
(function () {
  let pressed = false;
  window.addEventListener("mousedown", () => { pressed = true; }, true);
  window.addEventListener("mouseup", () => { pressed = false; }, true);
  window.addEventListener("blur", () => {
    if (!pressed) return;
    pressed = false;
    window.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
  });
  window.addEventListener("mousemove", (e) => {
    if (!pressed || e.buttons !== 0 || !e.isTrusted) return;
    pressed = false;
    window.dispatchEvent(new MouseEvent("mouseup", {
      bubbles: true, clientX: e.clientX, clientY: e.clientY,
    }));
  }, true);
})();

(function () {
  const $ = (id) => document.getElementById(id);

  /* ── dirty dot + undo/redo button state ── */
  function refreshChrome() {
    $("tb-dirty-dot").classList.toggle("on", EB.runtime.dirty);
    $("tb-undo").disabled = !EB._undoStack.length;
    $("tb-redo").disabled = !EB._redoStack.length;
    if (document.activeElement !== $("tb-project-name")) {
      $("tb-project-name").value = EB.project.name;
    }
    document.title = (EB.runtime.dirty ? "● " : "") + EB.project.name + " — BHR Experience Builder";
  }
  EB.on("project-changed", refreshChrome);

  /* ── save / open ── */
  function serialize() {
    return JSON.stringify(EB.project, null, 2);
  }

  async function saveProject() {
    EB.project.name = $("tb-project-name").value.trim() || "Untitled Experience";
    const suggested = EB.project.name.replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "_") + ".bhrx.json";
    if (window.showSaveFilePicker) {
      try {
        if (!EB.runtime.projectHandle) {
          EB.runtime.projectHandle = await window.showSaveFilePicker({
            suggestedName: suggested,
            types: [{ description: "BHR Experience", accept: { "application/json": [".json"] } }],
          });
        }
        const writable = await EB.runtime.projectHandle.createWritable();
        await writable.write(serialize());
        await writable.close();
        EB.runtime.dirty = false;
        refreshChrome();
        EB.toast("Project saved");
        return true;
      } catch (e) {
        if (e && e.name === "AbortError") return false;
        console.error(e);
        // fall through to download
      }
    }
    const blob = new Blob([serialize()], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = suggested;
    a.click();
    URL.revokeObjectURL(a.href);
    EB.runtime.dirty = false;
    refreshChrome();
    EB.toast("Project downloaded");
    return true;
  }

  async function openProject() {
    if (window.showOpenFilePicker) {
      try {
        const [handle] = await window.showOpenFilePicker({
          types: [{ description: "BHR Experience", accept: { "application/json": [".json", ".bhrx"] } }],
        });
        const file = await handle.getFile();
        loadProjectText(await file.text());
        EB.runtime.projectHandle = handle;
      } catch (e) { /* cancelled */ }
    } else {
      $("file-project").click();
    }
  }

  $("file-project").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (f) loadProjectText(await f.text());
    e.target.value = "";
  });

  function loadProjectText(text) {
    let p;
    try { p = JSON.parse(text); } catch (e) { return EB.toast("Not a valid project file", true); }
    if (!p || !Array.isArray(p.blocks)) return EB.toast("Not a BHR Experience project", true);
    if (p.version !== EB.PROJECT_VERSION) {
      EB.toast(`Project version ${p.version} — expected ${EB.PROJECT_VERSION}; loading anyway`, true);
    }
    EB.project = p;
    EB._undoStack.length = 0;
    EB._redoStack.length = 0;
    EB.runtime.selection = null;
    EB.runtime.dirty = false;
    EB.emit("project-changed", "open");
    EB.emit("selection-changed");
    EB.relinkAssets().then(() => {
      const missing = Object.keys(EB.runtime.missing).length;
      if (missing) EB.toast(`${missing} media file(s) need re-linking — click them in the library`, true);
    });
    EB.fitView();
    EB.toast(`Opened ${p.name}`);
  }

  /* ── export dialog ── */
  const SERVER = "http://127.0.0.1:8798";
  let exporting = false;

  async function serverAlive() {
    try {
      const ctl = new AbortController();
      const t = setTimeout(() => ctl.abort(), 800);
      const res = await fetch(SERVER + "/ping", { signal: ctl.signal });
      clearTimeout(t);
      return res.ok;
    } catch { return false; }
  }

  async function openExport() {
    const warnings = validateForExport();
    $("export-warnings").textContent = warnings.join("  ·  ");
    const fname = (EB.project.name.replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "_") || "project") + ".bhrx.json";
    $("export-cmd").textContent = `py -3.12 scripts/export_experience.py "${fname}"`;
    $("export-log").hidden = true;
    $("export-log").textContent = "";
    $("export-modal").classList.add("open");

    // With the helper server running (scripts/builder_server.py), the dialog
    // runs the export itself; otherwise it shows the command to run by hand.
    const live = await serverAlive();
    $("export-run-ui").hidden = !live;
    $("export-cmd-ui").hidden = live;
    $("export-run").hidden = !live;
    if (!live) saveProject();   // manual path: save so the command has a file
  }

  async function runExport() {
    if (exporting) return;
    exporting = true;
    const btn = $("export-run");
    const log = $("export-log");
    btn.disabled = true; btn.textContent = "Exporting…";
    log.hidden = false; log.textContent = "";
    const append = (txt) => {
      log.textContent += txt;
      log.scrollTop = log.scrollHeight;
    };
    try {
      EB.project.name = $("tb-project-name").value.trim() || EB.project.name;
      const res = await fetch(SERVER + "/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project: EB.project,
                               no_frames: $("export-no-frames").checked }),
      });
      if (!res.ok || !res.body) {
        append(`server error: HTTP ${res.status} ${await res.text()}\n`);
        return;
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let tail = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = dec.decode(value, { stream: true });
        append(chunk);
        tail = (tail + chunk).slice(-64);
      }
      if (/\[exit 0\]\s*$/.test(tail)) {
        EB.runtime.dirty = false;
        EB.emit("project-changed");
        EB.toast("Export finished — export/generated is up to date");
      } else {
        EB.toast("Export finished with errors — see the log", true);
      }
    } catch (err) {
      append(`\nconnection lost: ${err.message}\n(the export may still be ` +
             `running in the server terminal)\n`);
    } finally {
      exporting = false;
      btn.disabled = false; btn.textContent = "Run export";
    }
  }

  function validateForExport() {
    const p = EB.project;
    const warn = [];
    if (!p.start || !EB.getBlock(p.start)) warn.push("No start block.");
    const reachable = new Set();
    const stack = p.start ? [p.start] : [];
    while (stack.length) {
      const id = stack.pop();
      if (!id || reachable.has(id)) continue;
      reachable.add(id);
      const b = EB.getBlock(id);
      if (!b) continue;
      const e = p.edges.find(x => x.from === id);
      if (e) stack.push(e.to);
      if (b.type === "choice") {
        (b.branches || []).forEach(br => br.to && stack.push(br.to));
        if (b.timeout && b.timeout.to) stack.push(b.timeout.to);
      }
    }
    const unreachable = p.blocks.filter(b => !reachable.has(b.id));
    if (unreachable.length) warn.push(`${unreachable.length} block(s) unreachable from start.`);
    const noMedia = p.blocks.filter(b => b.type !== "merge" && !b.media);
    if (noMedia.length) warn.push(`${noMedia.length} block(s) have no clip.`);
    for (const b of p.blocks) {
      if (b.type === "choice" && !(b.branches || []).some(br => br.to)) {
        warn.push(`Choice "${b.name}" has no linked branches.`);
      }
    }
    return warn;
  }

  /* ── toolbar ── */
  document.addEventListener("DOMContentLoaded", () => {
    $("tb-save").addEventListener("click", saveProject);
    $("tb-open").addEventListener("click", openProject);
    $("tb-export").addEventListener("click", openExport);
    $("export-run").addEventListener("click", runExport);
    $("export-close").addEventListener("click", () => $("export-modal").classList.remove("open"));
    $("export-ok").addEventListener("click", () => $("export-modal").classList.remove("open"));
    $("tb-undo").addEventListener("click", () => EB.undo());
    $("tb-redo").addEventListener("click", () => EB.redo());
    $("tb-project-name").addEventListener("change", () => {
      EB.change("rename project", () => {
        EB.project.name = $("tb-project-name").value.trim() || "Untitled Experience";
      });
    });

    /* keyboard */
    window.addEventListener("keydown", (e) => {
      const inField = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
      const previewOpen = $("preview-overlay").classList.contains("open");
      const modalOpen = document.querySelector(".modal-scrim.open");

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault(); saveProject(); return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !inField) {
        e.preventDefault(); e.shiftKey ? EB.redo() : EB.undo(); return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y" && !inField) {
        e.preventDefault(); EB.redo(); return;
      }
      if (inField || previewOpen || modalOpen) return;

      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault(); EB.deleteSelection();
      } else if (e.key === " ") {
        e.preventDefault(); EB.tlTogglePlay();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault(); EB.tlStepFrame(-1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault(); EB.tlStepFrame(1);
      } else if (e.key === "Escape") {
        EB.select(null);
      }
    });

    window.addEventListener("beforeunload", (e) => {
      if (EB.runtime.dirty) { e.preventDefault(); e.returnValue = ""; }
    });

    /* boot: bundled project vs autosaved draft.
     *
     * js/project_data.js (generated by scripts/bundle_builder_project.py)
     * carries the .bhrx.json with a stamp. A NEW stamp means the file on
     * disk was regenerated (e.g. capcut_audio --to-builder) — it wins, and
     * the previous draft is backed up in localStorage. An unchanged stamp
     * means the draft is the user's own edits of this bundle — it wins. */
    const bundle = window.BHR_BUNDLED_PROJECT || null;
    const STAMP_KEY = "bhr_builder_bundle_stamp";
    const seenStamp = localStorage.getItem(STAMP_KEY);
    const bundleIsNew = bundle && bundle.stamp !== seenStamp;

    const announce = (msg, isErr) => EB.relinkAssets().then(() => {
      const missing = Object.keys(EB.runtime.missing).length;
      EB.toast(msg + (missing ? ` (${missing} file(s) need re-linking — `
        + "Media tab / Sounds 📁 link-folder)" : ""), isErr || !!missing);
    });

    if (bundleIsNew && bundle.project && Array.isArray(bundle.project.blocks)) {
      try {
        const draft = localStorage.getItem(EB.AUTOSAVE_KEY);
        if (draft) localStorage.setItem(EB.AUTOSAVE_KEY + "_backup", draft);
        localStorage.setItem(STAMP_KEY, bundle.stamp);
      } catch (e) { /* quota */ }
      EB.project = JSON.parse(JSON.stringify(bundle.project));
      EB._autosaveSoon();
      EB.emit("project-changed", "bundle");
      announce(`Loaded ${bundle.source} (bundled ${bundle.stamp})`
        + " — previous draft backed up");
      EB.runtime.dirty = false;
    } else if (EB.loadAutosave()) {
      EB.emit("project-changed", "autosave");
      // Be explicit: a refresh restores the in-browser DRAFT, including any
      // unsaved (possibly accidental) edits — NOT the .bhrx.json file.
      announce("Restored unsaved draft from your last session — use Open to "
        + "load the saved project file instead");
      EB.runtime.dirty = false;
    } else if (bundle && bundle.project && Array.isArray(bundle.project.blocks)) {
      // stamp already seen but no draft survived (cleared storage) — reload it
      EB.project = JSON.parse(JSON.stringify(bundle.project));
      EB._autosaveSoon();
      EB.emit("project-changed", "bundle");
      announce(`Loaded ${bundle.source} (bundled ${bundle.stamp})`);
      EB.runtime.dirty = false;
    }
    refreshChrome();
  });
})();
