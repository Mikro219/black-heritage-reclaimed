/* Boot, toolbar, keyboard shortcuts, project save/open, export dialog. */
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
  function openExport() {
    // save first so the exported command points at a real file
    const warnings = validateForExport();
    $("export-warnings").textContent = warnings.join("  ·  ");
    const fname = (EB.project.name.replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "_") || "project") + ".bhrx.json";
    $("export-cmd").textContent = `py -3.12 scripts/export_experience.py "${fname}"`;
    $("export-modal").classList.add("open");
    saveProject();
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

    /* boot: restore autosaved draft */
    if (EB.loadAutosave()) {
      EB.emit("project-changed", "autosave");
      EB.relinkAssets().then(() => {
        const missing = Object.keys(EB.runtime.missing).length;
        // Be explicit: a refresh restores the in-browser DRAFT, including any
        // unsaved (possibly accidental) edits — NOT the .bhrx.json file.
        EB.toast("Restored unsaved draft from your last session — use Open to "
          + "load the saved project file instead"
          + (missing ? ` (${missing} media file(s) need re-linking)` : ""), !!missing);
      });
      EB.runtime.dirty = false;
    }
    refreshChrome();
  });
})();
