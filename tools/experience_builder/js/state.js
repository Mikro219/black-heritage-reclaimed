/* Project store + undo/redo + persistence plumbing.
 *
 * EB.project is the single source of truth (the .bhrx.json document).
 * EB.runtime holds session-only things that never serialize: object URLs,
 * File handles, selection, zoom.  Mutations go through EB.commit(label) which
 * snapshots for undo, marks dirty, autosaves, and emits "project-changed".
 */
window.EB = window.EB || {};

EB.PROJECT_VERSION = 1;
EB.AUTOSAVE_KEY = "bhr_experience_builder_autosave";

EB.newProject = function () {
  return {
    version: EB.PROJECT_VERSION,
    name: "Untitled Experience",
    fps: 30,
    media: [],     // { id, name, duration_s }
    sounds: [],    // { id, name }
    settings: { global_detect_sound: null },
    start: null,   // block id
    blocks: [],    // see plan schema
    edges: [],     // { from, to }  (playback/merge flow; choice branches live on the block)
  };
};

EB.project = EB.newProject();

EB.runtime = {
  mediaURLs: {},    // media id  -> object URL
  soundURLs: {},    // sound id  -> object URL
  fileHandles: {},  // asset id  -> FileSystemFileHandle (persisted in IndexedDB)
  missing: {},      // asset id  -> true when the file needs relinking
  thumbs: {},       // "mediaId@t" -> dataURL
  projectHandle: null,
  selection: null,  // { kind: "block"|"window"|"edge", id, blockId? }
  dirty: false,
};

/* ── tiny event bus ── */
EB._listeners = {};
EB.on = function (ev, fn) { (EB._listeners[ev] = EB._listeners[ev] || []).push(fn); };
EB.emit = function (ev, arg) { (EB._listeners[ev] || []).forEach(fn => { try { fn(arg); } catch (e) { console.error(e); } }); };

/* ── ids ── */
EB.uid = function (prefix) {
  return prefix + "_" + Math.random().toString(36).slice(2, 8);
};

/* ── undo / redo ── */
EB._undoStack = [];
EB._redoStack = [];
EB.UNDO_CAP = 100;

EB.commit = function (label) {
  // snapshot the PRE-mutation state pushed by beginChange, or fall back to
  // post-state diffing being unnecessary: callers call EB.beginChange() before
  // mutating, then EB.commit() after.
  if (EB._pending != null) {
    EB._undoStack.push(EB._pending);
    if (EB._undoStack.length > EB.UNDO_CAP) EB._undoStack.shift();
    EB._redoStack.length = 0;
    EB._pending = null;
  }
  EB.runtime.dirty = true;
  EB._autosaveSoon();
  EB.emit("project-changed", label || "");
};

EB._pending = null;
EB.beginChange = function () {
  if (EB._pending == null) EB._pending = JSON.stringify(EB.project);
};
EB.cancelChange = function () { EB._pending = null; };

/* Convenience: snapshot + mutate + commit in one call. */
EB.change = function (label, fn) {
  EB.beginChange();
  fn();
  EB.commit(label);
};

EB.undo = function () {
  if (!EB._undoStack.length) return;
  EB._redoStack.push(JSON.stringify(EB.project));
  EB.project = JSON.parse(EB._undoStack.pop());
  EB._afterHistoryJump();
};

EB.redo = function () {
  if (!EB._redoStack.length) return;
  EB._undoStack.push(JSON.stringify(EB.project));
  EB.project = JSON.parse(EB._redoStack.pop());
  EB._afterHistoryJump();
};

EB._afterHistoryJump = function () {
  // selection may point at something that no longer exists
  const sel = EB.runtime.selection;
  if (sel && sel.kind === "block" && !EB.getBlock(sel.id)) EB.runtime.selection = null;
  if (sel && sel.kind === "window" && !EB.findWindow(sel.id)) EB.runtime.selection = null;
  EB.runtime.dirty = true;
  EB._autosaveSoon();
  EB.emit("project-changed", "history");
  EB.emit("selection-changed");
};

/* ── lookups ── */
EB.getBlock = function (id) { return EB.project.blocks.find(b => b.id === id) || null; };
EB.getMedia = function (id) { return EB.project.media.find(m => m.id === id) || null; };
EB.getSound = function (id) { return EB.project.sounds.find(s => s.id === id) || null; };

EB.findWindow = function (winId) {
  for (const b of EB.project.blocks) {
    const w = (b.windows || []).find(w => w.id === winId);
    if (w) return { block: b, win: w };
  }
  return null;
};

EB.blockLen = function (b) {
  if (!b.range_s) return 0;
  return Math.max(0, b.range_s[1] - b.range_s[0]);
};

EB.select = function (sel) {
  EB.runtime.selection = sel;
  EB.emit("selection-changed");
};

EB.selectedBlock = function () {
  const sel = EB.runtime.selection;
  if (!sel) return null;
  if (sel.kind === "block") return EB.getBlock(sel.id);
  if (sel.kind === "window") { const f = EB.findWindow(sel.id); return f ? f.block : null; }
  return null;
};

/* Remove a block and every reference to it. */
EB.deleteBlock = function (id) {
  const p = EB.project;
  p.blocks = p.blocks.filter(b => b.id !== id);
  p.edges = p.edges.filter(e => e.from !== id && e.to !== id);
  for (const b of p.blocks) {
    if (b.type === "choice") {
      (b.branches || []).forEach(br => { if (br.to === id) br.to = null; });
      if (b.timeout && b.timeout.to === id) b.timeout.to = null;
    }
  }
  if (p.start === id) p.start = p.blocks.length ? p.blocks[0].id : null;
};

/* ── time helpers ── */
EB.fmtTime = function (s, showHundredths) {
  if (s == null || isNaN(s)) return "–:––";
  const neg = s < 0; s = Math.abs(s);
  const m = Math.floor(s / 60), sec = s - m * 60;
  const body = m + ":" + (sec < 10 ? "0" : "") +
    (showHundredths === false ? Math.floor(sec) : sec.toFixed(2));
  return (neg ? "-" : "") + body;
};

EB.parseTime = function (str) {
  if (str == null) return null;
  str = String(str).trim();
  if (!str) return null;
  if (/^(until end|end)$/i.test(str)) return null;
  const parts = str.split(":");
  let s = 0;
  if (parts.length === 1) s = parseFloat(parts[0]);
  else if (parts.length === 2) s = parseInt(parts[0], 10) * 60 + parseFloat(parts[1]);
  else return null;
  return isNaN(s) ? null : Math.max(0, s);
};

/* ── autosave (localStorage draft) ── */
EB._autosaveTimer = null;
EB._autosaveSoon = function () {
  clearTimeout(EB._autosaveTimer);
  EB._autosaveTimer = setTimeout(() => {
    try {
      localStorage.setItem(EB.AUTOSAVE_KEY, JSON.stringify(EB.project));
    } catch (e) { /* quota — draft autosave is best-effort */ }
  }, 400);
};

EB.loadAutosave = function () {
  try {
    const raw = localStorage.getItem(EB.AUTOSAVE_KEY);
    if (!raw) return false;
    const p = JSON.parse(raw);
    if (!p || p.version !== EB.PROJECT_VERSION || !Array.isArray(p.blocks)) return false;
    if (!p.blocks.length && !p.media.length) return false; // empty draft, ignore
    EB.project = p;
    return true;
  } catch (e) { return false; }
};

/* ── IndexedDB: persist FileSystemFileHandles across reloads ── */
EB._idb = null;
EB._idbOpen = function () {
  return new Promise((resolve) => {
    if (EB._idb) return resolve(EB._idb);
    const req = indexedDB.open("bhr_experience_builder", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("handles");
    req.onsuccess = () => { EB._idb = req.result; resolve(EB._idb); };
    req.onerror = () => resolve(null);
  });
};

EB.storeHandle = async function (assetId, handle) {
  const db = await EB._idbOpen();
  if (!db) return;
  try {
    db.transaction("handles", "readwrite").objectStore("handles").put(handle, assetId);
  } catch (e) { /* handle not serializable in this browser — relink will prompt */ }
};

EB.loadHandle = function (assetId) {
  return new Promise(async (resolve) => {
    const db = await EB._idbOpen();
    if (!db) return resolve(null);
    try {
      const req = db.transaction("handles").objectStore("handles").get(assetId);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    } catch (e) { resolve(null); }
  });
};

/* Try to re-link every media/sound file from stored handles. Marks
 * runtime.missing for any asset that still needs a manual re-pick. */
EB.relinkAssets = async function () {
  const all = [...EB.project.media.map(m => ({ a: m, kind: "media" })),
               ...EB.project.sounds.map(s => ({ a: s, kind: "sound" }))];
  for (const { a, kind } of all) {
    const urls = kind === "media" ? EB.runtime.mediaURLs : EB.runtime.soundURLs;
    if (urls[a.id]) continue;
    const handle = await EB.loadHandle(a.id);
    let file = null;
    if (handle) {
      try {
        let perm = await handle.queryPermission({ mode: "read" });
        if (perm === "prompt") perm = await handle.requestPermission({ mode: "read" });
        if (perm === "granted") file = await handle.getFile();
      } catch (e) { /* fall through to missing */ }
    }
    if (file) {
      urls[a.id] = URL.createObjectURL(file);
      EB.runtime.fileHandles[a.id] = handle;
      delete EB.runtime.missing[a.id];
    } else {
      EB.runtime.missing[a.id] = true;
    }
  }
  EB.emit("assets-changed");
};

/* ── toasts ── */
EB.toast = function (msg, isError) {
  const box = document.getElementById("toasts");
  const el = document.createElement("div");
  el.className = "toast" + (isError ? " err" : "");
  el.innerHTML = `<span class="msr">${isError ? "error" : "check_circle"}</span><span></span>`;
  el.lastElementChild.textContent = msg;
  box.appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .25s"; }, 2600);
  setTimeout(() => el.remove(), 2900);
};
