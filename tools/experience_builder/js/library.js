/* Left library pane: Media / Sounds / Actions tabs.
 * MP4 import (drop or picker, File System Access when available), duration
 * probe via a scratch <video>, sound import, and the global detect sound. */
(function () {
  const $ = (id) => document.getElementById(id);

  /* ── tabs ── */
  document.querySelectorAll(".lib-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".lib-tab").forEach(t => t.classList.toggle("active", t === tab));
      document.querySelectorAll(".lib-page").forEach(p =>
        p.classList.toggle("active", p.dataset.page === tab.dataset.tab));
    });
  });

  /* ── import: video ── */
  async function pickVideo() {
    if (window.showOpenFilePicker) {
      try {
        const [handle] = await window.showOpenFilePicker({
          types: [{ description: "Video", accept: { "video/*": [".mp4", ".webm", ".mov", ".m4v"] } }],
        });
        importVideoFile(await handle.getFile(), handle);
      } catch (e) { /* user cancelled */ }
    } else {
      $("file-video").click();
    }
  }

  $("file-video").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) importVideoFile(f, null);
    e.target.value = "";
  });

  function probeDuration(url) {
    return new Promise((resolve) => {
      const v = document.createElement("video");
      v.preload = "metadata";
      v.onloadedmetadata = () => resolve(v.duration || 0);
      v.onerror = () => resolve(0);
      v.src = url;
    });
  }

  async function importVideoFile(file, handle, relinkId) {
    const url = URL.createObjectURL(file);
    // A project stores media by NAME — importing a file whose name matches an
    // existing entry is a re-link, never a duplicate.
    if (!relinkId) {
      const existing = EB.project.media.find(m => m.name === file.name);
      if (existing) relinkId = existing.id;
    }
    if (relinkId) {
      if (EB.runtime.mediaURLs[relinkId]) URL.revokeObjectURL(EB.runtime.mediaURLs[relinkId]);
      EB.runtime.mediaURLs[relinkId] = url;
      delete EB.runtime.missing[relinkId];
      if (handle) { EB.runtime.fileHandles[relinkId] = handle; EB.storeHandle(relinkId, handle); }
      EB.emit("assets-changed");
      EB.toast(`Re-linked ${file.name}`);
      return;
    }
    const dur = await probeDuration(url);
    const id = EB.uid("m");
    EB.change("import media", () => {
      EB.project.media.push({ id, name: file.name, duration_s: Math.round(dur * 100) / 100 });
    });
    EB.runtime.mediaURLs[id] = url;
    if (handle) { EB.runtime.fileHandles[id] = handle; EB.storeHandle(id, handle); }
    EB.emit("assets-changed");
    EB.toast(`Imported ${file.name} · ${EB.fmtTime(dur, false)}`);
  }

  /* ── import: sound (multi-select for bulk stem drops) ── */
  async function pickSound(relinkId) {
    if (window.showOpenFilePicker) {
      try {
        const handles = await window.showOpenFilePicker({
          multiple: !relinkId,
          types: [{ description: "Audio", accept: { "audio/*": [".mp3", ".wav", ".ogg", ".m4a"] } }],
        });
        for (const handle of handles) {
          importSoundFile(await handle.getFile(), handle, relinkId);
          relinkId = null;   // only the first file can be a relink target
        }
      } catch (e) { /* cancelled */ }
    } else {
      $("file-sound").dataset.relink = relinkId || "";
      $("file-sound").click();
    }
  }

  $("file-sound").addEventListener("change", (e) => {
    const relink = e.target.dataset.relink || null;
    [...e.target.files].forEach((f, i) => importSoundFile(f, null, i === 0 ? relink : null));
    e.target.value = "";
    e.target.dataset.relink = "";
  });

  /* Link a whole folder of audio files: every project sound whose name matches
   * a file in the folder (recursively) gets re-linked — the one-click path for
   * the 199-file stem delivery after a capcut_audio.py --to-builder import. */
  async function linkSoundFolder() {
    if (!window.showDirectoryPicker) {
      return EB.toast("Folder linking needs Chrome/Edge (File System Access API)", true);
    }
    let dir;
    try { dir = await window.showDirectoryPicker(); } catch (e) { return; }
    const wanted = new Map(EB.project.sounds.map(s => [s.name.toLowerCase(), s]));
    let linked = 0, seen = 0;
    async function walk(d) {
      for await (const entry of d.values()) {
        if (entry.kind === "directory") { await walk(entry); continue; }
        seen += 1;
        const s = wanted.get(entry.name.toLowerCase());
        if (!s || EB.runtime.soundURLs[s.id]) continue;
        try {
          const file = await entry.getFile();
          EB.runtime.soundURLs[s.id] = URL.createObjectURL(file);
          EB.runtime.fileHandles[s.id] = entry;
          EB.storeHandle(s.id, entry);
          delete EB.runtime.missing[s.id];
          linked += 1;
        } catch (e) { /* skip unreadable */ }
      }
    }
    await walk(dir);
    EB.emit("assets-changed");
    EB.toast(`Linked ${linked} of ${EB.project.sounds.length} sounds (${seen} files scanned)`);
  }

  function importSoundFile(file, handle, relinkId) {
    const url = URL.createObjectURL(file);
    if (!relinkId) {
      const existing = EB.project.sounds.find(s => s.name === file.name);
      if (existing) relinkId = existing.id;
    }
    if (relinkId) {
      if (EB.runtime.soundURLs[relinkId]) URL.revokeObjectURL(EB.runtime.soundURLs[relinkId]);
      EB.runtime.soundURLs[relinkId] = url;
      delete EB.runtime.missing[relinkId];
      if (handle) { EB.runtime.fileHandles[relinkId] = handle; EB.storeHandle(relinkId, handle); }
      EB.emit("assets-changed");
      return;
    }
    const id = EB.uid("s");
    EB.change("import sound", () => {
      EB.project.sounds.push({ id, name: file.name });
      // first sound imported becomes the detect sound automatically
      if (!EB.project.settings.global_detect_sound) EB.project.settings.global_detect_sound = id;
    });
    EB.runtime.soundURLs[id] = url;
    if (handle) { EB.runtime.fileHandles[id] = handle; EB.storeHandle(id, handle); }
    EB.emit("assets-changed");
    EB.toast(`Imported ${file.name}`);
  }

  /* ── drop zones ── */
  function wireDrop(el, onFiles) {
    el.addEventListener("click", () => onFiles === handleVideoDrop ? pickVideo() : pickSound());
    el.addEventListener("dragover", (e) => { e.preventDefault(); el.classList.add("dragover"); });
    el.addEventListener("dragleave", () => el.classList.remove("dragover"));
    el.addEventListener("drop", (e) => {
      e.preventDefault(); el.classList.remove("dragover");
      onFiles([...e.dataTransfer.files]);
    });
  }
  function handleVideoDrop(files) {
    files.filter(f => f.type.startsWith("video/")).forEach(f => importVideoFile(f, null));
  }
  function handleSoundDrop(files) {
    files.filter(f => f.type.startsWith("audio/")).forEach(f => importSoundFile(f, null, null));
  }
  wireDrop($("lib-drop"), handleVideoDrop);
  wireDrop($("lib-drop2"), handleSoundDrop);

  // dropping a video anywhere on the app also imports it
  document.body.addEventListener("dragover", (e) => e.preventDefault());
  document.body.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.target.closest("#lib-drop") || e.target.closest("#lib-drop2") || e.target.closest("#canvas-wrap")) return;
    handleVideoDrop([...e.dataTransfer.files]);
  });

  /* ── render: media list ── */
  function renderMedia() {
    const list = $("lib-media-list");
    list.innerHTML = "";
    $("lib-clip-count").textContent = EB.project.media.length;
    for (const m of EB.project.media) {
      const missing = EB.runtime.missing[m.id];
      const usedBy = EB.project.blocks.filter(b => b.media === m.id).length;
      const el = document.createElement("div");
      el.className = "lib-item" + (missing ? " missing" : "");
      el.draggable = !missing;
      el.innerHTML = `
        <span class="li-icon"><span class="msr">movie</span></span>
        <span class="li-meta">
          <span class="li-name"></span>
          <span class="li-sub">${missing ? "missing — click to relink" : EB.fmtTime(m.duration_s, false)}</span>
        </span>
        ${usedBy ? "" : '<button class="li-btn del" title="Remove (no block uses this clip)"><span class="msr">delete</span></button>'}
        <span class="li-grip msr">drag_indicator</span>`;
      el.querySelector(".li-name").textContent = m.name;
      const delBtn = el.querySelector(".li-btn.del");
      if (delBtn) delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        EB.change("remove media", () => {
          EB.project.media = EB.project.media.filter(x => x.id !== m.id);
        });
        if (EB.runtime.mediaURLs[m.id]) URL.revokeObjectURL(EB.runtime.mediaURLs[m.id]);
        delete EB.runtime.mediaURLs[m.id];
        delete EB.runtime.missing[m.id];
        EB.emit("assets-changed");
      });
      const thumb = EB.runtime.thumbs[m.id + "@0"];
      if (thumb) el.querySelector(".li-icon").innerHTML = `<img src="${thumb}" alt="">`;
      else EB.thumbnail(m.id, 0).then(t => { if (t) renderMedia(); });

      if (missing) {
        el.addEventListener("click", () => relinkMedia(m.id));
      } else {
        el.addEventListener("dragstart", (e) => {
          el.classList.add("dragging");
          e.dataTransfer.setData("application/x-bhr-media", m.id);
          e.dataTransfer.effectAllowed = "copy";
        });
        el.addEventListener("dragend", () => el.classList.remove("dragging"));
      }
      list.appendChild(el);
    }
  }

  async function relinkMedia(id) {
    if (window.showOpenFilePicker) {
      try {
        const [handle] = await window.showOpenFilePicker({
          types: [{ description: "Video", accept: { "video/*": [".mp4", ".webm", ".mov", ".m4v"] } }],
        });
        importVideoFile(await handle.getFile(), handle, id);
      } catch (e) { /* cancelled */ }
    } else {
      const input = $("file-video");
      const once = (e) => {
        input.removeEventListener("change", once, true);
        e.stopImmediatePropagation();
        const f = e.target.files[0];
        if (f) importVideoFile(f, null, id);
        e.target.value = "";
      };
      input.addEventListener("change", once, true);
      input.click();
    }
  }

  /* ── render: sound list ── */
  function renderSounds() {
    const list = $("lib-sound-list");
    list.innerHTML = "";
    $("lib-sound-count").textContent = EB.project.sounds.length;
    const detectId = EB.project.settings.global_detect_sound;
    const filter = ($("lib-sound-filter") ? $("lib-sound-filter").value : "").trim().toLowerCase();
    for (const s of EB.project.sounds) {
      if (filter && !s.name.toLowerCase().includes(filter)) continue;
      const missing = EB.runtime.missing[s.id];
      const isDetect = s.id === detectId;
      const el = document.createElement("div");
      el.className = "lib-item sound" + (missing ? " missing" : "");
      el.draggable = true;
      el.title = "Drag onto a timeline audio lane to place this sound on the selected block";
      el.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("application/x-bhr-sound", s.id);
        e.dataTransfer.effectAllowed = "copy";
      });
      el.innerHTML = `
        <span class="li-icon"><span class="msr">music_note</span></span>
        <span class="li-meta">
          <span class="li-name"></span>
          <span class="li-sub">${missing ? "missing — click name to relink" : "audio"}</span>
        </span>
        ${isDetect ? '<span class="li-badge">DETECT</span>' : ""}
        <button class="li-btn" title="Play"><span class="msr">play_arrow</span></button>
        <button class="li-btn bolt" title="Set as global detect sound"><span class="msr${isDetect ? " fill" : ""}" style="${isDetect ? "color:var(--amber);" : ""}">bolt</span></button>`;
      el.querySelector(".li-name").textContent = s.name;
      if (missing) el.querySelector(".li-name").addEventListener("click", () => pickSound(s.id));
      el.querySelector('.li-btn[title="Play"]').addEventListener("click", () => {
        const url = EB.runtime.soundURLs[s.id];
        if (url) new Audio(url).play().catch(() => {});
        else EB.toast("Sound file not linked — click its name to relink", true);
      });
      el.querySelector(".li-btn.bolt").addEventListener("click", () => {
        EB.change("set detect sound", () => { EB.project.settings.global_detect_sound = s.id; });
        EB.emit("assets-changed");
      });
      list.appendChild(el);
    }
  }

  /* ── render: actions (detector reference) ── */
  function renderActions() {
    const list = $("lib-actions-list");
    if (list.childElementCount) return; // static
    for (const d of EB.DETECTORS) {
      const el = document.createElement("div");
      el.className = "action-item";
      el.innerHTML = `
        <span class="li-icon"><span class="msr">${d.icon}</span></span>
        <span class="li-meta">
          <div class="a-name">${d.type}</div>
          <div class="a-desc"></div>
        </span>`;
      el.querySelector(".a-desc").textContent = d.hint || d.label;
      list.appendChild(el);
    }
  }

  /* ── thumbnails: capture a frame of a media at time t ── */
  const thumbQueue = [];
  let thumbBusy = false;
  EB.thumbnail = function (mediaId, t) {
    const key = mediaId + "@" + Math.round(t * 10) / 10;
    if (EB.runtime.thumbs[key]) return Promise.resolve(EB.runtime.thumbs[key]);
    return new Promise((resolve) => {
      thumbQueue.push({ mediaId, t, key, resolve });
      pumpThumbs();
    });
  };

  function pumpThumbs() {
    if (thumbBusy || !thumbQueue.length) return;
    const job = thumbQueue.shift();
    const url = EB.runtime.mediaURLs[job.mediaId];
    if (!url) { job.resolve(null); return pumpThumbs(); }
    if (EB.runtime.thumbs[job.key]) { job.resolve(EB.runtime.thumbs[job.key]); return pumpThumbs(); }
    thumbBusy = true;
    const v = document.createElement("video");
    v.preload = "auto";
    v.muted = true;
    const done = (dataURL) => {
      thumbBusy = false;
      v.removeAttribute("src"); v.load();
      if (dataURL) EB.runtime.thumbs[job.key] = dataURL;
      job.resolve(dataURL);
      EB.emit("thumb-ready", job);
      pumpThumbs();
    };
    v.onerror = () => done(null);
    v.onloadedmetadata = () => { v.currentTime = Math.min(job.t + 0.04, Math.max(0, v.duration - 0.05)); };
    v.onseeked = () => {
      try {
        const c = document.createElement("canvas");
        const w = 320, h = Math.round(w * v.videoHeight / Math.max(1, v.videoWidth));
        c.width = w; c.height = h || 180;
        c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
        done(c.toDataURL("image/jpeg", 0.6));
      } catch (e) { done(null); }
    };
    v.src = url;
  }

  /* ── wire up ── */
  EB.on("project-changed", () => { renderMedia(); renderSounds(); });
  EB.on("assets-changed", () => { renderMedia(); renderSounds(); });
  document.addEventListener("DOMContentLoaded", () => {
    renderMedia(); renderSounds(); renderActions();
    const filter = $("lib-sound-filter");
    if (filter) filter.addEventListener("input", renderSounds);
    const linkBtn = $("lib-link-folder");
    if (linkBtn) linkBtn.addEventListener("click", linkSoundFolder);
  });
})();
