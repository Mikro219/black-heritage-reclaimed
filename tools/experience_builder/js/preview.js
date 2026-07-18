/* Preview Flow — plays the experience from the Start block.
 *
 * Playback blocks play through and follow their flow edge. Choice blocks play,
 * pause on the last frame with their windows armed, and branch on click.
 * Clicking a window's region simulates the gesture detection: green edge flash
 * + the global detect sound — the same success language as the runtime.
 * Windows without a drawn region render as a clickable pill instead. */
(function () {
  const $ = (id) => document.getElementById(id);

  let running = false;
  let current = null;      // block id
  let raf = null;
  let timeoutTimer = null;
  let timeoutTick = null;
  let firedWindows = null; // Set of window ids already fired in this block
  let hops = 0;

  const video = () => $("pv-video");

  /* ── open / close ── */
  function start() {
    const p = EB.project;
    if (!p.start || !EB.getBlock(p.start)) {
      return EB.toast("No start block — add a block first", true);
    }
    const unlinked = p.blocks.some(b =>
      b.media && b.range_s && !EB.runtime.mediaURLs[b.media]);
    if (unlinked) {
      return EB.toast("Video file not linked — open the Media tab and click " +
        "the red clip to re-link it before previewing", true);
    }
    $("preview-overlay").classList.add("open");
    $("pv-ended").classList.remove("on");
    running = true;
    hops = 0;
    enterBlock(p.start);
  }

  function stopPreview() {
    running = false;
    cancelAnimationFrame(raf);
    clearTimeout(timeoutTimer);
    clearInterval(timeoutTick);
    if (EB.pvAudio) EB.pvAudio.stop(0);
    const v = video();
    v.pause();
    v.removeAttribute("src"); v.load();
    clearRegions();
    $("preview-overlay").classList.remove("open");
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("tb-preview-btn").addEventListener("click", start);
    $("pv-close").addEventListener("click", stopPreview);
    $("pv-restart").addEventListener("click", () => {
      $("pv-ended").classList.remove("on");
      hops = 0;
      enterBlock(EB.project.start);
    });
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && $("preview-overlay").classList.contains("open")) stopPreview();
    });
  });

  /* ── flow ── */
  function nextOf(blockId) {
    const e = EB.project.edges.find(x => x.from === blockId);
    return e ? e.to : null;
  }

  function enterBlock(id) {
    if (!running && !$("preview-overlay").classList.contains("open")) return;
    clearTimeout(timeoutTimer);
    clearInterval(timeoutTick);
    cancelAnimationFrame(raf);
    clearRegions();
    $("pv-timeout").style.visibility = "hidden";
    firedWindows = new Set();

    if (++hops > 200) { EB.toast("Flow loop detected — stopping preview", true); return endScreen(); }

    const b = id ? EB.getBlock(id) : null;
    if (!b) return endScreen();
    current = id;

    if (b.type === "merge") {
      crumb(b, "merge — passing through");
      return enterBlock(nextOf(b.id));
    }

    const url = b.media ? EB.runtime.mediaURLs[b.media] : null;
    crumb(b, b.type === "choice" ? "choice — plays, then waits for a decision" : "playback");
    if (EB.pvAudio) EB.pvAudio.enterBlock(b);
    if (!url || !b.range_s) {
      // no media: playback skips after a beat; choice waits for a click on pills
      if (b.type === "choice") { armWindows(b, true); armTimeout(b); }
      else timeoutTimer = setTimeout(() => enterBlock(nextOf(b.id)), 800);
      return;
    }

    const v = video();
    const src = url;
    // master_audio: the block plays the source video's own baked mix; other
    // blocks stay muted (their sound is the WebAudio lane clips)
    v.muted = !b.master_audio;
    const begin = () => {
      v.currentTime = b.range_s[0];
      v.play().catch(() => {});
      watch(b);
    };
    if (v.src !== src) {
      v.src = src;
      v.onloadedmetadata = begin;
    } else {
      begin();
    }
  }

  function watch(b) {
    const v = video();
    const step = () => {
      if (!running) return;
      const local = (v.currentTime || 0) - b.range_s[0];
      syncRegions(b, local, false);
      if ((v.currentTime || 0) >= b.range_s[1] - 0.02) {
        v.pause();
        if (b.type === "choice") {
          syncRegions(b, EB.blockLen(b), true);   // hold: all windows armed
          armTimeout(b);
        } else {
          enterBlock(nextOf(b.id));
        }
        return;
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  }

  function armTimeout(b) {
    const tt = b.timeout || {};
    const secs = tt.seconds != null ? tt.seconds : 30;
    let remain = secs;
    const lbl = $("pv-timeout");
    lbl.style.visibility = "visible";
    lbl.textContent = `auto-advance in ${remain}s`;
    timeoutTick = setInterval(() => {
      remain -= 1;
      lbl.textContent = `auto-advance in ${Math.max(0, remain)}s`;
    }, 1000);
    timeoutTimer = setTimeout(() => {
      clearInterval(timeoutTick);
      const fallback = tt.to
        || ((b.branches || [])[0] && (b.branches || [])[0].to)
        || nextOf(b.id);
      enterBlock(fallback);
    }, secs * 1000);
  }

  function crumb(b, note) {
    $("pv-crumb").innerHTML = `<b></b> <span style="color:var(--text-mute);">· ${note}</span>`;
    $("pv-crumb").querySelector("b").textContent = b.name || b.id;
  }

  function endScreen() {
    if (EB.pvAudio) EB.pvAudio.stop(800);   // let beds fade out musically
    $("pv-ended").classList.add("on");
    $("pv-crumb").innerHTML = "<b>End</b>";
  }

  /* ── windows / regions ── */
  function clearRegions() {
    document.querySelectorAll(".pv-region, .pv-pill").forEach(el => el.remove());
  }

  function activeWindows(b, local, holding) {
    return (b.windows || []).filter(w => {
      if (w.disabled) return false;
      const s = w.appears_s || 0;
      const e = w.duration_s == null ? Infinity : s + w.duration_s;
      if (holding) return w.duration_s == null || local <= e;
      return local >= s && local <= e;
    });
  }

  function syncRegions(b, local, holding) {
    const box = $("pv-video-box");
    const want = activeWindows(b, local, holding);
    const have = new Set([...document.querySelectorAll("[data-pv-win]")].map(el => el.dataset.pvWin));

    for (const w of want) {
      if (have.has(w.id)) continue;
      box.appendChild(regionEl(b, w));
    }
    document.querySelectorAll("[data-pv-win]").forEach(el => {
      if (!want.some(w => w.id === el.dataset.pvWin)) el.remove();
    });
  }

  function regionEl(b, w) {
    let el;
    if (w.region && w.region.w > 0.005) {
      el = document.createElement("div");
      el.className = "pv-region" + (w.region.shape === "circle" ? " circle" : "");
      el.style.left = w.region.x * 100 + "%";
      el.style.top = w.region.y * 100 + "%";
      el.style.width = w.region.w * 100 + "%";
      el.style.height = w.region.h * 100 + "%";
      el.innerHTML = `<span class="pv-rlabel"></span>`;
      el.querySelector(".pv-rlabel").textContent = w.label || w.detector;
    } else {
      el = document.createElement("button");
      el.className = "pv-region pv-pill";
      Object.assign(el.style, {
        left: "50%", top: "82%", transform: "translate(-50%,-50%)",
        width: "auto", height: "auto", padding: "10px 22px",
        borderRadius: "999px", background: "rgba(6,9,13,.72)",
        font: "600 14px 'IBM Plex Sans', sans-serif", color: "var(--text)",
        display: "flex", alignItems: "center", gap: "8px",
      });
      const det = EB.detectorByType(w.detector);
      el.innerHTML = `<span class="msr" style="color:var(--green);font-size:18px;">${det ? det.icon : "touch_app"}</span><span></span>`;
      el.lastElementChild.textContent = w.label || w.detector;
    }
    el.dataset.pvWin = w.id;
    el.title = `Click to simulate: ${w.detector}`;
    el.addEventListener("click", () => fireWindow(b, w));
    return el;
  }

  function fireWindow(b, w) {
    if (firedWindows.has(w.id)) return;
    firedWindows.add(w.id);

    // success language: green flash + global detect sound
    const flash = $("pv-flash");
    flash.classList.add("on");
    setTimeout(() => flash.classList.remove("on"), 320);
    const sid = EB.project.settings.global_detect_sound;
    const surl = sid ? EB.runtime.soundURLs[sid] : null;
    if (surl) new Audio(surl).play().catch(() => {});

    const branch = b.type === "choice" && (b.branches || []).find(br => br.window === w.id);
    if (branch && branch.to) {
      clearTimeout(timeoutTimer);
      clearInterval(timeoutTick);
      $("pv-timeout").style.visibility = "hidden";
      setTimeout(() => enterBlock(branch.to), 260);
    } else {
      // reaction-only window: it flashes and disappears
      const el = document.querySelector(`[data-pv-win="${w.id}"]`);
      if (el) el.remove();
    }
  }
})();
