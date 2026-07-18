/* Preview-flow audio engine (WebAudio).
 *
 * Plays each block's `audio` clips against the preview video with the same
 * semantics as the runtime's ShotAudioMixer: sustain beds (music/ambience)
 * loop and keep playing across block transitions when the next block has a
 * `continues` clip for the same sound (silent handover, gain adopted);
 * unclaimed beds fade out over their fade_out_ms. SFX are one-shots scheduled
 * at their at_s. VO clips play as exact slices on a virtual voice channel (a
 * new line replaces the previous one). Unlike the runtime — which plays a
 * whole pre-rendered line and marks `continues` pieces fired without
 * replaying — WebAudio can slice, so a split line's `continues` piece simply
 * plays its own offset+duration slice, which audibly matches the runtime's
 * line-runs-across-the-boundary behaviour. Clip gain and fades map to a
 * GainNode with linear ramps. `muted` clips are skipped (preview-only flag).
 */
(function () {
  const engine = {
    ctx: null,
    buffers: {},   // sound id -> Promise<AudioBuffer|null>
    beds: [],      // { soundId, clip, src, gain }
    oneshots: [],  // { src, gain }
    vo: null,      // { soundId, clip, src, gain } — live voice line
    blockNodes: [],// editor block-playback nodes (timeline transport)
  };
  EB.pvAudio = engine;

  function audible(c) {
    return !c.muted && !(EB.laneMuted && EB.laneMuted(c.role || "sfx"));
  }

  function ctx() {
    if (!engine.ctx) {
      engine.ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (engine.ctx.state === "suspended") engine.ctx.resume().catch(() => {});
    return engine.ctx;
  }

  function loadBuffer(soundId) {
    if (engine.buffers[soundId]) return engine.buffers[soundId];
    const url = EB.runtime.soundURLs[soundId];
    if (!url) return Promise.resolve(null);
    engine.buffers[soundId] = fetch(url)
      .then(r => r.arrayBuffer())
      .then(ab => ctx().decodeAudioData(ab))
      .catch(() => null);
    return engine.buffers[soundId];
  }

  function stopNode(entry, fadeMs) {
    const ac = ctx();
    const t = ac.currentTime;
    try {
      const fade = Math.max(0.01, (fadeMs || 0) / 1000);
      entry.gain.gain.cancelScheduledValues(t);
      entry.gain.gain.setValueAtTime(entry.gain.gain.value, t);
      entry.gain.gain.linearRampToValueAtTime(0, t + fade);
      entry.src.stop(t + fade + 0.02);
    } catch (e) { try { entry.src.stop(); } catch (e2) {} }
  }

  engine.stop = function (fadeMs) {
    for (const b of engine.beds) stopNode(b, fadeMs != null ? fadeMs : 0);
    for (const o of engine.oneshots) stopNode(o, 0);
    if (engine.vo) stopNode(engine.vo, 0);
    engine.beds = [];
    engine.oneshots = [];
    engine.vo = null;
  };

  /* Called by preview.js each time a block starts playing. */
  engine.enterBlock = function (block) {
    const ac = ctx();
    // master_audio blocks use the video's own mix — lane clips are ignored
    // (an empty clip list still fades out any beds carried from before)
    const clips = block.master_audio ? []
      : (block.audio || []).filter(audible);

    // Bed handover: keep beds the new block continues, fade the rest.
    const claimed = new Set();
    const kept = [];
    for (const bed of engine.beds) {
      const match = clips.find((c, i) =>
        !claimed.has(i) && c.continues && (c.role || "sfx") !== "sfx" &&
        c.sound === bed.soundId);
      if (match) {
        claimed.add(clips.indexOf(match));
        const g = match.gain != null ? match.gain : 1;
        bed.gain.gain.setTargetAtTime(g, ac.currentTime, 0.08);
        bed.clip = match;
        kept.push(bed);
      } else {
        stopNode(bed, bed.clip && bed.clip.fade_out_ms);
      }
    }
    engine.beds = kept;
    engine.oneshots = engine.oneshots.filter(o => o.src.__alive !== false);

    if (engine.vo && engine.vo.src.__alive === false) engine.vo = null;

    const t0 = ac.currentTime;
    clips.forEach((c, i) => {
      if (claimed.has(i)) return;
      loadBuffer(c.sound).then(buf => {
        if (!buf) return;
        const src = ac.createBufferSource();
        const gain = ac.createGain();
        src.buffer = buf;
        src.connect(gain).connect(ac.destination);

        const g = c.gain != null ? c.gain : 1;
        const when = t0 + Math.max(0, c.at_s || 0);
        const offset = Math.max(0, Math.min(buf.duration - 0.01, c.source_offset_s || 0));
        const fadeIn = Math.max(0, (c.fade_in_ms || 0) / 1000);
        gain.gain.setValueAtTime(fadeIn > 0 ? 0 : g, when);
        if (fadeIn > 0) gain.gain.linearRampToValueAtTime(g, when + fadeIn);

        const role = c.role || "sfx";
        const isBed = role !== "sfx" && role !== "vo" && c.sustain !== false;
        if (isBed) {
          src.loop = true;
          if (c.duration_s != null && offset + c.duration_s < buf.duration - 0.05) {
            src.loopStart = offset;
            src.loopEnd = offset + Math.max(0.1, c.duration_s);
          }
          src.start(when, offset);
          engine.beds.push({ soundId: c.sound, clip: c, src, gain });
        } else if (role === "vo") {
          // Voice channel: play the exact slice; a new line replaces the
          // previous one.
          if (engine.vo) stopNode(engine.vo, 0);
          const voDur = c.duration_s != null
            ? Math.max(0.05, Math.min(c.duration_s, buf.duration - offset))
            : undefined;
          src.start(when, offset, voDur);
          src.onended = () => { src.__alive = false; };
          engine.vo = { soundId: c.sound, clip: c, src, gain };
        } else {
          const dur = c.duration_s != null
            ? Math.max(0.05, Math.min(c.duration_s, buf.duration - offset))
            : undefined;
          src.start(when, offset, dur);
          src.onended = () => { src.__alive = false; };
          engine.oneshots.push({ src, gain });
        }
      });
    });
  };

  /* ── editor block playback (timeline transport) ──
   * Schedules the selected block's audio clips against the block-local
   * playhead so the bottom timeline's ▶ plays sound in sync with the scrub
   * video. Everything is stopped on pause/seek/stop; beds loop until then. */
  engine.stopBlock = function () {
    for (const n of engine.blockNodes) stopNode(n, 0);
    engine.blockNodes = [];
  };

  engine.playBlockAt = function (block, tLocal) {
    engine.stopBlock();
    if (!block || block.master_audio) return;   // video's own mix plays instead
    const ac = ctx();
    const len = EB.blockLen(block);
    const t0 = ac.currentTime;
    for (const c of (block.audio || [])) {
      if (!audible(c)) continue;
      const clipStart = Math.max(0, c.at_s || 0);
      const clipEnd = c.duration_s == null ? len
        : Math.min(len, clipStart + c.duration_s);
      if (clipEnd <= tLocal + 0.02) continue;          // already over
      loadBuffer(c.sound).then(buf => {
        if (!buf) return;
        const src = ac.createBufferSource();
        const gain = ac.createGain();
        src.buffer = buf;
        src.connect(gain).connect(ac.destination);

        const g = c.gain != null ? c.gain : 1;
        const into = Math.max(0, tLocal - clipStart);   // playhead inside clip
        const when = t0 + Math.max(0, clipStart - tLocal);
        const offset = Math.max(0, Math.min(buf.duration - 0.01,
                                            (c.source_offset_s || 0) + into));
        const fadeIn = into > 0 ? 0 : Math.max(0, (c.fade_in_ms || 0) / 1000);
        gain.gain.setValueAtTime(fadeIn > 0 ? 0 : g, when);
        if (fadeIn > 0) gain.gain.linearRampToValueAtTime(g, when + fadeIn);

        const role = c.role || "sfx";
        const isBed = role !== "sfx" && role !== "vo" && c.sustain !== false;
        if (isBed) {
          src.loop = true;
          if (c.duration_s != null &&
              (c.source_offset_s || 0) + c.duration_s < buf.duration - 0.05) {
            src.loopStart = c.source_offset_s || 0;
            src.loopEnd = (c.source_offset_s || 0) + Math.max(0.1, c.duration_s);
          }
          src.start(when, offset);
        } else {
          const remain = clipEnd - Math.max(tLocal, clipStart);
          src.start(when, offset, Math.max(0.05, remain));
        }
        src.onended = () => { src.__alive = false; };
        engine.blockNodes.push({ src, gain });
      });
    }
  };
})();
