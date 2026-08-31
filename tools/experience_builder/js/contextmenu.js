/* Tiny reusable right-click context menu.
 *
 *   EB.contextMenu(clientX, clientY, [
 *     { label: "Edit…",   icon: "edit",   onClick: fn },
 *     { sep: true },
 *     { label: "Delete",  icon: "delete", danger: true, onClick: fn },
 *   ]);
 *
 * Closes on outside click, Escape, scroll, or window blur. Used by the timeline
 * bars (interaction windows, audio clips, captions) for a Delete option. */
(function () {
  let menuEl = null;

  function close() {
    if (!menuEl) return;
    menuEl.remove();
    menuEl = null;
    document.removeEventListener("mousedown", onDoc, true);
    document.removeEventListener("keydown", onKey, true);
    window.removeEventListener("blur", close);
    window.removeEventListener("resize", close);
    document.removeEventListener("scroll", close, true);
  }

  function onDoc(e) { if (menuEl && !menuEl.contains(e.target)) close(); }
  function onKey(e) { if (e.key === "Escape") close(); }

  EB.contextMenu = function (x, y, items) {
    close();
    menuEl = document.createElement("div");
    menuEl.className = "ctx-menu";
    for (const it of items || []) {
      if (it.sep) {
        const s = document.createElement("div");
        s.className = "ctx-sep";
        menuEl.appendChild(s);
        continue;
      }
      const b = document.createElement("button");
      b.type = "button";
      b.className = "ctx-item" + (it.danger ? " danger" : "");
      b.innerHTML = (it.icon ? `<span class="msr">${it.icon}</span>` : "") +
        `<span>${(EB.escapeHtml ? EB.escapeHtml(it.label) : it.label)}</span>`;
      if (it.disabled) b.disabled = true;
      else b.addEventListener("click", () => { close(); it.onClick && it.onClick(); });
      menuEl.appendChild(b);
    }
    document.body.appendChild(menuEl);

    // clamp on-screen
    const r = menuEl.getBoundingClientRect();
    menuEl.style.left = Math.max(6, Math.min(x, window.innerWidth - r.width - 6)) + "px";
    menuEl.style.top = Math.max(6, Math.min(y, window.innerHeight - r.height - 6)) + "px";

    // defer listener attach so the opening click doesn't immediately close it
    setTimeout(() => {
      document.addEventListener("mousedown", onDoc, true);
      document.addEventListener("keydown", onKey, true);
      window.addEventListener("blur", close);
      window.addEventListener("resize", close);
      document.addEventListener("scroll", close, true);
    }, 0);
  };

  EB.closeContextMenu = close;
})();
