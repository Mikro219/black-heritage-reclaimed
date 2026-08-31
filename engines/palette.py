"""
Black Heritage Reclaimed — player-layer palette.

The piece is the Underground Railroad quilt-code story: lantern light against a
night sky, cloth, and the North Star. Every colour the VISITOR sees lives here
under a semantic name; the render layer never spells an RGB literal.

Scope — deliberately partial. This themes the player-facing surfaces (tutorial
card, captions, pause/volume UI, interaction indicators, the success flash).
The diagnostic surfaces are NOT themed and must stay high-contrast and ugly:
the debug overlay, the skeleton mini-panel, the camera-setup screen and the
gesture/voice tuners are instruments, not part of the piece.

Contrast: the work is projected in a lit public venue, so every text colour
carries its measured WCAG ratio against NIGHT below. tests/test_render.py pins
the floors (>= 7:1 for body text, >= 4.5:1 for the operator hint) so the look
can't erode one tweak at a time.
"""


class _Palette:
    """Frozen namespace — attribute access so typos raise instead of returning
    a silent None, and `__slots__` so nobody assigns a new colour at runtime."""

    __slots__ = ()

    # -- ground ---------------------------------------------------------
    NIGHT = (12, 14, 24)          # tutorial card fill — the night sky
    NIGHT_DEEP = (7, 9, 16)       # one step down, for edges that must separate
    CLOTH = (24, 20, 30)          # figure-box panel — a quilt patch on the night

    # A hairline of lamplight catching a seam.
    EDGE_RGBA = (255, 240, 214, 34)

    # -- lantern (the attention colour) ----------------------------------
    # Everything the visitor should look at or act on is amber. One colour,
    # one meaning: "here". 12.6:1 on NIGHT.
    LANTERN = (255, 201, 84)
    LANTERN_DIM = (176, 132, 52)  # ~0.7x — present, never competing (5.7:1)

    # -- light -----------------------------------------------------------
    NORTH_STAR = (255, 250, 235)  # 18:1 — the single most important glyph only
    LINEN = (240, 236, 228)       # 16:1 — body text, caption glyphs
    LINEN_DIM = (168, 162, 152)   # 7.6:1 — step counters, hints, labels
    LINEN_FAINT = (132, 128, 120)  # 4.9:1 — operator affordances

    # -- the success signal ----------------------------------------------
    # Visitors learn "green flash = you did it" in the tutorial. The signal is
    # hue + full-screen flash, so this must stay unmistakably a success green.
    SUCCESS = (0, 240, 96)
    SUCCESS_ALPHA = 80            # peak alpha of the flash overlay

    # -- hand cursors (FROZEN) -------------------------------------------
    # The hand icon PNGs in assets/hand_icons/ are pre-tinted green (*_l) and
    # blue (*_r) on disk. These dot-fallback colours must keep matching that
    # art or the L/R read breaks — do NOT restyle them with the palette.
    HAND_L = (60, 220, 90)
    HAND_R = (70, 160, 255)

    # -- panels ----------------------------------------------------------
    CAPTION_BG_RGBA = (9, 8, 14, 184)  # captions must survive a bright frame
    SHADOW_ALPHA = 130                 # caption glyph drop shadow
    VEIL_RGBA = (0, 0, 0, 140)         # pause veil
    TRACK = (58, 54, 66)               # volume slider track (fill is LANTERN)


PALETTE = _Palette()
