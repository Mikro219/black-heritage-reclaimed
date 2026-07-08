"""
TutorialEngine — code-rendered calibration/tutorial that runs when the attendant
starts the experience, before the first shot plays (July 2026 punch list).

Teaches the interaction vocabulary with one gesture per step; every success
fires the standard green flash so players learn "green flash = you did it".
Steps auto-advance after STEP_TIMEOUT_S so a confused visitor can never wedge
the installation, and the whole tutorial is skippable (S key).

How it runs: main.py keeps the shot player, render clock and audio PAUSED while
the tutorial is active. Gesture detection runs normally — the tutorial arms
ordinary CG windows on the event bus (ids prefixed "tutorial_"), listens for
cg_detected, and the render engine draws the card via update(tutorial_card=
tutorial.card_info()). The paused shot player ignores bus events, so tutorial
detections can't leak into the experience.

Config: "tutorial_enabled": true|false in config.json.
"""

import time

STEP_TIMEOUT_S   = 25.0   # per-step auto-advance for stuck/absent visitors
SUCCESS_LINGER_S = 1.2    # let the green flash land before the next card


def _steps() -> list:
    """Tutorial steps. Each arms a real detector, so the tutorial doubles as a
    live calibration check of the camera + pose pipeline."""
    return [
        {
            "title": "Welcome!",
            "prompt": "Raise BOTH hands above your shoulders. "
                      "When the screen flashes green, you did it!",
            "icon": "open_r",
            "type": "presence_bilateral",
            "params": {"y_threshold": "shoulder", "hold_ms": 500},
        },
        {
            "title": "Pointing",
            "prompt": "Point at the glowing box and hold still.",
            "icon": "point_r",
            # region_rect is RAW camera space; the card target_rect is the same
            # box in player/screen space (mirrored x).
            "type": "point_target_held",
            "params": {"region_rect": {"x": 0.18, "y": 0.28, "w": 0.24, "h": 0.34},
                       "hold_ms": 500},
            "target_rect": {"x": 0.58, "y": 0.28, "w": 0.24, "h": 0.34},
        },
        {
            "title": "Point LEFT",
            "prompt": "Reach out and point to the LEFT side of the screen.",
            "icon": "point_l",
            "type": "point_region",
            "params": {"directions": ["left"], "hold_ms": 500},
        },
        {
            "title": "Point RIGHT",
            "prompt": "Now point to the RIGHT side of the screen.",
            "icon": "point_r",
            "type": "point_region",
            "params": {"directions": ["right"], "hold_ms": 500},
        },
        {
            "title": "Point DOWN",
            "prompt": "Point DOWN toward the ground.",
            "icon": "point_r",
            "type": "directional_point",
            "params": {"directions": ["down", "down_left", "down_right"],
                       "hold_ms": 500},
        },
        {
            "title": "Reach IN",
            "prompt": "Reach straight toward the screen, like you're touching it.",
            "icon": "point_r",
            "type": "forward_point",
            "params": {"target_region": "center", "hold_ms": 400},
        },
    ]


class TutorialEngine:
    def __init__(self, config: dict, event_bus, gesture_engine):
        self.config = config
        self.bus = event_bus
        self.gesture = gesture_engine
        self.enabled = bool(config.get("tutorial_enabled", True))
        self.active = False
        self.done = False
        self._steps: list = []
        self._step_i = -1
        self._step_started = 0.0
        self._advance_at = None       # set on success; next step after linger
        self._succeeded = False
        self._was_locked = False
        event_bus.subscribe("cg_detected", self._on_cg_detected)

    # ------------------------------------------------------------------

    def begin(self) -> bool:
        if not self.enabled or self.done or self.active:
            return False
        self.active = True
        self._steps = _steps()
        self._step_i = -1
        # The shot player may have left input locked (prologue). Unlock for the
        # tutorial and restore on finish.
        self._was_locked = bool(getattr(self.gesture, "_input_locked", False))
        self.bus.emit("input_lock", {"locked": False})
        print("[Tutorial] started")
        self._next_step()
        return True

    def skip(self) -> None:
        if self.active:
            print("[Tutorial] skipped")
            self._finish()

    def update(self) -> None:
        if not self.active:
            return
        now = time.monotonic()
        if self._advance_at is not None:
            if now >= self._advance_at:
                self._next_step()
        elif now - self._step_started > STEP_TIMEOUT_S:
            print(f"[Tutorial] step {self._step_i + 1} timed out — advancing")
            self._next_step()

    def card_info(self):
        """Dict for RenderEngine._draw_tutorial_card, or None when inactive."""
        if not self.active or not (0 <= self._step_i < len(self._steps)):
            return None
        step = self._steps[self._step_i]
        return {
            "title":  step.get("title", ""),
            "prompt": "Nice!" if self._succeeded else step.get("prompt", ""),
            "icon":   step.get("icon"),
            "target_rect": None if self._succeeded else step.get("target_rect"),
            "step":   self._step_i + 1,
            "total":  len(self._steps),
        }

    # ------------------------------------------------------------------

    def _next_step(self) -> None:
        self._step_i += 1
        self._advance_at = None
        self._succeeded = False
        if self._step_i >= len(self._steps):
            self._finish()
            return
        step = self._steps[self._step_i]
        self._step_started = time.monotonic()
        print(f"[Tutorial] step {self._step_i + 1}/{len(self._steps)}: {step['title']}")
        self.bus.emit("cg_window_open", {
            "interaction": {
                "id":     f"tutorial_{self._step_i}",
                "type":   step["type"],
                "params": step["params"],
                "tier":   "cg",
            }
        })

    def _on_cg_detected(self, data: dict) -> None:
        gid = data.get("gesture_id", "")
        if not self.active or not gid.startswith("tutorial_"):
            return
        if gid != f"tutorial_{self._step_i}":
            return   # stale detection from an already-advanced step
        print(f"[Tutorial] step {self._step_i + 1} SUCCESS")
        self._succeeded = True
        self.bus.emit("oi_flash", {})   # the green "you did it" flash
        self._advance_at = time.monotonic() + SUCCESS_LINGER_S

    def _finish(self) -> None:
        self.active = False
        self.done = True
        self._advance_at = None
        self.bus.emit("cg_window_open", {"interaction": None})
        self.bus.emit("input_lock", {"locked": self._was_locked})
        print("[Tutorial] finished")
