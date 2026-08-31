"""
EventBus — lightweight publish/subscribe used by all engines.

All inter-engine communication goes through here so no engine holds a
direct reference to another.

Handlers are isolated: one raising subscriber must not abort the rest of a
synchronous emit cascade (a shot_load reaches render, narration and the
audio mixer — half-applying it leaves the engines inconsistent). Errors are
logged with a traceback instead; the kiosk degrades rather than crashes.
"""

import logging
import os
import time
from collections import defaultdict
from typing import Callable

log = logging.getLogger(__name__)

# Emits are synchronous and run on the caller's thread — a slow handler on the
# main thread freezes the picture while the audio runs on. Every handler is
# timed so a stall names itself instead of having to be guessed at. Override
# the threshold with BHR_SLOW_MS (0 disables the reporting entirely).
try:
    SLOW_HANDLER_MS = float(os.environ.get("BHR_SLOW_MS", "60"))
except ValueError:
    SLOW_HANDLER_MS = 60.0


class EventBus:
    def __init__(self):
        self._subscribers: dict = defaultdict(list)

    def subscribe(self, event: str, handler: Callable):
        self._subscribers[event].append(handler)

    def emit(self, event: str, data: dict = None):
        for handler in self._subscribers.get(event, []):
            t0 = time.perf_counter()
            try:
                handler(data or {})
            except Exception:
                log.exception("EventBus: handler %r failed for event %r",
                              getattr(handler, "__qualname__", handler), event)
            if SLOW_HANDLER_MS > 0:
                ms = (time.perf_counter() - t0) * 1000.0
                if ms >= SLOW_HANDLER_MS:
                    print(f"[perf] bus {event!r} -> "
                          f"{getattr(handler, '__qualname__', handler)} "
                          f"took {ms:.0f}ms", flush=True)
