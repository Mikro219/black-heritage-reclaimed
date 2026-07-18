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
from collections import defaultdict
from typing import Callable

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self):
        self._subscribers: dict = defaultdict(list)

    def subscribe(self, event: str, handler: Callable):
        self._subscribers[event].append(handler)

    def emit(self, event: str, data: dict = None):
        for handler in self._subscribers.get(event, []):
            try:
                handler(data or {})
            except Exception:
                log.exception("EventBus: handler %r failed for event %r",
                              getattr(handler, "__qualname__", handler), event)
