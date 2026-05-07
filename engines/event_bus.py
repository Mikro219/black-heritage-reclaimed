"""
EventBus — lightweight publish/subscribe used by all engines.

All inter-engine communication goes through here so no engine holds a
direct reference to another.
"""

from collections import defaultdict
from typing import Callable


class EventBus:
    def __init__(self):
        self._subscribers: dict = defaultdict(list)

    def subscribe(self, event: str, handler: Callable):
        self._subscribers[event].append(handler)

    def emit(self, event: str, data: dict = None):
        for handler in self._subscribers.get(event, []):
            handler(data or {})
