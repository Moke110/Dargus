"""ToolCache — session-scoped cache for heavy tool resources.

Created at session start, closed at ``SESSION_END``. Heavy resources
(embedding models) stay resident across PRA rounds instead of being
reloaded per call (design/6 §embedding tool).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ToolCache:
    """Named-resource cache with lazy factories and explicit close."""

    def __init__(self) -> None:
        self._resources: dict[str, Any] = {}
        self._closers: dict[str, Callable[[Any], None]] = {}
        self._closed = False

    def get(self, name: str, factory: Callable[[], Any] | None = None) -> Any:
        """Return the cached resource *name*, creating it via *factory* on
        first access. Raises KeyError when absent and no factory is given."""
        if self._closed:
            raise RuntimeError("ToolCache is closed")
        if name not in self._resources:
            if factory is None:
                raise KeyError(f"ToolCache has no resource {name!r}")
            logger.debug("ToolCache: creating resource %r", name)
            self._resources[name] = factory()
        return self._resources[name]

    def put(self, name: str, resource: Any, closer: Callable[[Any], None] | None = None) -> None:
        """Store *resource* under *name*, with an optional close callback."""
        if self._closed:
            raise RuntimeError("ToolCache is closed")
        self._resources[name] = resource
        if closer is not None:
            self._closers[name] = closer

    def has(self, name: str) -> bool:
        return name in self._resources

    def close(self) -> None:
        """Release all resources (idempotent)."""
        if self._closed:
            return
        self._closed = True
        for name, resource in self._resources.items():
            closer = self._closers.get(name)
            if closer is not None:
                try:
                    closer(resource)
                except Exception:
                    logger.warning("ToolCache: closer for %r failed", name, exc_info=True)
        self._resources.clear()
        self._closers.clear()
