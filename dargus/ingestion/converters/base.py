from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SkipRecord:
    """A structured, auditable skip decision from a converter."""

    source_entry: str
    source: str
    reason: str
    detail: str = ""


class BaseConverter(ABC):
    """Convert one raw provenance wrapper into evidence dicts or skips.

    ``convert`` is pure with respect to I/O: it takes a single raw wrapper
    dict (``{source, source_entry, source_time, data}``) and returns evidence
    dicts (to be run through ``DBaseStore.build_evidence`` by the pipeline)
    and/or ``SkipRecord`` decisions.
    """

    template_id: str

    @abstractmethod
    def convert(self, raw: dict[str, Any]) -> list[dict[str, Any] | SkipRecord]: ...
