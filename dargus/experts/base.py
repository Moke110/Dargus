"""v0.15.0 Expert abstract base class — evidence dict API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dargus.experts.protocol import ExpertContext, ExpertReport


class Expert(ABC):
    """Base class for workflow-oriented experts (v0.15.0).

    Each expert declares which biological levels it handles and
    provides rules for delegating records outside its scope.
    Works with keyed-object evidence dicts, not TemplateRecord.
    """

    SUPPORTED_LEVELS: tuple[str, ...] = ()
    DELEGATION_RULES: dict[str, str] = {}

    def __init__(self, dbase: Any = None):
        self.dbase = dbase

    @abstractmethod
    def assess(
        self,
        records: list[dict],
        context: ExpertContext,
    ) -> ExpertReport:
        """Assess evidence records and produce a structured report."""
        ...

    def can_handle(self, record: dict) -> bool:
        level = self._read_biological_level(record)
        return level in self.SUPPORTED_LEVELS

    def delegate_target(self, record: dict) -> str | None:
        level = self._read_biological_level(record)
        if level is None:
            return None
        return self.DELEGATION_RULES.get(level)

    def _read_biological_level(self, record: dict) -> str | None:
        return record.get("biological_level")

    def _read_field(self, record: dict, field_name: str) -> Any:
        return record.get(field_name)
