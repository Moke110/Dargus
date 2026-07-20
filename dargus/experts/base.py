"""v0.9.0 Expert abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dargus.dbase import TemplateRecord
from dargus.experts.protocol import ExpertContext, ExpertReport


class Expert(ABC):
    """Base class for workflow-oriented experts.

    Each expert declares which biological levels it handles and
    provides rules for delegating records outside its scope.
    """

    SUPPORTED_LEVELS: tuple[str, ...] = ()
    DELEGATION_RULES: dict[str, str] = {}

    def __init__(self, dbase: Any = None):
        self.dbase = dbase

    @abstractmethod
    def assess(
        self,
        records: list[TemplateRecord],
        context: ExpertContext,
    ) -> ExpertReport:
        """Assess records and produce a structured report.

        Used for both ingestion (validating extracted instances) and
        inference (evaluating evidence quality and relevance).
        """
        ...

    def can_handle(self, record: TemplateRecord) -> bool:
        """Return True if this Expert handles the record's biological level."""
        level = self._read_biological_level(record)
        return level in self.SUPPORTED_LEVELS

    def delegate_target(self, record: TemplateRecord) -> str | None:
        """Return the target Expert name for this record, or None."""
        level = self._read_biological_level(record)
        if level is None:
            return None
        return self.DELEGATION_RULES.get(level)

    def _read_biological_level(self, record: TemplateRecord) -> str | None:
        """Extract biological_level from a TemplateRecord."""
        return self._read_field(record, "biological_level")

    def _schema_for(self, record: TemplateRecord):
        if self.dbase is not None:
            return self.dbase._templates.get(record.template_id)
        return getattr(record, "_schema_cache", None)

    def _read_field(self, record: TemplateRecord, field_name: str) -> Any:
        schema = self._schema_for(record)
        if schema is None:
            return None
        try:
            idx = schema.field_index(field_name)
        except KeyError:
            return None
        indices = record.sparse_vector.get("indices", [])
        values = record.sparse_vector.get("values", [])
        if idx not in indices:
            return None
        field = schema.field_def(field_name)
        pos = indices.index(idx)
        value = values[pos]
        if field.type == "factor" and field.vocabulary:
            return field.vocabulary[int(value)]
        if field.type == "factor" and field.vocabulary_ref:
            return self.dbase.vocab.reverse_lookup(field.vocabulary_ref, int(value))
        return value
