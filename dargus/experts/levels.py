"""DEPRECATED: Level-specific expert implementations.

Replaced by v0.9.0 Expert system:
- MolecularExpert → MoleculeExpert (dargus.experts.molecule)
- CellularExpert, ExvivoExpert, AnimalExpert → BiomedExpert (dargus.experts.biomed)
- ClinicalExpert, EpiExpert → ClinicExpert (dargus.experts.clinic)

Kept for backward compatibility. Will be removed in a future version.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

from dargus.dbase import TemplateRecord
from dargus.experts.types import AnalysisReport, CurateResult, ExtractionReport


class LevelExpert(ABC):
    """DEPRECATED: Use dargus.experts.base.Expert instead."""

    def __init__(self, level_name: str | None = None, dbase: Any = None):
        warnings.warn(
            "LevelExpert is deprecated. Use dargus.experts.base.Expert instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.level_name = level_name or self._default_level_name()
        self.dbase = dbase

    def _default_level_name(self) -> str:
        return self.__class__.__name__.replace("Expert", "").lower()

    @abstractmethod
    def curate(self, records: list[TemplateRecord]) -> CurateResult:
        """Filter and validate records for this biological level."""
        ...

    @abstractmethod
    def analyze(self, curated: CurateResult) -> AnalysisReport:
        """Produce a level-specific analysis report."""
        ...

    @abstractmethod
    def extract(self, raw_data_dir: str) -> ExtractionReport:
        """Scan raw data directory and extract instances for this biological level."""
        ...

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
        return value

    def _detect_misclassification(self, record: TemplateRecord) -> dict[str, Any] | None:
        actual = self._read_field(record, "biological_level")
        if actual is None:
            return None
        if actual != self.level_name:
            return {
                "record_id": record.record_id,
                "expected_level": self.level_name,
                "actual_level": actual,
            }
        return None


class MolecularExpert(LevelExpert):
    def curate(self, records: list[TemplateRecord]) -> CurateResult:
        accepted: list[dict[str, Any]] = []
        misclassified: list[dict[str, Any]] = []
        notes: list[str] = []
        for record in records:
            misclass = self._detect_misclassification(record)
            if misclass is not None:
                misclassified.append(misclass)
                continue
            accepted.append({"record_id": record.record_id, "record": record})
        return CurateResult(records=accepted, misclassified=misclassified, notes=notes)

    def analyze(self, curated: CurateResult) -> AnalysisReport:
        if not curated.records:
            return AnalysisReport(
                level=self.level_name,
                summary="No molecular evidence available.",
                records=[],
                confidence="insufficient_data",
            )
        readouts = []
        for item in curated.records:
            record = item["record"]
            readout = self._read_field(record, "readout")
            if readout is not None:
                readouts.append(float(readout))
        no_quant_summary = (
            f"Molecular evidence: {len(curated.records)} record(s) " "without quantitative readout."
        )
        summary = (
            f"Molecular evidence: {len(readouts)} readout(s), "
            f"mean readout {sum(readouts) / len(readouts):.3f}."
            if readouts
            else no_quant_summary
        )
        return AnalysisReport(
            level=self.level_name,
            summary=summary,
            records=[item["record_id"] for item in curated.records],
            confidence="moderate" if readouts else "low",
        )

    def extract(self, raw_data_dir: str) -> ExtractionReport:
        from dargus.experts.toolrag.molecular_toolrag import MolecularToolRAG

        toolrag = MolecularToolRAG()
        return toolrag.extract(raw_data_dir)


class CellularExpert(LevelExpert):
    def curate(self, records: list[TemplateRecord]) -> CurateResult:
        return MolecularExpert(level_name=self.level_name, dbase=self.dbase).curate(records)

    def analyze(self, curated: CurateResult) -> AnalysisReport:
        return MolecularExpert(level_name=self.level_name, dbase=self.dbase).analyze(curated)

    def extract(self, raw_data_dir: str) -> ExtractionReport:
        from dargus.experts.toolrag.cellular_toolrag import CellularToolRAG

        toolrag = CellularToolRAG()
        return toolrag.extract(raw_data_dir)


class ExvivoExpert(LevelExpert):
    def curate(self, records: list[TemplateRecord]) -> CurateResult:
        return MolecularExpert(level_name=self.level_name, dbase=self.dbase).curate(records)

    def analyze(self, curated: CurateResult) -> AnalysisReport:
        return MolecularExpert(level_name=self.level_name, dbase=self.dbase).analyze(curated)

    def extract(self, raw_data_dir: str) -> ExtractionReport:
        from dargus.experts.toolrag.exvivo_toolrag import ExvivoToolRAG

        toolrag = ExvivoToolRAG()
        return toolrag.extract(raw_data_dir)


class AnimalExpert(LevelExpert):
    def curate(self, records: list[TemplateRecord]) -> CurateResult:
        return MolecularExpert(level_name=self.level_name, dbase=self.dbase).curate(records)

    def analyze(self, curated: CurateResult) -> AnalysisReport:
        return MolecularExpert(level_name=self.level_name, dbase=self.dbase).analyze(curated)

    def extract(self, raw_data_dir: str) -> ExtractionReport:
        from dargus.experts.toolrag.animal_toolrag import AnimalToolRAG

        toolrag = AnimalToolRAG()
        return toolrag.extract(raw_data_dir)


class ClinicalExpert(LevelExpert):
    def curate(self, records: list[TemplateRecord]) -> CurateResult:
        return MolecularExpert(level_name=self.level_name, dbase=self.dbase).curate(records)

    def analyze(self, curated: CurateResult) -> AnalysisReport:
        readouts = []
        for item in curated.records:
            record = item["record"]
            readout = self._read_field(record, "readout")
            if readout is not None:
                readouts.append(float(readout))
        contradictions = []
        if readouts and any(r > 0 for r in readouts) and any(r < 0 for r in readouts):
            contradictions.append("Mixed clinical effect directions detected.")
        no_quant_summary = (
            f"Clinical evidence: {len(curated.records)} record(s) " "without quantitative readout."
        )
        summary = (
            f"Clinical evidence: {len(readouts)} readout(s), "
            f"mean readout {sum(readouts) / len(readouts):.3f}."
            if readouts
            else no_quant_summary
        )
        return AnalysisReport(
            level=self.level_name,
            summary=summary,
            records=[item["record_id"] for item in curated.records],
            contradictions=contradictions,
            confidence="high" if len(readouts) > 1 else "low",
        )

    def extract(self, raw_data_dir: str) -> ExtractionReport:
        from dargus.experts.toolrag.clinical_toolrag import ClinicalToolRAG

        toolrag = ClinicalToolRAG()
        return toolrag.extract(raw_data_dir)


class EpiExpert(LevelExpert):
    def curate(self, records: list[TemplateRecord]) -> CurateResult:
        return MolecularExpert(level_name=self.level_name, dbase=self.dbase).curate(records)

    def analyze(self, curated: CurateResult) -> AnalysisReport:
        return MolecularExpert(level_name=self.level_name, dbase=self.dbase).analyze(curated)

    def extract(self, raw_data_dir: str) -> ExtractionReport:
        from dargus.experts.toolrag.epi_toolrag import EpiToolRAG

        toolrag = EpiToolRAG()
        return toolrag.extract(raw_data_dir)
