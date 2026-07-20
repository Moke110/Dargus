"""DEPRECATED: DiseaseExpert — replaced by IrisExpert + FourDExpert in v0.9.0.

Kept for backward compatibility. Will be removed in a future version.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from dargus.dbase.manager import DBaseManager
from dargus.dbase.record import TemplateRecord
from dargus.experts.levels import (
    AnimalExpert,
    CellularExpert,
    ClinicalExpert,
    EpiExpert,
    ExvivoExpert,
    LevelExpert,
    MolecularExpert,
)
from dargus.experts.types import (
    ExtractionReport,
    IngestionResult,
    IngestionSummary,
    PlanProposal,
)

LEVEL_ORDER = ["molecular", "cellular", "exvivo", "animal", "clinical", "epi"]


class DiseaseExpert:
    """DEPRECATED: Use IrisExpert + FourDExpert instead.

    This shim maintains the original interface while delegating to the
    v0.9.0 Expert system internally.
    """

    def __init__(
        self,
        manager: DBaseManager,
        level_experts: dict[str, LevelExpert] | None = None,
    ):
        warnings.warn(
            "DiseaseExpert is deprecated. Use IrisExpert + FourDExpert instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.manager = manager
        self.level_experts = level_experts or self._default_level_experts(manager)

    def _default_level_experts(self, manager: DBaseManager) -> dict[str, LevelExpert]:
        return {
            "molecular": MolecularExpert(dbase=manager.dbase),
            "cellular": CellularExpert(dbase=manager.dbase),
            "exvivo": ExvivoExpert(dbase=manager.dbase),
            "animal": AnimalExpert(dbase=manager.dbase),
            "clinical": ClinicalExpert(dbase=manager.dbase),
            "epi": EpiExpert(dbase=manager.dbase),
        }

    def ingest(self, file_paths: list[str]) -> IngestionResult:
        """Parse data files and write complete TemplateRecords to D-Base."""
        n_records = 0
        errors: list[str] = []
        for path_str in file_paths:
            path = Path(path_str)
            if not path.exists():
                errors.append(f"File not found: {path_str}")
                continue
            try:
                instances = self._parse_data_file(path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Failed to parse {path_str}: {exc}")
                continue
            for raw in instances:
                try:
                    record = self.manager.fill_template(
                        raw,
                        source_metadata=raw.get("source", {"type": "user_upload"}),
                    )
                    self.manager.write_record(record)
                    n_records += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"Failed to ingest record from {path_str}: {exc}")
        self.manager.dbase.save()
        return IngestionResult(n_records=n_records, files=file_paths, errors=errors)

    def _parse_data_file(self, path: Path) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif suffix in {".tsv", ".tab"}:
            df = pd.read_csv(path, sep="\t")
        else:
            return []
        instances: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            instance = {"source": {"file": str(path), "type": "user_upload"}}
            instance.update(row.dropna().to_dict())
            instances.append(instance)
        return instances

    def ingest_from_dir(
        self,
        datadir: str,
        disease_kb_dir: str | None = None,
        auto_confirm: bool = False,
    ) -> IngestionSummary:
        """Extract instances from raw data directory using parallel level experts.

        Each LevelExpert.extract() runs in its own thread. Results are aggregated
        into an IngestionSummary. When ``auto_confirm`` is True, all extracted
        instances are written to D-Base via ``DBaseManager.write_record()``.
        """
        raw_data_dir = Path(datadir)

        if disease_kb_dir:
            self._load_disease_kb(disease_kb_dir)

        reports: dict[str, ExtractionReport] = {}
        with ThreadPoolExecutor(max_workers=len(self.level_experts)) as executor:
            future_to_level = {
                executor.submit(expert.extract, str(raw_data_dir)): level
                for level, expert in self.level_experts.items()
            }
            for future in as_completed(future_to_level):
                level = future_to_level[future]
                try:
                    reports[level] = future.result()
                except Exception as exc:  # noqa: BLE001
                    reports[level] = ExtractionReport(
                        level=level,
                        files_considered=[],
                        files_selected=[],
                        source_types={},
                        instances=[],
                        notes=[f"Extraction failed: {exc}"],
                    )

        summary = IngestionSummary(per_level=reports)

        if auto_confirm or self._confirm_with_user(summary):
            self._write_instances(summary)

        return summary

    def _confirm_with_user(self, summary: IngestionSummary) -> bool:
        """Placeholder: returns False by default. Override for interactive use."""
        return False

    def _write_instances(self, summary: IngestionSummary) -> None:
        """Write all extracted instances from the summary to D-Base."""
        for report in summary.per_level.values():
            for inst in report.instances:
                try:
                    record = self.manager.fill_template(
                        inst.raw_fields,
                        source_metadata={
                            "type": "auto_extract",
                            "file": inst.source_file,
                            "row": inst.source_row,
                            "level": report.level,
                        },
                        suggested_template=inst.template_id,
                    )
                    self.manager.write_record(record)
                except Exception:  # noqa: BLE001
                    summary.warnings.append(
                        f"Failed to write {inst.template_id} from "
                        f"{inst.source_file}:{inst.source_row}"
                    )
        self.manager.dbase.save()

    def plan(
        self,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str] | None = None,
    ) -> PlanProposal:
        """Propose a prediction plan."""
        if endpoints is None:
            endpoints = ["primary_endpoint_change"]

        records = self.manager.read_records(disease_id=disease_id)
        available_agents = ["Iris-search", "Iris-expert"]
        weights = {"Iris-search": 0.6, "Iris-expert": 0.4}
        if records:
            available_agents.extend(["Iris-analog", "Iris-bayes", "Iris-gnn", "Iris-llm"])
            weights = {
                "Iris-search": 0.15,
                "Iris-analog": 0.15,
                "Iris-bayes": 0.25,
                "Iris-gnn": 0.15,
                "Iris-llm": 0.15,
                "Iris-expert": 0.15,
            }

        return PlanProposal(
            drug_ids=drug_ids,
            disease_id=disease_id,
            endpoints=endpoints,
            level_experts=list(self.level_experts.keys()),
            agents=available_agents,
            weights=weights,
            reasoning=(
                "Aggregate evidence across biological levels."
                if records
                else "Insufficient data; using conservative agents only."
            ),
        )

    def predict(
        self,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str] | None = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Predict efficacy intervals. Delegates to IrisExpert v0.9.0."""
        if endpoints is None:
            endpoints = ["primary_endpoint_change"]

        from dargus.experts.bioinfo import BioinfoExpert
        from dargus.experts.biomed import BiomedExpert
        from dargus.experts.clinic import ClinicExpert
        from dargus.experts.director import FourDExpert
        from dargus.experts.iris_expert import IrisExpert
        from dargus.experts.molecule import MoleculeExpert
        from dargus.experts.protocol import ExpertContext

        iris = IrisExpert(
            molecule=MoleculeExpert(dbase=self.manager.dbase),
            biomed=BiomedExpert(dbase=self.manager.dbase),
            bioinfo=BioinfoExpert(dbase=self.manager.dbase),
            clinic=ClinicExpert(dbase=self.manager.dbase),
            director=FourDExpert(dbase=self.manager.dbase),
        )

        result: dict[str, dict[str, dict[str, Any]]] = {drug: {} for drug in drug_ids}
        for drug in drug_ids:
            for endpoint in endpoints:
                ctx = ExpertContext(
                    drug_ids=[drug],
                    disease_id=disease_id,
                    endpoints=[endpoint],
                )
                records = self.manager.read_records(disease_id=disease_id)
                final = iris.run(records, ctx)
                result[drug][endpoint] = {
                    "efficacy_low": final.efficacy_low,
                    "efficacy_up": final.efficacy_up,
                    "supporting_records": final.supporting_records,
                    "reasoning_mode": "Iris-expert",
                    "confidence_level": final.confidence_level,
                    "level_reports": {},
                }
        return result

    def _record_drug_id(self, record: TemplateRecord) -> str | None:
        return self._record_field(record, "drug_id")

    def _record_endpoint(self, record: TemplateRecord) -> str | None:
        return self._record_field(record, "endpoint")

    def _record_value(self, record: TemplateRecord, field_name: str) -> float | None:
        value = self._record_field(record, field_name)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _load_disease_kb(self, disease_kb_dir: str) -> None:
        """Load disease knowledge documents into DiseaseRAG instances.

        Each .md/.txt file becomes one DiseaseRAG instance keyed by its stem
        (filename without extension). Instance is stored on self._disease_kbs
        for use by level experts during extraction and analysis.
        """
        from dargus.diseaserag.kb import DiseaseRAG

        kb_path = Path(disease_kb_dir)
        if not kb_path.exists():
            return

        if not hasattr(self, "_disease_kbs"):
            self._disease_kbs: dict[str, DiseaseRAG] = {}

        for doc_path in kb_path.rglob("*"):
            if not doc_path.is_file():
                continue
            if doc_path.suffix.lower() not in {".md", ".txt"}:
                continue
            disease_id = doc_path.stem
            if disease_id not in self._disease_kbs:
                self._disease_kbs[disease_id] = DiseaseRAG(disease_id=disease_id)
            self._disease_kbs[disease_id].add_documents([str(doc_path)])

    def _record_field(self, record: TemplateRecord, field_name: str) -> Any:
        schema = self.manager.dbase._templates.get(record.template_id)
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
            return self.manager.dbase.vocab.reverse_lookup(field.vocabulary_ref, int(value))
        return value
