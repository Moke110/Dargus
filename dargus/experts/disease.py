"""DiseaseExpert — coordinates level experts for a disease."""

from __future__ import annotations

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
    AnalysisReport,
    ExtractionReport,
    IngestionResult,
    IngestionSummary,
    PlanProposal,
)
from dargus.iris.probability_utils import probability_interval_from_effect

LEVEL_ORDER = ["molecular", "cellular", "exvivo", "animal", "clinical", "epi"]


class DiseaseExpert:
    """Reads D-Base, dispatches records to level experts, and synthesizes predictions."""

    def __init__(
        self,
        manager: DBaseManager,
        level_experts: dict[str, LevelExpert] | None = None,
    ):
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
        """Placeholder for user confirmation via coding agent conversation."""
        return True

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
        """Predict efficacy intervals per drug and endpoint."""
        if endpoints is None:
            endpoints = ["primary_endpoint_change"]
        records = self.manager.read_records(disease_id=disease_id)
        result: dict[str, dict[str, dict[str, Any]]] = {drug: {} for drug in drug_ids}

        for drug in drug_ids:
            drug_records = [r for r in records if self._record_drug_id(r) == drug]
            for endpoint in endpoints:
                endpoint_records = [r for r in drug_records if self._record_endpoint(r) == endpoint]
                if not endpoint_records:
                    result[drug][endpoint] = {
                        "efficacy_low": 0.0,
                        "efficacy_up": 1.0,
                        "supporting_records": [],
                        "reasoning_mode": "DiseaseExpert",
                        "confidence_level": "insufficient_data",
                        "level_reports": {},
                    }
                    continue

                # Level curation/analysis for reporting
                level_reports: dict[str, AnalysisReport] = {}
                for level, expert in self.level_experts.items():
                    curated = expert.curate(endpoint_records)
                    if curated.records:
                        level_reports[level] = expert.analyze(curated)

                # Simple endpoint aggregation: mean effect with max CI range
                effects = [self._record_value(r, "fold_change") for r in endpoint_records]
                effects = [e for e in effects if e is not None]
                mean_effect = sum(effects) / len(effects) if effects else 0.0
                lowers = [self._record_value(r, "ci95_lower") for r in endpoint_records]
                uppers = [self._record_value(r, "ci95_upper") for r in endpoint_records]
                lowers = [lo for lo in lowers if lo is not None]
                uppers = [up for up in uppers if up is not None]
                ci_lower = min(lowers) if lowers else mean_effect - 0.5
                ci_upper = max(uppers) if uppers else mean_effect + 0.5
                efficacy_low, efficacy_up = probability_interval_from_effect(
                    mean_effect, ci_lower, ci_upper
                )

                confidence = "low"
                if "clinical" in level_reports:
                    confidence = "high"
                elif level_reports:
                    confidence = "moderate"

                result[drug][endpoint] = {
                    "efficacy_low": efficacy_low,
                    "efficacy_up": efficacy_up,
                    "supporting_records": [r.record_id for r in endpoint_records],
                    "reasoning_mode": "DiseaseExpert",
                    "confidence_level": confidence,
                    "level_reports": {
                        level: report.summary for level, report in level_reports.items()
                    },
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
