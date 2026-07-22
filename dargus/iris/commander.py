"""Iris — orchestrates Expert assessment via BaseAgent Harness (v0.16.0)."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

from dargus.agents.base import BaseAgent
from dargus.dbase import DBase
from dargus.dbase.manager import DBaseManager
from dargus.dbase.paths import default_dargus_home, working_dbase
from dargus.iris.ensemble import IrisEnsemble
from dargus.workflows.train import TrainingReport
from dargus.workflows.train import run as run_train

logger = logging.getLogger(__name__)


class Iris(BaseAgent):
    """Coordinates D-Base, Expert system, and Iris prediction agents via Harness."""

    name = "Iris"
    PERMITTED_TOOLS = [
        "dbase_query",
        "pubmed_search",
        "iris_search",
        "iris_analog",
        "iris_bayes",
        "iris_gnn",
        "iris_llm",
    ]
    PERMITTED_KNOWLEDGE = ["dbase", "disease_rag"]
    SUPPORTED_SKILLS = []  # Iris orchestrates; doesn't execute skills directly

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config)

    def _global_manager(self) -> DBaseManager:
        dbase = DBase.global_instance()
        return DBaseManager(dbase)

    def status(self) -> dict[str, Any]:
        """Report global D-Base status."""
        dbase = DBase.global_instance()
        records = dbase.read_shards()
        return {
            "dargus_home": str(default_dargus_home()),
            "working_dbase": working_dbase(),
            "dbase_dir": str(dbase.dbase_dir),
            "n_records": len(records),
        }

    def train(self, datadir: str) -> TrainingReport:
        """Run the Train workflow on the global D-Base."""
        return run_train(datadir)

    def predict(
        self,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
        max_rounds: int = 5,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Run full multi-Expert assessment."""
        from dargus.experts.bioinfo import BioinfoExpert
        from dargus.experts.biomed import BiomedExpert
        from dargus.experts.clinic import ClinicExpert
        from dargus.experts.director import FourDExpert
        from dargus.experts.molecule import MoleculeExpert
        from dargus.experts.protocol import ExpertContext

        manager = self._global_manager()

        result: dict[str, dict[str, dict[str, Any]]] = {}
        for drug_id in drug_ids:
            result[drug_id] = {disease_id: {}}
            for endpoint in endpoints:
                ctx = ExpertContext(
                    drug_ids=[drug_id],
                    disease_id=disease_id,
                    endpoints=[endpoint],
                )
                records = manager.read_records(disease_id=disease_id)
                molecule = MoleculeExpert(dbase=manager.dbase)
                biomed = BiomedExpert(dbase=manager.dbase)
                bioinfo = BioinfoExpert(dbase=manager.dbase)
                clinic = ClinicExpert(dbase=manager.dbase)
                director = FourDExpert(dbase=manager.dbase)

                mol_report = molecule.assess(records, ctx)
                bio_report = biomed.assess(records, ctx)
                bioinfo_report = bioinfo.assess(records, ctx)
                clinic_report = clinic.assess(records, ctx)

                all_reports: dict[str, list] = {
                    "MoleculeExpert": [mol_report],
                    "BiomedExpert": [bio_report],
                    "BioinfoExpert": [bioinfo_report],
                    "ClinicExpert": [clinic_report],
                }
                final = director.conclude(drug_id, disease_id, endpoint, all_reports)
                result[drug_id][disease_id][endpoint] = {
                    "efficacy_low": final.efficacy_low,
                    "efficacy_up": final.efficacy_up,
                    "confidence_level": final.confidence_level,
                    "reasoning_mode": "Iris-expert",
                    "supporting_records": final.supporting_records,
                    "expert_consensus": final.expert_consensus,
                    "contradictions": final.contradictions,
                    "data_gaps": final.data_gaps,
                }
        return result

    def infer(
        self,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str] | None = None,
        datadir: str | None = None,
        confirm_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Deprecated: use Iris.predict() instead."""
        warnings.warn(
            "Iris.infer() is deprecated, use Iris.predict() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        new_result = self.predict(
            drug_ids=drug_ids,
            disease_id=disease_id,
            endpoints=endpoints or [],
        )
        old_result: dict[str, dict[str, dict[str, Any]]] = {}
        for drug, diseases in new_result.items():
            old_result[drug] = {}
            for _disease, eps in diseases.items():
                old_result[drug].update(eps)
        return old_result

    def benchmark(
        self,
        strip: dict[str, Any],
        split: dict[str, Any] | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Run the bench-full-stack workflow (deprecated in v0.15.2)."""
        raise NotImplementedError(
            "bench-full-stack workflow removed in v0.15.2. "
            "Use 'dargus test-dbase' for single-evidence testing."
        )

    def _empty_pred(self) -> dict[str, Any]:
        return {
            "efficacy_low": 0.0,
            "efficacy_up": 1.0,
            "supporting_records": [],
            "reasoning_mode": self.name,
            "confidence_level": "insufficient_data",
        }

    def ensemble(
        self,
        predictions: dict[str, dict[str, dict[str, Any]]],
        weights: dict[str, float] | None = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Aggregate predictions from multiple Iris-* agents."""
        ensembler = IrisEnsemble()
        aggregated = ensembler.aggregate(predictions)
        if weights:
            for drug in aggregated:
                for endpoint in aggregated[drug]:
                    entry = aggregated[drug][endpoint]
                    components = entry.get("component_predictions", {})
                    weighted_low: list[float] = []
                    weighted_up: list[float] = []
                    denom = 0.0
                    for mode, interval in components.items():
                        w = weights.get(mode, 1.0)
                        weighted_low.append(interval["efficacy_low"] * w)
                        weighted_up.append(interval["efficacy_up"] * w)
                        denom += w
                    if denom > 0:
                        entry["efficacy_low"] = sum(weighted_low) / denom
                        entry["efficacy_up"] = sum(weighted_up) / denom
        return aggregated

    def process_query(self, query: str) -> str:
        """Parse NL query via LLM and route to predict/status."""
        import json
        from pathlib import Path

        import yaml

        from dargus.llm import llm_from_config

        config = self.config
        if not config:
            config_path = Path(__file__).resolve().parent.parent / "config" / "dargus_config.yaml"
            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as fh:
                    config = yaml.safe_load(fh) or {}

        SYSTEM_PROMPT = """You are Iris, the clinical efficacy prediction assistant for Dargus.
Available actions: "predict", "status", "clarify", "chat".
Return ONLY valid JSON:
{"intent": "predict", "drugs": ["aspirin"], "disease": "headache", "endpoints": []}
{"intent": "status"}
{"intent": "clarify", "question": "Which disease are you interested in?"}
{"intent": "chat", "message": "I can help you predict drug efficacy..."}"""

        backend = llm_from_config(config)
        if backend is None:
            return (
                "Iris: No LLM backend configured.\n\n"
                "Set your API key with:\n"
                "  dargus config set-api-key <provider> <key>\n\n"
                "Or use CLI subcommands directly:\n"
                "  dargus predict --drugs aspirin --disease headache\n"
                "  dargus status\n"
                "  dargus --help"
            )

        try:
            raw = backend.complete(SYSTEM_PROMPT + "\n\nUser query: " + query)
            parsed = json.loads(raw.strip())
        except Exception:
            return (
                "Iris: I had trouble understanding that. Could you rephrase?\n\n"
                "Examples:\n"
                "  predict aspirin for migraine\n"
                "  what's the evidence for metformin in type 2 diabetes?\n"
                "  status"
            )

        intent = parsed.get("intent", "chat")

        if intent == "predict":
            drugs = parsed.get("drugs", [])
            disease = parsed.get("disease", "")
            endpoints = parsed.get("endpoints", [])
            if not drugs or not disease:
                question = parsed.get("question", "Which drug and disease are you interested in?")
                return f"Iris: {question}"

            try:
                result = self.predict(drug_ids=drugs, disease_id=disease, endpoints=endpoints or [])
            except Exception as exc:
                return f"Iris: Prediction failed: {exc}"

            lines = [f"Iris: Prediction for {', '.join(drugs)} on {disease}:"]
            for drug, diseases in result.items():
                for disease_name, eps in diseases.items():
                    for ep, pred in eps.items():
                        ci = f"[{pred['efficacy_low']:.3f}, {pred['efficacy_up']:.3f}]"
                        conf = pred.get("confidence_level", "unknown")
                        lines.append(f"  {drug} / {disease_name} / {ep}: {ci} (confidence: {conf})")
            return "\n".join(lines)

        if intent == "status":
            status = self.status()
            return (
                f"Iris: D-Base status:\n"
                f"  Records:   {status['n_records']}\n"
                f"  Working:   {status['working_dbase']}\n"
                f"  Location:  {status['dbase_dir']}"
            )

        if intent == "clarify":
            return f"Iris: {parsed.get('question', 'Can you provide more details?')}"

        return f"Iris: {parsed.get('message', 'I help with drug efficacy prediction.')}"
