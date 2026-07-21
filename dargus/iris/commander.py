"""Iris commander — orchestrates global D-Base workflows."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

from dargus.dbase import DBase
from dargus.dbase.manager import DBaseManager
from dargus.dbase.paths import default_dargus_home
from dargus.experts.disease import DiseaseExpert
from dargus.iris.analog import IrisAnalog
from dargus.iris.bayes import IrisBayes
from dargus.iris.ensemble import IrisEnsemble
from dargus.iris.expert import IrisExpert
from dargus.iris.gnn import IrisGnn
from dargus.iris.llm import IrisLlm
from dargus.iris.search import IrisSearch
from dargus.workflows.train import TrainingReport
from dargus.workflows.train import run as run_train

logger = logging.getLogger(__name__)


class Iris:
    """Coordinates D-Base, expert system, and Iris-* agents."""

    name = "Iris"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def _global_manager(self) -> DBaseManager:
        dbase = DBase.global_instance()
        self._ensure_default_templates(dbase)
        return DBaseManager(dbase)

    def status(self) -> dict[str, Any]:
        """Report global D-Base status."""
        dbase = DBase.global_instance()
        return {
            "dargus_home": str(default_dargus_home()),
            "dbase_dir": str(dbase.dbase_dir),
            "n_records": len(dbase.list_records()),
            "n_templates": len(dbase._templates),
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
        """Run full Iris -> IrisExpert multi-round prediction.

        This is the unified prediction path. It delegates to IrisExpert's
        multi-round Expert dialog protocol (Molecule -> Biomed -> Bioinfo ->
        Clinic -> FourDExpert).
        """
        from dargus.experts.bioinfo import BioinfoExpert
        from dargus.experts.biomed import BiomedExpert
        from dargus.experts.clinic import ClinicExpert
        from dargus.experts.director import FourDExpert
        from dargus.experts.iris_expert import IrisExpert as IrisOrchestrator
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
                orchestrator = IrisOrchestrator(
                    molecule=MoleculeExpert(dbase=manager.dbase),
                    biomed=BiomedExpert(dbase=manager.dbase),
                    bioinfo=BioinfoExpert(dbase=manager.dbase),
                    clinic=ClinicExpert(dbase=manager.dbase),
                    director=FourDExpert(dbase=manager.dbase),
                )
                orchestrator.max_rounds = max_rounds
                final = orchestrator.run(records, ctx)
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
        """Deprecated: use ``Iris.predict()`` instead."""
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
        # Flatten disease dimension for backward compat:
        # {drug: {disease: {ep: ...}}} -> {drug: {ep: ...}}
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
        """Run the bench-full-stack workflow."""
        from dargus.workflows.bench_full_stack import run as run_bench_full_stack

        return run_bench_full_stack(strip=strip, split=split, output_dir=output_dir)

    def _run_agent(
        self,
        name: str,
        manager: DBaseManager,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
    ) -> dict[str, dict[str, dict[str, Any]]] | None:
        dbase = manager.dbase
        if name == "Iris-expert":
            agent = IrisExpert(DiseaseExpert(manager))
            return agent.predict(dbase, drug_ids, disease_id, endpoints)
        if name == "Iris-search":
            return IrisSearch().predict(dbase, drug_ids, disease_id, endpoints)
        if name == "Iris-analog":
            return IrisAnalog().predict(dbase, drug_ids, disease_id, endpoints)
        if name == "Iris-bayes":
            return IrisBayes().predict(dbase, drug_ids, disease_id, endpoints)
        if name == "Iris-gnn":
            return IrisGnn().predict(dbase, drug_ids, disease_id, endpoints)
        if name == "Iris-llm":
            config = self.config if self.config else None
            return IrisLlm(config=config).predict(dbase, drug_ids, disease_id, endpoints)
        return None

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
        ensemble = IrisEnsemble()
        aggregated = ensemble.aggregate(predictions)
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
        """Parse a natural language query using LLM and route to the appropriate handler.

        Returns a human-readable response string.
        """
        import json

        from dargus.llm_backends import llm_backend_from_config

        SYSTEM_PROMPT = """You are Iris, the clinical efficacy prediction assistant for Dargus.
You help researchers predict drug efficacy for diseases using a multi-level
evidence analysis system.

Available actions:
- "predict": Run efficacy prediction for one or more drugs against a disease
- "status": Check the current state of the global evidence database (D-Base)
- "clarify": Ask the user for more information if the query is ambiguous
- "chat": Respond conversationally to general questions about Dargus capabilities

When the user asks to predict drug efficacy, extract:
- drugs: list of drug names/IDs mentioned
- disease: the disease name/ID
- endpoints: any specific clinical endpoints mentioned
  (e.g., UPDRS-III, ADAS-Cog), or empty list for defaults

Return ONLY valid JSON, no other text. Format:
{"intent": "predict", "drugs": ["aspirin"], "disease": "headache", "endpoints": []}
{"intent": "status"}
{"intent": "clarify", "question": "Which disease are you interested in?"}
{"intent": "chat", "message": "I can help you predict drug efficacy..."}"""

        try:
            backend = llm_backend_from_config(self.config)
        except Exception:
            backend = None

        # Detect cases where no real LLM is configured
        from dargus.llm_backends import MockLLMBackend

        if backend is None or isinstance(backend, MockLLMBackend):
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
                f"> {query}\n"
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
                return f"> {query}\nIris: {question}"

            try:
                result = self.predict(
                    drug_ids=drugs,
                    disease_id=disease,
                    endpoints=endpoints or [],
                )
            except Exception as exc:
                return f"> {query}\nIris: Prediction failed: {exc}"

            lines = [f"> {query}", f"Iris: Prediction for {', '.join(drugs)} on {disease}:"]
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
                f"> {query}\n"
                f"Iris: D-Base status:\n"
                f"  Records:   {status['n_records']}\n"
                f"  Templates: {status['n_templates']}\n"
                f"  Location:  {status['dbase_dir']}"
            )

        if intent == "clarify":
            question = parsed.get("question", "Can you provide more details?")
            return f"> {query}\nIris: {question}"

        # Default: chat
        message = parsed.get(
            "message",
            (
                "I'm here to help with drug efficacy prediction. "
                "Try asking me to predict a drug's effect on a disease."
            ),
        )
        return f"> {query}\nIris: {message}"

    def _ensure_default_templates(self, dbase: DBase) -> None:
        drug_vocab = "global_drug_vocab"
        disease_vocab = "global_disease_vocab"
        endpoint_vocab = "global_endpoint_vocab"
        if "clinical_trial_outcome_v1" not in dbase._templates:
            from dargus.dbase import TemplateSchema

            dbase.add_template(
                TemplateSchema(
                    template_id="clinical_trial_outcome_v1",
                    fields=[
                        {
                            "name": "biological_level",
                            "type": "factor",
                            "vocabulary": [
                                "molecular",
                                "cellular",
                                "exvivo",
                                "animal",
                                "clinical",
                                "epi",
                            ],
                        },
                        {"name": "drug_id", "type": "factor", "vocabulary_ref": drug_vocab},
                        {"name": "disease_id", "type": "factor", "vocabulary_ref": disease_vocab},
                        {"name": "endpoint", "type": "factor", "vocabulary_ref": endpoint_vocab},
                        {"name": "fold_change", "type": "float"},
                        {"name": "ci95_lower", "type": "float"},
                        {"name": "ci95_upper", "type": "float"},
                    ],
                )
            )
