"""Iris — orchestrates Expert assessment via BaseAgent Harness (v0.16.0)."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable

from dargus.agents.base import BaseAgent
from dargus.dbase import DBase
from dargus.dbase.manager import DBaseManager
from dargus.dbase.paths import default_dargus_home, working_dbase
from dargus.workflows.ingest import IngestionReport, run_ingest

if TYPE_CHECKING:
    from dargus.runtime.lifecycle import LifecycleManager

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

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        lifecycle_manager: LifecycleManager | None = None,
    ):
        super().__init__(config=config)
        self._lifecycle_manager: LifecycleManager | None = lifecycle_manager

        # ------------------------------------------------------------------
        # Phase D/E: Skill loading for predict/ingest workflows.
        #
        # TODO(Phase E): Create Skill .md files in dargus/agents/skills/
        #   - predict.md  → predict workflow Skill
        #   - ingest.md   → ingest workflow Skill
        # Once Phase E ships, Iris will load these Skills and pass them to
        # LifecycleManager.run_predict / run_ingest. Until then the Skills
        # are attempted but gracefully skipped when missing.
        # ------------------------------------------------------------------
        try:
            from pathlib import Path

            from dargus.agents.skill_registry import SkillRegistry

            _skills_dir = Path(__file__).resolve().parent.parent / "agents" / "skills"
            self._skill_registry = SkillRegistry(_skills_dir)
            _loaded = {s.name for s in self._skill_registry.list_all()}
            if _loaded:
                logger.info("Iris loaded Skills: %s", sorted(_loaded))
            else:
                logger.debug("Iris: no Skill files found in %s", _skills_dir)
        except Exception:
            logger.debug("Iris: SkillRegistry init skipped (skills dir missing)", exc_info=True)
            self._skill_registry = None

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

    def ingest(self, datadir: str, disease_kb_dir: str | None = None) -> IngestionReport:
        """Run the Ingest workflow on the global D-Base.

        When a :class:`LifecycleManager` is injected (Phase D+), delegates
        to ``lifecycle_manager.run_ingest``.  Falls back to the direct
        ``run_ingest`` call for backward compatibility.
        """
        # ------------------------------------------------------------------
        # Phase D/E: Attempt to load the ingest Skill.  If present, it will
        # be passed to LifecycleManager.run_ingest once Phase E ships.
        # ------------------------------------------------------------------
        _ingest_skill = None
        if self._skill_registry is not None:
            try:
                _ingest_skill = self._skill_registry.get("ingest")
            except KeyError:
                logger.debug("Iris: 'ingest' Skill not found in registry")

        if self._lifecycle_manager is not None:
            try:
                _task_spec: dict = {"datadir": datadir}
                if disease_kb_dir is not None:
                    _task_spec["disease_kb_dir"] = disease_kb_dir
                result = self._lifecycle_manager.run_ingest(_task_spec)
                if result is not None:
                    return result
            except NotImplementedError:
                logger.warning("LifecycleManager.run_ingest is not implemented — falling back")
            except Exception:
                logger.exception("LifecycleManager.run_ingest failed — falling back")
        return run_ingest(datadir, disease_kb_dir=disease_kb_dir)

    def predict(
        self,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
        max_rounds: int = 5,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Run full multi-Expert assessment.

        When a :class:`LifecycleManager` is injected (Phase D+), delegates
        to ``lifecycle_manager.run_predict``.  Falls back to the direct
        implementation for backward compatibility.
        """
        # ------------------------------------------------------------------
        # Phase D/E: Attempt to load the predict Skill.  If present, it will
        # be passed to LifecycleManager.run_predict once Phase E ships.
        # ------------------------------------------------------------------
        _predict_skill = None
        if self._skill_registry is not None:
            try:
                _predict_skill = self._skill_registry.get("predict")
            except KeyError:
                logger.debug("Iris: 'predict' Skill not found in registry")

        if self._lifecycle_manager is not None:
            try:
                result = self._lifecycle_manager.run_predict(
                    {
                        "drug_ids": drug_ids,
                        "disease_id": disease_id,
                        "endpoints": endpoints,
                        "max_rounds": max_rounds,
                    }
                )
                if result is not None:
                    return result
            except NotImplementedError:
                logger.warning("LifecycleManager.run_predict is not implemented — falling back")
            except Exception:
                logger.exception("LifecycleManager.run_predict failed — falling back")

        # --- Fallback: direct implementation (backward compat) ---
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
                    "efficacy_score": final.efficacy_score,
                    "confidence_score": final.confidence_score,
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
            "efficacy_score": None,
            "confidence_score": None,
            "supporting_records": [],
            "reasoning_mode": self.name,
            "confidence_level": "insufficient_data",
        }

    def process_query(self, query: str) -> str:
        """Parse NL query via LLM and route to predict/status."""
        import json
        from pathlib import Path

        import yaml

        from dargus.models.compat import llm_from_config

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
                        des = pred.get("efficacy_score")
                        dcs = pred.get("confidence_score")
                        if des is None or dcs is None:
                            score = "insufficient data"
                        else:
                            score = f"DES {des:.3f} ± DCS {dcs:.3f}"
                        conf = pred.get("confidence_level", "unknown")
                        label = f"{drug} / {disease_name} / {ep}"
                        lines.append(f"  {label}: {score} (confidence: {conf})")
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
