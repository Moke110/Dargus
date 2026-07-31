"""Iris — orchestrates Expert assessment via the BaseAgent harness."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dargus.agents.base import BaseAgent
from dargus.dbase import DBase
from dargus.dbase.paths import default_dargus_home, working_dbase
from dargus.dbase.store import DBaseStore
from dargus.workflows.ingest import run_ingest

if TYPE_CHECKING:
    from dargus.runtime.lifecycle import LifecycleManager

logger = logging.getLogger(__name__)


class Iris(BaseAgent):
    """Coordinates D-Base, Expert system, and Iris prediction agents via Harness."""

    name = "Iris"
    PERMITTED_TOOLS = [
        "dbase_query",
        "pubmed_search",
        "read_file",
        "write_file",
        "switch_mode",
    ]
    SUPPORTED_SKILLS = []  # Iris orchestrates; doesn't execute skills directly

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        lifecycle_manager: LifecycleManager | None = None,
        agent_factory: Any | None = None,
        **di_kwargs: Any,
    ):
        super().__init__(config=config, **di_kwargs)
        self._lifecycle_manager: LifecycleManager | None = lifecycle_manager
        self._agent_factory = agent_factory

        # ------------------------------------------------------------------
        # Skill loading: prefer the injected registry (AgentFactory path);
        # fall back to a local registry over the packaged skills directory.
        # ------------------------------------------------------------------
        if self._skill_registry is None:
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

    def _global_manager(self) -> DBaseStore:
        dbase = DBase.global_instance()
        return DBaseStore(dbase)

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

    def ingest(self, datadir: str, disease_kb_dir: str | None = None) -> dict[str, Any]:
        """Run the Ingest workflow on the global D-Base.

        Delegates to ``run_ingest(task_spec)``.
        """
        _task_spec: dict = {"workflow": "ingest", "source_path": datadir}
        if disease_kb_dir is not None:
            _task_spec["disease_kb_dir"] = disease_kb_dir
        if self._lifecycle_manager is not None:
            return self._lifecycle_manager.run_ingest(_task_spec)
        return run_ingest(_task_spec)

    def predict(
        self,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
        max_rounds: int = 5,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Run full multi-Expert assessment.

        With an injected LifecycleManager this delegates to
        ``run_predict(task_spec)``; without one it runs the direct
        expert-loop implementation. Experts are created through the
        AgentFactory when one is wired (design/2_runtime_structure.md: the factory
        is the single creation point for every Agent).
        """
        if self._lifecycle_manager is not None:
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

        from dargus.experts.protocol import ExpertContext

        factory = self._agent_factory
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
                if factory is not None:
                    molecule = factory.expert("molecular")
                    biomed = factory.expert("biomedical")
                    bioinfo = factory.expert("bioinformatics")
                    clinic = factory.expert("clinical")
                    director = factory.d4_expert()
                else:
                    from dargus.experts.bioinfo import BioinfoExpert
                    from dargus.experts.biomed import BiomedExpert
                    from dargus.experts.clinic import ClinicExpert
                    from dargus.experts.director import D4Expert
                    from dargus.experts.molecule import MoleculeExpert

                    molecule = MoleculeExpert(dbase=manager.dbase)
                    biomed = BiomedExpert(dbase=manager.dbase)
                    bioinfo = BioinfoExpert(dbase=manager.dbase)
                    clinic = ClinicExpert(dbase=manager.dbase)
                    director = D4Expert(dbase=manager.dbase)

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
                if factory is not None:
                    for agent in (molecule, biomed, bioinfo, clinic, director):
                        factory.terminate(agent)
        return result

    def _empty_pred(self) -> dict[str, Any]:
        return {
            "efficacy_score": None,
            "confidence_score": None,
            "supporting_records": [],
            "reasoning_mode": self.name,
            "confidence_level": "insufficient_data",
        }
