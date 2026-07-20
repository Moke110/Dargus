from __future__ import annotations

from typing import Any

from dargus.dbase.manager import DBaseManager
from dargus.iris.base import IrisAgent, PredictionMatrix


class IrisExpert(IrisAgent):
    """Iris-* agent that wraps the v0.9.0 Expert system.

    Delegates to the new IrisExpert orchestration layer (dargus.experts.iris_expert)
    which manages MoleculeExpert, BiomedExpert, BioinfoExpert, ClinicExpert,
    and FourDExpert through a round-based dialog protocol.
    """

    name = "Iris-expert"

    def predict(
        self,
        dbase: Any,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
        embeddings: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PredictionMatrix:
        from dargus.experts.biomed import BiomedExpert
        from dargus.experts.bioinfo import BioinfoExpert
        from dargus.experts.clinic import ClinicExpert
        from dargus.experts.director import FourDExpert
        from dargus.experts.iris_expert import IrisExpert as IrisOrchestrator
        from dargus.experts.molecule import MoleculeExpert
        from dargus.experts.protocol import ExpertContext

        manager = DBaseManager(dbase)
        orchestrator = IrisOrchestrator(
            molecule=MoleculeExpert(dbase=dbase),
            biomed=BiomedExpert(dbase=dbase),
            bioinfo=BioinfoExpert(dbase=dbase),
            clinic=ClinicExpert(dbase=dbase),
            director=FourDExpert(dbase=dbase),
        )

        result: PredictionMatrix = {}
        for drug_id in drug_ids:
            result[drug_id] = {}
            for endpoint in endpoints:
                ctx = ExpertContext(
                    drug_ids=[drug_id],
                    disease_id=disease_id,
                    endpoints=[endpoint],
                )
                records = manager.read_records(disease_id=disease_id)
                final = orchestrator.run(records, ctx)
                result[drug_id][endpoint] = {
                    "efficacy_low": final.efficacy_low,
                    "efficacy_up": final.efficacy_up,
                    "confidence_level": final.confidence_level,
                    "reasoning_mode": self.name,
                    "supporting_records": final.supporting_records,
                    "expert_consensus": final.expert_consensus,
                    "contradictions": final.contradictions,
                    "data_gaps": final.data_gaps,
                }
        return result
