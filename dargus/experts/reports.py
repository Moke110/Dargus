"""ExpertReport serialization and the expert domain registry (shared).

Single source of truth for the shapes the harness review (#92) found
duplicated across ``dargus/tools/spawn.py``, ``dargus/iris/commander.py``,
``dargus/runtime/factory.py``, ``dargus/experts/director.py`` and
``dargus/workflows/predict.py``:

- :func:`expert_report_to_dict` / :func:`expert_report_from_dict` — the
  field-for-field serialization of :class:`~dargus.experts.protocol.ExpertReport`
  used both by the ``spawn_expert`` Tool result and by Iris's stub path.
- :data:`EXPERT_DOMAINS` / :data:`DOMAIN_EXPERT_PATHS` /
  :data:`EXPERT_NAME_TO_DOMAIN` — the domain key ↔ expert class mapping
  consumed by the spawn tool, the factory, the D4 director and the predict
  workflow.
- :func:`predict_task_spec` — the one helper that builds the predict
  ``task_spec`` dict, shared by the spawn tool and Iris.
"""

from __future__ import annotations

from typing import Any

from dargus.experts.protocol import (
    ConfidenceInterval,
    EvidenceAssessment,
    ExpertReport,
    FinalReport,
    TaskDelegation,
)

#: The four domain expert keys (excludes the D4 director). Used where the
#: spawn/stub path consults the full domain set.
DOMAIN_EXPERTS: list[str] = ["molecular", "biomedical", "bioinformatics", "clinical"]

#: The expert keys the spawn tool accepts: the four domain experts + the D4
#: director (SPEC-C). Consumed by the ``spawn_expert`` Tool's ``enum``.
EXPERT_DOMAINS: list[str] = DOMAIN_EXPERTS + ["d4"]

#: Domain key → expert class path. Single registry; ``AgentFactory.expert()``
#: and ``D4Expert.delegate_to_expert()`` resolve from here.
DOMAIN_EXPERT_PATHS: dict[str, str] = {
    "molecular": "dargus.experts.molecule.MoleculeExpert",
    "biomedical": "dargus.experts.biomed.BiomedExpert",
    "bioinformatics": "dargus.experts.bioinfo.BioinfoExpert",
    "clinical": "dargus.experts.clinic.ClinicExpert",
}

#: Expert class name → domain key (inverse of :data:`DOMAIN_EXPERT_PATHS`).
#: ``AgentFactory.expert()`` accepts class-name aliases via this map.
EXPERT_NAME_TO_DOMAIN: dict[str, str] = {
    "MoleculeExpert": "molecular",
    "BiomedExpert": "biomedical",
    "BioinfoExpert": "bioinformatics",
    "ClinicExpert": "clinical",
}


def expert_report_to_dict(report: ExpertReport) -> dict[str, Any]:
    """Serialize an :class:`ExpertReport` into a plain dict (Tool result)."""
    return {
        "expert": report.expert,
        "round": report.round,
        "findings": [
            {
                "record_ids": f.record_ids,
                "biological_level": f.biological_level,
                "relevance": f.relevance,
                "quality_score": f.quality_score,
                "limitations": list(f.limitations),
            }
            for f in report.findings
        ],
        "confidence": {
            "low": report.confidence.low,
            "high": report.confidence.high,
            "sources": list(report.confidence.sources),
        },
        "delegations": [
            {
                "target_expert": d.target_expert,
                "record_ids": list(d.record_ids),
                "reason": d.reason,
                "priority": d.priority,
            }
            for d in report.delegations
        ],
        "data_gaps": list(report.data_gaps),
        "bias_notes": list(report.bias_notes),
    }


def expert_report_from_dict(payload: dict) -> ExpertReport:
    """Rebuild an :class:`ExpertReport` from its serialized dict.

    Mirrors :func:`expert_report_to_dict` field-for-field.
    """
    findings = [
        EvidenceAssessment(
            record_ids=f.get("record_ids", []),
            biological_level=f.get("biological_level", ""),
            relevance=f.get("relevance", "medium"),
            quality_score=f.get("quality_score", 0.5),
            limitations=f.get("limitations", []),
        )
        for f in payload.get("findings", [])
    ]
    conf = payload.get("confidence") or {}
    delegations = [
        TaskDelegation(
            target_expert=d.get("target_expert", ""),
            record_ids=d.get("record_ids", []),
            reason=d.get("reason", ""),
            priority=d.get("priority", "medium"),
        )
        for d in payload.get("delegations", [])
    ]
    return ExpertReport(
        expert=payload.get("expert", ""),
        round=payload.get("round", 0),
        findings=findings,
        confidence=ConfidenceInterval(
            low=conf.get("low", 0.0),
            high=conf.get("high", 1.0),
            sources=conf.get("sources", []),
        ),
        delegations=delegations,
        data_gaps=payload.get("data_gaps", []),
        bias_notes=payload.get("bias_notes", []),
    )


def final_report_to_dict(report: FinalReport) -> dict[str, Any]:
    """Serialize a :class:`FinalReport` into the universal contract dict.

    Mirrors the DES ± DCS nested shape ``predict()`` returns
    (``{drug_id: {disease_id: {endpoint: {...}}}}``) so a D4 spawn's result
    can feed the final prediction (SPEC-C / #96).
    """
    return {
        report.drug_id: {
            report.disease_id: {
                report.endpoint: {
                    "efficacy_score": report.efficacy_score,
                    "confidence_score": report.confidence_score,
                    "confidence_level": report.confidence_level,
                    "reasoning_mode": report.reasoning_mode,
                    "supporting_records": report.supporting_records,
                    "expert_consensus": report.expert_consensus,
                    "contradictions": report.contradictions,
                    "data_gaps": report.data_gaps,
                }
            }
        }
    }


def collect_spawned_reports(conv: Any, skip: str | None = None) -> dict[str, list[ExpertReport]]:
    """Collect the ExpertReports carried by a Conversation's ``spawn_expert``
    Tool Messages into ``{expert_name: [ExpertReport, ...]}``.

    Shared by Iris's predict loop and the D4 spawn (SPEC-C / #95, #96): every
    spawn is a Tool Message in the parent Conversation, and the D4 director's
    ``conclude()`` consumes this dict keyed by expert class name.
    """
    reports_by_expert: dict[str, list[ExpertReport]] = {}
    for msg in getattr(conv, "messages", []):
        if msg.tool_call is None or msg.tool_call.name != "spawn_expert":
            continue
        if msg.tool_result is None or msg.tool_result.error is not None:
            continue
        payload = msg.tool_result.output
        if not isinstance(payload, dict) or "report" not in payload:
            continue
        if skip is not None and payload.get("expert") == skip:
            continue  # never feed the director its own report
        expert_name = payload.get("expert", "unknown")
        reports_by_expert.setdefault(expert_name, []).append(
            expert_report_from_dict(payload["report"])
        )
    return reports_by_expert


def final_report_from_dict(payload: dict) -> FinalReport:
    """Rebuild a :class:`FinalReport` from the universal contract dict.

    Mirrors :func:`final_report_to_dict` — accepts the nested
    ``{drug_id: {disease_id: {endpoint: {...}}}}`` shape a D4 spawn returns.
    """
    ((drug_id, diseases),) = payload.items()
    ((disease_id, endpoints),) = diseases.items()
    ((endpoint, entry),) = endpoints.items()
    return FinalReport(
        drug_id=drug_id,
        disease_id=disease_id,
        endpoint=endpoint,
        efficacy_score=entry.get("efficacy_score"),
        confidence_score=entry.get("confidence_score"),
        confidence_level=entry.get("confidence_level", "insufficient_data"),
        reasoning_mode=entry.get("reasoning_mode", "Iris-expert"),
        supporting_records=entry.get("supporting_records", []),
        expert_consensus=entry.get("expert_consensus", ""),
        contradictions=entry.get("contradictions", []),
        data_gaps=entry.get("data_gaps", []),
    )


def predict_task_spec(
    *,
    drug: str,
    disease: str,
    endpoint: str,
    session_id: str,
) -> dict[str, Any]:
    """Build the predict ``task_spec`` dict for one drug/disease/endpoint.

    Shared by the ``spawn_expert`` Tool (the Expert's sub-session) and by
    Iris's predict loop (the parent run) so the shape is built in exactly
    one place.
    """
    return {
        "workflow": "predict",
        "drug_ids": [drug],
        "disease_id": disease,
        "endpoints": [endpoint],
        "session_id": session_id,
    }
