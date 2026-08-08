"""Iris — orchestrates Expert assessment via the BaseAgent harness."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dargus.agents.base import BaseAgent
from dargus.dbase import DBase
from dargus.dbase.paths import default_dargus_home, working_dbase
from dargus.dbase.store import DBaseStore
from dargus.experts.reports import DOMAIN_EXPERTS, expert_report_from_dict, predict_task_spec
from dargus.models.conversation import ToolCall, ToolResult
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

        Delegates to ``run_ingest(task_spec)``. Runs through the reused
        runtime's hook registry when one is wired (SPEC-B), matching predict
        and ask.
        """
        _task_spec: dict = {"workflow": "ingest", "source_path": datadir}
        if disease_kb_dir is not None:
            _task_spec["disease_kb_dir"] = disease_kb_dir
        if self._lifecycle_manager is not None:
            return self._lifecycle_manager.run_ingest(_task_spec)
        return run_ingest(_task_spec, runtime=self._runtime)

    def predict(
        self,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
        max_rounds: int = 5,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Run model-driven multi-Expert assessment (SPEC-C).

        Replaces the hardcoded expert loop: predict-mode Iris runs its PRA
        loop, the model emits ``spawn_expert`` calls (each creates a
        parent-linked subagent and returns an ExpertReport as the Tool
        result), and final synthesis consumes the collected ExpertReports.

        When no reasoning LLM is wired (stub mode), Iris deterministically
        spawns all four domain Experts plus the D4 director so a usable
        DES ± DCS prediction is still produced.
        """
        from dargus.models.conversation import Conversation

        result: dict[str, dict[str, dict[str, Any]]] = {}
        # Predict runs in predict mode so the spawn_expert tool is available
        # and the model's mode-tag validates (SPEC-C).
        self._mode = "predict"
        for drug_id in drug_ids:
            result[drug_id] = {disease_id: {}}
            for endpoint in endpoints:
                # ── Model-driven run: Iris reasons and emits spawn_expert ──
                task_spec = predict_task_spec(
                    drug=drug_id,
                    disease=disease_id,
                    endpoint=endpoint,
                    session_id=f"predict:{drug_id}:{disease_id}:{endpoint}",
                )
                self.run(task_spec)

                # Collect ExpertReports from the spawn_expert Tool results in
                # Iris's Conversation (each spawn is a Tool Message).
                reports_by_expert: dict[str, list[Any]] = {}
                conv: Conversation = self._resolve_conversation(task_spec)
                for msg in conv.messages:
                    if msg.tool_call is None or msg.tool_call.name != "spawn_expert":
                        continue
                    if msg.tool_result is None or msg.tool_result.error is not None:
                        continue
                    payload = msg.tool_result.output
                    if not isinstance(payload, dict) or "report" not in payload:
                        continue
                    expert_name = payload.get("expert", "unknown")
                    reports_by_expert.setdefault(expert_name, []).append(
                        expert_report_from_dict(payload["report"])
                    )

                # ── Final synthesis: consume the collected ExpertReports ──
                if not reports_by_expert and not self._llm_available():
                    # No usable reasoning LLM: deterministically spawn all four
                    # domain experts so a usable prediction is produced. Gated on
                    # _llm_available() (not just "no LLM wired") because a
                    # keyless-but-configured backend is equally unusable.
                    reports_by_expert = self._stub_spawn_all(drug_id, disease_id, endpoint, conv)

                # Surface any TaskDelegations carried by ExpertReports to Iris
                # as synthetic Messages — Iris (the sole orchestrator) decides
                # whether to spawn the target Expert (SPEC-C).
                self._surface_delegations(conv, reports_by_expert)

                final = self._synthesize(drug_id, disease_id, endpoint, reports_by_expert)
                result[drug_id][disease_id][endpoint] = {
                    "efficacy_score": final.efficacy_score,
                    "confidence_score": final.confidence_score,
                    "confidence_level": final.confidence_level,
                    "reasoning_mode": "Iris-model-driven",
                    "supporting_records": final.supporting_records,
                    "expert_consensus": final.expert_consensus,
                    "contradictions": final.contradictions,
                    "data_gaps": final.data_gaps,
                }
        return result

    def _stub_spawn_all(
        self, drug_id: str, disease_id: str, endpoint: str, conv: Any
    ) -> dict[str, list[Any]]:
        """Stub-mode fallback: spawn every domain Expert via the spawn tool.

        Used when no usable reasoning LLM is available — the model cannot
        emit spawn_expert calls, so Iris deterministically consults all four
        domain Experts (SPEC-C preserves a usable no-model path). Each spawn
        is recorded as a Tool Message in Iris's Conversation, mirroring the
        model-driven path (T6: spawns are inspectable in the log).
        """
        tool = getattr(self._runtime, "_spawn_tool", None) if self._runtime is not None else None
        reports: dict[str, list[Any]] = {}
        if tool is None:
            return reports
        for domain in DOMAIN_EXPERTS:
            try:
                out = tool.execute(
                    expert=domain, drug=drug_id, disease=disease_id, endpoint=endpoint
                )
            except Exception:
                logger.warning("stub spawn of %s failed", domain, exc_info=True)
                continue
            if not isinstance(out, dict) or "report" not in out:
                continue
            report = expert_report_from_dict(out["report"])
            reports.setdefault(report.expert, []).append(report)
            # Record the spawn as a Tool Message so the stub path is as
            # inspectable as the model-driven one (T6).
            conv.add_tool(
                ToolCall(
                    name="spawn_expert",
                    params={
                        "expert": domain,
                        "drug": drug_id,
                        "disease": disease_id,
                        "endpoint": endpoint,
                    },
                ),
                ToolResult(output=out),
                mode=self._mode,
            )
        return reports

    @staticmethod
    def _surface_delegations(conv: Any, reports_by_expert: dict[str, list[Any]]) -> None:
        """Surface TaskDelegations from ExpertReports as synthetic Messages.

        Because Experts cannot spawn, a TaskDelegation travels as data in the
        delegating Expert's ExpertReport; the harness surfaces it to Iris as a
        ``synthetic`` Message so Iris can decide whether to spawn the target
        (SPEC-C).
        """
        for expert_name, reports in reports_by_expert.items():
            for report in reports:
                for delegation in getattr(report, "delegations", []) or []:
                    conv.add_synthetic(
                        f"[TaskDelegation] {expert_name} requests {delegation.target_expert} "
                        f"for records {list(delegation.record_ids)}: {delegation.reason}"
                    )

    def _synthesize(
        self,
        drug_id: str,
        disease_id: str,
        endpoint: str,
        reports_by_expert: dict[str, list[Any]],
    ) -> Any:
        """Synthesize collected ExpertReports into a FinalReport via D4Expert.

        When the D4 director was not itself spawned (no LLM), it is created
        directly and its ``conclude`` produces the DES ± DCS contract.
        """
        from dargus.experts.director import D4Expert

        director = self._agent_factory.d4_expert() if self._agent_factory is not None else None
        if director is None:
            director = D4Expert(dbase=self._global_manager().dbase)
        return director.conclude(drug_id, disease_id, endpoint, reports_by_expert)

    def _empty_pred(self) -> dict[str, Any]:
        return {
            "efficacy_score": None,
            "confidence_score": None,
            "supporting_records": [],
            "reasoning_mode": self.name,
            "confidence_level": "insufficient_data",
        }
