"""spawn_expert Tool — the opencode ``task`` analogue for Iris (SPEC-C).

In predict mode, Iris's PRA loop exposes ``spawn_expert(expert, drug,
disease, endpoint)``. On invocation it:

1. Enforces the depth-1 guard: only Iris may spawn. A Subagent mid-run
   (runtime.spawn_stack non-empty) is denied.
2. Creates a ``parent_id``-linked subagent Conversation for that Expert.
3. Runs the Expert's full PRA loop in its expert Mode, where it self-serves
   evidence from the shared D-Base via ``dbase_query``.
4. Derives the ``ExpertReport`` from the records the Expert assessed.
5. Returns the ExpertReport as the Tool result (a Message in Iris's
   Conversation).

Reference: opencode ``packages/opencode/src/tool/task.ts`` (subagent spawn,
``parentID`` linking, depth guard, result returned to parent).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dargus.experts.protocol import ExpertContext
from dargus.experts.reports import EXPERT_DOMAINS, expert_report_to_dict, predict_task_spec
from dargus.tools.base import Tool, ToolParam

if TYPE_CHECKING:
    from dargus.iris.commander import Iris
    from dargus.runtime.context import DargusRuntime
    from dargus.runtime.factory import AgentFactory

logger = logging.getLogger(__name__)


def _derive_expert_report(expert: Any, drug: str, disease: str, endpoint: str) -> Any:
    """Run the Expert's evidence logic over the scope and return an ExpertReport.

    Reuses each Expert's existing ``assess()`` (which carries the per-domain
    quality/delegation evidence logic) against the records D-Base holds for
    the drug/disease/endpoint scope. The Expert self-serves: it queries D-Base
    for its scope rather than Iris brokering records in (SPEC-C).

    Returns the ExpertReport dataclass.
    """
    ctx = ExpertContext(
        drug_ids=[drug],
        disease_id=disease,
        endpoints=[endpoint],
    )
    manager = getattr(expert, "_runtime", None)
    records: list[dict] = []
    if manager is not None:
        store = getattr(manager, "dbase_store", None)
        if store is not None:
            try:
                records = store.read_records(disease_id=disease)
            except Exception:
                logger.warning("spawn_expert: D-Base query failed", exc_info=True)
                records = []
    return expert.assess(records, ctx)


def _run_expert_subagent(expert: Any, session_id: str, task_spec: dict[str, Any]) -> Any:
    """Run the Expert's PRA loop inside its own parent-linked Conversation.

    The sub-session is made current for the duration of the run so the
    Expert's Conversation is created under its own ``session_id``; the
    parent's ``current_session_id`` is restored on the way out so a later
    spawn in the same run still links to the same parent (SPEC-C).
    """
    runtime = expert._runtime
    parent_session = getattr(runtime, "current_session_id", None)
    runtime.current_session_id = session_id
    try:
        report = expert.run(task_spec)
    finally:
        runtime.current_session_id = parent_session
    return report


def make_spawn_expert_tool(factory: AgentFactory, iris: Iris) -> Tool:
    """Create the ``spawn_expert`` Tool bound to a factory + Iris.

    Args:
        factory: The AgentFactory used to create Expert subagents.
        iris: The Iris commander whose Conversation is the parent.
    """
    runtime: DargusRuntime | None = getattr(iris, "_runtime", None)

    def _impl(expert: str, drug: str, disease: str, endpoint: str) -> dict[str, Any]:
        # ── Depth-1 guard (SPEC-C): only Iris may spawn. ──────────────
        if runtime is not None and runtime.spawn_stack:
            return {
                "error": (
                    f"spawn_expert denied: agent depth would exceed 1 "
                    f"(active subagents: {runtime.spawn_stack})"
                )
            }

        # Resolve the Expert instance through the factory.
        if expert == "d4":
            expert_agent = factory.d4_expert()
        else:
            try:
                expert_agent = factory.expert(expert)
            except ValueError as exc:
                return {"error": str(exc)}

        # ── parent_id-linked subagent Conversation (SPEC-C) ──────────
        parent_session = getattr(iris._runtime, "current_session_id", None) or "dialogue"
        sub_session = f"{parent_session}:{expert}:{drug}:{disease}:{endpoint}"
        if runtime is not None:
            runtime.get_conversation(sub_session, expert_agent.name, parent_id=parent_session)

        # Mark the subagent as active so a nested spawn is denied.
        if runtime is not None:
            runtime.spawn_stack.append(sub_session)

        task_spec = predict_task_spec(
            drug=drug, disease=disease, endpoint=endpoint, session_id=sub_session
        )

        try:
            # ── Run the Expert's PRA loop (self-serves evidence) ─────
            _run_expert_subagent(expert_agent, sub_session, task_spec)

            # ── Derive the ExpertReport from the Expert's assessment ──
            report = _derive_expert_report(expert_agent, drug, disease, endpoint)
        finally:
            if runtime is not None and runtime.spawn_stack:
                runtime.spawn_stack.pop()

        return {
            "expert": expert_agent.name,
            "session_id": sub_session,
            "report": expert_report_to_dict(report),
        }

    tool = Tool(
        name="spawn_expert",
        description=(
            "Spawn a domain Expert (or the D4 director) to assess drug efficacy "
            "for a drug/disease/endpoint. The Expert self-serves evidence from "
            "the shared D-Base and returns an ExpertReport you can reason over. "
            "Available experts: molecular, biomedical, bioinformatics, clinical, d4."
        ),
        parameters=[
            ToolParam(
                name="expert",
                type="string",
                required=True,
                description=(
                    "Domain expert to spawn: molecular, biomedical, "
                    "bioinformatics, clinical, or d4."
                ),
                enum=EXPERT_DOMAINS,
            ),
            ToolParam("drug", "string", required=True, description="Drug identifier (CURIE)."),
            ToolParam(
                "disease", "string", required=True, description="Disease identifier (CURIE)."
            ),
            ToolParam("endpoint", "string", required=True, description="Endpoint name."),
        ],
        output={
            "type": "object",
            "properties": {
                "expert": {"type": "string"},
                "session_id": {"type": "string"},
                "report": {"type": "object"},
            },
        },
        timeout_ms=60_000,
        fallback="null_result",
        side_effect="read",
    )
    # Restrict to predict mode (SPEC-C).
    tool._modes = ["predict"]
    tool.bind(_impl)
    return tool
