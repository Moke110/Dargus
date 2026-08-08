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
from dargus.experts.reports import (
    EXPERT_DOMAINS,
    expert_report_to_dict,
    final_report_to_dict,
    predict_task_spec,
)
from dargus.tools.base import Tool, ToolParam

if TYPE_CHECKING:
    from dargus.iris.commander import Iris
    from dargus.runtime.context import DargusRuntime
    from dargus.runtime.factory import AgentFactory

logger = logging.getLogger(__name__)


def _self_serve(expert: Any, task_spec: dict[str, Any]) -> None:
    """Deterministic self-serve fallback: the Expert issues ``dbase_query``
    itself, through its own Conversation (SPEC-C / #95).

    When no usable reasoning LLM is wired, the model cannot emit the tool call,
    so this routine drives the same PRA loop the model would: the Expert's ACT
    runs ``dbase_query`` for its scope (drug, disease, endpoint, level) and the
    round is settled in the Expert's Conversation exactly like a model-driven
    tool round. This replaces the old Iris-brokered ``store.read_records``.

    The query is issued via the Expert's own tool registry, and the result
    lands in its Conversation — so the derived ExpertReport (below) is built
    from what the Expert actually did.
    """
    conv = expert._resolve_conversation(task_spec)
    level = expert.SUPPORTED_LEVELS[0] if getattr(expert, "SUPPORTED_LEVELS", ()) else None
    # The Expert self-serves within its biological-level scope (SPEC-C US8):
    # query by level, letting the Routing-Skill ranking narrow to the
    # drug/disease. Over-constraining on drug/disease/endpoint here would
    # (a) exceed the self-serve mandate and (b) empty the result set on a
    # mismatch — the parquet path collapses zero-row frames.
    params: dict[str, Any] = {"limit": 100}
    if level is not None:
        params["level"] = level

    from dargus.models.conversation import ToolCall, ToolResult

    try:
        tool = expert._get_tool("dbase_query")
        output = tool.execute(**params)
    except Exception as exc:
        logger.warning("spawn_expert: %s self-serve dbase_query failed", expert.name, exc_info=True)
        conv.add_tool(
            ToolCall(name="dbase_query", params=params),
            ToolResult(error=str(exc)),
            mode=getattr(expert, "_mode", "expert"),
        )
        return

    records = output.get("records", []) if isinstance(output, dict) else []
    conv.add_tool(
        ToolCall(name="dbase_query", params=params),
        ToolResult(output={"records": records, "count": len(records)}),
        mode=getattr(expert, "_mode", "expert"),
    )


def _derive_report_from_conversation(
    expert: Any, session_id: str, drug: str, disease: str, endpoint: str
) -> Any:
    """Derive the ExpertReport from the Expert's own Conversation (SPEC-C / #95).

    Collects the records the Expert gathered via its own ``dbase_query`` Tool
    Messages from the Conversation at *session_id*, then runs the Expert's
    ``assess()`` evidence logic over them. The report is built from what the
    Expert actually did in its PRA loop, not from records the tool fetched on
    its behalf.
    """
    from dargus.models.conversation import Conversation

    conv: Conversation = expert._resolve_conversation(
        predict_task_spec(drug=drug, disease=disease, endpoint=endpoint, session_id=session_id)
    )
    records: list[dict] = []
    for msg in conv.messages:
        if msg.tool_call is None or msg.tool_call.name != "dbase_query":
            continue
        if msg.tool_result is None or msg.tool_result.error is not None:
            continue
        output = msg.tool_result.output
        if isinstance(output, dict):
            records.extend(output.get("records", []) or [])

    ctx = ExpertContext(
        drug_ids=[drug],
        disease_id=disease,
        endpoints=[endpoint],
    )
    return expert.assess(records, ctx)


def _collect_domain_reports(iris: Iris, session_id: str) -> dict[str, Any]:
    """Collect the domain ExpertReports Iris has already gathered.

    Reads the ``spawn_expert`` Tool Messages from Iris's Conversation at
    *session_id* (the parent predict session) and rebuilds the
    ``{expert_name: [ExpertReport, ...]}`` dict the D4 director's ``conclude()``
    consumes (SPEC-C / #96). The D4 director's own report is excluded so the
    director is never fed its own output.
    """
    from dargus.experts.reports import collect_spawned_reports

    conv = iris._resolve_conversation({"session_id": session_id})
    return collect_spawned_reports(conv, skip="D4Expert")


def _run_expert_subagent(expert: Any, session_id: str, task_spec: dict[str, Any]) -> Any:
    """Run the Expert's PRA loop inside its own parent-linked Conversation.

    The sub-session is made current for the duration of the run so the
    Expert's Conversation is created under its own ``session_id``; the
    parent's ``current_session_id`` is restored on the way out so a later
    spawn in the same run still links to the same parent (SPEC-C).

    When no usable reasoning LLM is wired, the Expert's PRA loop converges
    after a deterministic self-serve ``dbase_query`` round (#95).
    """
    runtime = expert._runtime
    parent_session = getattr(runtime, "current_session_id", None)
    runtime.current_session_id = session_id
    try:
        if expert._llm_available():
            report = expert.run(task_spec)
        else:
            _self_serve(expert, task_spec)
            report = expert.run(task_spec)  # converges on the settled round
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
            if expert == "d4":
                # ── D4 synthesis (SPEC-C / #96): run the director as a
                # subagent, then pass Iris's collected ExpertReports to its
                # conclude() — the DES ± DCS contract is produced by the D4
                # spawn, not by Iris constructing the director directly.
                _run_expert_subagent(expert_agent, sub_session, task_spec)
                all_reports = _collect_domain_reports(iris, parent_session)
                conclusion = expert_agent.conclude(drug, disease, endpoint, all_reports)
                return {
                    "expert": expert_agent.name,
                    "session_id": sub_session,
                    "report": expert_report_to_dict(
                        _derive_report_from_conversation(
                            expert_agent, sub_session, drug, disease, endpoint
                        )
                    ),
                    "final_report": final_report_to_dict(conclusion),
                }

            # ── Domain Expert (self-serves evidence) ────────────────
            _run_expert_subagent(expert_agent, sub_session, task_spec)
            report = _derive_report_from_conversation(
                expert_agent, sub_session, drug, disease, endpoint
            )
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
