"""delegate_to_expert — Tool wrapper for D4Expert coordination."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dargus.tools.base import Tool, ToolParam

if TYPE_CHECKING:
    from dargus.experts.director import D4Expert


def delegate_to_expert(
    d4_expert: D4Expert,
    domain: str,
    records: list,
    question: str,
) -> dict[str, Any]:
    """Delegate evidence assessment to a DomainExpert. Returns ExpertReport.

    This is a thin wrapper — the real logic is in :class:`D4Expert`.
    The tool function exists so it can be registered and called by agents
    in the P-R-A loop.

    Args:
        d4_expert: The D4Expert coordinator instance.
        domain: Domain key (``"molecular"``, ``"biomedical"``, etc.).
        records: List of evidence records to assess.
        question: The assessment question.

    Returns:
        Expert report dict with keys: ``domain``, ``conclusion``,
        ``confidence``, ``supporting_evidence``.
    """
    return d4_expert.delegate_to_expert(domain, records, question)


def make_delegate_tool(d4_expert: D4Expert) -> Tool:
    """Create a Tool wrapping :func:`delegate_to_expert` for the P-R-A loop.

    Args:
        d4_expert: The D4Expert coordinator instance to bind.

    Returns:
        A :class:`Tool` ready for registration and execution.
    """
    tool = Tool(
        name="delegate_to_expert",
        description=(
            "Delegate evidence assessment to the appropriate DomainExpert "
            "for a given domain (molecular, biomed, bioinformatics, clinical). "
            "Returns a structured expert report with conclusion, confidence "
            "interval, and supporting evidence."
        ),
        parameters=[
            ToolParam(
                name="domain",
                type="string",
                required=True,
                description="Target domain key (molecular, biomed, bioinformatics, clinical)",
                enum=["molecular", "biomed", "bioinformatics", "clinical"],
            ),
            ToolParam(
                name="records",
                type="array",
                required=True,
                description="List of evidence records to assess",
            ),
            ToolParam(
                name="question",
                type="string",
                required=True,
                description="The assessment question to answer",
            ),
        ],
        output={
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "conclusion": {"type": "string"},
                "confidence": {
                    "type": "object",
                    "properties": {
                        "low": {"type": "number"},
                        "high": {"type": "number"},
                    },
                },
                "supporting_evidence": {"type": "array"},
            },
        },
        timeout_ms=30_000,
        fallback="null_result",
    )
    tool.bind(
        lambda domain, records, question: d4_expert.delegate_to_expert(domain, records, question)
    )
    return tool
