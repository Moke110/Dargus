"""delegate_to_expert — Tool wrapper for D4Expert coordination."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dargus.experts.director import FourDExpert


def delegate_to_expert(
    d4_expert: FourDExpert,
    domain: str,
    records: list,
    question: str,
) -> dict[str, Any]:
    """Delegate evidence assessment to a DomainExpert. Returns ExpertReport.

    This is a thin wrapper — the real logic is in :class:`FourDExpert`.
    The tool function exists so it can be registered and called by agents
    in the P-R-A loop.

    Args:
        d4_expert: The FourDExpert coordinator instance.
        domain: Domain key (``"molecular"``, ``"biomedical"``, etc.).
        records: List of evidence records to assess.
        question: The assessment question.

    Returns:
        Expert report dict with keys: ``domain``, ``conclusion``,
        ``confidence``, ``supporting_evidence``.
    """
    return d4_expert.delegate_to_expert(domain, records, question)
