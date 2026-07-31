"""Test Expert base class."""

from dargus.experts.base import Expert
from dargus.experts.protocol import ExpertContext, ExpertReport


class _TestExpert(Expert):
    SUPPORTED_LEVELS = ("molecular",)
    DELEGATION_RULES = {"cellular": "BiomedExpert"}

    def assess(self, records, context):
        return ExpertReport(
            expert=self.__class__.__name__,
            round=context.round,
            findings=[],
            confidence=None,
        )


def test_expert_cannot_instantiate_abstract():
    # Expert is no longer abstract — it inherits BaseAgent's unified PRA loop.
    # Subclasses override _build_reason_prompt() for domain-specific behavior.
    # Bare Expert can be instantiated and runs the default PRA loop.
    expert = Expert()
    assert expert.SUPPORTED_LEVELS == ()
    assert expert.DELEGATION_RULES == {}


def test_concrete_expert_has_levels():
    expert = _TestExpert()
    assert expert.SUPPORTED_LEVELS == ("molecular",)


def test_expert_delegation_rules():
    expert = _TestExpert()
    assert expert.DELEGATION_RULES["cellular"] == "BiomedExpert"


def test_expert_assess_returns_report():
    expert = _TestExpert()
    ctx = ExpertContext(
        drug_ids=["D1"],
        disease_id="Dis1",
        endpoints=["EP1"],
        round=1,
    )
    report = expert.assess([], ctx)
    assert isinstance(report, ExpertReport)
    assert report.expert == "_TestExpert"
    assert report.round == 1
