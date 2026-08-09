"""Test Expert base class."""

from dargus.experts.base import BIOLOGICAL_LEVEL_DELEGATION, Expert
from dargus.experts.protocol import ExpertContext, ExpertReport


class _TestExpert(Expert):
    SUPPORTED_LEVELS = ("molecular",)

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
    assert expert.SIM_PENALTY == 0.2
    assert expert.RELEVANCE_MAP == {}


def test_concrete_expert_has_levels():
    expert = _TestExpert()
    assert expert.SUPPORTED_LEVELS == ("molecular",)


def test_expert_delegate_target_uses_single_map():
    """Routing resolves through the single canonical delegation map (#100)."""
    expert = _TestExpert()
    assert expert.delegate_target({"biological_level": "cellular"}) == "BiomedExpert"
    assert expert.delegate_target({"biological_level": "rct"}) == "ClinicExpert"
    assert expert.delegate_target({"biological_level": "molecular"}) == "MoleculeExpert"
    assert expert.delegate_target({}) is None


def test_single_canonical_delegation_map():
    """One canonical biological_level -> Expert map on the base; the per-expert
    DELEGATION_RULES copies are gone."""
    assert "cellular" in BIOLOGICAL_LEVEL_DELEGATION
    assert BIOLOGICAL_LEVEL_DELEGATION["cellular"] == "BiomedExpert"
    assert BIOLOGICAL_LEVEL_DELEGATION["rct"] == "ClinicExpert"
    assert BIOLOGICAL_LEVEL_DELEGATION["epi"] == "ClinicExpert"
    assert BIOLOGICAL_LEVEL_DELEGATION["molecular"] == "MoleculeExpert"
    # Every delegation target is a real level Expert class name. Bioinfo is a
    # cross-cutting expert — it is never a delegation *target* of the level map.
    from dargus.experts.biomed import BiomedExpert
    from dargus.experts.clinic import ClinicExpert
    from dargus.experts.molecule import MoleculeExpert

    targets = set(BIOLOGICAL_LEVEL_DELEGATION.values())
    assert targets == {cls.name for cls in (MoleculeExpert, BiomedExpert, ClinicExpert)}


def test_no_per_expert_delegation_rules_remain():
    """None of the domain Experts carry their own DELEGATION_RULES (#100)."""
    from dargus.experts.bioinfo import BioinfoExpert
    from dargus.experts.biomed import BiomedExpert
    from dargus.experts.clinic import ClinicExpert
    from dargus.experts.molecule import MoleculeExpert

    for cls in (MoleculeExpert, BiomedExpert, ClinicExpert, BioinfoExpert):
        assert not hasattr(cls, "DELEGATION_RULES")


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
