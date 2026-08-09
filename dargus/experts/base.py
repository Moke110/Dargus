"""Expert abstract base class — domain agents inheriting Harness from BaseAgent."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dargus.agents.base import BaseAgent
from dargus.agents.skill_registry import SkillRegistry
from dargus.experts.protocol import (
    ConfidenceInterval,
    EvidenceAssessment,
    ExpertContext,
    ExpertReport,
    TaskDelegation,
)
from dargus.models.reasoning import ReasoningLLM
from dargus.runtime.hooks import HookRegistry
from dargus.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-domain default stub evidence dicts keyed by SUPPORTED_LEVELS[0]
# ---------------------------------------------------------------------------

_STUB_EVIDENCE: dict[str, dict[str, Any]] = {
    "molecular": {
        "biological_level": "molecular",
        "evidence_design": "descriptive",
        "sources": [{"rank": 1, "type": "journal", "name": "stub"}],
        "source_entry": "stub://molecular",
        "source_time": "2024-01-01",
        "xy": {"count": 1},
        "x": {"type": "drug", "value": [{"entity_id": "chembl:1", "entity_label": "test_drug"}]},
        "y": {"type": "binding_affinity", "category": "binding", "value": [1.0]},
        "bg": {"dose_value": 1.0, "dose_unit": "µM"},
    },
    "cellular": {
        "biological_level": "cellular",
        "evidence_design": "descriptive",
        "sources": [{"rank": 1, "type": "journal", "name": "stub"}],
        "source_entry": "stub://cellular",
        "source_time": "2024-01-01",
        "xy": {"count": 1},
        "x": {"type": "drug", "value": [{"entity_id": "chembl:1", "entity_label": "test_drug"}]},
        "y": {"type": "viability_72h", "category": "viability", "value": [50.0], "assay": "MTT"},
        "cell_line_id": "cellosaurus:CVCL_0030",
    },
    "animal": {
        "biological_level": "animal",
        "evidence_design": "two_arm_comparison",
        "sources": [{"rank": 1, "type": "journal", "name": "stub"}],
        "source_entry": "stub://animal",
        "source_time": "2024-01-01",
        "xy": {"count": 2},
        "x": {
            "type": "drug",
            "value": [
                {"entity_id": "chembl:1", "entity_label": "test_drug"},
                {"entity_id": None, "entity_label": "vehicle"},
            ],
        },
        "y": {
            "type": "tumor_volume_change",
            "category": "clinic_efficacy_secondary",
            "value": [-30.0, 5.0],
            "direction": "beneficial",
        },
        "model_organism": "NCBITaxon:10090",
    },
    "rct": {
        "biological_level": "rct",
        "evidence_design": "two_arm_comparison",
        "sources": [{"rank": 1, "type": "journal", "name": "stub"}],
        "source_entry": "stub://rct",
        "source_time": "2024-01-01",
        "xy": {"count": 2},
        "x": {
            "type": "drug",
            "value": [
                {"entity_id": "chembl:1", "entity_label": "test_drug"},
                {"entity_id": None, "entity_label": "placebo"},
            ],
        },
        "y": {
            "type": "overall_survival",
            "category": "clinic_efficacy_primary",
            "value": [0.75, 0.60],
            "direction": "beneficial",
        },
        "bg": {
            "disease_id": ["mondo:0005249"],
            "drugs": [{"entity_id": "chembl:1", "entity_label": "test_drug"}],
        },
        "clinical_design": {
            "comparator_type": "placebo",
            "blinding": "double",
            "randomized": True,
            "phase": "phase_3",
            "n_arms": 2,
            "population": "adults",
        },
        "is_primary_endpoint": True,
    },
    "epi": {
        "biological_level": "epi",
        "evidence_design": "observational_association",
        "sources": [{"rank": 1, "type": "journal", "name": "stub"}],
        "source_entry": "stub://epi",
        "source_time": "2024-01-01",
        "xy": {"count": 2},
        "x": {
            "type": "drug",
            "value": [
                {"entity_id": "chembl:1", "entity_label": "test_drug"},
                {"entity_id": None, "entity_label": "untreated"},
            ],
        },
        "y": {
            "type": "incidence_rate_ratio",
            "category": "clinic_efficacy_secondary",
            "value": [0.82, 1.0],
            "direction": "beneficial",
        },
        "bg": {
            "disease_id": ["doid:9352"],
            "drugs": [{"entity_id": "chembl:1", "entity_label": "test_drug"}],
        },
        "clinical_design": {"population": "adults"},
    },
}

# Domain -> canonical biological level used in stub dicts
_DOMAIN_STUB_LEVEL: dict[str, str] = {
    "molecule": "molecular",
    "biomedical": "cellular",
    "bioinformatics": "cellular",
    "clinical": "rct",
}

#: The single canonical biological level -> target Expert routing table (ADR-0004).
#: Routing is wiring/behaviour and lives in code next to the Expert classes, NOT
#: in ``vocabularies.json`` (which owns the levels themselves and their
#: clinical/simulation flags). Re-routing or adding a level is a one-line change.
#: Consumed by :meth:`Expert.delegate_target`; previously duplicated five times
#: as per-expert ``DELEGATION_RULES`` dicts (molecule / biomed / clinic / bioinfo
#: and the D4 director).
BIOLOGICAL_LEVEL_DELEGATION: dict[str, str] = {
    "molecular": "MoleculeExpert",
    "molecular-sim": "MoleculeExpert",
    "cellular": "BiomedExpert",
    "cellular-sim": "BiomedExpert",
    "exvivo": "BiomedExpert",
    "exvivo-sim": "BiomedExpert",
    "animal": "BiomedExpert",
    "animal-sim": "BiomedExpert",
    "rct": "ClinicExpert",
    "epi": "ClinicExpert",
    "rct-sim": "ClinicExpert",
}


class Expert(BaseAgent):
    """Domain expert with biological level declarations and an assessment loop.

    The base owns the single canonical ``biological_level → Expert`` routing
    table (:data:`BIOLOGICAL_LEVEL_DELEGATION`) and the ``assess()`` template
    method. Each Expert declares only what makes it different:

      - SUPPORTED_LEVELS: which biological levels it can assess
      - SIM_PENALTY / SIM_BIAS_MSG: simulation-derived record quality penalty
        and the bias-note text for it
      - RELEVANCE_MAP: level → relevance label (default ``medium``)
      - PERMITTED_TOOLS: tools this expert may call during execution
      - SUPPORTED_SKILLS: skills this expert may load during planning

    Experts may override one optional hook, ``_collect_gaps`` (post-loop data
    gap / bias logic), and one optional gate, ``_gate`` (per-record admission;
    defaults to ``can_handle``). ``_assess_quality`` and ``_assess_confidence``
    remain overridable per domain.
    """

    SUPPORTED_LEVELS: tuple[str, ...] = ()
    SIM_PENALTY: float = 0.2
    SIM_BIAS_MSG: str = "Record {eid}: simulation-derived data ({level})"
    RELEVANCE_MAP: dict[str, str] = {}

    def __init__(
        self,
        dbase: Any = None,
        config: dict[str, Any] | None = None,
        reasoning_llm: ReasoningLLM | None = None,
        tool_registry: ToolRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        hook_registry: HookRegistry | None = None,
        mode: str = "auto",
        mode_config: dict[str, Any] | None = None,
    ):
        super().__init__(
            config=config,
            reasoning_llm=reasoning_llm,
            tool_registry=tool_registry,
            skill_registry=skill_registry,
            hook_registry=hook_registry,
            mode=mode,
            mode_config=mode_config,
        )
        self.dbase = dbase

    # ------------------------------------------------------------------
    # Extract — convert domain files into structured evidence instances
    # ------------------------------------------------------------------

    def can_handle(self, record: dict) -> bool:
        level = self._read_biological_level(record)
        return level in self.SUPPORTED_LEVELS

    def delegate_target(self, record: dict) -> str | None:
        level = self._read_biological_level(record)
        if level is None:
            return None
        return BIOLOGICAL_LEVEL_DELEGATION.get(level)

    def _read_biological_level(self, record: dict) -> str | None:
        return record.get("biological_level")

    def _read_field(self, record: dict, field_name: str) -> Any:
        return record.get(field_name)

    # ------------------------------------------------------------------
    # Shared assessment loop — the template method every domain Expert uses
    # ------------------------------------------------------------------

    def assess(
        self,
        records: list[dict],
        context: ExpertContext,
    ) -> ExpertReport:
        """Assess a batch of records, delegating out-of-scope ones.

        Template method shared by all domain Experts: for each record, read
        its biological level, skip when absent, gate admission, assess quality,
        apply the simulation penalty and bias note, and append a finding. After
        the loop, collect domain data gaps via ``_collect_gaps``, assess
        confidence, and build the :class:`ExpertReport`.

        Domain-specific behaviour is declared as class attributes
        (``SIM_PENALTY``, ``SIM_BIAS_MSG``, ``RELEVANCE_MAP``) and the two
        overridable hooks ``_gate`` and ``_collect_gaps``.
        """
        findings: list[EvidenceAssessment] = []
        delegations: list[TaskDelegation] = []
        data_gaps: list[str] = []
        bias_notes: list[str] = []

        for record in records:
            eid = record.get("evidence_id", "")
            level = self._read_biological_level(record)
            if level is None:
                continue

            if not self._gate(record):
                target = self.delegate_target(record)
                if target:
                    delegations.append(
                        TaskDelegation(
                            target_expert=target,
                            record_ids=[eid],
                            reason=self._delegation_reason(level, target),
                        )
                    )
                continue

            quality = self._assess_quality(record)
            if "-sim" in (level or "") and self.SIM_PENALTY > 0:
                bias_notes.append(self._sim_bias_msg(eid, level))
                quality = max(0.0, quality - self.SIM_PENALTY)

            findings.append(
                EvidenceAssessment(
                    record_ids=[eid],
                    biological_level=level or "unknown",
                    relevance=self._relevance(level or "unknown"),
                    quality_score=quality,
                    limitations=[],
                )
            )

        extra_gaps, extra_bias = self._collect_gaps(records, findings)
        data_gaps.extend(extra_gaps)
        bias_notes.extend(extra_bias)

        confidence = self._assess_confidence(findings)
        return ExpertReport(
            expert=self.name,
            round=context.round,
            findings=findings,
            confidence=confidence,
            delegations=delegations,
            data_gaps=data_gaps,
            bias_notes=bias_notes,
        )

    def _gate(self, record: dict) -> bool:
        """Admission gate for a record: default is level scope (``can_handle``).

        Bioinfo overrides this with its high-throughput assay test.
        """
        return self.can_handle(record)

    def _relevance(self, level: str) -> str:
        """Relevance label for a biological level (default ``medium``)."""
        return self.RELEVANCE_MAP.get(level, "medium")

    def _sim_bias_msg(self, eid: str, level: str) -> str:
        """Format the simulation-derived bias note for a record."""
        try:
            return self.SIM_BIAS_MSG.format(eid=eid, level=level)
        except (KeyError, IndexError):
            return self.SIM_BIAS_MSG

    def _delegation_reason(self, level: str, target: str) -> str:
        """Reason text attached to a delegation (default scope message)."""
        return f"Record level '{level}' outside {self.name} scope"

    def _assess_quality(self, record: dict) -> float:
        """Assess a single record's evidence quality (overridden per domain)."""
        raise NotImplementedError(f"{self.name} must implement _assess_quality")

    def _assess_confidence(self, findings: list[EvidenceAssessment]) -> ConfidenceInterval:
        """Assess overall confidence from the findings (overridden per domain)."""
        raise NotImplementedError(f"{self.name} must implement _assess_confidence")

    def _collect_gaps(
        self, records: list[dict], findings: list[EvidenceAssessment]
    ) -> tuple[list[str], list[str]]:
        """Post-loop domain logic: extra data gaps and bias notes.

        Default is a no-op. Biomed overrides it for the in-vivo/in-vitro data
        gap; Clinic overrides it for mixed-direction detection and the "no real
        clinical evidence" gap. Receives both *records* and *findings* because
        Clinic's mixed-direction detection reads raw records.
        """
        return [], []

    # ------------------------------------------------------------------
    # Prompt overrides — domain-specific system prompts
    # ------------------------------------------------------------------

    def _build_reason_prompt(self) -> str:
        return (
            f"You are {self.name}, a biomedical domain expert specializing in "
            f"evidence at biological levels: {', '.join(self.SUPPORTED_LEVELS)}.\n"
            "Given a task specification and available tools, "
            "return a JSON response.\n\n"
            "Output format:\n"
            '{"mode": "<current_mode>", "action": "<text|tool_call>", '
            '"text": "<response if action is text>", '
            '"tool": "<tool name if action is tool_call>", '
            '"params": {}}'
        )

    # ------------------------------------------------------------------
    # Extract — convert domain files into structured evidence instances
    # ------------------------------------------------------------------

    def extract(self, source: str | list[str]) -> list[dict[str, Any]]:
        """Extract structured evidence instances from one or more source files.

        When an LLM is wired, sends each file through the reasoning LLM
        with a domain-specific extraction prompt.  When no LLM is present
        (stub/debug mode), returns a single stub instance per domain
        matching the 50-field schema shape.

        Individual file failures **raise** so that callers (e.g. the
        Convert phase) can count errors and skip the file.  No internal
        catching is done here — the workflow layer handles per-file
        resilience.

        Args:
            source: A single file path or a list of file paths to process.

        Returns:
            A list of evidence instance dicts, each with the 50-field
            schema shape defined in ``design/3.2_D-Base_field_reference.md``.
        """
        if isinstance(source, str):
            sources: list[str] = [source]
        else:
            sources = source

        if not self._reasoning_llm:
            return self._extract_stub()

        instances: list[dict[str, Any]] = []
        for path_str in sources:
            result = self._extract_one(path_str)
            if result is not None:
                instances.append(result)
        return instances

    def _extract_one(self, file_path: str) -> dict[str, Any] | None:
        """Extract evidence from a single file via LLM.

        Returns the parsed evidence dict, or ``None`` if the file cannot
        be read.  Raises :class:`ValueError` when the LLM call fails or
        produces unparseable output (so the caller can log and skip).
        """
        content = self._read_file_content(file_path)
        if not content:
            return None

        system_prompt = self._build_extract_prompt()
        user_prompt = f"File: {Path(file_path).name}\n\nContent:\n{content[:8000]}"

        response = self._llm_call(system_prompt, user_prompt)
        try:
            parsed = json.loads(response.strip())
        except json.JSONDecodeError:
            raise ValueError(f"{self.name}: unparseable LLM output for {file_path}") from None

        # Detect stub error payloads returned by _llm_call on exception
        if isinstance(parsed, dict) and "error" in parsed:
            raise ValueError(f"{self.name}: LLM call error for {file_path}: {parsed['error']}")

        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed:
            return parsed[0]

        raise ValueError(
            f"{self.name}: unexpected LLM output shape for {file_path}: " f"{type(parsed).__name__}"
        )

    def _extract_stub(self) -> list[dict[str, Any]]:
        """Return a stub evidence instance per domain when no LLM is wired."""
        primary_level = self.SUPPORTED_LEVELS[0] if self.SUPPORTED_LEVELS else "molecular"
        stub = _STUB_EVIDENCE.get(primary_level)
        if stub is not None:
            return [dict(stub)]
        return []

    @staticmethod
    def _read_file_content(file_path: str) -> str | None:
        """Read a file into a string, returning None on any error."""
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return None
        try:
            return p.read_text(encoding="utf-8")[:8000]
        except Exception:
            logger.warning("Failed to read %s", file_path)
            return None

    def _build_extract_prompt(self) -> str:
        """Build the extraction prompt for this expert's domain."""
        domain = self.SUPPORTED_LEVELS[0] if self.SUPPORTED_LEVELS else "molecular"
        return (
            f"You are {self.name}, a biomedical expert extracting structured evidence "
            f"from source files in the {domain} domain.\n"
            "Extract a single D-Base evidence instance as a JSON object covering "
            "these top-level keys:\n"
            "  biological_level, evidence_design, sources, source_entry, source_time,\n"
            "  xy, x, y,\n"
            "  bg (drugs, genes, disease_id, dose_value, dose_unit),\n"
            "  cell_line_id, model_organism, strain, sex, age, tissue, cell_type,\n"
            "  clinical_design (for rct/epi levels),\n"
            "  related_evidence_id, is_primary_endpoint, p_value_adjusted.\n"
            "Use vocabularies from dargus/dbase/vocabularies.json.\n"
            "Return ONLY the JSON object, no commentary."
        )
