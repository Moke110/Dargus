"""Dargus expert system."""

from dargus.experts.base import Expert
from dargus.experts.biomed import BiomedExpert
from dargus.experts.bioinfo import BioinfoExpert
from dargus.experts.clinic import ClinicExpert
from dargus.experts.director import FourDExpert
from dargus.experts.iris_expert import IrisExpert
from dargus.experts.molecule import MoleculeExpert
from dargus.experts.protocol import (
    ConfidenceInterval,
    EvidenceAssessment,
    ExpertContext,
    ExpertReport,
    FinalReport,
    TaskDelegation,
)

# Backward-compatible re-exports
from dargus.experts.disease import DiseaseExpert
from dargus.experts.levels import (
    AnimalExpert,
    CellularExpert,
    ClinicalExpert,
    EpiExpert,
    ExvivoExpert,
    LevelExpert,
    MolecularExpert,
)
from dargus.experts.types import (
    AnalysisReport,
    CurateResult,
    ExtractedInstance,
    ExtractionReport,
    IngestionResult,
    IngestionSummary,
    PlanProposal,
)

__all__ = [
    # New v0.9.0 Expert system
    "Expert",
    "IrisExpert",
    "MoleculeExpert",
    "BiomedExpert",
    "BioinfoExpert",
    "ClinicExpert",
    "FourDExpert",
    # Protocol types
    "ExpertReport",
    "EvidenceAssessment",
    "TaskDelegation",
    "ConfidenceInterval",
    "FinalReport",
    "ExpertContext",
    # Backward-compatible (deprecated)
    "DiseaseExpert",
    "LevelExpert",
    "MolecularExpert",
    "CellularExpert",
    "ExvivoExpert",
    "AnimalExpert",
    "ClinicalExpert",
    "EpiExpert",
    "AnalysisReport",
    "CurateResult",
    "ExtractedInstance",
    "ExtractionReport",
    "IngestionResult",
    "IngestionSummary",
    "PlanProposal",
]
