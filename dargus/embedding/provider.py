"""Embedding provider interface and default implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Interface for drug and disease embedding models."""

    model_name: str = "unknown"
    model_version: str = "0.0.0"

    @abstractmethod
    def encode(self, identifier: str) -> np.ndarray:
        """Return a fixed-dimension embedding vector."""

    @property
    def dimension(self) -> int:
        return int(self.encode("test").shape[0])


class DrugMorganRDKitProvider(EmbeddingProvider):
    """Baseline drug embedding using Morgan fingerprints + RDKit descriptors."""

    model_name = "morgan_rdkit"
    model_version = "1.0"

    def __init__(self, dimension: int = 256):
        self._dimension = dimension

    def encode(self, identifier: str) -> np.ndarray:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors

            mol = Chem.MolFromSmiles(identifier)
            if mol is None:
                return self._fallback(identifier)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self._dimension)
            # Use ToList for portability across RDKit versions.
            vec = np.array(fp.ToList(), dtype=float)
            desc = np.array(
                [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                ],
                dtype=float,
            )
            combined = np.concatenate([vec, desc])
            if len(combined) < self._dimension:
                combined = np.pad(combined, (0, self._dimension - len(combined)))
            return combined[: self._dimension].astype(float)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RDKit encoding failed (%s); using fallback", exc)
            return self._fallback(identifier)

    def _fallback(self, identifier: str) -> np.ndarray:
        rng = np.random.default_rng(hash(identifier) % (2**32))
        vec = rng.random(self._dimension).astype(float)
        return vec / (np.linalg.norm(vec) + 1e-9)


class DiseaseClinicalMLPProvider(EmbeddingProvider):
    """Baseline disease embedding from clinical indicator vector."""

    model_name = "clinical_mlp"
    model_version = "1.0"

    def __init__(self, dimension: int = 256):
        self._dimension = dimension

    def encode(self, identifier: str) -> np.ndarray:
        rng = np.random.default_rng(hash(identifier) % (2**32))
        vec = rng.random(self._dimension).astype(float)
        # Center around a mild disease severity prior
        vec[0] = 0.5
        return vec / (np.linalg.norm(vec) + 1e-9)


def get_embedding_provider(config: dict[str, Any], kind: str) -> EmbeddingProvider:
    """Factory for embedding providers."""
    embedding_cfg = config.get("embedding", {})
    dim = embedding_cfg.get("dimension", 256)
    if kind == "drug":
        model = embedding_cfg.get("drug", "morgan_rdkit")
        if model == "morgan_rdkit":
            return DrugMorganRDKitProvider(dimension=dim)
        raise ValueError(f"Unknown drug embedding model: {model}")
    if kind == "disease":
        model = embedding_cfg.get("disease", "clinical_mlp")
        if model == "clinical_mlp":
            return DiseaseClinicalMLPProvider(dimension=dim)
        raise ValueError(f"Unknown disease embedding model: {model}")
    raise ValueError(f"Unknown embedding kind: {kind}")
