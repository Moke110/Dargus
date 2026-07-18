from __future__ import annotations

import hashlib

import numpy as np
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

DRUG_EMBEDDING_DIM = 2048


def drug_morgan_embedding(
    smiles: str, radius: int = 2, n_bits: int = DRUG_EMBEDDING_DIM
) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=float)
    fp = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=float)


def disease_onehot_embedding(disease_name: str, dim: int = 128) -> np.ndarray:
    h = hashlib.sha1(disease_name.encode("utf-8")).hexdigest()
    idx = int(h, 16) % dim
    vec = np.zeros(dim, dtype=float)
    vec[idx] = 1.0
    return vec
