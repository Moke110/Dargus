"""Standalone smoke: Models — config parsing + offline embedding round-trip.

Pins two model invariants: (1) ``load_model_config`` parses a real temp config
into a ModelConfig with the expected provider/model, and (2) the embedding
wiring does a deterministic offline round-trip through a stubbed backend,
producing stable, self-consistent vectors. No network, no real model download.

Contract: prints a ``PASS`` / ``FAIL`` / ``SKIP`` verdict line and exits 0 on
pass/skip, non-zero on fail. Run directly:  python smoke_models.py
"""

from __future__ import annotations

import sys

from _bootstrap import ensure_dargus_on_path

ensure_dargus_on_path()


class _OfflineEmbeddingBackend:
    """Deterministic offline stand-in for the default embedding backend.

    Mirrors the unit-suite hash-embedding stub (conftest.py) so the smoke
    proves the EmbeddingModel wiring without a Hugging Face download.
    """

    _model_name = "smoke-offline"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * 384
            for i, b in enumerate(text.encode("utf-8")):
                vec[i % 384] += float(b)
            norm = sum(v * v for v in vec) ** 0.5
            vectors.append([v / norm for v in vec] if norm else vec)
        return vectors


def main() -> int:
    from dargus.models.config import EnvSecretsManager, load_model_config
    from dargus.models.embedding import EmbeddingModel

    # 1. Config → ModelConfig wiring from a real temp config dict.
    config = {
        "models": {
            "reasoning": {"provider": "deepseek", "model": "deepseek-v4-pro", "temperature": 0.0},
            "embedding": {"provider": "sentence_transformers", "model": "all-MiniLM-L6-v2"},
        }
    }
    secrets = EnvSecretsManager()
    model_config = load_model_config(config, secrets)
    assert model_config.reasoning_provider == "deepseek"
    assert model_config.reasoning_model == "deepseek-v4-pro"
    assert model_config.embedding_model == "all-MiniLM-L6-v2"

    # 2. Offline embedding round-trip through the real EmbeddingModel wiring.
    model = EmbeddingModel(_OfflineEmbeddingBackend())
    texts = ["aspirin reduces inflammation", "placebo control arm"]
    vectors = model.embed(texts)
    assert len(vectors) == len(texts)
    assert all(len(v) == 384 for v in vectors), "embedding dim != 384"

    # Deterministic: embedding the same text twice gives the same vector.
    again = model.embed(["aspirin reduces inflammation"])
    assert vectors[0] == again[0], "embedding not deterministic"

    # Self-consistent: a text is most similar to itself.
    s = EmbeddingModel.similarity(vectors[0], vectors[0])
    assert abs(s - 1.0) < 1e-6, f"self-similarity {s} != 1"

    # Dimension mismatch is rejected, not silently coerced.
    short = vectors[0][:100]
    try:
        EmbeddingModel.similarity(vectors[0], short)
    except ValueError:
        pass
    else:
        raise AssertionError("similarity accepted mismatched dimensions")

    # model_name exposes the backend identity (used by the D-Base fingerprint).
    assert model.model_name == "smoke-offline"

    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — smoke scripts report any failure as FAIL
        print(f"FAIL: {exc.__class__.__name__}: {exc}")
        sys.exit(1)
