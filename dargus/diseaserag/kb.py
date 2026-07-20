from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from dargus.dbase.nlp import DBaseNLP, MockNLP
from dargus.dbase.paths import default_dargus_home
from dargus.diseaserag.chunker import MarkdownChunker


class DiseaseRAG:
    """Local vector knowledge base for one disease. Supports add/query with idempotent adds."""

    def __init__(
        self,
        disease_id: str,
        kb_dir: Path | None = None,
        chunker: MarkdownChunker | None = None,
        embedder: DBaseNLP | MockNLP | None = None,
    ):
        self.disease_id = disease_id
        self.kb_root = Path(kb_dir or default_dargus_home()) / "diseaserag"
        self.disease_dir = self.kb_root / disease_id
        self.disease_dir.mkdir(parents=True, exist_ok=True)

        self.chunker = chunker or MarkdownChunker(max_tokens=512, overlap=50)
        self.embedder = embedder or DBaseNLP()

        self.chunks_path = self.disease_dir / "chunks.jsonl"
        self.embeddings_path = self.disease_dir / "embeddings.npz"
        self.metadata_path = self.disease_dir / "metadata.json"

        self._chunks: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None
        self._file_hashes: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self.chunks_path.exists():
            with self.chunks_path.open("r", encoding="utf-8") as fh:
                self._chunks = [json.loads(line) for line in fh if line.strip()]
        self._file_hashes = {c["file_hash"] for c in self._chunks}
        if self.embeddings_path.exists():
            self._embeddings = sp.load_npz(self.embeddings_path).toarray()

    def add_documents(self, file_paths: list[str | Path]) -> None:
        new_chunks: list[dict[str, Any]] = []
        for path in file_paths:
            path = Path(path)
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if file_hash in self._file_hashes:
                continue
            text_chunks = self.chunker.split(content)
            for idx, chunk_text in enumerate(text_chunks):
                new_chunks.append(
                    {
                        "file_hash": file_hash,
                        "file_path": str(path),
                        "chunk_index": idx,
                        "text": chunk_text,
                    }
                )

        if not new_chunks:
            return

        vectors_list = [self.embedder.embed_text(c["text"]) for c in new_chunks]
        vectors = np.array(vectors_list, dtype=np.float32)

        self._chunks.extend(new_chunks)
        if self._embeddings is None:
            self._embeddings = vectors
        else:
            self._embeddings = np.vstack([self._embeddings, vectors])

        self._file_hashes.update(c["file_hash"] for c in new_chunks)
        self._save()

    def query(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        query_vec = self.embedder.embed_text(text).astype(np.float32)
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm == 0:
            return []

        norms = np.linalg.norm(self._embeddings, axis=1)
        valid = norms > 0
        similarities = np.zeros(len(self._chunks))
        if valid.any():
            similarities[valid] = (
                (self._embeddings[valid] @ query_vec) / (norms[valid] * query_norm)
            )

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self._chunks[i]["text"], float(similarities[i])) for i in top_indices]

    def _save(self) -> None:
        with self.chunks_path.open("w", encoding="utf-8") as fh:
            for chunk in self._chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        if self._embeddings is not None and self._embeddings.shape[0] > 0:
            sp.save_npz(self.embeddings_path, sp.csr_matrix(self._embeddings))

        metadata = {
            "disease_id": self.disease_id,
            "n_chunks": len(self._chunks),
            "embedding_dim": (
                int(self._embeddings.shape[1]) if self._embeddings is not None else 0
            ),
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
