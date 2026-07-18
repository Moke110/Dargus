from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from dargus.dbase.record import TemplateRecord
from dargus.dbase.template import TemplateSchema
from dargus.dbase.vocabulary import VocabularyManager


class DBase:
    """Experiment-level conclusion store: sparse matrix + factor vocabularies."""

    def __init__(self, project_id: str, root_dir: str | Path):
        self.project_id = project_id
        self.root = Path(root_dir)
        self.dbase_dir = self.root / "dbase"
        self.templates_dir = self.dbase_dir / "templates"
        self.records_path = self.dbase_dir / "records.jsonl"
        self.vocab_path = self.dbase_dir / "vocabularies.json"
        self.matrix_path = self.dbase_dir / "records.npz"
        self.index_dir = self.dbase_dir / "index"

        self._records: list[TemplateRecord] = []
        self._templates: dict[str, TemplateSchema] = {}
        self._vocab: VocabularyManager | None = None
        self._matrix: sp.csr_matrix | None = None
        self._dirty = True

        self._ensure_dirs()
        self._load()

    def _ensure_dirs(self) -> None:
        for d in [self.templates_dir, self.index_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if self.vocab_path.exists():
            self._vocab = VocabularyManager.load(self.vocab_path)
        else:
            self._vocab = VocabularyManager()

        for yaml_path in self.templates_dir.glob("*.yaml"):
            schema = TemplateSchema.from_yaml(yaml_path)
            self._templates[schema.template_id] = schema

        if self.records_path.exists():
            with self.records_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self._records.append(TemplateRecord(**data))

        if self.matrix_path.exists():
            loaded = sp.load_npz(self.matrix_path)
            self._matrix = loaded

    @property
    def vocab(self) -> VocabularyManager:
        if self._vocab is None:
            raise RuntimeError("VocabularyManager not initialized")
        return self._vocab

    def add_template(self, schema: TemplateSchema) -> None:
        self._templates[schema.template_id] = schema
        schema.to_yaml(self.templates_dir / f"{schema.template_id}.yaml")

    def get_template(self, template_id: str) -> TemplateSchema:
        if template_id not in self._templates:
            raise KeyError(f"Template {template_id!r} not found")
        return self._templates[template_id]

    def add_record(self, record: TemplateRecord) -> None:
        self._records.append(record)
        self._dirty = True

    def query(
        self,
        template_id: str | None = None,
        drug_id: str | None = None,
        disease_id: str | None = None,
    ) -> list[TemplateRecord]:
        results = []
        for rec in self._records:
            if template_id and rec.template_id != template_id:
                continue
            if drug_id and not self._record_has_factor(rec, "drug_id", drug_id):
                continue
            if disease_id and not self._record_has_factor(rec, "disease_id", disease_id):
                continue
            results.append(rec)
        return results

    def _record_has_factor(self, record: TemplateRecord, field_name: str, term: str) -> bool:
        schema = self._templates.get(record.template_id)
        if schema is None:
            return False
        try:
            idx = schema.field_index(field_name)
        except KeyError:
            return False
        indices = record.sparse_vector.get("indices", [])
        values = record.sparse_vector.get("values", [])
        if idx not in indices:
            return False
        pos = indices.index(idx)
        field = schema.field_def(field_name)
        if field.type != "factor":
            return False
        factor_value = values[pos]
        expected = self.vocab.get(field.vocabulary_ref or field_name, term)
        return factor_value == expected

    def to_sparse_matrix(self) -> sp.csr_matrix:
        if not self._dirty and self._matrix is not None:
            return self._matrix

        if not self._records:
            return sp.csr_matrix((0, 0))

        n_cols = max((t.n_fields for t in self._templates.values()), default=0)
        data: list[float] = []
        indices: list[int] = []
        indptr: list[int] = [0]

        for rec in self._records:
            rec_indices = rec.sparse_vector.get("indices", [])
            rec_values = rec.sparse_vector.get("values", [])
            for i, v in zip(rec_indices, rec_values, strict=True):
                if i >= n_cols:
                    continue
                data.append(float(v))
                indices.append(i)
            indptr.append(len(data))

        self._matrix = sp.csr_matrix((data, indices, indptr), shape=(len(self._records), n_cols))
        self._dirty = False
        return self._matrix

    def save(self) -> None:
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        for schema in self._templates.values():
            schema.to_yaml(self.templates_dir / f"{schema.template_id}.yaml")

        self.vocab.save(self.vocab_path)

        with self.records_path.open("w", encoding="utf-8") as fh:
            for rec in self._records:
                fh.write(rec.model_dump_json() + "\n")

        matrix = self.to_sparse_matrix()
        if matrix.shape[0] > 0 and matrix.shape[1] > 0:
            sp.save_npz(self.matrix_path, matrix)

        self._save_index()

    def _save_index(self) -> None:
        by_template: dict[str, list[int]] = {}
        for i, rec in enumerate(self._records):
            by_template.setdefault(rec.template_id, []).append(i)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        (self.index_dir / "by_template.json").write_text(
            json.dumps(by_template), encoding="utf-8"
        )

    def list_records(self) -> list[TemplateRecord]:
        return list(self._records)
