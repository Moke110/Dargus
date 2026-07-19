from __future__ import annotations

import json
from pathlib import Path

import scipy.sparse as sp

from dargus.dbase.paths import global_dbase_root
from dargus.dbase.record import TemplateRecord
from dargus.dbase.template import TemplateSchema
from dargus.dbase.vocabulary import VocabularyManager


class DBase:
    """Experiment-level conclusion store: factorized sparse matrix + vocabularies."""

    def __init__(self, project_id: str, root_dir: str | Path):
        self.project_id = project_id
        self.root = Path(root_dir)
        self.dbase_dir = self.root / "dbase"
        self.templates_dir = self.dbase_dir / "templates"
        self.vocab_path = self.dbase_dir / "vocabularies.json"
        self.matrix_path = self.dbase_dir / "records.npz"
        self.manifest_path = self.dbase_dir / "records_manifest.json"

        self._templates: dict[str, TemplateSchema] = {}
        self._vocab: VocabularyManager | None = None
        self._records: list[TemplateRecord] = []
        self._record_ids: set[str] = set()
        self._manifest: list[dict] = []
        self._matrix: sp.csr_matrix | None = None
        self._dirty = True

        self._ensure_dirs()
        self._load()

    def _ensure_dirs(self) -> None:
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if self.vocab_path.exists():
            self._vocab = VocabularyManager.load(self.vocab_path)
        else:
            self._vocab = VocabularyManager()

        for yaml_path in self.templates_dir.glob("*.yaml"):
            schema = TemplateSchema.from_yaml(yaml_path)
            self._templates[schema.template_id] = schema

        if self.manifest_path.exists() and self.matrix_path.exists():
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._matrix = sp.load_npz(self.matrix_path)
            self._dirty = False
            self._reconstruct_records_from_matrix()

    def _reconstruct_records_from_matrix(self) -> None:
        self._records = []
        self._record_ids = set()
        if self._matrix is None or self._matrix.shape[0] == 0:
            return
        n_manifest = len(self._manifest)
        n_matrix = self._matrix.shape[0]
        if n_manifest != n_matrix:
            raise RuntimeError(
                f"Manifest/matrix row count mismatch: "
                f"manifest has {n_manifest} rows but matrix has {n_matrix} rows"
            )
        for row_idx, entry in enumerate(self._manifest):
            row = self._matrix[row_idx]
            self._records.append(
                TemplateRecord(
                    template_id=entry["template_id"],
                    record_id=entry["record_id"],
                    source=entry["source"],
                    sparse_vector={
                        "indices": row.indices.tolist(),
                        "values": row.data.tolist(),
                    },
                    provenance_note=entry.get("provenance_note", ""),
                )
            )
            self._record_ids.add(entry["record_id"])

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
        """Add a single TemplateRecord. One record per call for safety."""
        if not isinstance(record, TemplateRecord):
            raise TypeError("record must be a TemplateRecord")
        if record.template_id not in self._templates:
            raise KeyError(f"Template {record.template_id!r} not registered")
        if record.record_id in self._record_ids:
            raise ValueError(f"Record with record_id {record.record_id!r} already exists")

        schema = self._templates[record.template_id]
        n_fields = schema.n_fields
        for idx in record.sparse_vector.get("indices", []):
            if idx < 0 or idx >= n_fields:
                raise ValueError(
                    f"sparse_vector index {idx} out of range for template "
                    f"{record.template_id!r} (n_fields={n_fields})"
                )

        self._records.append(record)
        self._record_ids.add(record.record_id)
        self._manifest.append(self._record_to_manifest_entry(record))
        self._dirty = True

    def _record_to_manifest_entry(self, record: TemplateRecord) -> dict:
        return {
            "record_id": record.record_id,
            "template_id": record.template_id,
            "source": record.source,
            "provenance_note": record.provenance_note,
        }

    def _record_to_json(self, record: TemplateRecord) -> dict:
        return {
            "template_id": record.template_id,
            "record_id": record.record_id,
            "source": record.source,
            "sparse_vector": record.sparse_vector,
            "provenance_note": record.provenance_note,
        }

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

        matrix = self.to_sparse_matrix()
        if matrix.shape[0] > 0 and matrix.shape[1] > 0:
            sp.save_npz(self.matrix_path, matrix)
        elif self.matrix_path.exists():
            self.matrix_path.unlink()

        self.manifest_path.write_text(json.dumps(self._manifest, indent=2), encoding="utf-8")

        records_jsonl_path = self.dbase_dir / "records.jsonl"
        with records_jsonl_path.open("w", encoding="utf-8") as fh:
            for record in self._records:
                fh.write(json.dumps(self._record_to_json(record)) + "\n")

    def list_records(self) -> list[TemplateRecord]:
        return list(self._records)

    @classmethod
    def global_instance(cls) -> "DBase":
        """Return the singleton-like global D-Base for this installation."""
        return cls(project_id="global", root_dir=global_dbase_root())
