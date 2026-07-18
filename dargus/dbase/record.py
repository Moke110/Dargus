from __future__ import annotations

from pydantic import BaseModel, model_validator


class TemplateRecord(BaseModel):
    template_id: str
    record_id: str
    source: dict
    sparse_vector: dict[str, list]
    provenance_note: str = ""

    @model_validator(mode="after")
    def _check_sparse_vector(self) -> "TemplateRecord":
        sv = self.sparse_vector
        indices = sv.get("indices")
        values = sv.get("values")
        if indices is not None and values is not None and len(indices) != len(values):
            raise ValueError(
                f"sparse_vector indices length ({len(indices)}) != values length ({len(values)})"
            )
        return self
