from __future__ import annotations

from pydantic import BaseModel


class TemplateRecord(BaseModel):
    template_id: str
    record_id: str
    source: dict
    sparse_vector: dict[str, list]
    provenance_note: str = ""
