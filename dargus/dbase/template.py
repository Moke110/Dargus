from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class FieldDef(BaseModel):
    name: str
    type: Literal["factor", "float", "int"]
    vocabulary: list[str] | None = None
    vocabulary_ref: str | None = None
    optional: bool = False
    meaning: str = ""


class TemplateSchema(BaseModel):
    template_id: str
    description: str = ""
    fields: list[FieldDef] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TemplateSchema":
        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(), fh, sort_keys=False)

    def field_index(self, name: str) -> int:
        for i, field in enumerate(self.fields):
            if field.name == name:
                return i
        raise KeyError(f"Field {name!r} not in template {self.template_id}")

    def field_def(self, name: str) -> FieldDef:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(f"Field {name!r} not in template {self.template_id}")

    @property
    def n_fields(self) -> int:
        return len(self.fields)
