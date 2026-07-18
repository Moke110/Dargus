from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseConverter(ABC):
    template_id: str

    @abstractmethod
    def convert(self, path: Path) -> list[dict[str, Any]]: ...
