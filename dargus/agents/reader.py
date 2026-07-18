from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pandas as pd

from dargus.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class ReaderAgent(BaseAgent):
    """Scans datadir, classifies files, disassembles literature/data."""

    name = "ReaderAgent"

    # NOTE: The brief skeleton lists `.txt` as literature, but the existing test
    # suite expects `.txt` to be classified as unknown. We preserve that
    # behaviour for the MVP to avoid breaking existing assertions.
    LITERATURE_SUFFIXES = {".pdf", ".html"}
    DATA_SUFFIXES = {".csv", ".xlsx", ".xls", ".tsv"}

    def scan_directory(self, datadir: str) -> dict[str, list[str]]:
        path = Path(datadir)
        result: dict[str, list[str]] = {
            "literature_files": [],
            "data_files": [],
            "unknown_files": [],
        }
        for item in path.iterdir():
            if not item.is_file():
                continue
            suffix = item.suffix.lower()
            if suffix in self.LITERATURE_SUFFIXES:
                result["literature_files"].append(str(item))
            elif suffix in self.DATA_SUFFIXES:
                result["data_files"].append(str(item))
            else:
                result["unknown_files"].append(str(item))
        return result

    def disassemble_literature(self, lit_files: list[str]) -> list[dict[str, Any]]:
        instances: list[dict[str, Any]] = []
        for path_str in lit_files:
            path = Path(path_str)
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                text = self._extract_pdf_text(path)
            elif suffix == ".html":
                text = self._read_text_file(path)
            else:
                continue
            instances.extend(
                self._extract_experiment_instances(text, source={"file": path_str})
            )
        return instances

    def parse_data_file(self, data_file: str) -> list[dict[str, Any]]:
        path = Path(data_file)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif suffix == ".tsv":
            df = pd.read_csv(path, sep="\t")
        else:
            return []
        return self._dataframe_to_instances(df, source={"file": data_file})

    def _extract_pdf_text(self, path: Path) -> str:
        try:
            doc = fitz.open(path)
            return "\n".join(page.get_text() for page in doc)
        except Exception as exc:
            logger.warning("Failed to parse PDF %s: %s", path, exc)
            return ""

    def _read_text_file(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read text file %s: %s", path, exc)
            return ""

    def _extract_experiment_instances(
        self, text: str, source: dict[str, Any]
    ) -> list[dict[str, Any]]:
        # MVP: return a single unstructured instance; later use LLM/regex extraction.
        return [{"source": source, "raw_text": text[:2000], "note": "MVP extraction stub"}]

    def _dataframe_to_instances(
        self, df: pd.DataFrame, source: dict[str, Any]
    ) -> list[dict[str, Any]]:
        instances: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            instance = {"source": source}
            instance.update(row.dropna().to_dict())
            instances.append(instance)
        return instances

    def run(self, task_spec: dict[str, Any]) -> dict[str, Any]:
        """Execute the reader agent task."""
        datadir = task_spec.get("datadir", "")
        scan = self.scan_directory(datadir)
        literature_instances = self.disassemble_literature(scan["literature_files"])
        data_instances: list[dict[str, Any]] = []
        for data_file in scan["data_files"]:
            data_instances.extend(self.parse_data_file(data_file))
        return {
            "literature_files": scan["literature_files"],
            "data_files": scan["data_files"],
            "unknown_files": scan["unknown_files"],
            "literature_instances": literature_instances,
            "data_instances": data_instances,
        }
