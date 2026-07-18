"""DataMaster: unified sample-level project database."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class DataMaster:
    """Manage the project-level SQLite database."""

    def __init__(self, project_id: str, projects_root: str = "projects"):
        self.project_id = project_id
        self.db_path = Path(projects_root) / project_id / "project_db.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        schema_path = Path(__file__).resolve().parent / "schema.sql"
        with self._connection() as conn:
            conn.executescript(schema_path.read_text(encoding="utf-8"))
            conn.commit()

    def ingest(
        self,
        source: str | pd.DataFrame | list[dict[str, Any]],
        source_type: str,
        biological_level: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Ingest data into the samples table.

        Args:
            source: File path, DataFrame, or list of dicts.
            source_type: published_literature | public_db | user_upload.
            biological_level: molecular | cellular | exvivo | animal | clinical | epidemiology.
            **kwargs: Default values for fields not present in source rows.

        Returns:
            Dict with sample_ids and count.
        """
        rows = self._normalize_source(source)
        sample_ids = []
        with self._connection() as conn:
            for row in rows:
                record = self._build_record(row, source_type, biological_level, kwargs)
                conn.execute(
                    """
                    INSERT INTO samples (
                        sample_id, sample_type, species, strain,
                        tissue_organ, cell_type, cell_line_id, sex, age,
                        disease, disease_subtype, model_system, model_induction,
                        genetic_background, treatment, treatment_dose,
                        treatment_dose_unit, treatment_route, treatment_duration,
                        treatment_regimen, assay_type, assay_platform,
                        assay_endpoint, endpoint_category, source_type,
                        source_id, source_url, source_table_figure,
                        extraction_method, extraction_date, data_quality,
                        quality_notes, n_replicates, biological_level,
                        project_id, curator_agent, curation_date, data_type,
                        data_label, data_value, data_unit, data_uncertainty,
                        uncertainty_type
                    ) VALUES (
                        :sample_id, :sample_type, :species, :strain,
                        :tissue_organ, :cell_type, :cell_line_id, :sex, :age,
                        :disease, :disease_subtype, :model_system, :model_induction,
                        :genetic_background, :treatment, :treatment_dose,
                        :treatment_dose_unit, :treatment_route, :treatment_duration,
                        :treatment_regimen, :assay_type, :assay_platform,
                        :assay_endpoint, :endpoint_category, :source_type,
                        :source_id, :source_url, :source_table_figure,
                        :extraction_method, :extraction_date, :data_quality,
                        :quality_notes, :n_replicates, :biological_level,
                        :project_id, :curator_agent, :curation_date, :data_type,
                        :data_label, :data_value, :data_unit, :data_uncertainty,
                        :uncertainty_type
                    )
                    """,
                    record,
                )
                sample_ids.append(record["sample_id"])
            conn.commit()
        return {"sample_ids": sample_ids, "count": len(sample_ids)}

    def _normalize_source(
        self, source: str | pd.DataFrame | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if isinstance(source, str):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(source)
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path).to_dict("records")
            if path.suffix.lower() in (".xlsx", ".xls"):
                return pd.read_excel(path).to_dict("records")
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        if isinstance(source, pd.DataFrame):
            return source.to_dict("records")
        if isinstance(source, list):
            return source
        raise TypeError(f"Unsupported source type: {type(source)}")

    def _build_record(
        self,
        row: dict[str, Any],
        source_type: str,
        biological_level: str,
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        merged = {**defaults, **row}
        value = merged.get("data_value")
        if value is not None and not isinstance(value, str):
            value = json.dumps(value)
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "sample_id": merged.get("sample_id") or str(uuid.uuid4()),
            "sample_type": merged.get("sample_type", "unknown"),
            "species": merged.get("species"),
            "strain": merged.get("strain"),
            "tissue_organ": merged.get("tissue_organ"),
            "cell_type": merged.get("cell_type"),
            "cell_line_id": merged.get("cell_line_id"),
            "sex": merged.get("sex"),
            "age": merged.get("age"),
            "disease": merged.get("disease", "unknown"),
            "disease_subtype": merged.get("disease_subtype"),
            "model_system": merged.get("model_system"),
            "model_induction": merged.get("model_induction"),
            "genetic_background": merged.get("genetic_background"),
            "treatment": merged.get("treatment"),
            "treatment_dose": merged.get("treatment_dose"),
            "treatment_dose_unit": merged.get("treatment_dose_unit"),
            "treatment_route": merged.get("treatment_route"),
            "treatment_duration": merged.get("treatment_duration"),
            "treatment_regimen": merged.get("treatment_regimen"),
            "assay_type": merged.get("assay_type"),
            "assay_platform": merged.get("assay_platform"),
            "assay_endpoint": merged.get("assay_endpoint"),
            "endpoint_category": merged.get("endpoint_category"),
            "source_type": source_type,
            "source_id": merged.get("source_id"),
            "source_url": merged.get("source_url"),
            "source_table_figure": merged.get("source_table_figure"),
            "extraction_method": merged.get("extraction_method", "user_provided"),
            "extraction_date": merged.get("extraction_date", now),
            "data_quality": merged.get("data_quality", "medium"),
            "quality_notes": merged.get("quality_notes"),
            "n_replicates": merged.get("n_replicates"),
            "biological_level": biological_level,
            "project_id": self.project_id,
            "curator_agent": merged.get("curator_agent", "DataMaster"),
            "curation_date": now,
            "data_type": merged.get("data_type", "continuous"),
            "data_label": merged.get("data_label", "unknown"),
            "data_value": value,
            "data_unit": merged.get("data_unit"),
            "data_uncertainty": merged.get("data_uncertainty"),
            "uncertainty_type": merged.get("uncertainty_type"),
        }
        return record

    def query(
        self,
        biological_level: str | None = None,
        disease: str | None = None,
        treatment: str | None = None,
        assay_endpoint: str | None = None,
        species: str | None = None,
        sample_type: str | None = None,
        source_type: str | None = None,
        data_quality: str | None = None,
        **extra_filters: Any,
    ) -> pd.DataFrame:
        """Query samples and return a DataFrame with deserialized data_value."""
        conditions = ["project_id = :project_id"]
        params: dict[str, Any] = {"project_id": self.project_id}

        if biological_level is not None:
            conditions.append("biological_level = :biological_level")
            params["biological_level"] = biological_level
        if disease is not None:
            conditions.append("disease = :disease")
            params["disease"] = disease
        if treatment is not None:
            conditions.append("treatment = :treatment")
            params["treatment"] = treatment
        if assay_endpoint is not None:
            conditions.append("assay_endpoint = :assay_endpoint")
            params["assay_endpoint"] = assay_endpoint
        if species is not None:
            conditions.append("species = :species")
            params["species"] = species
        if sample_type is not None:
            conditions.append("sample_type = :sample_type")
            params["sample_type"] = sample_type
        if source_type is not None:
            conditions.append("source_type = :source_type")
            params["source_type"] = source_type
        if data_quality is not None:
            conditions.append("data_quality = :data_quality")
            params["data_quality"] = data_quality

        for key, value in extra_filters.items():
            conditions.append(f"{key} = :{key}")
            params[key] = value

        sql = "SELECT * FROM samples WHERE " + " AND ".join(conditions)
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        df = pd.DataFrame([dict(row) for row in rows])
        if not df.empty and "data_value" in df.columns:
            df["data_value"] = df["data_value"].apply(self._deserialize_value)
        return df

    @staticmethod
    def _deserialize_value(value: str | None) -> Any:
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def get_summary_stats(self, **filters: Any) -> dict[str, Any]:
        """Return summary statistics for the current query."""
        df = self.query(**filters)
        if df.empty:
            return {
                "n_samples": 0,
                "source_distribution": {},
                "quality_distribution": {},
                "assay_distribution": {},
            }
        return {
            "n_samples": len(df),
            "source_distribution": df["source_type"].value_counts().to_dict(),
            "quality_distribution": df["data_quality"].value_counts().to_dict(),
            "assay_distribution": df["assay_endpoint"].value_counts().to_dict(),
        }

    def export(self, output_path: str | Path, **filters: Any) -> Path:
        """Export query result to CSV."""
        df = self.query(**filters)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path
