"""D-Base Tool wrappers — plain functions wrapping DBaseManager methods."""

from __future__ import annotations

from typing import Any

from dargus.dbase.manager import DBaseManager
from dargus.tools.base import Tool, ToolParam


def dbase_query(manager: DBaseManager, query: dict) -> dict:
    """Query D-Base. Returns matching records.

    Args:
        manager: A DBaseManager instance wired to a DBase store.
        query: Filter dict with optional keys: x_entity, disease_id, y_type,
               y_category, level, evidence_design, limit.

    Returns:
        Dict with ``records`` (list of matching evidence dicts) and ``count`` (int).
    """
    limit: int | None = query.get("limit")
    records = manager.read_records(
        x_entity=query.get("x_entity"),
        disease_id=query.get("disease_id"),
        y_type=query.get("y_type"),
        y_category=query.get("y_category"),
        level=query.get("level"),
        evidence_design=query.get("evidence_design"),
    )
    if limit is not None and limit > 0:
        records = records[:limit]
    return {"records": records, "count": len(records)}


def dbase_write(manager: DBaseManager, record: dict) -> dict:
    """Write evidence record. Internally calls EmbeddingModel for embedding generation.

    Args:
        manager: A DBaseManager instance wired to a DBase store.
        record: Three-axis evidence dict to write.

    Returns:
        Dict with ``evidence_id``, ``written`` (bool), and ``embedding_generated`` (bool).
    """
    result = manager.write_record(record)
    if isinstance(result, bool):
        written = result
    else:
        # DuplicateReviewRequest — record was already present
        written = False

    if written:
        manager.dbase.rebuild_view()

    embedding_generated = False
    if written:
        fp = manager.dbase.sidecars.active_fingerprint()
        if fp:
            embedding_generated = record.get(
                "evidence_id"
            ) in manager.dbase.sidecars.read_embeddings(fp)

    return {
        "evidence_id": record.get("evidence_id", ""),
        "written": written,
        "embedding_generated": embedding_generated,
    }


def dbase_status(manager: DBaseManager) -> dict:
    """Get D-Base status: record count, index info, etc.

    Args:
        manager: A DBaseManager instance wired to a DBase store.

    Returns:
        Dict with ``record_count``, ``shard_count``, ``has_parquet_view``.
    """
    manifest = manager.dbase.read_manifest()
    shards = list(manager.dbase.data_dir.glob("shard-*.jsonl"))
    return {
        "record_count": manifest.get("row_count", 0),
        "shard_count": len(shards),
        "has_parquet_view": manager.dbase.parquet_path.exists(),
    }


def dbase_update_status(
    manager: DBaseManager,
    evidence_id: str,
    status: str,
    superseded_by: str | None = None,
) -> dict:
    """Append a lifecycle status transition to the status sidecar.

    Args:
        manager: A DBaseManager instance wired to a DBase store.
        evidence_id: Target record id.
        status: New status — active / superseded / retracted /
            holdout-test / holdout-valid.
        superseded_by: Replacement evidence_id when *status* is
            ``"superseded"``.

    Returns:
        Dict with ``evidence_id`` and the resulting ``status``.
    """
    manager.update_status(evidence_id, status, superseded_by=superseded_by)
    return {"evidence_id": evidence_id, "status": manager.get_status(evidence_id)["status"]}


def dbase_write_summary(manager: DBaseManager, evidence_id: str, summary: str) -> dict:
    """Write or replace the LLM summary sidecar entry for a record.

    Args:
        manager: A DBaseManager instance wired to a DBase store.
        evidence_id: Target record id.
        summary: Summary text.

    Returns:
        Dict with ``evidence_id`` and ``written`` (bool).
    """
    manager.write_summary(evidence_id, summary)
    return {"evidence_id": evidence_id, "written": True}


# ---------------------------------------------------------------------------
# Tool object bindings (design/6: dbase_* are real Tool instances)
# ---------------------------------------------------------------------------


def make_dbase_tools(manager: DBaseManager) -> list[Tool]:
    """Bind the dbase_* functions to :class:`Tool` objects for a registry."""
    query_tool = Tool(
        name="dbase_query",
        description="Query the D-Base evidence store by drug/disease/level filters",
        parameters=[
            ToolParam("x_entity", "string", description="Intervention entity id (CURIE)"),
            ToolParam("disease_id", "string", description="Disease id (CURIE)"),
            ToolParam("y_type", "string", description="Readout/endpoint type"),
            ToolParam("y_category", "string", description="Readout category"),
            ToolParam("level", "string", description="Biological evidence level"),
            ToolParam("evidence_design", "string", description="Evidence design"),
            ToolParam("limit", "integer", default=100, description="Max records"),
        ],
        output={"type": "object", "properties": {"records": {"type": "array"}}},
    )
    query_tool.bind(lambda **kw: dbase_query(manager, kw))

    write_tool = Tool(
        name="dbase_write",
        description="Write an evidence record through the single-writer D-Base API",
        parameters=[
            ToolParam("record", "object", required=True, description="Three-axis evidence dict"),
        ],
        output={"type": "object", "properties": {"evidence_id": {"type": "string"}}},
    )
    write_tool.bind(lambda record: dbase_write(manager, record))

    status_tool = Tool(
        name="dbase_status",
        description="Report D-Base state (record count, shards, parquet view)",
        parameters=[],
        output={"type": "object", "properties": {"record_count": {"type": "integer"}}},
    )
    status_tool.bind(lambda: dbase_status(manager))

    update_status_tool = Tool(
        name="dbase_update_status",
        description="Append a lifecycle status transition (supersede, retract, holdout)",
        parameters=[
            ToolParam("evidence_id", "string", required=True),
            ToolParam(
                "status",
                "string",
                required=True,
                enum=["active", "superseded", "retracted", "holdout-test", "holdout-valid"],
            ),
            ToolParam("superseded_by", "string"),
        ],
        output={"type": "object", "properties": {"status": {"type": "string"}}},
    )
    update_status_tool.bind(
        lambda evidence_id, status, superseded_by=None: dbase_update_status(
            manager, evidence_id, status, superseded_by
        )
    )

    summary_tool = Tool(
        name="dbase_write_summary",
        description="Write or replace the LLM summary sidecar entry for a record",
        parameters=[
            ToolParam("evidence_id", "string", required=True),
            ToolParam("summary", "string", required=True),
        ],
        output={"type": "object", "properties": {"written": {"type": "boolean"}}},
    )
    summary_tool.bind(
        lambda evidence_id, summary: dbase_write_summary(manager, evidence_id, summary)
    )

    return [query_tool, write_tool, status_tool, update_status_tool, summary_tool]


def register_dbase_tools(manager: DBaseManager, registry: Any) -> None:
    """Register bound dbase_* Tools into a ToolRegistry (replacing stubs)."""
    for tool in make_dbase_tools(manager):
        registry._tools[tool.name] = tool
