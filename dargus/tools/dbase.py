"""D-Base Tool wrappers — plain functions wrapping DBaseManager methods."""

from __future__ import annotations

from dargus.dbase.manager import DBaseManager


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

    embedding_generated = "embedding" in record

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
