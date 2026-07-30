"""D-Base record text serialization — the semantic surface for embedding.

``record_to_text`` flattens a v1.0.0 three-axis evidence dict into a single
text string. The ``embedding`` Tool (design/4_harness.md)
embeds this text; D-Base stores the resulting vectors in the fingerprinted
embeddings sidecar.
"""

from __future__ import annotations


def record_to_text(record: dict) -> str:
    """Serialize a v1.0.0 three-axis evidence dict to text for embedding.

    Incorporates llm_summary when present.
    """
    parts: list[str] = []

    # llm_summary — primary semantic surface if present
    llm = record.get("llm_summary", "")
    if llm:
        parts.append(f"summary: {llm}")

    level = record.get("biological_level", "")
    if level:
        parts.append(f"level: {level}")

    design = record.get("evidence_design", "")
    if design:
        parts.append(f"design: {design}")

    # x axis (three-axis)
    x = record.get("x") or {}
    xtype = x.get("type", "")
    if xtype:
        parts.append(f"x.type: {xtype}")
    for item in x.get("value") or []:
        if isinstance(item, dict):
            eid = item.get("entity_id", "")
            elabel = item.get("entity_label", "")
            alt = item.get("alteration", "")
            parts.append(f"x: id={eid or elabel} alt={alt}")

    # y axis (three-axis)
    y = record.get("y") or {}
    yt = y.get("type", "")
    if yt:
        parts.append(f"y.type: {yt}")
    yc = y.get("category", "")
    if yc:
        parts.append(f"y.category: {yc}")
    yv = y.get("value") or []
    yu = y.get("unit", "")
    if yv:
        parts.append(f"y.value: {yv} {yu}".strip())
    ye = y.get("effect")
    if ye:
        parts.append(f"y.effect: {ye.get('value_type', '')}={ye.get('value')}")

    # bg axis
    bg = record.get("bg") or {}
    dids = bg.get("disease_id") or []
    if dids:
        parts.append(f"bg.disease_id: {dids}")

    # sample identity
    for k in ("cell_line_id", "model_organism", "strain", "sex", "tissue", "cell_type"):
        v = record.get(k)
        if v:
            parts.append(f"{k}: {v}")

    # clinical design
    cd = record.get("clinical_design") or {}
    for k, v in cd.items():
        if v:
            parts.append(f"{k}: {v}")

    # sources
    sources = record.get("sources", [])
    if sources:
        rank1 = next((s for s in sources if s.get("rank") == 1), None)
        if rank1:
            parts.append(f"source: {rank1.get('type')}:{rank1.get('name', '')}")

    return "; ".join(parts)
