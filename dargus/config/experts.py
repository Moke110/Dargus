"""Expert domain registry — loaded from the project-level config file.

The domain key ↔ expert class path mapping lives in the project config
(``dargus/config/dargus_config.yaml``, ``experts:`` block) so
``AgentFactory.expert()`` has no dependency on the (deleted) expert report
serialization layer. This module is the single reader of that block.
"""

from __future__ import annotations

from typing import Any

import yaml

from dargus.config.paths import get_config_path


def _load_experts_block() -> dict[str, Any]:
    """Read the ``experts:`` block from the resolved config file."""
    cfg_path = get_config_path()
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("experts") or {}


def domain_to_expert_path(domain: str) -> str | None:
    """Return the expert class path for *domain*, or None if unknown.

    Accepts either the domain key (``"molecular"``) or an expert class name
    alias (``"MoleculeExpert"``).
    """
    block = _load_experts_block()
    domains: dict[str, str] = block.get("domains") or {}
    if domain in domains:
        return domains[domain]
    # Class-name alias → domain key
    for key, path in domains.items():
        if path.rsplit(".", 1)[-1] == domain:
            return path
    return None


def domain_to_expert_class(domain: str) -> type | None:
    """Return the expert class for *domain* (importing it), or None."""
    path = domain_to_expert_path(domain)
    if path is None:
        return None
    module_path, class_name = path.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)
