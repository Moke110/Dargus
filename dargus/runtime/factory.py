"""AgentFactory — the single creation/termination point for every Agent.

v1.0.0 (design/2_runtime_structure.md): the factory creates and terminates every
Agent — Iris, Domain Experts, and D4Expert — injecting runtime-provided
dependencies so any dependency can be replaced with a fake or stub without
changing Agent code.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dargus.agents.base import BaseAgent
    from dargus.runtime.context import DargusRuntime

logger = logging.getLogger(__name__)

# Domain key → expert class path, used by ``expert()``. Also consulted by
# D4Expert when delegating a task to a domain expert.
_DOMAIN_EXPERT_PATHS: dict[str, str] = {
    "molecular": "dargus.experts.molecule.MoleculeExpert",
    "biomedical": "dargus.experts.biomed.BiomedExpert",
    "bioinformatics": "dargus.experts.bioinfo.BioinfoExpert",
    "clinical": "dargus.experts.clinic.ClinicExpert",
}

_DOMAIN_ALIASES: dict[str, str] = {
    "MoleculeExpert": "molecular",
    "BiomedExpert": "biomedical",
    "BioinfoExpert": "bioinformatics",
    "ClinicExpert": "clinical",
}


def _import_class(path: str) -> type:
    import importlib

    module_path, class_name = path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


class AgentFactory:
    """Creates and terminates Agents, wiring shared resources from the runtime."""

    def __init__(self, runtime: DargusRuntime) -> None:
        self._runtime = runtime

    # ------------------------------------------------------------------
    # Dependency bundle
    # ------------------------------------------------------------------

    def _di_kwargs(self) -> dict[str, Any]:
        rt = self._runtime
        return {
            "config": rt.config,
            "reasoning_llm": rt.reasoning_llm,
            "tool_registry": rt.tool_registry,
            "skill_registry": rt.skill_registry,
            "hook_registry": rt.hook_registry,
        }

    def _dbase(self) -> Any:
        """D-Base for Expert wiring: the runtime's DBaseStore, else global."""
        store = self._runtime.dbase_store
        if store is not None:
            return store.dbase
        from dargus.dbase import DBase

        return DBase.global_instance()

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def base_agent(self, name: str) -> BaseAgent:
        """Create a BaseAgent with dependencies injected from the runtime."""
        from dargus.agents.base import BaseAgent

        return BaseAgent(name=name, **self._di_kwargs())

    def expert(self, domain: str):
        """Create the DomainExpert for *domain*.

        Args:
            domain: Domain key (``"molecular"``, ``"biomedical"``,
                ``"bioinformatics"``, ``"clinical"``) or an expert class
                name (``"MoleculeExpert"`` …).

        Raises:
            ValueError: If *domain* is not recognised.
        """
        domain = _DOMAIN_ALIASES.get(domain, domain)
        path = _DOMAIN_EXPERT_PATHS.get(domain)
        if path is None:
            raise ValueError(
                f"Unknown expert domain {domain!r}. "
                f"Known domains: {sorted(_DOMAIN_EXPERT_PATHS)}"
            )
        expert_cls = _import_class(path)
        return expert_cls(dbase=self._dbase(), **self._di_kwargs())

    def d4_expert(self):
        """Create the D4Expert coordinator."""
        from dargus.experts.director import D4Expert

        return D4Expert(dbase=self._dbase(), agent_factory=self, **self._di_kwargs())

    def iris(self):
        """Create the Iris commander Agent."""
        from dargus.iris.commander import Iris

        return Iris(agent_factory=self, **self._di_kwargs())

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def terminate(self, agent: Any) -> None:
        """Terminate an Agent created by this factory.

        Agents hold no external resources today, so termination is a
        best-effort ``close()`` hook plus a log line; it exists so the
        factory remains the single lifecycle boundary required by
        design/2_runtime_structure.md.
        """
        close = getattr(agent, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.warning(
                    "Agent %r raised during close()", getattr(agent, "name", agent), exc_info=True
                )
        logger.debug("AgentFactory terminated agent %r", getattr(agent, "name", agent))
