"""AgentFactory — the single creation point for every Agent.

The factory creates every Agent — Iris, Domain Experts, and D4Expert —
injecting runtime-provided dependencies so any dependency can be replaced
with a fake or stub without changing Agent code.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dargus.agents.base import BaseAgent
    from dargus.iris.commander import Iris
    from dargus.models.session import Session
    from dargus.runtime.context import DargusRuntime

logger = logging.getLogger(__name__)


def _import_class(path: str) -> type:
    import importlib

    module_path, class_name = path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


class AgentFactory:
    """Creates Agents, wiring shared resources from the runtime."""

    def __init__(self, runtime: DargusRuntime) -> None:
        self._runtime = runtime
        self._iris_cache: Any | None = None

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
        }

    def _dbase(self) -> Any:
        """D-Base for Expert wiring: the runtime's DBaseStore, else global."""
        store = self._runtime.dbase_store
        if store is not None:
            return store.dbase
        from dargus.dbase import DBase

        return DBase.global_instance()

    def _wire(self, agent: Any) -> Any:
        """Attach the runtime back-reference for runtime-dependent behaviour."""
        agent._runtime = self._runtime
        return agent

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def base_agent(self, name: str) -> BaseAgent:
        """Create a BaseAgent with dependencies injected from the runtime."""
        from dargus.agents.base import BaseAgent

        return self._wire(BaseAgent(name=name, **self._di_kwargs()))

    def expert(self, domain: str):
        """Create the DomainExpert for *domain*.

        Args:
            domain: Domain key (``"molecular"``, ``"biomedical"``,
                ``"bioinformatics"``, ``"clinical"``) or an expert class
                name (``"MoleculeExpert"`` …).

        Raises:
            ValueError: If *domain* is not recognised.
        """
        from dargus.config.experts import domain_to_expert_path

        path = domain_to_expert_path(domain)
        if path is None:
            raise ValueError(f"Unknown expert domain {domain!r}. Known domains: see config.")
        expert_cls = _import_class(path)
        return self._wire(expert_cls(dbase=self._dbase(), **self._di_kwargs()))

    def d4_expert(self):
        """Create the D4Expert coordinator."""
        from dargus.experts.director import D4Expert

        return self._wire(D4Expert(dbase=self._dbase(), agent_factory=self, **self._di_kwargs()))

    def iris(self):
        """Create — and then cache — the Iris commander Agent.

        Iris is long-lived (SPEC-B): the same instance is returned on
        subsequent calls so her identity stays stable within a process, and
        **at most one live Iris** exists at a time. The cache is reset by
        :meth:`terminate` (the swap path) — a fresh Iris then owns a fresh
        Session.
        """
        if self._iris_cache is not None:
            return self._iris_cache
        from dargus.iris.commander import Iris

        iris = self._wire(Iris(agent_factory=self, **self._di_kwargs()))
        self._iris_cache = iris
        return iris

    # ------------------------------------------------------------------
    # Swap (ADR-0005 single session-swap verb)
    # ------------------------------------------------------------------

    def swap(self, *, hydrate: "Session | None" = None) -> Iris:
        """Persist-then-end the current live Iris and start a fresh one.

        This is the single session-swap verb shared by ``/new`` and
        ``/resume <id>``: the one-live-Iris invariant holds before, during,
        and after the swap.

        Args:
            hydrate: An optional loaded Session to seed the fresh Iris with
                (resume). ``None`` starts a fresh empty Session (``/new``).

        Returns:
            The new live Iris.
        """
        if self._iris_cache is not None:
            self.terminate(self._iris_cache)
        iris = self.iris()
        if hydrate is not None:
            iris._session = hydrate  # fresh Iris, resumed history
        return iris

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def terminate(self, agent: Any) -> None:
        """Terminate an Agent created by this factory.

        Agents hold no external resources today, so termination is a
        best-effort ``close()`` hook plus a log line.
        """
        close = getattr(agent, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.warning(
                    "Agent %r raised during close()", getattr(agent, "name", agent), exc_info=True
                )
        if self._iris_cache is agent:
            self._iris_cache = None
        logger.debug("AgentFactory terminated agent %r", getattr(agent, "name", agent))
