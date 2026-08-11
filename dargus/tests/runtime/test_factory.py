"""Tests for AgentFactory — the single Agent creation/termination point."""

import pytest

from dargus.agents.base import BaseAgent
from dargus.experts.bioinfo import BioinfoExpert
from dargus.experts.biomed import BiomedExpert
from dargus.experts.clinic import ClinicExpert
from dargus.experts.director import D4Expert
from dargus.experts.molecule import MoleculeExpert
from dargus.iris.commander import Iris
from dargus.runtime.context import DargusRuntime
from dargus.runtime.factory import AgentFactory


def _runtime(tmp_path: pytest.TempPathFactory | None = None) -> DargusRuntime:
    """A runtime rooted at a tmp workspace so agent persistence stays hermetic."""
    if tmp_path is None:
        return DargusRuntime()
    return DargusRuntime(config={"workspace_root": str(tmp_path)})


class TestAgentFactory:
    """Tests for AgentFactory."""

    def test_constructor_stores_runtime(self):
        rt = DargusRuntime()
        factory = AgentFactory(rt)
        assert factory._runtime is rt

    def test_base_agent_creates_instance(self):
        rt = DargusRuntime()
        factory = AgentFactory(rt)
        agent = factory.base_agent("TestAgent")

        assert isinstance(agent, BaseAgent)
        assert agent.name == "TestAgent"

    def test_base_agent_with_di_kwargs(self):
        rt = DargusRuntime()
        rt.reasoning_llm = object()  # type: ignore[assignment]
        factory = AgentFactory(rt)
        agent = factory.base_agent("DI_Agent")

        assert isinstance(agent, BaseAgent)
        assert agent.name == "DI_Agent"


class TestExpertCreation:
    """expert()/d4_expert()/iris() create wired agents (no more stubs)."""

    @pytest.mark.parametrize(
        "domain,cls",
        [
            ("molecular", MoleculeExpert),
            ("biomedical", BiomedExpert),
            ("bioinformatics", BioinfoExpert),
            ("clinical", ClinicExpert),
        ],
    )
    def test_expert_creates_domain_expert(self, domain, cls):
        factory = AgentFactory(DargusRuntime())
        expert = factory.expert(domain)
        assert isinstance(expert, cls)

    @pytest.mark.parametrize(
        "alias,cls",
        [
            ("MoleculeExpert", MoleculeExpert),
            ("BiomedExpert", BiomedExpert),
            ("BioinfoExpert", BioinfoExpert),
            ("ClinicExpert", ClinicExpert),
        ],
    )
    def test_expert_accepts_class_name_alias(self, alias, cls):
        factory = AgentFactory(DargusRuntime())
        assert isinstance(factory.expert(alias), cls)

    def test_expert_unknown_domain_raises(self):
        factory = AgentFactory(DargusRuntime())
        with pytest.raises(ValueError, match="Unknown expert domain"):
            factory.expert("cardiology")

    def test_d4_expert_creates_coordinator_with_factory(self):
        factory = AgentFactory(DargusRuntime())
        d4 = factory.d4_expert()
        assert isinstance(d4, D4Expert)
        assert d4._agent_factory is factory

    def test_iris_creates_commander_with_factory(self):
        factory = AgentFactory(DargusRuntime())
        iris = factory.iris()
        assert isinstance(iris, Iris)
        assert iris._agent_factory is factory

    def test_iris_cached_returns_same_instance(self):
        """SPEC-B: AgentFactory.iris() caches and returns the same Iris."""
        factory = AgentFactory(DargusRuntime())
        iris1 = factory.iris()
        iris2 = factory.iris()
        assert iris1 is iris2

    def test_iris_cache_reset_on_terminate(self, tmp_path):
        """SPEC-B: terminating Iris clears the factory cache."""
        factory = AgentFactory(_runtime(tmp_path))
        iris1 = factory.iris()
        factory.terminate(iris1)
        iris2 = factory.iris()
        assert iris2 is not iris1

    def test_iris_cache_is_per_factory(self):
        """Two factories on different runtimes each cache their own Iris."""
        f1 = AgentFactory(DargusRuntime())
        f2 = AgentFactory(DargusRuntime())
        assert f1.iris() is not f2.iris()

    def test_experts_receive_dbase_from_runtime_manager(self):
        class _Manager:
            dbase = object()

        rt = DargusRuntime(dbase_store=_Manager())
        factory = AgentFactory(rt)
        expert = factory.expert("molecular")
        assert expert.dbase is rt.dbase_store.dbase


class TestTermination:
    def test_terminate_without_close_is_noop(self, tmp_path):
        factory = AgentFactory(_runtime(tmp_path))
        agent = factory.base_agent("TermAgent")
        factory.terminate(agent)  # must not raise

    def test_terminate_calls_close_when_present(self):
        class _Agent:
            name = "closable"
            closed = False

            def close(self):
                self.closed = True

        agent = _Agent()
        factory = AgentFactory(_runtime())
        factory.terminate(agent)
        assert agent.closed is True

    def test_terminate_swallows_close_errors(self):
        class _Agent:
            name = "broken"

            def close(self):
                raise RuntimeError("boom")

        factory = AgentFactory(_runtime())
        factory.terminate(_Agent())  # logs but must not raise


class TestLiveAccess:
    """end_live()/live_session_id() — the factory owns live-Iris access."""

    def test_live_session_id_none_without_iris(self):
        factory = AgentFactory(_runtime())
        assert factory.live_session_id() is None

    def test_live_session_id_returns_live_id(self):
        factory = AgentFactory(_runtime())
        iris = factory.iris()
        assert factory.live_session_id() == iris._session.metadata.session_id

    def test_live_session_id_none_after_swap(self):
        """After a swap (terminate + fresh iris), the id is the *new* iris's."""
        factory = AgentFactory(_runtime())
        iris1 = factory.iris()
        factory.swap(hydrate=None)
        assert factory.live_session_id() != iris1._session.metadata.session_id

    def test_end_live_none_without_iris(self):
        factory = AgentFactory(_runtime())
        assert factory.end_live() is None

    def test_end_live_persists_and_returns_id(self, tmp_path):
        """end_live() persists the live Session and returns its id."""
        factory = AgentFactory(_runtime(tmp_path))
        iris = factory.iris()
        iris._session.metadata.workspace_root = str(tmp_path)
        iris._session.add_user("q1")
        iris._session.add_assistant("reply")

        session_id = factory.end_live()
        assert session_id == iris._session.metadata.session_id

        from dargus.sessions.store import SessionStore

        assert session_id in SessionStore(str(tmp_path)).list_ids()

    def test_end_live_idempotent(self, tmp_path):
        """A second end_live is a no-op (append-only archive)."""
        factory = AgentFactory(_runtime(tmp_path))
        iris = factory.iris()
        iris._session.metadata.workspace_root = str(tmp_path)
        iris._session.add_user("q1")
        iris._session.add_assistant("reply")

        first = factory.end_live()
        assert factory.end_live() == first  # same id, no error
