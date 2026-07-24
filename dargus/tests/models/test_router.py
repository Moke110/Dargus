"""Tests for ModelRouter — phase-based routing and usage tracking."""

import pytest

from dargus.models.reasoning import (
    LLMResponse,
    LLMUsage,
    Message,
    ReasoningOptions,
)
from dargus.models.router import ModelRouter


class CountingBackend:
    """Mock ReasoningBackend that counts calls and returns predictable responses.

    Each call returns usage proportional to a multiplier, so we can verify
    that total_usage aggregates correctly.
    """

    def __init__(self, label: str, multiplier: int = 1):
        self.label = label
        self.multiplier = multiplier
        self.call_count = 0
        self.last_messages: list[Message] = []
        self.last_options: ReasoningOptions | None = None

    def chat(self, messages: list[Message], options: ReasoningOptions | None = None) -> LLMResponse:
        self.call_count += 1
        self.last_messages = messages
        self.last_options = options
        return LLMResponse(
            content=f"response from {self.label}",
            usage=LLMUsage(
                input_tokens=10 * self.multiplier,
                output_tokens=5 * self.multiplier,
                cost=0.001 * self.multiplier,
            ),
            model=self.label,
            finish_reason="stop",
        )


class TestModelRouter:
    """Tests for ModelRouter."""

    def test_routes_to_correct_phase(self):
        planner = CountingBackend("planner")
        executor = CountingBackend("executor", multiplier=2)

        router = ModelRouter({"planner": planner, "executor": executor})

        msgs = [Message(role="user", content="plan this")]
        resp = router.route("executor", msgs)

        assert executor.call_count == 1
        assert planner.call_count == 0
        assert resp.content == "response from executor"
        assert resp.model == "executor"

    def test_fallback_to_planner(self):
        planner = CountingBackend("planner")

        router = ModelRouter({"planner": planner})

        msgs = [Message(role="user", content="test")]
        resp = router.route("unknown_phase", msgs)

        assert planner.call_count == 1
        assert resp.content == "response from planner"

    def test_raises_when_no_planner_fallback(self):
        executor = CountingBackend("executor")
        router = ModelRouter({"executor": executor})

        msgs = [Message(role="user", content="test")]
        with pytest.raises(KeyError, match="planner"):
            router.route("unknown_phase", msgs)

    def test_tracks_total_usage(self):
        planner = CountingBackend("planner")
        critic = CountingBackend("critic", multiplier=3)

        router = ModelRouter({"planner": planner, "critic": critic})

        msgs = [Message(role="user", content="a")]
        router.route("planner", msgs)
        router.route("planner", msgs)
        router.route("critic", msgs)

        usage = router.total_usage
        # planner: 2 calls x 1x => input=20, output=10, cost=0.002
        # critic: 1 call x 3x => input=30, output=15, cost=0.003
        # total: input=50, output=25, cost=0.005
        assert usage.input_tokens == 50
        assert usage.output_tokens == 25
        assert usage.cost == 0.005

    def test_total_usage_returns_copy(self):
        planner = CountingBackend("planner")
        router = ModelRouter({"planner": planner})

        router.route("planner", [Message(role="user", content="x")])

        usage1 = router.total_usage
        usage2 = router.total_usage

        assert usage1 == usage2
        # Modifying the copy should NOT affect the router's internal state
        usage1.input_tokens = 999
        assert router.total_usage.input_tokens == 10

    def test_reset_usage(self):
        planner = CountingBackend("planner")
        router = ModelRouter({"planner": planner})

        router.route("planner", [Message(role="user", content="x")])
        assert router.total_usage.input_tokens == 10

        router.reset_usage()
        assert router.total_usage.input_tokens == 0
        assert router.total_usage.output_tokens == 0
        assert router.total_usage.cost == 0.0

    def test_passes_options_to_backend(self):
        planner = CountingBackend("planner")
        router = ModelRouter({"planner": planner})

        opts = ReasoningOptions(temperature=0.5, max_tokens=100)
        msgs = [Message(role="user", content="test")]
        router.route("planner", msgs, options=opts)

        assert planner.last_options is opts

    def test_multiple_phases_with_same_backend(self):
        shared = CountingBackend("shared")
        router = ModelRouter(
            {
                "planner": shared,
                "executor": shared,
                "critic": shared,
            }
        )

        router.route("planner", [Message(role="user", content="p")])
        router.route("executor", [Message(role="user", content="e")])
        router.route("critic", [Message(role="user", content="c")])

        assert shared.call_count == 3
        assert router.total_usage.input_tokens == 30
