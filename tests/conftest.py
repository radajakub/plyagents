"""Real game states for the components that read the true state (the oracle ones), without the plugin players; an offline
LLM router for the components that take one but make no call; and a stub LLM answering from a queue, as plyllm would."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from plybench.core.engine import TurnBasedEngine
from plybench.games.builtins import register_builtin_games
from plybench.registry import Registry
from plybench.trackers.llm_call import LLMCall
from plyllm import LLM, FailureKind, LLMCallError, LLMConfig, LLMResponse, LLMTokens
from plyllm.response import OutputText
from pydantic import ValidationError

from plyagents.tac.configs.components import Services
from plyagents.tac.utils.calls import Calls


@pytest.fixture
def tic_tac_toe(tmp_path, monkeypatch) -> TurnBasedEngine:
    # the solved game is cached under ./cache, kept out of the repository
    monkeypatch.chdir(tmp_path)
    registry = Registry()
    register_builtin_games(registry)
    engine = registry.build_engine(registry.game_config("tic_tac_toe:"))
    engine.reset()
    return engine


@pytest.fixture
def services() -> Services:
    # a router without providers: building the tabular and oracle components needs one, calling it does not happen
    return Services(LLM(LLMConfig()))


@pytest.fixture
def calls(services: Services) -> Calls:
    # unseeded, and the calls are not kept
    return Calls(services.llm, None, lambda call: None)


class StubLLM:
    # answers from a queue: a string is the output text, an exception is raised
    def __init__(self, answers: list[str | Exception]) -> None:
        self.answers = answers
        self.seeds: list[int | None] = []

    async def generate(self, model_config, system, messages, output_schema=None, seed=None):
        self.seeds.append(seed)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        response = LLMResponse(
            provider=model_config.provider, model_string="stub", tokens=LLMTokens(input_tokens=5, output_tokens=3), items=[OutputText([answer])], output_text=answer, seed=seed
        )
        # as plyllm does: a structured answer that does not match the schema is an UNPARSEABLE failure carrying the answer
        if output_schema is not None:
            try:
                output_schema.model_validate_json(answer)
            except ValidationError as error:
                raise LLMCallError(FailureKind.UNPARSEABLE, str(error), response) from error
        return response


type StubCalls = Callable[..., tuple[Calls, StubLLM, list[LLMCall]]]


@pytest.fixture
def stub_calls() -> StubCalls:
    # Calls over a stub LLM: make(answers, seed=7) gives the Calls, the stub (its seeds) and the calls recorded
    def make(answers: list[str | Exception], seed: int | None = 7) -> tuple[Calls, StubLLM, list[LLMCall]]:
        stub, recorded = StubLLM(answers), []
        return Calls(cast(LLM, stub), seed, recorded.append), stub, recorded

    return make
