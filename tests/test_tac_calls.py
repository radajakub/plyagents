"""The LLM calls of a proposal: seeded from the game's seed per role and index, each recorded, a structured answer parsed, and an
unusable one logged and asked again before the proposal fails."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
from plybench.trackers.llm_call import LLMCall
from plybench.utils.seed import derive_seed
from plyllm import LLM, FailureKind, LLMCallError, LLMResponse, LLMTokens, ModelConfig, Provider
from plyllm.response import OutputText
from pydantic import BaseModel, ValidationError

from plyagents.tac.utils.calls import ATTEMPTS, Calls

MODEL = ModelConfig(Provider.METACENTRUM, "gpt-oss-120b")


class _Verdict(BaseModel):
    grade: str


class _StubLLM:
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


def _calls(answers: list[str | Exception], seed: int | None = 7) -> tuple[Calls, _StubLLM, list[LLMCall]]:
    stub, recorded = _StubLLM(answers), []
    return Calls(cast(LLM, stub), seed, recorded.append), stub, recorded


def test_each_call_is_seeded_per_role_and_index_and_recorded():
    calls, stub, recorded = _calls(["a", "b", "c"])
    for role in ("apply", "apply", "feedback"):
        asyncio.run(calls.text(role, MODEL, "system", "prompt"))
    assert stub.seeds == [derive_seed(7, "apply", 0), derive_seed(7, "apply", 1), derive_seed(7, "feedback", 0)]
    assert [(call.role, call.full_output, call.failure) for call in recorded] == [("apply", "a", None), ("apply", "b", None), ("feedback", "c", None)]


def test_an_unseeded_game_sends_no_seed():
    calls, stub, _ = _calls(["a"], seed=None)
    asyncio.run(calls.text("apply", MODEL, "system", "prompt"))
    assert stub.seeds == [None]


def test_a_structured_answer_is_parsed():
    calls, _, recorded = _calls(['{"grade": "good"}'])
    assert asyncio.run(calls.structured("feedback", MODEL, "system", "prompt", _Verdict)) == _Verdict(grade="good")
    assert recorded[0].failure is None


def test_an_unusable_answer_is_logged_recorded_and_asked_again(capsys):
    refusal = LLMCallError(FailureKind.REFUSAL, "declined", LLMResponse(provider=Provider.METACENTRUM, model_string="stub", tokens=LLMTokens(), items=[], output_text=""))
    calls, stub, recorded = _calls(["not json", refusal, '{"grade": "mistake"}'])
    assert asyncio.run(calls.structured("feedback", MODEL, "system", "prompt", _Verdict)) == _Verdict(grade="mistake")

    assert [call.failure for call in recorded] == ["unparseable", "refusal", None]
    # each attempt has its own seed
    assert len(set(stub.seeds)) == 3
    assert capsys.readouterr().err.count("unusable answer") == 2


def test_the_proposal_fails_after_the_last_attempt():
    calls, _, recorded = _calls(["no"] * ATTEMPTS)
    with pytest.raises(ValueError, match=f"after {ATTEMPTS} attempts"):
        asyncio.run(calls.structured("feedback", MODEL, "system", "prompt", _Verdict))
    assert len(recorded) == ATTEMPTS


def test_a_transport_failure_is_not_retried():
    calls, _, recorded = _calls([LLMCallError(FailureKind.TIMEOUT, "slow")])
    with pytest.raises(LLMCallError):
        asyncio.run(calls.structured("feedback", MODEL, "system", "prompt", _Verdict))
    assert recorded == []
