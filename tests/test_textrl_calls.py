"""The LLM calls of a proposal: seeded from the game's seed per role and index, each recorded, a structured answer parsed, and an
unusable one logged and asked again before the proposal fails."""

from __future__ import annotations

import asyncio

import pytest
from plybench.utils.seed import derive_seed
from plyllm import FailureKind, LLMCallError, LLMResponse, LLMTokens, ModelConfig, Provider
from pydantic import BaseModel

from plyagents.textrl.utils.calls import ATTEMPTS

MODEL = ModelConfig(Provider.METACENTRUM, "gpt-oss-120b")


class _Verdict(BaseModel):
    grade: str


def test_each_call_is_seeded_per_role_and_index_and_recorded(stub_calls):
    calls, stub, recorded = stub_calls(["a", "b", "c"])
    for role in ("apply", "apply", "feedback"):
        asyncio.run(calls.text(role, MODEL, "system", "prompt"))
    assert stub.seeds == [derive_seed(7, "apply", 0), derive_seed(7, "apply", 1), derive_seed(7, "feedback", 0)]
    assert [(call.role, call.full_output, call.failure) for call in recorded] == [("apply", "a", None), ("apply", "b", None), ("feedback", "c", None)]


def test_an_unseeded_game_sends_no_seed(stub_calls):
    calls, stub, _ = stub_calls(["a"], seed=None)
    asyncio.run(calls.text("apply", MODEL, "system", "prompt"))
    assert stub.seeds == [None]


def test_a_structured_answer_is_parsed(stub_calls):
    calls, _, recorded = stub_calls(['{"grade": "good"}'])
    answer = asyncio.run(calls.structured("feedback", MODEL, "system", "prompt", _Verdict))
    assert answer.output == _Verdict(grade="good") and answer.index == 0 and recorded[0].failure is None


def test_an_unusable_answer_is_logged_recorded_and_asked_again(capsys, stub_calls):
    refusal = LLMCallError(FailureKind.REFUSAL, "declined", LLMResponse(provider=Provider.METACENTRUM, model_string="stub", tokens=LLMTokens(), items=[], output_text=""))
    calls, stub, recorded = stub_calls(["not json", refusal, '{"grade": "mistake"}'])
    # the index of the call that answered, among all the calls recorded
    answer = asyncio.run(calls.structured("feedback", MODEL, "system", "prompt", _Verdict))
    assert answer.output == _Verdict(grade="mistake") and answer.index == 2

    assert [call.failure for call in recorded] == ["unparseable", "refusal", None]
    # each attempt has its own seed
    assert len(set(stub.seeds)) == 3
    assert capsys.readouterr().err.count("unusable answer") == 2


def test_the_proposal_fails_after_the_last_attempt(stub_calls):
    calls, _, recorded = stub_calls(["no"] * ATTEMPTS)
    with pytest.raises(ValueError, match=f"after {ATTEMPTS} attempts"):
        asyncio.run(calls.structured("feedback", MODEL, "system", "prompt", _Verdict))
    assert len(recorded) == ATTEMPTS


def test_a_transport_failure_is_not_retried(stub_calls):
    calls, _, recorded = stub_calls([LLMCallError(FailureKind.TIMEOUT, "slow")])
    with pytest.raises(LLMCallError):
        asyncio.run(calls.structured("feedback", MODEL, "system", "prompt", _Verdict))
    assert recorded == []
