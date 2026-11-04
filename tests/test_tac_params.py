"""The TAC params: one config per role, written as only what differs from the defaults, with the checkpoint the harness sets."""

from __future__ import annotations

from pathlib import Path

import pytest

from plyagents.tac.actors.tabular import TabularActorParams
from plyagents.tac.configs.components import ComponentConfig
from plyagents.tac.enums import CriticType, SchedulerType, SelectorType, TargetType
from plyagents.tac.params import HASH_LENGTH, MAX_SUFFIX, TACParams
from plyagents.tac.schedulers.constant import ConstantParams


def test_params_round_trip_and_name_only_what_differs_from_the_defaults():
    text = "actor=tabular[target=critic,scheduler=constant[beta=0.2]],critic=tabular,train_selector=eps_greedy[eps=0.2]"
    params = TACParams.from_string(text)
    actor = params.actor.params
    assert isinstance(actor, TabularActorParams)
    assert actor.target.kind is TargetType.CRITIC and actor.scheduler == ComponentConfig(SchedulerType.CONSTANT, ConstantParams(beta=0.2))
    assert params.critic.kind is CriticType.TABULAR and params.eval_selector.kind is SelectorType.GREEDY
    assert params.to_string() == text
    assert params.path_suffix == "actor-tabular-target-critic-scheduler-constant-beta-0.2_critic-tabular_train_selector-eps_greedy-eps-0.2"
    assert TACParams.from_string(params.to_string()) == params
    assert TACParams.from_string("actor=tabular").to_string() == "actor=tabular"


def test_the_checkpoint_round_trips_and_stays_out_of_the_path():
    params = TACParams.from_string("actor=tabular,critic=oracle")
    with_checkpoint = params.with_checkpoint(Path("results/x/checkpoint_1"))
    assert with_checkpoint.checkpoint == Path("results/x/checkpoint_1")
    assert TACParams.from_string(with_checkpoint.to_string()) == with_checkpoint
    assert with_checkpoint.path_suffix == params.path_suffix


def test_params_refuse_unknown_options_and_types_and_a_missing_actor():
    with pytest.raises(ValueError, match="Unknown"):
        TACParams.from_string("actor=tabular,betta=0.2")
    with pytest.raises(ValueError, match="Unknown"):
        TACParams.from_string("actor=tabular[betta=0.2]")
    with pytest.raises(ValueError, match="Invalid critic"):
        TACParams.from_string("actor=tabular,critic=numeric")
    with pytest.raises(ValueError, match="Invalid TAC key"):
        TACParams.from_string("actor=tabular,key=pixels")
    with pytest.raises(ValueError, match="actor"):
        TACParams.from_string("critic=oracle")


def test_debug_is_read_but_stays_out_of_the_identity():
    params = TACParams.from_string("actor=tabular,debug=true")
    assert params.debug
    assert params.to_string() == "actor=tabular"
    assert params == TACParams.from_string("actor=tabular")
    assert params.path_suffix == TACParams.from_string("actor=tabular").path_suffix


def test_a_long_suffix_keeps_its_start_and_a_hash_of_the_whole():
    model = "vllm:qwen-3.8-27b-fp8[thinking_enabled=True,reasoning_effort=low]"
    text = f"actor=llm[model={model},budget=600,apply=compress[model={model}]],critic=oracle,feedback=llm[model={model},lesson_chars=200]"
    params = TACParams.from_string(text)
    suffix = params.path_suffix
    assert len(suffix) == MAX_SUFFIX and suffix.startswith("actor-llm-model-vllm_qwen-3-8-27b-fp8")
    assert not any(char in suffix for char in "=,[]")
    # another long spec that shares the start gets another hash
    other = TACParams.from_string(text.replace("lesson_chars=200", "lesson_chars=300")).path_suffix
    assert other[:-HASH_LENGTH] == suffix[:-HASH_LENGTH] and other != suffix
    # a short one is kept whole
    assert TACParams.from_string("actor=tabular").path_suffix == "actor-tabular"
