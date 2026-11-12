"""The TextRL params: one config per role, written as only what differs from the defaults, with the checkpoint the harness sets."""

from __future__ import annotations

from pathlib import Path

import pytest

from plyagents.textrl.actors.tabular import TabularActorParams
from plyagents.textrl.configs.components import ComponentConfig
from plyagents.textrl.critics.llm import LLMCriticParams
from plyagents.textrl.enums import BackwardType, CriticType, SchedulerType, SelectorType, SweepType, TargetType
from plyagents.textrl.params import HASH_LENGTH, MAX_SUFFIX, TextRLParams
from plyagents.textrl.schedulers.constant import ConstantParams


def test_params_round_trip_and_name_only_what_differs_from_the_defaults():
    text = "actor=tabular[target=critic,scheduler=constant[beta=0.2]],critic=tabular,train_selector=eps_greedy[eps=0.2]"
    params = TextRLParams.from_string(text)
    actor = params.actor.params
    assert isinstance(actor, TabularActorParams)
    assert actor.target.kind is TargetType.CRITIC and actor.scheduler == ComponentConfig(SchedulerType.CONSTANT, ConstantParams(beta=0.2))
    assert params.critic.kind is CriticType.TABULAR and params.eval_selector.kind is SelectorType.GREEDY
    assert params.to_string() == text
    assert params.path_suffix == "actor-tabular-target-critic-scheduler-constant-beta-0.2_critic-tabular_train_selector-eps_greedy-eps-0.2"
    assert TextRLParams.from_string(params.to_string()) == params
    assert TextRLParams.from_string("actor=tabular").to_string() == "actor=tabular"


def test_the_checkpoint_round_trips_and_stays_out_of_the_path():
    params = TextRLParams.from_string("actor=tabular,critic=oracle")
    with_checkpoint = params.with_checkpoint(Path("results/x/checkpoint_1"))
    assert with_checkpoint.checkpoint == Path("results/x/checkpoint_1")
    assert TextRLParams.from_string(with_checkpoint.to_string()) == with_checkpoint
    assert with_checkpoint.path_suffix == params.path_suffix


def test_params_refuse_unknown_options_and_types_and_a_missing_actor():
    with pytest.raises(ValueError, match="Unknown"):
        TextRLParams.from_string("actor=tabular,betta=0.2")
    with pytest.raises(ValueError, match="Unknown"):
        TextRLParams.from_string("actor=tabular[betta=0.2]")
    with pytest.raises(ValueError, match="Invalid critic"):
        TextRLParams.from_string("actor=tabular,critic=numeric")
    with pytest.raises(ValueError, match="Invalid TextRL key"):
        TextRLParams.from_string("actor=tabular,key=pixels")
    with pytest.raises(ValueError, match="actor"):
        TextRLParams.from_string("critic=oracle")


def test_debug_is_read_but_stays_out_of_the_identity():
    params = TextRLParams.from_string("actor=tabular,debug=true")
    assert params.debug
    assert params.to_string() == "actor=tabular"
    assert params == TextRLParams.from_string("actor=tabular")
    assert params.path_suffix == TextRLParams.from_string("actor=tabular").path_suffix


def test_a_long_suffix_keeps_its_start_and_a_hash_of_the_whole():
    model = "vllm:qwen-3.8-27b-fp8[thinking_enabled=True,reasoning_effort=low]"
    text = f"actor=llm[model={model},budget=600,apply=compress[model={model}]],critic=oracle,feedback=llm[model={model},lesson_chars=200]"
    params = TextRLParams.from_string(text)
    suffix = params.path_suffix
    assert len(suffix) == MAX_SUFFIX and suffix.startswith("actor-llm-model-vllm_qwen-3-8-27b-fp8")
    assert not any(char in suffix for char in "=,[]")
    # another long spec that shares the start gets another hash
    other = TextRLParams.from_string(text.replace("lesson_chars=200", "lesson_chars=300")).path_suffix
    assert other[:-HASH_LENGTH] == suffix[:-HASH_LENGTH] and other != suffix
    # a short one is kept whole
    assert TextRLParams.from_string("actor=tabular").path_suffix == "actor-tabular"


@pytest.mark.parametrize(
    ("spec", "name"),
    [
        (
            "actor=llm[model=gemini:gemini-3.5-flash-lite[thinking_enabled=True],budget=600],critic=oracle,feedback=oracle",
            "TextRL: LLM actor, oracle critic, oracle feedback (gemini-3.5-flash-lite)",
        ),
        ("actor=tabular[target=critic],critic=tabular,train_selector=eps_greedy[eps=0.3]", "TextRL: tabular actor, tabular critic, no feedback"),
        (
            "actor=llm[model=gemini:gemini-3.5-flash-lite],critic=llm_v[model=metacentrum:gpt-oss-120b]",
            "TextRL: LLM actor (gemini-3.5-flash-lite), LLM V critic (gpt-oss-120b), no feedback",
        ),
    ],
)
def test_the_display_name_is_textrl_and_its_components(spec, name):
    # the model once when the LLM components share it, else after each; params, selectors, reward and key left out
    assert TextRLParams.from_string(spec).display_name() == name


def test_the_models_are_every_llm_component_once_nested_ones_included():
    # rung 4: the actor and its compress, the critic and its compress, the feedback; two models, the second only in the critic's compress
    flash, gpt = "gemini:gemini-3.5-flash-lite[thinking_enabled=True]", "metacentrum:gpt-oss-120b"
    spec = f"actor=llm[model={flash},budget=600,apply=compress[model={flash}]],critic=llm_q[model={flash},budget=600,apply=compress[model={gpt}]],feedback=llm[model={flash}]"
    models = TextRLParams.from_string(spec).models()
    assert [f"{model.provider.value}:{model.model_name}" for model in models] == ["gemini:gemini-3.5-flash-lite", "metacentrum:gpt-oss-120b"]
    assert TextRLParams.from_string("actor=tabular,critic=oracle").models() == []


def test_the_sweep_and_the_backward_passes_are_named_only_when_they_differ_from_the_defaults():
    # the defaults are the TAC before the blocks (the episode sweep, the identity backward), so existing specs and paths do not change
    model = "metacentrum:gpt-oss-120b"
    plain = f"actor=llm[model={model}],critic=llm_v[model={model}],feedback=llm[model={model}]"
    explicit = f"actor=llm[model={model},backward=identity],critic=llm_v[model={model},backward=identity],feedback=llm[model={model}],sweep=episode"
    params = TextRLParams.from_string(plain)
    critic = params.critic.params
    assert isinstance(critic, LLMCriticParams) and params.sweep.kind is SweepType.EPISODE and critic.backward.kind is BackwardType.IDENTITY
    assert TextRLParams.from_string(explicit) == params
    assert TextRLParams.from_string(explicit).to_string() == plain and TextRLParams.from_string(explicit).path_suffix == params.path_suffix
    with pytest.raises(ValueError, match="Invalid sweep"):
        TextRLParams.from_string("actor=tabular,sweep=forward")
    with pytest.raises(ValueError, match="Invalid backward"):
        TextRLParams.from_string(f"actor=llm[model={model},backward=free]")
