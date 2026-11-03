"""The TAC params: one config per role, written as only what differs from the defaults, with the checkpoint the harness sets."""

from __future__ import annotations

from pathlib import Path

import pytest

from plyagents.tac.actors.tabular import TabularActorParams
from plyagents.tac.configs.components import ComponentConfig
from plyagents.tac.enums import CriticType, SchedulerType, SelectorType, TargetType
from plyagents.tac.params import TACParams
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
