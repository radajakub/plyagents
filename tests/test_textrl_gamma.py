"""The discount gamma: one option of the TextRL, named only when it discounts. It reaches the losses, which show the return of a move discounted over the
agent's later moves and a TD target as gamma times the next value, and the tabular targets; with gamma = 1 nothing they show or compute changes."""

from __future__ import annotations

import asyncio

import pytest
from plybench.app import PlyBench
from plyllm import LLMConfig

from plyagents.textrl.actors.tabular import TabularActor
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.critics.tabular import TabularCritic
from plyagents.textrl.enums import TargetType
from plyagents.textrl.losses.llm_actor import LLMActorLoss
from plyagents.textrl.losses.llm_critic import LLMCriticLoss
from plyagents.textrl.losses.llm_joint import LLMJointLoss
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.params import TextRLParams
from plyagents.textrl.player import TextRLPlayer
from plyagents.textrl.schedulers.constant import ConstantScheduler
from plyagents.textrl.targets.critic import CriticTarget
from plyagents.textrl.targets.outcome import OutcomeTarget

M = "metacentrum:gpt-oss-120b"
MODEL = ModelSpec.from_string(M)
STEPS = [TextRLStep(key, "<1>", 0, "s", legal=("<1>", "<2>")) for key in ("a", "b", "c")]
VALUES = [Assessment(0.2, "fine"), Assessment(0.4, "good"), Assessment(-0.6, "bad")]


def test_gamma_is_named_only_when_it_discounts_and_reads_back():
    plain = TextRLParams.from_string("actor=tabular")
    assert plain.gamma == 1.0 and "gamma" not in plain.to_string() and TextRLParams.from_string("actor=tabular,gamma=1") == plain
    params = TextRLParams.from_string("actor=tabular,gamma=0.9")
    assert params.gamma == 0.9 and TextRLParams.from_string(params.to_string()) == params and params.path_suffix.endswith("_gamma-0.9")
    # a preset's string names it with every role
    assert "gamma=1.0" in TextRLParams.from_string(f"preset=tac,model={M}").to_string()
    for bad in ("0", "1.5", "-0.1"):
        with pytest.raises(ValueError, match="Invalid TextRL gamma"):
            TextRLParams.from_string(f"actor=tabular,gamma={bad}")


def _player(spec: str) -> TextRLPlayer:
    registry = PlyBench(LLMConfig()).registry
    game = registry.build_engine(registry.game_config("tic_tac_toe:")).game
    player = registry.build_player(game, registry.player_config(f"textrl:{spec}"), "i")
    assert isinstance(player, TextRLPlayer)
    return player


def test_the_discount_reaches_the_losses_and_the_tabular_targets():
    tabular = _player("actor=tabular[target=critic],critic=tabular,gamma=0.5")
    assert isinstance(tabular.actor, TabularActor) and isinstance(tabular.actor.target, CriticTarget) and tabular.actor.target.gamma == 0.5
    assert isinstance(tabular.critic, TabularCritic) and tabular.critic.gamma == 0.5
    own = _player(f"actor=llm[model={M}],critic=llm_q[model={M}],actor_loss=llm[model={M}],critic_loss=llm[model={M}],gamma=0.5")
    assert isinstance(own.losses.actor, LLMActorLoss) and own.losses.actor.gamma == 0.5
    assert isinstance(own.losses.critic, LLMCriticLoss) and own.losses.critic.gamma == 0.5
    joint = _player(f"actor=llm[model={M}],feedback=llm[model={M}],gamma=0.5")
    assert isinstance(joint.losses.joint, LLMJointLoss) and joint.losses.joint.gamma == 0.5


def test_the_tabular_targets_are_the_discounted_return_and_gamma_times_the_next_value(calls):
    assert OutcomeTarget(0.5).targets(VALUES, 1.0) == [0.25, 0.5, 1.0] and OutcomeTarget().targets(VALUES, 1.0) == [1.0, 1.0, 1.0]
    assert CriticTarget(0.5).targets(VALUES, -1.0) == [0.2, -0.3, -1.0]
    update = asyncio.run(TabularCritic(ConstantScheduler(0.4), 0.5).propose(STEPS, VALUES, -1.0, [], calls))
    assert update is not None and [sample.target for sample in update.samples if sample.action is None] == [-0.25, -0.5, -1.0]


def test_the_losses_show_the_discounted_return_and_the_discounted_next_value():
    outcome = LLMActorLoss(MODEL, None, {"outcome"}, gamma=0.5)
    first, last = outcome.step_prompt(0, STEPS, VALUES, 1.0, {}), outcome.step_prompt(2, STEPS, VALUES, 1.0, {})
    assert "Discounted by gamma=0.5 over the agent's 2 later moves, the return of this move: +0.25." in first and "Discounted" not in last
    # the state-value critic's next assessment, a TD target in text, and its number in the critic's change
    assert "It counts gamma=0.5 times as much as this situation" in outcome.step_prompt(0, STEPS, VALUES, 1.0, {})
    delta = LLMActorLoss(MODEL, None, {"delta"}, gamma=0.5).step_prompt(0, STEPS, VALUES, 1.0, {})
    assert "after +0.20 (gamma=0.5 times the next value); change +0.00" in delta

    td = LLMCriticLoss(MODEL, None, set(), TargetType.CRITIC, 0.5).step_prompt(0, STEPS, VALUES, 1.0, {})
    mc = LLMCriticLoss(MODEL, None, set(), TargetType.MC, 0.5).step_prompt(1, STEPS, VALUES, 1.0, {})
    assert "It counts gamma=0.5 times" in td and "The game ended later. Outcome for the agent: +1" in mc and "the return of this move: +0.50." in mc

    joint = LLMJointLoss(MODEL, None, {"value"}, gamma=0.5).game_prompt(STEPS, VALUES, 1.0, {})
    assert "The outcome is discounted by gamma=0.5 per move of the agent" in joint


def test_without_a_discount_the_losses_show_what_they_showed_before():
    prompts = [
        LLMActorLoss(MODEL, None, {"outcome", "delta"}).step_prompt(0, STEPS, VALUES, 1.0, {}),
        LLMCriticLoss(MODEL, None, {"outcome"}, TargetType.CRITIC).step_prompt(0, STEPS, VALUES, 1.0, {}),
        LLMJointLoss(MODEL, None, {"value"}).game_prompt(STEPS, VALUES, 1.0, {}),
    ]
    assert not any("gamma" in prompt or "Discounted" in prompt or "discounted" in prompt for prompt in prompts)
    assert "after +0.40; change +0.20" in prompts[0]
