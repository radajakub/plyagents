"""The actors: the oracle plays the solved policy through its selector, the tabular one learns Q(z, a) from its targets."""

from __future__ import annotations

import asyncio

import numpy as np
from plybench.core.minimax import load_or_solve

from plyagents.textrl.actors.build import ACTORS
from plyagents.textrl.actors.oracle import OracleActor
from plyagents.textrl.actors.tabular import TabularActor
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.step import Situation, TextRLStep
from plyagents.textrl.schedulers.constant import ConstantScheduler
from plyagents.textrl.selectors.greedy import Greedy
from plyagents.textrl.targets.critic import CriticTarget
from plyagents.textrl.targets.outcome import OutcomeTarget


def _first_situation(engine) -> Situation:
    game = engine.game
    os_observation = game.get_observation(0)
    observation = engine.observation_class.from_openspiel(os_observation, engine.interface_transformer)
    legal = [engine.action_class.from_openspiel(move, engine.interface_transformer) for move in game.get_legal_moves(0)]
    return Situation("start", observation, legal, [action.to_llm().string for action in legal], 0, os_observation.state, False)


def test_the_oracle_actor_with_greedy_plays_an_optimal_action(tic_tac_toe):
    situation = _first_situation(tic_tac_toe)
    actor = OracleActor()
    actor.initialize(tic_tac_toe.game, tic_tac_toe.prompt_adapter)
    optimal = load_or_solve(tic_tac_toe.game)[0, situation.os_state]
    assert optimal is not None

    for seed in range(10):
        output = asyncio.run(actor.act(situation, Greedy(), np.random.default_rng(seed)))
        assert output.action is not None and optimal.check_optimal(output.action.number)


def test_the_tabular_actor_learns_from_its_targets_and_then_prefers_the_action(tic_tac_toe, calls):
    situation = _first_situation(tic_tac_toe)
    actor = TabularActor(OutcomeTarget(), ConstantScheduler(0.5))
    chosen = situation.actions[4]
    update = asyncio.run(actor.propose([TextRLStep("start", chosen, 0, situation.os_state)], [Assessment(None, "")], 1.0, [StepFeedback()], calls))
    assert update is not None
    actor.commit(update)

    assert actor.tables["start"].get(chosen).value == 0.5
    output = asyncio.run(actor.act(situation, Greedy(), np.random.default_rng(0)))
    assert output.action is not None and output.action.to_llm().string == chosen


def test_the_tabular_actor_skips_failed_moves_and_uses_its_target(calls):
    actor = TabularActor(CriticTarget(), ConstantScheduler(1.0))
    steps = [TextRLStep("z", "a", 0, "s0"), TextRLStep("y", None, 0, "s1"), TextRLStep("x", "c", 0, "s2")]
    update = asyncio.run(actor.propose(steps, [Assessment(0.1, ""), Assessment(0.2, ""), Assessment(0.3, "")], -1.0, [StepFeedback()] * 3, calls))
    assert update is not None
    assert [(sample.key, sample.action, sample.target) for sample in update.samples] == [("z", "a", 0.2), ("x", "c", -1.0)]


def test_the_tabular_actor_restores_its_tables(calls):
    actor = TabularActor(OutcomeTarget(), ConstantScheduler(0.5))
    update = asyncio.run(actor.propose([TextRLStep("z", "a", 0, "s")], [Assessment(None, "")], 1.0, [StepFeedback()], calls))
    assert update is not None
    actor.commit(update)
    restored = TabularActor(OutcomeTarget(), ConstantScheduler(0.5))
    restored.load_state(actor.state())
    assert restored.state() == actor.state()


def test_the_collection_builds_each_actor(services):
    assert isinstance(ACTORS.build(ACTORS.config("oracle"), services), OracleActor)
    tabular = ACTORS.build(ACTORS.config("tabular[target=critic]"), services)
    assert isinstance(tabular, TabularActor) and isinstance(tabular.target, CriticTarget)
