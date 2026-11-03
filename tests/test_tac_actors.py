"""The actors: the oracle plays the solved policy through its selector, the tabular one learns Q(z, a) from its targets."""

from __future__ import annotations

import asyncio

import numpy as np
from plybench.core.minimax import load_or_solve

from plyagents.tac.actors.build import ACTORS
from plyagents.tac.actors.oracle import OracleActor
from plyagents.tac.actors.tabular import TabularActor
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.step import Situation, TACStep
from plyagents.tac.schedulers.constant import ConstantScheduler
from plyagents.tac.selectors.greedy import Greedy
from plyagents.tac.targets.critic import CriticTarget
from plyagents.tac.targets.outcome import OutcomeTarget


def _first_situation(engine) -> Situation:
    game = engine.game
    observation = game.get_observation(0)
    legal = [engine.action_class.from_openspiel(move, engine.interface_transformer) for move in game.get_legal_moves(0)]
    return Situation("start", legal, [action.to_llm().string for action in legal], 0, observation.state)


def test_the_oracle_actor_with_greedy_plays_an_optimal_action(tic_tac_toe):
    situation = _first_situation(tic_tac_toe)
    actor = OracleActor()
    actor.initialize(tic_tac_toe.game)
    optimal = load_or_solve(tic_tac_toe.game)[0, situation.os_state]
    assert optimal is not None

    for seed in range(10):
        output = asyncio.run(actor.act(situation, Greedy(), np.random.default_rng(seed)))
        assert output.action is not None and optimal.check_optimal(output.action.number)


def test_the_tabular_actor_learns_from_its_targets_and_then_prefers_the_action(tic_tac_toe):
    situation = _first_situation(tic_tac_toe)
    actor = TabularActor(OutcomeTarget(), ConstantScheduler(0.5))
    chosen = situation.actions[4]
    update = actor.propose([TACStep("start", chosen, 0, situation.os_state)], [Assessment(None, "")], 1.0)
    assert update is not None
    actor.commit(update)

    assert actor.tables["start"].get(chosen).value == 0.5
    output = asyncio.run(actor.act(situation, Greedy(), np.random.default_rng(0)))
    assert output.action is not None and output.action.to_llm().string == chosen


def test_the_tabular_actor_skips_failed_moves_and_uses_its_target():
    actor = TabularActor(CriticTarget(), ConstantScheduler(1.0))
    steps = [TACStep("z", "a", 0, "s0"), TACStep("y", None, 0, "s1"), TACStep("x", "c", 0, "s2")]
    update = actor.propose(steps, [Assessment(0.1, ""), Assessment(0.2, ""), Assessment(0.3, "")], -1.0)
    assert update is not None
    assert [(sample.key, sample.action, sample.target) for sample in update.samples] == [("z", "a", 0.2), ("x", "c", -1.0)]


def test_the_tabular_actor_restores_its_tables():
    actor = TabularActor(OutcomeTarget(), ConstantScheduler(0.5))
    update = actor.propose([TACStep("z", "a", 0, "s")], [Assessment(None, "")], 1.0)
    assert update is not None
    actor.commit(update)
    restored = TabularActor(OutcomeTarget(), ConstantScheduler(0.5))
    restored.load_state(actor.state())
    assert restored.state() == actor.state()


def test_the_collection_builds_each_actor():
    assert isinstance(ACTORS.build(ACTORS.config("oracle")), OracleActor)
    tabular = ACTORS.build(ACTORS.config("tabular[target=critic]"))
    assert isinstance(tabular, TabularActor) and isinstance(tabular.target, CriticTarget)
