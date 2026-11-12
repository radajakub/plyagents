"""The critics: none gives no value, the oracle the solved value V*, the tabular one a Monte Carlo estimate that it learns."""

from __future__ import annotations

import asyncio

from plybench.core.minimax import load_or_solve

from plyagents.textrl.critics.build import CRITICS
from plyagents.textrl.critics.none import NoCritic
from plyagents.textrl.critics.oracle import OracleCritic
from plyagents.textrl.critics.tabular import TabularCritic
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.model.table import TableUpdate
from plyagents.textrl.schedulers.constant import ConstantScheduler


def _steps(*keys: str) -> list[TextRLStep]:
    return [TextRLStep(key, f"a{index}", 0, f"s{index}") for index, key in enumerate(keys)]


def test_no_critic_gives_no_value(calls):
    assert [(assessment.value, assessment.text) for assessment in asyncio.run(NoCritic().assess(_steps("z", "y"), calls))] == [(None, ""), (None, "")]


def test_the_oracle_critic_gives_the_solved_value_of_the_true_state(tic_tac_toe, calls):
    game = tic_tac_toe.game
    observation = game.get_observation(0)
    critic = OracleCritic()
    critic.initialize(game, tic_tac_toe.prompt_adapter)
    [assessment] = asyncio.run(critic.assess([TextRLStep("any key", None, 0, observation.state)], calls))

    solved = load_or_solve(game)[0, observation.state]
    assert solved is not None
    # tic-tac-toe is a draw with best play
    assert assessment.value == solved.V() == 0.0
    assert assessment.text == "value +0"


def test_the_tabular_critic_assesses_before_it_learns_from_the_outcome(calls):
    critic = TabularCritic(ConstantScheduler(0.5))
    steps = _steps("z", "y")
    assert [assessment.value for assessment in asyncio.run(critic.assess(steps, calls))] == [0.0, 0.0]

    update = asyncio.run(critic.propose(steps, [Assessment(None, "")] * 2, 1.0, [StepFeedback()] * 2, calls))
    assert update is not None
    critic.commit(update)
    assert [assessment.value for assessment in asyncio.run(critic.assess(steps, calls))] == [0.5, 0.5]
    assert asyncio.run(critic.assess(steps, calls))[0].text == "value +0.50"


def test_the_tabular_critic_restores_its_table(calls):
    critic = TabularCritic(ConstantScheduler(0.5))
    update = asyncio.run(critic.propose(_steps("z"), [Assessment(None, "")], -1.0, [StepFeedback()], calls))
    assert update is not None
    critic.commit(update)
    restored = TabularCritic(ConstantScheduler(0.5))
    restored.load_state(critic.state())
    assert restored.state() == critic.state()


def test_the_collection_builds_each_critic(services):
    assert isinstance(CRITICS.build(CRITICS.config("none"), services), NoCritic)
    assert isinstance(CRITICS.build(CRITICS.config("oracle"), services), OracleCritic)
    tabular = CRITICS.build(CRITICS.config("tabular[scheduler=constant[beta=0.05]]"), services)
    assert isinstance(tabular, TabularCritic) and isinstance(tabular.scheduler, ConstantScheduler) and tabular.scheduler.beta == 0.05


def test_the_oracle_critic_gives_the_solved_action_values_aligned_with_the_legal_moves(tic_tac_toe, calls):
    game = tic_tac_toe.game
    observation = game.get_observation(0)
    numbers = tuple(move.number for move in game.get_legal_moves(0))
    critic = OracleCritic()
    critic.initialize(game, tic_tac_toe.prompt_adapter)
    [assessment] = asyncio.run(critic.assess([TextRLStep("k", None, 0, observation.state, tuple(str(number) for number in numbers), numbers)], calls))

    solved = load_or_solve(game)[0, observation.state]
    assert solved is not None and assessment.action_values == tuple(solved.Q(number) for number in numbers)


def test_the_tabular_critic_learns_q_of_the_moves_played_beside_v(calls):
    critic = TabularCritic(ConstantScheduler(0.5))
    steps = [TextRLStep("z", "a", 0, "s", ("a", "b"), (0, 1))]
    assert asyncio.run(critic.assess(steps, calls))[0].action_values == (None, None)

    update = asyncio.run(critic.propose(steps, [Assessment(None, "")], 1.0, [StepFeedback()], calls))
    assert update is not None
    critic.commit(TableUpdate.from_dict(update.to_dict()))
    [assessment] = asyncio.run(critic.assess(steps, calls))
    # Q only for the move played, the other stays unknown
    assert assessment.value == 0.5 and assessment.action_values == (0.5, None)
    restored = TabularCritic(ConstantScheduler(0.5))
    restored.load_state(critic.state())
    assert restored.state() == critic.state() and asyncio.run(restored.assess(steps, calls)) == [assessment]
