"""The critics: none gives no value, the oracle the solved value V*, the tabular one a Monte Carlo estimate that it learns."""

from __future__ import annotations

from plybench.core.minimax import load_or_solve

from plyagents.tac.critics.build import CRITICS
from plyagents.tac.critics.none import NoCritic
from plyagents.tac.critics.oracle import OracleCritic
from plyagents.tac.critics.tabular import TabularCritic
from plyagents.tac.model.step import TACStep
from plyagents.tac.schedulers.constant import ConstantScheduler


def _steps(*keys: str) -> list[TACStep]:
    return [TACStep(key, f"a{index}", 0, f"s{index}") for index, key in enumerate(keys)]


def test_no_critic_gives_no_value():
    assert [(assessment.value, assessment.text) for assessment in NoCritic().assess(_steps("z", "y"))] == [(None, ""), (None, "")]


def test_the_oracle_critic_gives_the_solved_value_of_the_true_state(tic_tac_toe):
    game = tic_tac_toe.game
    observation = game.get_observation(0)
    critic = OracleCritic()
    critic.initialize(game)
    [assessment] = critic.assess([TACStep("any key", None, 0, observation.state)])

    solved = load_or_solve(game)[0, observation.state]
    assert solved is not None
    # tic-tac-toe is a draw with best play
    assert assessment.value == solved.V() == 0.0
    assert assessment.text == "value +0"


def test_the_tabular_critic_assesses_before_it_learns_from_the_outcome():
    critic = TabularCritic(ConstantScheduler(0.5))
    steps = _steps("z", "y")
    assert [assessment.value for assessment in critic.assess(steps)] == [0.0, 0.0]

    update = critic.propose(steps, 1.0)
    assert update is not None
    critic.commit(update)
    assert [assessment.value for assessment in critic.assess(steps)] == [0.5, 0.5]
    assert critic.assess(steps)[0].text == "value +0.50"


def test_the_tabular_critic_restores_its_table():
    critic = TabularCritic(ConstantScheduler(0.5))
    update = critic.propose(_steps("z"), -1.0)
    assert update is not None
    critic.commit(update)
    restored = TabularCritic(ConstantScheduler(0.5))
    restored.load_state(critic.state())
    assert restored.state() == critic.state()


def test_the_collection_builds_each_critic():
    assert isinstance(CRITICS.build(CRITICS.config("none")), NoCritic)
    assert isinstance(CRITICS.build(CRITICS.config("oracle")), OracleCritic)
    tabular = CRITICS.build(CRITICS.config("tabular[scheduler=constant[beta=0.05]]"))
    assert isinstance(tabular, TabularCritic) and isinstance(tabular.scheduler, ConstantScheduler) and tabular.scheduler.beta == 0.05
