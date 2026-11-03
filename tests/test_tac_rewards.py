"""The reward shapers: the outcome G of a game for the agent, on the game's own scale."""

from __future__ import annotations

import pytest
from plybench.common.enums import GameResults

from plyagents.tac.model.step import TACStep
from plyagents.tac.rewards.build import REWARDS
from plyagents.tac.rewards.length_penalty import LengthPenaltyShaper
from plyagents.tac.rewards.result import ResultShaper


def _steps(count: int) -> list[TACStep]:
    return [TACStep(f"z{index}", f"a{index}", 0, f"s{index}") for index in range(count)]


@pytest.mark.parametrize(
    ("result", "outcome"),
    [(GameResults.WIN, 1.0), (GameResults.DRAW, 0.0), (GameResults.LOSS, -1.0), (GameResults.MY_FAIL, -1.0), (GameResults.OPPONENT_FAIL, 1.0)],
)
def test_the_result_shaper_uses_the_scale_of_the_game(result, outcome):
    assert ResultShaper().outcome(result, _steps(2)) == outcome


def test_the_result_shaper_takes_its_fail_values_and_ignores_the_length():
    assert ResultShaper(my_fail=-2.0).outcome(GameResults.MY_FAIL, _steps(1)) == -2.0
    assert ResultShaper(opponent_fail=0.5).outcome(GameResults.OPPONENT_FAIL, _steps(1)) == 0.5
    assert ResultShaper().outcome(GameResults.WIN, _steps(1)) == ResultShaper().outcome(GameResults.WIN, _steps(7)) == 1.0


def test_the_length_penalty_shaper_subtracts_the_penalty_per_agent_move():
    assert LengthPenaltyShaper(penalty=0.1).outcome(GameResults.WIN, _steps(3)) == pytest.approx(0.7)
    assert LengthPenaltyShaper(penalty=0.1).outcome(GameResults.LOSS, _steps(2)) == pytest.approx(-1.2)
    assert LengthPenaltyShaper(my_fail=-2.0, penalty=0.0).outcome(GameResults.MY_FAIL, _steps(4)) == -2.0


def test_the_collection_builds_each_shaper_with_its_params():
    result = REWARDS.build(REWARDS.config("result[my_fail=-2.0]"))
    assert isinstance(result, ResultShaper) and not isinstance(result, LengthPenaltyShaper)
    assert result.outcome(GameResults.MY_FAIL, _steps(2)) == -2.0
    shaper = REWARDS.build(REWARDS.config("length_penalty[penalty=0.1,my_fail=-2.0]"))
    assert isinstance(shaper, LengthPenaltyShaper)
    assert shaper.outcome(GameResults.MY_FAIL, _steps(2)) == pytest.approx(-2.2)
    assert REWARDS.config("length_penalty").to_string() == "length_penalty"
    with pytest.raises(ValueError, match="Unknown options"):
        REWARDS.config("result[penalty=0.1]")
