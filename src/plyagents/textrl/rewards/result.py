from __future__ import annotations

from dataclasses import dataclass

from plybench.common.enums import GameResults

from plyagents.textrl.configs.components import ComponentParams
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.rewards.base import RewardShaper
from plyagents.textrl.utils.options import read_options

RESULT_OPTIONS = {"my_fail", "opponent_fail"}


@dataclass(frozen=True, eq=True)
class ResultParams(ComponentParams):
    my_fail: float = -1.0
    opponent_fail: float = 1.0

    @classmethod
    def from_string(cls, params_string: str) -> ResultParams:
        options = read_options(params_string, RESULT_OPTIONS, "result")
        return cls(float(options.get("my_fail", -1.0)), float(options.get("opponent_fail", 1.0)))


class ResultShaper(RewardShaper):
    # the result alone, on the game's own scale (OpenSpiel rewards, as the minimax values): win +1, draw 0, loss -1, and the
    # fail values for a failed move. plybench's GameResults.to_reward (draw 0.5, loss 0) is another scale and would not match V*
    def __init__(self, my_fail: float = -1.0, opponent_fail: float = 1.0) -> None:
        self.values = {GameResults.WIN: 1.0, GameResults.DRAW: 0.0, GameResults.LOSS: -1.0, GameResults.MY_FAIL: my_fail, GameResults.OPPONENT_FAIL: opponent_fail}

    def outcome(self, result: GameResults, steps: list[TextRLStep]) -> float:
        return self.values[result]

    def value_text(self, value: float, certain: bool) -> str:
        win, loss = self.values[GameResults.WIN], self.values[GameResults.LOSS]
        if not certain:
            # with chance, 0 is not a draw but an even chance to win: the value is named as an expectation between its ends
            return f"has an expected outcome of {value:+.2f} (a sure loss is {loss:+g}, a sure win {win:+g})"
        names = {win: "wins", self.values[GameResults.DRAW]: "draws", loss: "loses"}
        return names.get(value, f"is worth {value:+g}")

    def describe(self) -> str:
        return f"win +1, draw 0, loss -1, own fail {self.values[GameResults.MY_FAIL]:+g}, opponent fail {self.values[GameResults.OPPONENT_FAIL]:+g}"
