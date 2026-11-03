from __future__ import annotations

from dataclasses import dataclass

from plybench.common.enums import GameResults

from plyagents.tac.model.step import TACStep
from plyagents.tac.rewards.result import RESULT_OPTIONS, ResultParams, ResultShaper
from plyagents.tac.utils.options import read_options


@dataclass(frozen=True, eq=True)
class LengthPenaltyParams(ResultParams):
    penalty: float = 0.01

    @classmethod
    def from_string(cls, params_string: str) -> LengthPenaltyParams:
        options = read_options(params_string, RESULT_OPTIONS | {"penalty"}, "length_penalty")
        return cls(float(options.get("my_fail", -1.0)), float(options.get("opponent_fail", 1.0)), float(options.get("penalty", 0.01)))


class LengthPenaltyShaper(ResultShaper):
    # the result minus penalty * (agent moves), so a shorter win is worth more and a longer loss costs more. The shaped G
    # leaves the scale of V*, so a critic target from the oracle critic mixes two scales
    def __init__(self, my_fail: float = -1.0, opponent_fail: float = 1.0, penalty: float = 0.01) -> None:
        super().__init__(my_fail, opponent_fail)
        self.penalty = penalty

    def outcome(self, result: GameResults, steps: list[TACStep]) -> float:
        return super().outcome(result, steps) - self.penalty * len(steps)
