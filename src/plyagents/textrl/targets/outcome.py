from __future__ import annotations

from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.targets.base import Target


def discounted(outcome: float, index: int, steps: int, gamma: float) -> float:
    # the return of step index of a game of `steps` moves of the agent, whose one reward is the outcome after the last: gamma^k G, k the later moves
    return gamma ** (steps - 1 - index) * outcome


class OutcomeTarget(Target):
    # Monte Carlo: y_t = gamma^(T-1-t) G, the outcome discounted over the agent's later moves (G itself without a discount)
    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def targets(self, assessments: list[Assessment], outcome: float) -> list[float]:
        return [discounted(outcome, index, len(assessments), self.gamma) for index in range(len(assessments))]

    def sources(self, steps: int) -> list[str]:
        return ["outcome"] * steps

    def describe(self) -> str:
        discount = f", discounted by gamma={self.gamma:g} per later move" if self.gamma != 1 else ""
        return f"y_t = G, the outcome of the game (Monte Carlo){discount}"
