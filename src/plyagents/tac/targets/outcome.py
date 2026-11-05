from __future__ import annotations

from plyagents.tac.model.assessment import Assessment
from plyagents.tac.targets.base import Target


class OutcomeTarget(Target):
    # Monte Carlo: y_t = G
    def targets(self, assessments: list[Assessment], outcome: float) -> list[float]:
        return [outcome for _ in assessments]

    def describe(self) -> str:
        return "y_t = G, the outcome of the game (Monte Carlo)"
