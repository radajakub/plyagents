from __future__ import annotations

from plyagents.tac.model.assessment import Assessment
from plyagents.tac.targets.base import Target


class CriticTarget(Target):
    # y_t = v_{t+1}, the critic's value of the agent's next state, and G after the last move
    def targets(self, assessments: list[Assessment], outcome: float) -> list[float]:
        if not assessments:
            return []
        values = [assessment.value for assessment in assessments[1:]]
        if any(value is None for value in values):
            raise ValueError("target=critic needs a critic that gives values (oracle or tabular)")
        return [float(value) for value in values if value is not None] + [outcome]

    def sources(self, steps: int) -> list[str]:
        return ["critic"] * (steps - 1) + ["outcome"] if steps else []

    def describe(self) -> str:
        return "y_t = v_{t+1}, the critic's value of the next own state, G after the last move"
