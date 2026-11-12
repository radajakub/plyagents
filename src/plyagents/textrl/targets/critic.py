from __future__ import annotations

from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.targets.base import Target


class CriticTarget(Target):
    # y_t = gamma v_{t+1}, the critic's value of the agent's next state, and G after the last move
    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def targets(self, assessments: list[Assessment], outcome: float) -> list[float]:
        if not assessments:
            return []
        values = [assessment.value for assessment in assessments[1:]]
        if any(value is None for value in values):
            raise ValueError("target=critic needs a critic that gives values (oracle or tabular)")
        return [self.gamma * float(value) for value in values if value is not None] + [outcome]

    def sources(self, steps: int) -> list[str]:
        return ["critic"] * (steps - 1) + ["outcome"] if steps else []

    def describe(self) -> str:
        discount = f", v_{{t+1}} discounted by gamma={self.gamma:g}" if self.gamma != 1 else ""
        return f"y_t = v_{{t+1}}, the critic's value of the next own state, G after the last move{discount}"
