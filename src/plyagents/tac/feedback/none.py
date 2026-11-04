from __future__ import annotations

from plyagents.tac.feedback.base import Feedback
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.step import TACStep
from plyagents.tac.utils.calls import Calls


class NoFeedback(Feedback):
    # no verdict and no call: the components learn from the outcome and the critic's values alone (the tabular ones)
    async def give(self, steps: list[TACStep], assessments: list[Assessment], outcome: float, calls: Calls) -> list[StepFeedback]:
        return [StepFeedback() for _ in steps]

    def describe(self) -> str:
        return "no verdict, no call"
