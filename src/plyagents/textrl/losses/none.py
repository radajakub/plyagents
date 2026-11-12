from __future__ import annotations

from collections.abc import Mapping

from plyagents.textrl.losses.base import ActorLoss, CriticLoss, JointLoss, Loss
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.counts import KeyCounts
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.flow import Given
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.utils.calls import Calls


class NoLoss(Loss):
    # the loss of type none of every role: no verdict and no call
    judges = False

    async def give(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], calls: Calls) -> Given:
        return Given([StepFeedback() for _ in steps])

    async def judge_step(
        self, index: int, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], corrections: list[str | None], calls: Calls
    ) -> StepFeedback:
        return StepFeedback()


class NoJointLoss(NoLoss, JointLoss):
    # the components learn from the outcome and the critic's values alone (the tabular ones), or from their own losses
    def describe(self) -> str:
        return "no verdict, no call"


class NoActorLoss(NoLoss, ActorLoss):
    def describe(self) -> str:
        return "no verdict, no call"


class NoCriticLoss(NoLoss, CriticLoss):
    def describe(self) -> str:
        return "no note, no call"
