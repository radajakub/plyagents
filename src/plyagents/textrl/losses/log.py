from __future__ import annotations

from collections.abc import Mapping

from plyagents.textrl.losses.base import JointLoss
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.counts import KeyCounts, result
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.flow import Given
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.utils.calls import Calls


class LogJointLoss(JointLoss):
    # no grade and no call: a fixed lesson per step, the move and the game's result ("Played <button:red>: win."). With Apply truncate and no budget,
    # the actor's entry is the full history of its moves in order: the raw history in the prompt, as in the bandit literature (H-raw)
    async def give(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], calls: Calls) -> Given:
        return Given([StepFeedback(lesson=f"Played {step.action or 'no valid move'}: {result(outcome)}.") for step in steps])

    def describe(self) -> str:
        return "no verdict, no call: logs every move and the game's result as a lesson"
