from __future__ import annotations

from collections.abc import Mapping

from plyagents.tac.feedback.base import Feedback
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.counts import KeyCounts, result
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.step import TACStep
from plyagents.tac.utils.calls import Calls


class LogFeedback(Feedback):
    # no grade and no call: a fixed lesson per step, the move and the game's result ("Played <button:red>: win."). With Apply truncate and no budget,
    # the actor's entry is the full history of its moves in order: the raw history in the prompt, as in the bandit literature (H-raw)
    async def give(self, steps: list[TACStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], calls: Calls) -> list[StepFeedback]:
        return [StepFeedback(lesson=f"Played {step.action or 'no valid move'}: {result(outcome)}.") for step in steps]

    def describe(self) -> str:
        return "no verdict, no call: logs every move and the game's result as a lesson"
