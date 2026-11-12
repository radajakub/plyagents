from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.losses.base import Loss
from plyagents.textrl.losses.inputs import deltas
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.counts import KeyCounts, counts_to_dict
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.flow import Given
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.utils.calls import Calls
from plyagents.textrl.utils.rules import with_rules


class LLMLoss(Loss):
    # a loss that asks an LLM (F, F_pi, F_Q): one call for the whole game, its steps numbered (the joint loss, or a loss shown the trajectory), or one
    # call per step, shown that step alone (judge_step), the calls started in the order of the steps so each takes its seed by it. Every call has the
    # loss's system prompt with the rules of the game; role names its calls and its errors. A step it is not asked about gets no verdict. known: the
    # inputs it can be shown besides the moves, in their fixed order; inputs: the ones it is shown; gamma: the discount in the targets it is shown
    role: str
    system: str
    known: tuple[str, ...]

    def __init__(self, model: ModelSpec, inputs: set[str], gamma: float) -> None:
        self.model = model
        self.inputs = inputs
        self.gamma = gamma

    @property
    def whole_game(self) -> bool:
        return "trajectory" in self.inputs

    def asked(self, step: TextRLStep, assessment: Assessment, whole_game: bool) -> bool:
        return True

    @abstractmethod
    def game_prompt(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts]) -> str:
        raise NotImplementedError

    # the call for the whole game: the verdicts by step number, and the answers it dropped (for a step that does not exist, or one already answered)
    @abstractmethod
    async def ask_game(self, system: str, prompt: str, calls: Calls, about: dict[str, Any], steps: int) -> tuple[dict[int, StepFeedback], list[dict[str, Any]]]:
        raise NotImplementedError

    def step_prompt(
        self, index: int, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], corrections: list[str | None] | None = None
    ) -> str:
        raise NotImplementedError(f"{type(self).__name__} judges whole games only")

    async def ask_step(self, system: str, prompt: str, calls: Calls, about: dict[str, Any]) -> StepFeedback:
        raise NotImplementedError(f"{type(self).__name__} judges whole games only")

    async def give(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], calls: Calls) -> Given:
        if not self.whole_game:
            verdicts = await asyncio.gather(*(self.judge_step(index, steps, assessments, outcome, history, [], calls) for index in range(len(steps))))
            return Given(list(verdicts), [], self.inputs_record(steps, assessments, outcome, history))
        prompt = self.game_prompt(steps, assessments, outcome, history)
        found, dropped = await self.ask_game(with_rules(self.system, steps), prompt, calls, {"steps": list(range(1, len(steps) + 1))}, len(steps))
        verdicts = [
            found.get(number, StepFeedback()) if self.asked(step, assessment, True) else StepFeedback()
            for number, (step, assessment) in enumerate(zip(steps, assessments), start=1)
        ]
        return Given(verdicts, dropped, self.inputs_record(steps, assessments, outcome, history))

    async def judge_step(
        self, index: int, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], corrections: list[str | None], calls: Calls
    ) -> StepFeedback:
        if not self.asked(steps[index], assessments[index], False):
            return StepFeedback()
        prompt = self.step_prompt(index, steps, assessments, outcome, history, corrections)
        return await self.ask_step(with_rules(self.system, steps), prompt, calls, {"steps": [index + 1]})

    def inputs_record(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts]) -> dict[str, Any]:
        # what the prompt showed besides the moves and the outcome, by part: the inputs named, the critic's change per step, the earlier games per step
        inputs: dict[str, Any] = {"shown": [name for name in self.known if name in self.inputs]}
        if "delta" in self.inputs:
            inputs["deltas"] = [{"before": before, "after": after, "last": last} for before, after, last in deltas(assessments, outcome, self.role, self.gamma)]
        if "history" in self.inputs:
            inputs["history"] = [counts_to_dict(history.get(step.key, {})) for step in steps]
        return inputs
