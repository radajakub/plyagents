from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from plybench.utils.text import to_bool
from pydantic import BaseModel

from plyagents.textrl.configs.components import ComponentParams
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.losses.base import ActorLoss
from plyagents.textrl.losses.inputs import (
    CHECKED,
    NO_INPUTS,
    TRACK,
    ShownInputs,
    action_values_line,
    by_step,
    delta_line,
    deltas,
    discount_line,
    history_block,
    history_text,
    next_discount,
    outcome_line,
    parse_inputs,
    return_line,
)
from plyagents.textrl.losses.llm import LLMLoss
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.counts import KeyCounts
from plyagents.textrl.model.feedback import GRADES, Grade, StepFeedback
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.utils.calls import Calls
from plyagents.textrl.utils.options import read_options

SYSTEM = (
    "You judge the moves an agent played in a game, by what a critic says of them. Your lessons are added to the agent's notes for the situation"
    " they are about, and the agent reads them the next time it is in that situation."
)
LESSON = (
    "Where the agent should play differently next time, or keep playing the same, add a lesson: one or two sentences it can act on when it meets"
    " this situation again{limit}.{track}"
)
STEP_PROMPT = "{step}\n\nGrade the move ({grades}) and give a short reason. " + LESSON
GAME_PROMPT = (
    "The agent's moves in one game, in order. Each shows the situation as the agent saw it, the move it played and what the critic says of it.\n\n"
    "{steps}{history}{outcome}\n\nFor every move give its step number, a grade ({grades}) and a short reason. " + LESSON
)
# the TD target of the state-value critic after the last move: the game's end, whatever the inputs
ENDED = "After this move the game ended. "

# what F_pi is shown besides the situation, the move and the critic's assessment (the strict loss of SAC and A2C), in a fixed order:
#   outcome     the game's outcome at every step (Monte Carlo)
#   trajectory  the whole game in one call (hindsight); without it every step is judged in its own call, from that step alone
#   memory      the entry of the actor's memory it read before the move
#   delta       the critic's value before and after the move (a numeric critic)
#   q           the critic's value of every legal move (a numeric critic)
#   history     the moves played in the same situation in the earlier games and their results
#   reasoning   the actor's reasoning
INPUTS = ("outcome", "trajectory", "memory", "delta", "q", "history", "reasoning")


def _at(corrections: list[str | None], index: int, shown: list[int] | None) -> str | None:
    # the correction of step index, noted in shown (1-based) when there is one, as the prompt shows it
    note = corrections[index] if index < len(corrections) else None
    if note and shown is not None:
        shown.append(index + 1)
    return note


class _Judged(BaseModel):
    grade: Grade
    reason: str
    lesson: str | None = None


class _NumberedJudged(_Judged):
    step: int


class _Judgements(BaseModel):
    verdicts: list[_NumberedJudged]


def _verdict(item: _Judged) -> StepFeedback:
    return StepFeedback(item.grade, item.reason, item.lesson or None)


def read_inputs(text: str) -> str:
    return parse_inputs(text, INPUTS, "actor_loss")


@dataclass(frozen=True, eq=True)
class LLMActorLossParams(ShownInputs, ComponentParams):
    model: ModelSpec | None = None
    # the length the lessons are asked to keep to, in characters; None states no limit
    lesson_chars: int | None = None
    # what F_pi is shown besides the step and the critic's assessment (INPUTS), joined by +; "none": the strict loss
    inputs: str = NO_INPUTS
    # asks the lessons to keep track of the moves tried and how sure they are (TRACK)
    track: bool = False

    @classmethod
    def from_string(cls, params_string: str) -> LLMActorLossParams:
        options = read_options(params_string, {"model", "lesson_chars", "inputs", "track"}, "llm actor_loss")
        return cls(
            ModelSpec.from_string(options["model"]) if "model" in options else None,
            int(options["lesson_chars"]) if "lesson_chars" in options else None,
            read_inputs(options.get("inputs", cls.inputs)),
            to_bool(options.get("track", False)),
        )


class LLMActorLoss(LLMLoss, ActorLoss):
    # F_pi: grades every move by the critic's assessment. The Q critic's text assesses every legal move, so the step is judged against its alternatives
    # (SAC); the state-value critic's is compared with its assessment of the next situation the agent moved in, or with the outcome after the last
    # move (the TD error of A2C). One call per step, shown that step alone, or with `trajectory` one call for the whole game. A failed move has
    # nothing to judge, so no call of its own
    role = "actor_loss"
    system = SYSTEM
    known = INPUTS

    def __init__(self, model: ModelSpec, lesson_chars: int | None, inputs: set[str], track: bool = False, gamma: float = 1.0) -> None:
        super().__init__(model, inputs, gamma)
        self.lesson_chars = lesson_chars
        self.track = track

    def asked(self, step: TextRLStep, assessment: Assessment, whole_game: bool) -> bool:
        return whole_game or step.action is not None

    def _critic(
        self, index: int, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, alone: bool, corrections: list[str | None], shown: list[int] | None
    ) -> list[str]:
        # what the critic says of the step: every legal move (Q), or the situation before and after the move (V); alone: the step is shown without the
        # rest of the game, so the next situation is written out. A correction of an assessment by the critic loss (the backward sweep) follows it
        assessment = assessments[index]
        if not assessment.text:
            return []
        checked = [CHECKED.format(note=note)] if (note := _at(corrections, index, shown)) else []
        if assessment.moves:
            return [f"The critic's assessment of each legal move:\n{assessment.text}", *checked]
        lines = [f"The critic's assessment of the situation: {assessment.text}", *checked]
        if index + 1 == len(steps):
            lines.append(ENDED + outcome_line(outcome))
        elif alone:
            following = steps[index + 1]
            lines.append(f"The next situation the agent moved in:\n{following.key}\nThe critic's assessment of it: {assessments[index + 1].text}{next_discount(self.gamma)}")
            if note := _at(corrections, index + 1, shown):
                lines.append(CHECKED.format(note=note))
        return lines

    def _step(
        self,
        index: int,
        steps: list[TextRLStep],
        assessments: list[Assessment],
        outcome: float,
        history: Mapping[str, KeyCounts],
        alone: bool,
        corrections: list[str | None],
        shown: list[int] | None = None,
    ) -> str:
        step, assessment = steps[index], assessments[index]
        lines = [f"The situation, as the agent saw it:\n{step.key}" if alone else f"Step {index + 1}:\n{step.key}"]
        if "memory" in self.inputs:
            lines.append(f"The notes the agent read here:\n{step.context}" if step.context else "The agent read no notes here.")
        if "reasoning" in self.inputs and step.reasoning:
            lines.append(f"The agent's reasoning for this move:\n{step.reasoning.strip()}")
        lines.append(f"Move: {step.action or 'no valid move'}")
        lines.extend(self._critic(index, steps, assessments, outcome, alone, corrections, shown))
        if "delta" in self.inputs:
            lines.append(delta_line(deltas(assessments, outcome, self.role, self.gamma)[index], self.gamma))
        if "q" in self.inputs:
            lines.append(action_values_line(step, assessment, self.role))
        if alone and "history" in self.inputs:
            lines.append(f"Earlier games in this situation: {history_text(history.get(step.key, {})) or 'none'}.")
        if alone and "outcome" in self.inputs:
            lines.append(return_line(outcome, index, len(steps), self.gamma))
        return "\n".join(lines)

    def _instructions(self) -> dict[str, str]:
        return {"grades": ", ".join(GRADES), "limit": f", at most {self.lesson_chars} characters" if self.lesson_chars is not None else "", "track": TRACK if self.track else ""}

    def step_prompt(
        self,
        index: int,
        steps: list[TextRLStep],
        assessments: list[Assessment],
        outcome: float,
        history: Mapping[str, KeyCounts],
        corrections: list[str | None] | None = None,
        shown: list[int] | None = None,
    ) -> str:
        return STEP_PROMPT.format(step=self._step(index, steps, assessments, outcome, history, True, corrections or [], shown), **self._instructions())

    def game_prompt(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts]) -> str:
        shown = "\n\n".join(self._step(index, steps, assessments, outcome, history, False, []) for index in range(len(steps)))
        return GAME_PROMPT.format(
            steps=shown,
            history=history_block(history, [step.key for step in steps]) if "history" in self.inputs else "",
            outcome=f"\n\n{outcome_line(outcome)}{discount_line(self.gamma)}" if "outcome" in self.inputs else "",
            **self._instructions(),
        )

    async def ask_game(self, system: str, prompt: str, calls: Calls, about: dict[str, Any], steps: int) -> tuple[dict[int, StepFeedback], list[dict[str, Any]]]:
        answer = await calls.structured(self.role, self.model.config, system, prompt, _Judgements, about)
        found, dropped = by_step(answer.output.verdicts, steps)
        return {number: _verdict(item) for number, item in found.items()}, dropped

    async def ask_step(self, system: str, prompt: str, calls: Calls, about: dict[str, Any]) -> StepFeedback:
        return _verdict((await calls.structured(self.role, self.model.config, system, prompt, _Judged, about)).output)

    def describe(self) -> str:
        limit = f", lessons of at most {self.lesson_chars} characters" if self.lesson_chars is not None else ""
        calls = "one call per game" if self.whole_game else "one call per step, shown that step alone"
        shown = "+".join(name for name in INPUTS if name in self.inputs) or NO_INPUTS
        return f"{self.model.to_string()} grades every move by the critic's assessment, {calls}, also shown {shown}{limit}"
