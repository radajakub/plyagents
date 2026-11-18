from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from plyagents.textrl.configs.components import ComponentParams
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.enums import TargetType
from plyagents.textrl.losses.base import CriticLoss
from plyagents.textrl.losses.inputs import (
    CHECKED,
    NO_INPUTS,
    ShownInputs,
    by_step,
    discount_line,
    history_block,
    history_text,
    next_discount,
    outcome_line,
    parse_inputs,
    results_text,
    return_line,
)
from plyagents.textrl.losses.llm import LLMLoss
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.counts import KeyCounts, MoveCounts, counts_to_dict
from plyagents.textrl.model.feedback import Error, StepFeedback
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.utils.calls import Calls
from plyagents.textrl.utils.options import read_options

SYSTEM = (
    "You check a critic's estimates of how good the situations and moves of a game are for an agent that plays it, from -1, a sure loss, to +1, a"
    " sure win. Your notes are added to the critic's notes for the situation they are about, and the critic reads them the next time it assesses it."
    " One game's outcome, or the critic's assessment one step later, is one sample of that worth, not the worth itself: a good move can lose and a"
    " bad one win by chance."
)
ANSWER = "Give the error, your corrected estimate (from -1 to +1) and a short reason."
NOTE = "Where it was too high or too low, add a critic note: your estimate of what {worth} is worth here, and why{limit}."
STEP_PROMPT = "{step}\n\n{target}\n\nWas the critic's assessment too high, too low or right? {how} " + ANSWER + " " + NOTE
GAME_PROMPT = (
    "A critic's assessments in one game an agent played, in order. Each step shows the situation as the agent saw it, the move it played and the"
    " critic's assessment of {assessed}.\n\n{steps}{history}\n\n{outcome}\n\nFor every step give its step number and the error of the critic's"
    " assessment compared with {compared}: too high, too low or right. {how} Also give your corrected estimate (from -1 to +1) and a short reason. " + NOTE
)
ENDED = "After this move the game ended. "
# how the estimate follows the target: a step towards one sample (mc, critic; the size is left to the model, a step size alpha is planned), or the
# record's mean
SAMPLE = "The target is one sample of the worth: correct the estimate by a step towards it, not onto it."
MEAN = "The record's mean outcome is the best estimate of the worth: the assessment is right when it is close to it."

# what F_Q is shown besides the step, the critic's assessment and its target:
#   outcome     the game's outcome at every step, beside the temporal-difference target
#   trajectory  the whole game in one call (hindsight); without it every step is checked in its own call, from that step and its target alone
#   history     the moves played in the same situation in all earlier games and their results, so a note can rest on the record, not on one game
INPUTS = ("outcome", "trajectory", "history")


class _Checked(BaseModel):
    error: Error
    estimate: float = Field(ge=-1.0, le=1.0)
    reason: str
    note: str | None = None


class _NumberedChecked(_Checked):
    step: int


class _Checks(BaseModel):
    checks: list[_NumberedChecked]


def _verdict(item: _Checked) -> StepFeedback:
    return StepFeedback(reason=item.reason, critic_note=item.note or None, error=item.error, estimate=item.estimate)


def read_inputs(text: str) -> str:
    return parse_inputs(text, INPUTS, "critic_loss")


@dataclass(frozen=True, eq=True)
class LLMCriticLossParams(ShownInputs, ComponentParams):
    model: ModelSpec | None = None
    # the length the critic notes are asked to keep to, in characters; None states no limit
    note_chars: int | None = None
    inputs: str = NO_INPUTS
    # what the assessment is checked against: mc, the outcome; critic, the critic's own assessment at the next step and the outcome after the last
    # (temporal difference); record, the mean outcome of the move (of the situation, for V) in the earlier games and this one
    target: TargetType = TargetType.MC

    @classmethod
    def from_string(cls, params_string: str) -> LLMCriticLossParams:
        options = read_options(params_string, {"model", "note_chars", "inputs", "target"}, "llm critic_loss")
        target = TargetType.from_value(options.get("target", cls.target.value))
        if target is None:
            raise ValueError(f"Invalid llm critic_loss target {options['target']!r}; choose from {TargetType.values()}")
        return cls(
            ModelSpec.from_string(options["model"]) if "model" in options else None,
            int(options["note_chars"]) if "note_chars" in options else None,
            read_inputs(options.get("inputs", cls.inputs)),
            target,
        )


def record(step: TextRLStep, assessment: Assessment, history: Mapping[str, KeyCounts], outcome: float) -> MoveCounts:
    # the games of the assessed move (for V, of every move of the situation) before this one, and this game
    counts = history.get(step.key, {})
    if assessment.moves:
        earlier = counts.get(step.action, MoveCounts()) if step.action is not None else MoveCounts()
    else:
        earlier = MoveCounts()
        for move in counts.values():
            results = {name: earlier.results.get(name, 0) + move.results.get(name, 0) for name in {*earlier.results, *move.results}}
            earlier = MoveCounts(results, earlier.total + move.total, earlier.squares + move.squares)
    return earlier.plus(outcome)


def assessed(step: TextRLStep, assessment: Assessment) -> str | None:
    # the critic's assessment of the step: the situation's (V), or the move played's (Q); None for a move it left out or a failed move
    if not assessment.moves:
        return assessment.text or None
    if step.action is None or assessment.move_texts is None or step.action not in step.legal:
        return None
    return assessment.move_texts[step.legal.index(step.action)]


class LLMCriticLoss(LLMLoss, CriticLoss):
    # F_Q: checks the critic's assessment of every step against its target, the outcome (mc), the critic's own next assessment and the outcome after
    # the last step (critic, temporal difference) or the mean outcome of the move's record (record), gives the sign of the error and a corrected
    # estimate, and writes a critic note where it was off. One call per step, shown that step and its target alone, or with `trajectory` one call for
    # the whole game. A step whose move the Q critic did not assess (or a failed one) is not checked
    role = "critic_loss"
    system = SYSTEM
    known = INPUTS

    def __init__(self, model: ModelSpec, note_chars: int | None, inputs: set[str], target: TargetType, gamma: float = 1.0) -> None:
        super().__init__(model, inputs, gamma)
        if target is TargetType.RECORD and gamma != 1:
            raise ValueError("llm critic_loss target=record needs gamma=1: the record keeps the outcomes undiscounted")
        self.note_chars = note_chars
        self.target = target

    def asked(self, step: TextRLStep, assessment: Assessment, whole_game: bool) -> bool:
        return assessed(step, assessment) is not None

    def _step(self, index: int, steps: list[TextRLStep], assessments: list[Assessment], alone: bool, history: Mapping[str, KeyCounts], outcome: float) -> str:
        step, assessment = steps[index], assessments[index]
        what = "the move played" if assessment.moves else "the situation"
        head = f"The situation, as the agent saw it:\n{step.key}" if alone else f"Step {index + 1}:\n{step.key}"
        text = f"{head}\nMove: {step.action or 'no valid move'}\nThe critic's assessment of {what}: {assessed(step, assessment) or 'none'}"
        if alone and "history" in self.inputs:
            text += f"\nEarlier games in this situation: {history_text(history.get(step.key, {})) or 'none'}."
        if not alone and self.target is TargetType.RECORD:
            text += "\n" + self._record_line(step, assessment, history, outcome)
        return text

    def _record_line(self, step: TextRLStep, assessment: Assessment, history: Mapping[str, KeyCounts], outcome: float) -> str:
        counts = record(step, assessment, history, outcome)
        what = "the move played" if assessment.moves else "the situation"
        return f"The record of {what} here, this game included: {results_text(counts)}, mean outcome {counts.total / counts.visits:+.2f}."

    def _target(
        self,
        index: int,
        steps: list[TextRLStep],
        assessments: list[Assessment],
        outcome: float,
        history: Mapping[str, KeyCounts],
        corrections: list[str | None],
        shown: list[int] | None = None,
    ) -> str:
        # the record: the outcome and the record with it; the TD target: the critic's next assessment, with what the check of it found when the sweep
        # checked it first (the backward sweep)
        if self.target is TargetType.RECORD:
            ended = ENDED + outcome_line(outcome) if index + 1 == len(steps) else f"The game ended later. {outcome_line(outcome)}"
            return f"{ended}\n{self._record_line(steps[index], assessments[index], history, outcome)}"
        if self.target is TargetType.MC or index + 1 == len(steps):
            return ENDED + outcome_line(outcome) if index + 1 == len(steps) else f"The game ended later. {return_line(outcome, index, len(steps), self.gamma)}"
        following, assessment = steps[index + 1], assessments[index + 1]
        what = f"its next move, {following.action or 'no valid move'}" if assessment.moves else "it"
        target = f"The next situation the agent moved in:\n{following.key}\nThe critic's assessment of {what}: {assessed(following, assessment) or 'none'}"
        target += next_discount(self.gamma)
        if index + 1 < len(corrections) and (note := corrections[index + 1]):
            target += "\n" + CHECKED.format(note=note)
            if shown is not None:
                shown.append(index + 2)
        return f"{target}\n{return_line(outcome, index, len(steps), self.gamma)}" if "outcome" in self.inputs else target

    def _how(self) -> str:
        return MEAN if self.target is TargetType.RECORD else SAMPLE

    def _worth(self, assessments: list[Assessment]) -> dict[str, str]:
        moves = any(assessment.moves for assessment in assessments)
        return {"worth": "this move" if moves else "the situation", "limit": f", at most {self.note_chars} characters" if self.note_chars is not None else ""}

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
        target = self._target(index, steps, assessments, outcome, history, corrections or [], shown)
        step = self._step(index, steps, assessments, True, history, outcome)
        return STEP_PROMPT.format(step=step, target=target, how=self._how(), **self._worth(assessments))

    def game_prompt(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts]) -> str:
        moves = any(assessment.moves for assessment in assessments)
        compared = {
            TargetType.MC: "the outcome",
            TargetType.CRITIC: "the critic's assessment at the next step (the outcome after the last step)",
            TargetType.RECORD: "the record of its step",
        }[self.target]
        return GAME_PROMPT.format(
            assessed="the move played" if moves else "the situation",
            steps="\n\n".join(self._step(index, steps, assessments, False, history, outcome) for index in range(len(steps))),
            history=history_block(history, [step.key for step in steps]) if "history" in self.inputs else "",
            outcome=outcome_line(outcome) + discount_line(self.gamma),
            compared=compared,
            how=self._how(),
            **self._worth(assessments),
        )

    async def ask_game(self, system: str, prompt: str, calls: Calls, about: dict[str, Any], steps: int) -> tuple[dict[int, StepFeedback], list[dict[str, Any]]]:
        answer = await calls.structured(self.role, self.model.config, system, prompt, _Checks, about)
        found, dropped = by_step(answer.output.checks, steps)
        return {number: _verdict(item) for number, item in found.items()}, dropped

    async def ask_step(self, system: str, prompt: str, calls: Calls, about: dict[str, Any]) -> StepFeedback:
        return _verdict((await calls.structured(self.role, self.model.config, system, prompt, _Checked, about)).output)

    def inputs_record(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts]) -> dict[str, Any]:
        checked = [number for number, (step, assessment) in enumerate(zip(steps, assessments), start=1) if self.asked(step, assessment, self.whole_game)]
        inputs: dict[str, Any] = {"shown": [name for name in INPUTS if name in self.inputs], "target": self.target.value, "checked": checked}
        if "history" in self.inputs or self.target is TargetType.RECORD:
            inputs["history"] = [counts_to_dict(history.get(step.key, {})) for step in steps]
        return inputs

    def describe(self) -> str:
        target = {
            TargetType.MC: "the outcome",
            TargetType.CRITIC: "its own next assessment (the outcome after the last step)",
            TargetType.RECORD: "the mean outcome of the move's record",
        }[self.target]
        calls = "one call per game" if self.whole_game else "one call per step, shown that step and its target alone"
        limit = f", notes of at most {self.note_chars} characters" if self.note_chars is not None else ""
        return f"{self.model.to_string()} checks the critic's assessments against {target}, {calls}{limit}"
