from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from plyagents.tac.configs.components import ComponentParams
from plyagents.tac.configs.model import ModelSpec
from plyagents.tac.feedback.base import Feedback
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.feedback import GRADES, Grade, StepFeedback
from plyagents.tac.model.step import TACStep
from plyagents.tac.utils.calls import Calls
from plyagents.tac.utils.options import read_options

SYSTEM = (
    "You review a game an agent played and judge each of its moves. Your lessons are added to the agent's notes for the situation"
    " they are about, and the agent reads them the next time it is in that situation."
)
PROMPT = """The agent's moves, in order. Each shows the situation as the agent saw it and the move it played{estimates}.

{steps}

Outcome for the agent: {outcome:+g} (+1 is a win, 0 a draw, -1 a loss).

For every move give its step number, a grade ({grades}) and a short reason. Where the agent should play differently next time, or keep playing the same, add a lesson: one or two sentences it can act on when it meets this situation again{limit}.{critic}"""
CRITIC = " Where the value estimate of a move was wrong, add a critic note: what the position was really worth, and why."


class _Verdict(BaseModel):
    step: int
    grade: Grade
    reason: str
    lesson: str | None = None
    critic_note: str | None = None


class _Verdicts(BaseModel):
    verdicts: list[_Verdict]


@dataclass(frozen=True, eq=True)
class LLMFeedbackParams(ComponentParams):
    model: ModelSpec | None = None
    # the length the lessons are asked to keep to, in characters; None states no limit
    lesson_chars: int | None = None

    @classmethod
    def from_string(cls, params_string: str) -> LLMFeedbackParams:
        options = read_options(params_string, {"model", "lesson_chars"}, "llm feedback")
        return cls(ModelSpec.from_string(options["model"]) if "model" in options else None, int(options["lesson_chars"]) if "lesson_chars" in options else None)


class LLMFeedback(Feedback):
    # F as in the paper: one call per game, the moves numbered, the critic's estimates and the outcome shown; it answers per
    # step number (keys are long, the model never copies them). A step it leaves out gets no feedback
    def __init__(self, model: ModelSpec, lesson_chars: int | None) -> None:
        self.model = model
        self.lesson_chars = lesson_chars

    def prompt(self, steps: list[TACStep], assessments: list[Assessment], outcome: float) -> str:
        shown = [
            f"Step {number}:\n{step.key}\nMove: {step.action or 'no valid move'}" + (f"\nValue estimate: {assessment.text}" if assessment.text else "")
            for number, (step, assessment) in enumerate(zip(steps, assessments), start=1)
        ]
        estimated = any(assessment.text for assessment in assessments)
        return PROMPT.format(
            estimates=" and the value estimate of the position" if estimated else "",
            steps="\n\n".join(shown),
            outcome=outcome,
            grades=", ".join(GRADES),
            limit=f", at most {self.lesson_chars} characters" if self.lesson_chars is not None else "",
            critic=CRITIC if estimated else "",
        )

    async def give(self, steps: list[TACStep], assessments: list[Assessment], outcome: float, calls: Calls) -> list[StepFeedback]:
        answer = await calls.structured("feedback", self.model.config, SYSTEM, self.prompt(steps, assessments, outcome), _Verdicts)
        by_step: dict[int, _Verdict] = {}
        for verdict in answer.verdicts:
            by_step.setdefault(verdict.step, verdict)
        return [
            StepFeedback(verdict.grade, verdict.reason, verdict.lesson or None, verdict.critic_note or None) if (verdict := by_step.get(number)) else StepFeedback()
            for number in range(1, len(steps) + 1)
        ]

    def describe(self) -> str:
        limit = f", lessons of at most {self.lesson_chars} characters" if self.lesson_chars is not None else ""
        return f"llm: {self.model.to_string()} grades every move of the game in one call{limit}"
