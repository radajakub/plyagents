from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from plybench.utils.text import to_bool
from pydantic import BaseModel

from plyagents.textrl.configs.components import ComponentParams
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.losses.base import JointLoss
from plyagents.textrl.losses.inputs import NO_INPUTS, TRACK, ShownInputs, action_values_line, by_step, delta_line, deltas, discount_line, history_block, parse_inputs
from plyagents.textrl.losses.llm import LLMLoss
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.counts import KeyCounts
from plyagents.textrl.model.feedback import GRADES, Grade, StepFeedback
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.utils.calls import Calls
from plyagents.textrl.utils.options import read_options

SYSTEM = (
    "You review a game an agent played and judge each of its moves. Your lessons are added to the agent's notes for the situation"
    " they are about, and the agent reads them the next time it is in that situation."
)
PROMPT = """The agent's moves, in order. Each shows the situation as the agent saw it and the move it played{estimates}.

{steps}{history}

Outcome for the agent: {outcome:+g} (+1 is a win, 0 a draw, -1 a loss).{discount}

For every move give its step number, a grade ({grades}) and a short reason. Where the agent should play differently next time, or keep playing the same, add a lesson: one or two sentences it can act on when it meets this situation again{limit}.{track}{critic}"""
CRITIC = " Where the value estimate of a move was wrong, add a critic note: what the position was really worth, and why."
# with the Q critic, whose text assesses every legal move
CRITIC_Q = " Where the critic's assessment of the move played was wrong, add a critic note: what that move was really worth here, and why."

# what F is shown besides the moves and the outcome, each a block in a fixed format; in the order of the information ladder
# (experiments/tac-feedback-fixes.md in the phd repository), from REINFORCE's information to more than an estimator gets:
#   value      the critic's text for the position (what F was always shown)
#   memory     the entry of the actor's memory it read before the move: the parameters the gradient is taken at
#   delta      the critic's value before and after the move and the change, the TD error of actor-critic
#   q          the critic's value of every legal move, as the all-action policy gradient uses it
#   history    the moves played in the same situation in the earlier games and their results: the batch
#   reasoning  the actor's reasoning, and an LLM critic's notes and reasoning: more than an estimator gets
INPUTS = ("value", "memory", "delta", "q", "history", "reasoning")


def read_inputs(text: str) -> str:
    # "q+memory" -> "memory+q": the inputs in their fixed order, so equal sets give equal configs and paths; the moves and the outcome alone for "none"
    return parse_inputs(text, INPUTS, "feedback")


class _Verdict(BaseModel):
    step: int
    grade: Grade
    reason: str
    lesson: str | None = None
    critic_note: str | None = None


class _Verdicts(BaseModel):
    verdicts: list[_Verdict]


@dataclass(frozen=True, eq=True)
class LLMJointLossParams(ShownInputs, ComponentParams):
    model: ModelSpec | None = None
    # the length the lessons are asked to keep to, in characters; None states no limit
    lesson_chars: int | None = None
    # what F is shown besides the moves and the outcome (INPUTS), joined by +; "none" for nothing
    inputs: str = "value"
    # asks the lessons to keep track of the moves tried and how sure they are (TRACK)
    track: bool = False

    @classmethod
    def from_string(cls, params_string: str) -> LLMJointLossParams:
        options = read_options(params_string, {"model", "lesson_chars", "inputs", "track"}, "llm feedback")
        return cls(
            ModelSpec.from_string(options["model"]) if "model" in options else None,
            int(options["lesson_chars"]) if "lesson_chars" in options else None,
            read_inputs(options.get("inputs", cls.inputs)),
            to_bool(options.get("track", False)),
        )


class LLMJointLoss(LLMLoss, JointLoss):
    # F as in the paper: one call per game, the moves numbered, the outcome and what its inputs name shown; it answers per step
    # number (keys are long, the model never copies them). A step it leaves out gets no feedback. The history is the player's
    # counts before this game, so F keeps no state of its own
    role = "feedback"
    system = SYSTEM
    known = INPUTS

    def __init__(self, model: ModelSpec, lesson_chars: int | None, inputs: set[str] | None = None, track: bool = False, gamma: float = 1.0) -> None:
        super().__init__(model, {"value"} if inputs is None else inputs, gamma)
        self.lesson_chars = lesson_chars
        self.track = track

    @property
    def whole_game(self) -> bool:
        return True

    def _step(self, number: int, step: TextRLStep, assessment: Assessment, delta: tuple[float, float, bool] | None) -> str:
        # one move: the situation, what the actor read and thought, the move, then what the critic says of it
        lines = [f"Step {number}:", step.key]
        if "memory" in self.inputs:
            lines.append(step.context or "The agent read no notes here.")
        if "reasoning" in self.inputs and step.reasoning:
            lines.append(f"The agent's reasoning for this move:\n{step.reasoning.strip()}")
        lines.append(f"Move: {step.action or 'no valid move'}")
        if "value" in self.inputs and assessment.text:
            lines.append(f"Move assessments:\n{assessment.text}" if assessment.moves else f"Value estimate: {assessment.text}")
        if delta is not None:
            lines.append(delta_line(delta, self.gamma))
        if "q" in self.inputs:
            lines.append(action_values_line(step, assessment, self.role))
        if "reasoning" in self.inputs and assessment.context:
            lines.append(f"The critic's notes for this situation:\n{assessment.context}")
        if "reasoning" in self.inputs and assessment.reasoning:
            lines.append(f"The critic's reasoning:\n{assessment.reasoning.strip()}")
        return "\n".join(lines)

    def game_prompt(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts]) -> str:
        # V(s_t) against gamma V(s_t+1), the next position the agent moves in; after its last move, the outcome (the one reward)
        changes: list[tuple[float, float, bool] | None] = list(deltas(assessments, outcome, self.role, self.gamma)) if "delta" in self.inputs else [None] * len(steps)
        shown = [self._step(number, step, assessment, delta) for number, (step, assessment, delta) in enumerate(zip(steps, assessments, changes), start=1)]
        estimated = ("value" in self.inputs and any(assessment.text for assessment in assessments)) or "delta" in self.inputs
        moves = any(assessment.moves for assessment in assessments)
        return PROMPT.format(
            estimates=(" and the critic's assessment of each legal move" if moves else " and the value estimate of the position") if estimated else "",
            steps="\n\n".join(shown),
            history=history_block(history, [step.key for step in steps]) if "history" in self.inputs else "",
            outcome=outcome,
            discount=discount_line(self.gamma),
            grades=", ".join(GRADES),
            limit=f", at most {self.lesson_chars} characters" if self.lesson_chars is not None else "",
            track=TRACK if self.track else "",
            critic=(CRITIC_Q if moves else CRITIC) if estimated else "",
        )

    async def ask_game(self, system: str, prompt: str, calls: Calls, about: dict[str, Any], steps: int) -> tuple[dict[int, StepFeedback], list[dict[str, Any]]]:
        answer = await calls.structured(self.role, self.model.config, system, prompt, _Verdicts, about)
        found, dropped = by_step(answer.output.verdicts, steps)
        return {number: StepFeedback(item.grade, item.reason, item.lesson or None, item.critic_note or None) for number, item in found.items()}, dropped

    def describe(self) -> str:
        limit = f", lessons of at most {self.lesson_chars} characters" if self.lesson_chars is not None else ""
        shown = "+".join(name for name in INPUTS if name in self.inputs) or NO_INPUTS
        track = ", lessons keep track of the moves tried" if self.track else ""
        return f"{self.model.to_string()} grades every move of the game in one call, shown {shown}{limit}{track}"
