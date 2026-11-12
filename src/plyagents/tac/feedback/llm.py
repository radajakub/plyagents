from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from plybench.utils.text import to_bool
from pydantic import BaseModel

from plyagents.tac.configs.components import ComponentParams
from plyagents.tac.configs.model import ModelSpec
from plyagents.tac.feedback.base import Feedback, Given
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.counts import KeyCounts, counts_to_dict
from plyagents.tac.model.feedback import GRADES, Grade, StepFeedback
from plyagents.tac.model.step import TACStep
from plyagents.tac.utils.calls import Calls
from plyagents.tac.utils.options import read_options

SYSTEM = (
    "You review a game an agent played and judge each of its moves. Your lessons are added to the agent's notes for the situation"
    " they are about, and the agent reads them the next time it is in that situation."
)
PROMPT = """The agent's moves, in order. Each shows the situation as the agent saw it and the move it played{estimates}.

{steps}{history}

Outcome for the agent: {outcome:+g} (+1 is a win, 0 a draw, -1 a loss).

For every move give its step number, a grade ({grades}) and a short reason. Where the agent should play differently next time, or keep playing the same, add a lesson: one or two sentences it can act on when it meets this situation again{limit}.{track}{critic}"""
CRITIC = " Where the value estimate of a move was wrong, add a critic note: what the position was really worth, and why."
# with the Q critic, whose text assesses every legal move
CRITIC_Q = " Where the critic's assessment of the move played was wrong, add a critic note: what that move was really worth here, and why."
# with `track` (the exploration experiment's T-text): the lessons keep what was tried and how sure they are, so the actor can judge what is still uncertain
TRACK = " In each lesson, also keep track of which moves were tried in this situation and how sure the lesson is."
HISTORY = "What the agent's earlier games showed in these situations (the moves it played there and how those games ended):"

# what F is shown besides the moves and the outcome, each a block in a fixed format; in the order of the information ladder
# (experiments/tac-feedback-fixes.md in the phd repository), from REINFORCE's information to more than an estimator gets:
#   value      the critic's text for the position (what F was always shown)
#   memory     the entry of the actor's memory it read before the move: the parameters the gradient is taken at
#   delta      the critic's value before and after the move and the change, the TD error of actor-critic
#   q          the critic's value of every legal move, as the all-action policy gradient uses it
#   history    the moves played in the same situation in the earlier games and their results: the batch
#   reasoning  the actor's reasoning, and an LLM critic's notes and reasoning: more than an estimator gets
INPUTS = ("value", "memory", "delta", "q", "history", "reasoning")
# written for an empty set of inputs: the moves and the outcome alone
NO_INPUTS = "none"


def read_inputs(text: str) -> str:
    # "q+memory" -> "memory+q": the inputs in their fixed order, so equal sets give equal configs and paths
    names = set() if text == NO_INPUTS else set(text.split("+"))
    unknown = sorted(names - set(INPUTS))
    if unknown:
        raise ValueError(f"Unknown feedback inputs {unknown}; known: {list(INPUTS)} or {NO_INPUTS!r}, joined by +")
    return "+".join(name for name in INPUTS if name in names) or NO_INPUTS


def history_text(counts: KeyCounts) -> str | None:
    # "<door:1> in 3 games (2 loss, 1 win); <door:2> in 1 games (1 win)", the moves in the order first played; None when the key was never seen
    if not counts:
        return None
    parts = (
        f"{move} in {move_counts.visits} games ({', '.join(f'{count} {name}' for name, count in Counter(move_counts.results).most_common())})"
        for move, move_counts in counts.items()
    )
    return "; ".join(parts)


def history_block(history: Mapping[str, KeyCounts], keys: list[str]) -> str:
    # per step of the game, the moves played in its situation in the earlier games; put after the moves
    lines = [f"Step {number}: {history_text(history.get(key, {})) or 'no earlier games in this situation'}." for number, key in enumerate(keys, start=1)]
    return f"\n\n{HISTORY}\n" + "\n".join(lines)


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
    # what F is shown besides the moves and the outcome (INPUTS), joined by +; "none" for nothing
    inputs: str = "value"
    # asks the lessons to keep track of the moves tried and how sure they are (TRACK)
    track: bool = False

    @classmethod
    def from_string(cls, params_string: str) -> LLMFeedbackParams:
        options = read_options(params_string, {"model", "lesson_chars", "inputs", "track"}, "llm feedback")
        return cls(
            ModelSpec.from_string(options["model"]) if "model" in options else None,
            int(options["lesson_chars"]) if "lesson_chars" in options else None,
            read_inputs(options.get("inputs", cls.inputs)),
            to_bool(options.get("track", False)),
        )

    @property
    def shown(self) -> set[str]:
        return set() if self.inputs == NO_INPUTS else set(self.inputs.split("+"))


class LLMFeedback(Feedback):
    # F as in the paper: one call per game, the moves numbered, the outcome and what its inputs name shown; it answers per step
    # number (keys are long, the model never copies them). A step it leaves out gets no feedback. The history is the player's
    # counts before this game, so F keeps no state of its own
    def __init__(self, model: ModelSpec, lesson_chars: int | None, inputs: set[str] | None = None, track: bool = False) -> None:
        self.model = model
        self.lesson_chars = lesson_chars
        self.inputs = {"value"} if inputs is None else inputs
        self.track = track

    @staticmethod
    def system(steps: list[TACStep]) -> str:
        # the rules of the game as the agent read them, from the record (the player keeps them with its first step of a game)
        rules = next((step.rules for step in steps if step.rules), None)
        return f"{SYSTEM}\n\nThe rules, as the agent was given them:\n{rules}" if rules else SYSTEM

    def _values(self, assessments: list[Assessment]) -> list[float]:
        values = [assessment.value for assessment in assessments]
        if any(value is None for value in values):
            raise ValueError("feedback inputs=delta needs a critic that gives numbers (oracle, tabular, or llm_v[value=true])")
        return [value for value in values if value is not None]

    @staticmethod
    def _action_values(step: TACStep, assessment: Assessment) -> str:
        if assessment.action_values is None:
            raise ValueError("feedback inputs=q needs a critic with action values (oracle or tabular)")
        shown = [f"{action} {value:+.2f}" if value is not None else f"{action} unknown" for action, value in zip(step.legal, assessment.action_values)]
        return f"Critic's value of each move here: {', '.join(shown)}."

    def _step(self, number: int, step: TACStep, assessment: Assessment, delta: tuple[float, float, bool] | None) -> str:
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
            before, after, last = delta
            lines.append(f"Critic: value before the move {before:+.2f}, after {after:+.2f}{' (the outcome)' if last else ''}; change {after - before:+.2f}.")
        if "q" in self.inputs:
            lines.append(self._action_values(step, assessment))
        if "reasoning" in self.inputs and assessment.context:
            lines.append(f"The critic's notes for this situation:\n{assessment.context}")
        if "reasoning" in self.inputs and assessment.reasoning:
            lines.append(f"The critic's reasoning:\n{assessment.reasoning.strip()}")
        return "\n".join(lines)

    def _deltas(self, steps: list[TACStep], assessments: list[Assessment], outcome: float) -> list[tuple[float, float, bool] | None]:
        # V(s_t) against V(s_t+1), the next position the agent moves in; after its last move, the outcome (the one reward, no discount)
        if "delta" not in self.inputs:
            return [None] * len(steps)
        values = self._values(assessments)
        return [(value, values[index + 1] if index + 1 < len(values) else outcome, index + 1 == len(values)) for index, value in enumerate(values)]

    def prompt(self, steps: list[TACStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts]) -> str:
        deltas = self._deltas(steps, assessments, outcome)
        shown = [self._step(number, step, assessment, delta) for number, (step, assessment, delta) in enumerate(zip(steps, assessments, deltas), start=1)]
        estimated = ("value" in self.inputs and any(assessment.text for assessment in assessments)) or "delta" in self.inputs
        moves = any(assessment.moves for assessment in assessments)
        return PROMPT.format(
            estimates=(" and the critic's assessment of each legal move" if moves else " and the value estimate of the position") if estimated else "",
            steps="\n\n".join(shown),
            history=history_block(history, [step.key for step in steps]) if "history" in self.inputs else "",
            outcome=outcome,
            grades=", ".join(GRADES),
            limit=f", at most {self.lesson_chars} characters" if self.lesson_chars is not None else "",
            track=TRACK if self.track else "",
            critic=(CRITIC_Q if moves else CRITIC) if estimated else "",
        )

    async def give(self, steps: list[TACStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], calls: Calls) -> list[StepFeedback]:
        return (await self.give_recorded(steps, assessments, outcome, history, calls)).verdicts

    async def give_recorded(self, steps: list[TACStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], calls: Calls) -> Given:
        prompt = self.prompt(steps, assessments, outcome, history)
        about = {"steps": list(range(1, len(steps) + 1))}
        answer = await calls.structured("feedback", self.model.config, self.system(steps), prompt, _Verdicts, about)
        by_step: dict[int, _Verdict] = {}
        dropped: list[dict[str, Any]] = []
        for verdict in answer.verdicts:
            # a verdict for a step the game does not have, or for a step already answered, is not used
            if not 1 <= verdict.step <= len(steps) or verdict.step in by_step:
                dropped.append(verdict.model_dump())
                continue
            by_step[verdict.step] = verdict
        verdicts = [
            StepFeedback(verdict.grade, verdict.reason, verdict.lesson or None, verdict.critic_note or None) if (verdict := by_step.get(number)) else StepFeedback()
            for number in range(1, len(steps) + 1)
        ]
        return Given(verdicts, dropped, self._inputs(steps, assessments, outcome, history))

    def _inputs(self, steps: list[TACStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts]) -> dict[str, Any]:
        # what the prompt showed besides the moves and the outcome, by part: the inputs named, the critic's change per step, the earlier games per step
        inputs: dict[str, Any] = {"shown": [name for name in INPUTS if name in self.inputs]}
        if "delta" in self.inputs:
            deltas = self._deltas(steps, assessments, outcome)
            inputs["deltas"] = [None if delta is None else {"before": delta[0], "after": delta[1], "last": delta[2]} for delta in deltas]
        if "history" in self.inputs:
            inputs["history"] = [counts_to_dict(history.get(step.key, {})) for step in steps]
        return inputs

    def describe(self) -> str:
        limit = f", lessons of at most {self.lesson_chars} characters" if self.lesson_chars is not None else ""
        shown = "+".join(name for name in INPUTS if name in self.inputs) or NO_INPUTS
        track = ", lessons keep track of the moves tried" if self.track else ""
        return f"{self.model.to_string()} grades every move of the game in one call, shown {shown}{limit}{track}"
