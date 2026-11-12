from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from plybench.utils.text import to_bool
from pydantic import BaseModel, Field

from plyagents.textrl.apply.base import Apply
from plyagents.textrl.apply.build import APPLIES
from plyagents.textrl.backward.base import Backward
from plyagents.textrl.backward.build import BACKWARDS
from plyagents.textrl.block import MemoryBlock
from plyagents.textrl.configs.components import ComponentConfig, ComponentParams
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.critics.base import Critic
from plyagents.textrl.enums import ApplyType, BackwardType, SchedulerType
from plyagents.textrl.losses.llm_critic import assessed
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.flow import Tape
from plyagents.textrl.model.memory import MemoryUpdate
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.schedulers.base import Scheduler
from plyagents.textrl.schedulers.build import SCHEDULERS
from plyagents.textrl.utils.calls import Calls
from plyagents.textrl.utils.options import read_options
from plyagents.textrl.utils.rules import with_rules

SYSTEM = (
    "You estimate how good a situation in a game is for an agent that plays it, under the agent's current way of playing: from -1, a sure"
    " loss, to +1, a sure win. You see the situation and your own notes from past assessments of it."
)
# put before the entry, as the actor's label; no instruction to follow it
LABEL = "Notes from past assessments of this situation:"
REQUEST = "How good is this situation for the agent? Give a short assessment with its reasons{limit}."
VALUE = " Also give your estimate as a number from -1 to +1."

# the Q critic: every legal move of the situation in one call
SYSTEM_Q = (
    "You estimate how good each legal move in a situation of a game is for an agent that plays it: the outcome it can expect after the move, under"
    " its current way of playing, from -1, a sure loss, to +1, a sure win. You see the situation, its legal moves and your own notes from past assessments."
)
# before the entries of the moves, each under its move
MOVE_LABEL = "Notes from past assessments of these moves:"
REQUEST_Q = "How good is each legal move here for the agent? For every move give a short assessment with its reasons{limit}."
VALUE_Q = " Also give your estimate for each move as a number from -1 to +1."
# the memory key of the Q critic's entry of one move of a situation: Q(s, a) is kept per state and move
MOVE_KEY = "{key}|{move}"


class _Valued(BaseModel):
    assessment: str
    value: float = Field(ge=-1.0, le=1.0)


class _Move(BaseModel):
    move: str
    assessment: str


class _ValuedMove(_Move):
    value: float = Field(ge=-1.0, le=1.0)


class _Moves(BaseModel):
    moves: list[_Move]


class _ValuedMoves(BaseModel):
    moves: list[_ValuedMove]


def _plain(move: str) -> str:
    # a move as the model may write it back ("Door 1" for "<door:1>"): letters and digits only, lower case
    return "".join(char for char in move.lower() if char.isalnum())


def match_moves(legal: tuple[str, ...], answered: list[_Move]) -> list[_Move | None]:
    # the answer of each legal move, aligned with legal; a move the model left out gets None, one it made up is dropped, the first answer of a move counts
    by_move: dict[str, _Move] = {}
    for item in answered:
        by_move.setdefault(_plain(item.move), item)
    return [by_move.get(_plain(move)) for move in legal]


@dataclass(frozen=True, eq=True)
class LLMCriticParams(ComponentParams):
    model: ModelSpec | None = None
    # the entry budget B in characters; None: notes are only appended
    budget: int | None = None
    # alpha_V, above the actor's 0.25: the critic tracks a policy that changes
    scheduler: ComponentConfig[SchedulerType] = SCHEDULERS.config("constant[beta=0.4]")
    apply: ComponentConfig[ApplyType] = APPLIES.default(ApplyType.TRUNCATE)
    # the length the assessments are asked to keep to, in characters; None states no limit
    assessment_chars: int | None = None
    # also a number in [-1, 1] (per move for the Q critic), kept for diagnostics (in the recorded call and the reference record); the feedback model
    # is shown the text only
    value: bool = False
    # the backward pass: turns the critic notes of the loss into the text written to the memory
    backward: ComponentConfig[BackwardType] = BACKWARDS.default(BackwardType.IDENTITY)

    @classmethod
    def from_string(cls, params_string: str) -> LLMCriticParams:
        options = read_options(params_string, {"model", "budget", "scheduler", "apply", "assessment_chars", "value", "backward"}, "llm critic")
        return cls(
            ModelSpec.from_string(options["model"]) if "model" in options else None,
            int(options["budget"]) if "budget" in options else None,
            SCHEDULERS.config(options["scheduler"]) if "scheduler" in options else cls.scheduler,
            APPLIES.config(options["apply"]) if "apply" in options else cls.apply,
            int(options["assessment_chars"]) if "assessment_chars" in options else None,
            to_bool(options.get("value", False)),
            BACKWARDS.config(options["backward"]) if "backward" in options else cls.backward,
        )


class LLMCritic(MemoryBlock, Critic[MemoryUpdate]):
    # the critic of tac.tex: the LLM assesses each visited state from its observation and the entry of the critic memory M^V alone (not the trajectory,
    # the outcome or the actor's memory), one call per key, M^V from before the game's update (the forward pass); learns the losses' critic notes as a
    # MemoryBlock. The V critic assesses the situation, the Q critic each of its legal moves
    task = SYSTEM
    kind = "critic"
    backward_role = "backward_critic"
    apply_role = "apply_critic"

    def __init__(
        self, model: ModelSpec, budget: int | None, scheduler: Scheduler, apply: Apply, assessment_chars: int | None, value: bool, backward: Backward | None = None
    ) -> None:
        super().__init__(budget, scheduler, apply, backward)
        self.model = model
        self.assessment_chars = assessment_chars
        self.value = value

    # one call for the key of step, the first step of the game at it; about: the key and the steps at it (LLMCall.about)
    @abstractmethod
    async def _assess(self, system: str, step: TextRLStep, calls: Calls, about: dict[str, Any]) -> Assessment:
        raise NotImplementedError

    async def assess(self, steps: list[TextRLStep], calls: Calls) -> list[Assessment]:
        # one call per key, in the order of first visit; the calls start in that order, so each takes its seed by it
        system, first, numbers = with_rules(self.task, steps), {}, {}
        for number, step in enumerate(steps, start=1):
            first.setdefault(step.key, step)
            numbers.setdefault(step.key, []).append(number)
        assessed = dict(zip(first, await asyncio.gather(*(self._assess(system, step, calls, {"key": key, "steps": numbers[key]}) for key, step in first.items()))))
        return [assessed[step.key] for step in steps]

    def tape(self, step: TextRLStep, assessment: Assessment, limit: int | None) -> Tape:
        # what the critic read, thought and assessed
        return Tape("critic", step.key, assessment.context, assessment.text, assessment.reasoning, limit, step.action)

    def describe(self) -> str:
        budget = f"budget {self.budget} characters" if self.budget is not None else "no budget"
        value = ", with a number for diagnostics" if self.value else ""
        return (
            f"{self.model.to_string()} assesses {self.assessed()} from its notes{value}; notes written by {self.apply.describe()}; alpha_V: {self.scheduler.describe()}; {budget}"
        )

    @abstractmethod
    def assessed(self) -> str:
        raise NotImplementedError


class LLMVCritic(LLMCritic):
    # V: one assessment of the situation
    def prompt(self, key: str) -> str:
        entry = self.memory.get(key).text
        notes = f"\n\n{LABEL}\n{entry}" if entry else ""
        limit = f", at most {self.assessment_chars} characters" if self.assessment_chars is not None else ""
        return f"The situation:\n{key}{notes}\n\n{REQUEST.format(limit=limit)}" + (VALUE if self.value else "")

    async def _assess(self, system: str, step: TextRLStep, calls: Calls, about: dict[str, Any]) -> Assessment:
        # with the entry it read and its reasoning, for a feedback model shown them
        entry = self.memory.get(step.key).text or None
        read = (self.memory.read(step.key, "critic"),)
        if not self.value:
            text = await calls.text("critic", self.model.config, system, self.prompt(step.key), about)
            return Assessment(None, text.output.strip(), context=entry, reasoning=text.reasoning or None, read=read)
        answer = await calls.structured("critic", self.model.config, system, self.prompt(step.key), _Valued, about)
        return Assessment(answer.output.value, answer.output.assessment.strip(), context=entry, reasoning=answer.reasoning or None, read=read)

    def assessed(self) -> str:
        return "each visited state"


class LLMQCritic(LLMCritic):
    # Q: one assessment per legal move of the situation, in one call; no value of the situation (it would need the policy)
    task = SYSTEM_Q

    def entry_key(self, step: TextRLStep) -> str | None:
        # the entry of the move played; a failed move has none, so its note is not written
        return MOVE_KEY.format(key=step.key, move=step.action) if step.action is not None else None

    def tape(self, step: TextRLStep, assessment: Assessment, limit: int | None) -> Tape:
        # what the critic read and assessed of the move played: the entry its note is written to and its assessment of that move, q_t(a_t), from the
        # entries it read (Assessment.read); not the other moves', so a note cannot go to the wrong move
        key = self.entry_key(step)
        read = next((entry["text"] for entry in assessment.read if entry["key"] == key), "") or None
        return Tape("critic", step.key, read, assessed(step, assessment) or "none", assessment.reasoning, limit, step.action, per_move=True)

    def context(self, step: TextRLStep) -> str | None:
        # the notes the critic reads: the entry of every legal move that has one, each under its move
        entries = [(move, self.memory.get(MOVE_KEY.format(key=step.key, move=move)).text) for move in step.legal]
        return "\n".join(f"{move}: {entry}" for move, entry in entries if entry) or None

    def prompt(self, step: TextRLStep) -> str:
        moves = "\n".join(f"- {move}" for move in step.legal)
        context = self.context(step)
        notes = f"\n\n{MOVE_LABEL}\n{context}" if context else ""
        limit = f", at most {self.assessment_chars} characters each" if self.assessment_chars is not None else ""
        return f"The situation:\n{step.key}\n\nLegal moves:\n{moves}{notes}\n\n{REQUEST_Q.format(limit=limit)}" + (VALUE_Q if self.value else "")

    def _read(self, step: TextRLStep) -> tuple[dict[str, Any], ...]:
        # every entry the notes are read from: each legal move's
        return tuple(self.memory.read(MOVE_KEY.format(key=step.key, move=move), "critic") for move in step.legal)

    async def _assess(self, system: str, step: TextRLStep, calls: Calls, about: dict[str, Any]) -> Assessment:
        # the text of every answered move in the order of legal, the numbers aligned with legal (None for a move left out); the moves it named
        # that are not legal, or named again, are dropped and recorded
        schema = _ValuedMoves if self.value else _Moves
        answered = await calls.structured("critic", self.model.config, system, self.prompt(step), schema, about)
        answer, reasoning = answered.output, answered.reasoning
        matched = match_moves(step.legal, list(answer.moves))
        text = "\n".join(f"{move}: {item.assessment.strip()}" for move, item in zip(step.legal, matched) if item is not None)
        values = tuple(item.value if isinstance(item, _ValuedMove) else None for item in matched) if self.value else None
        legal = {_plain(move) for move in step.legal}
        named = [_plain(item.move) for item in answer.moves]
        unmatched = tuple(item.move for item in answer.moves if _plain(item.move) not in legal)
        repeated = tuple(move for move in step.legal if named.count(_plain(move)) > 1)
        texts = tuple(item.assessment.strip() if item is not None else None for item in matched)
        return Assessment(None, text, values, self.context(step), reasoning or None, moves=True, read=self._read(step), unmatched=unmatched, repeated=repeated, move_texts=texts)

    def assessed(self) -> str:
        return "each legal move of each visited state (notes per move)"
