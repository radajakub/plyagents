from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Any

from plybench.utils.text import to_bool
from pydantic import BaseModel, Field

from plyagents.tac.apply.base import Apply
from plyagents.tac.apply.build import APPLIES
from plyagents.tac.configs.components import ComponentConfig, ComponentParams
from plyagents.tac.configs.model import ModelSpec
from plyagents.tac.critics.base import Critic
from plyagents.tac.enums import ApplyType, NotesType, SchedulerType
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.memory import KeyedMemory, MemoryUpdate
from plyagents.tac.model.step import TACStep
from plyagents.tac.schedulers.base import Scheduler
from plyagents.tac.schedulers.build import SCHEDULERS
from plyagents.tac.utils.calls import Calls
from plyagents.tac.utils.options import read_options

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
# with notes=move, before the entries of the moves, each under its move
MOVE_LABEL = "Notes from past assessments of these moves:"
REQUEST_Q = "How good is each legal move here for the agent? For every move give a short assessment with its reasons{limit}."
VALUE_Q = " Also give your estimate for each move as a number from -1 to +1."
# the memory key of one move of a situation (notes=move)
MOVE_KEY = "{key}\nMove: {move}"
# the options both LLM critics read; the Q critic also reads notes
_OPTIONS = {"model", "budget", "scheduler", "apply", "assessment_chars", "value"}


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

    @staticmethod
    def _read(options: dict[str, str]) -> dict[str, Any]:
        return {
            "model": ModelSpec.from_string(options["model"]) if "model" in options else None,
            "budget": int(options["budget"]) if "budget" in options else None,
            "scheduler": SCHEDULERS.config(options["scheduler"]) if "scheduler" in options else LLMCriticParams.scheduler,
            "apply": APPLIES.config(options["apply"]) if "apply" in options else LLMCriticParams.apply,
            "assessment_chars": int(options["assessment_chars"]) if "assessment_chars" in options else None,
            "value": to_bool(options.get("value", False)),
        }

    @classmethod
    def from_string(cls, params_string: str) -> LLMCriticParams:
        return cls(**cls._read(read_options(params_string, _OPTIONS, "llm_v critic")))


@dataclass(frozen=True, eq=True)
class LLMQCriticParams(LLMCriticParams):
    # one entry per situation, as the V critic, or one per move of a situation
    notes: NotesType = NotesType.SITUATION

    @classmethod
    def from_string(cls, params_string: str) -> LLMQCriticParams:
        options = read_options(params_string, _OPTIONS | {"notes"}, "llm_q critic")
        notes = NotesType.from_value(options.get("notes", NotesType.SITUATION.value))
        if notes is None:
            raise ValueError(f"Invalid llm_q critic notes {options['notes']!r}; choose from {NotesType.values()}")
        return cls(**cls._read(options), notes=notes)


class LLMCritic(Critic[MemoryUpdate]):
    # the critic of tac.tex: the LLM assesses each visited state from its observation and the entry of the critic memory M^V alone (not the trajectory,
    # the outcome or the actor's memory), one call per key, M^V from before the game's update; propose writes the feedback's critic notes into M^V.
    # The V critic assesses the situation, the Q critic each of its legal moves
    task = SYSTEM

    def __init__(self, model: ModelSpec, budget: int | None, scheduler: Scheduler, apply: Apply, assessment_chars: int | None, value: bool) -> None:
        self.model = model
        self.budget = budget
        self.scheduler = scheduler
        self.apply = apply
        self.assessment_chars = assessment_chars
        self.value = value
        self.memory = KeyedMemory()

    @classmethod
    def system(cls, steps: list[TACStep]) -> str:
        # the rules of the game as the agent read them, from the record (the player keeps them with its first step of a game)
        rules = next((step.rules for step in steps if step.rules), None)
        return f"{cls.task}\n\nThe rules, as the agent was given them:\n{rules}" if rules else cls.task

    # one call for the key of step, the first step of the game at it
    @abstractmethod
    async def _assess(self, system: str, step: TACStep, calls: Calls) -> Assessment:
        raise NotImplementedError

    def note_key(self, step: TACStep) -> str:
        # the entry the feedback's critic note of a step is written to
        return step.key

    async def assess(self, steps: list[TACStep], calls: Calls) -> list[Assessment]:
        # one call per key, in the order of first visit; the calls start in that order, so each takes its seed by it
        system, first = self.system(steps), {}
        for step in steps:
            first.setdefault(step.key, step)
        assessed = dict(zip(first, await asyncio.gather(*(self._assess(system, step, calls) for step in first.values()))))
        return [assessed[step.key] for step in steps]

    async def propose(self, steps: list[TACStep], outcome: float, feedback: list[StepFeedback], calls: Calls) -> MemoryUpdate | None:
        noted = [replace(step, key=self.note_key(step)) for step in steps]
        return await self.apply.update(self.memory, noted, feedback, [verdict.critic_note for verdict in feedback], self.scheduler, self.budget, calls, "apply_critic")

    def commit(self, update: MemoryUpdate) -> None:
        update.apply_to(self.memory)

    def parse_update(self, data: dict[str, Any]) -> MemoryUpdate:
        return MemoryUpdate.from_dict(data)

    def state(self) -> dict[str, Any] | None:
        return self.memory.to_dict()

    def load_state(self, state: dict[str, Any] | None) -> None:
        self.memory = KeyedMemory.from_dict(state or {})

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

    async def _assess(self, system: str, step: TACStep, calls: Calls) -> Assessment:
        # with the entry it read and its reasoning, for a feedback model shown them
        entry = self.memory.get(step.key).text or None
        if not self.value:
            text, reasoning = await calls.reasoned_text("critic", self.model.config, system, self.prompt(step.key))
            return Assessment(None, text.strip(), context=entry, reasoning=reasoning or None)
        answer, reasoning = await calls.reasoned_structured("critic", self.model.config, system, self.prompt(step.key), _Valued)
        return Assessment(answer.value, answer.assessment.strip(), context=entry, reasoning=reasoning or None)

    def assessed(self) -> str:
        return "each visited state"


class LLMQCritic(LLMCritic):
    # Q: one assessment per legal move of the situation, in one call; no value of the situation (it would need the policy)
    task = SYSTEM_Q

    def __init__(self, model: ModelSpec, budget: int | None, scheduler: Scheduler, apply: Apply, assessment_chars: int | None, value: bool, notes: NotesType) -> None:
        super().__init__(model, budget, scheduler, apply, assessment_chars, value)
        self.notes = notes

    def note_key(self, step: TACStep) -> str:
        # with notes=move, the entry of the move played; a failed move has none, so its note goes to the situation's
        return MOVE_KEY.format(key=step.key, move=step.action) if self.notes is NotesType.MOVE and step.action is not None else step.key

    def context(self, step: TACStep) -> str | None:
        # the notes the critic reads: the situation's entry, or the entry of every legal move that has one, each under its move
        if self.notes is NotesType.SITUATION:
            return self.memory.get(step.key).text or None
        entries = [(move, self.memory.get(MOVE_KEY.format(key=step.key, move=move)).text) for move in step.legal]
        return "\n".join(f"{move}: {entry}" for move, entry in entries if entry) or None

    def prompt(self, step: TACStep) -> str:
        moves = "\n".join(f"- {move}" for move in step.legal)
        context = self.context(step)
        notes = f"\n\n{LABEL if self.notes is NotesType.SITUATION else MOVE_LABEL}\n{context}" if context else ""
        limit = f", at most {self.assessment_chars} characters each" if self.assessment_chars is not None else ""
        return f"The situation:\n{step.key}\n\nLegal moves:\n{moves}{notes}\n\n{REQUEST_Q.format(limit=limit)}" + (VALUE_Q if self.value else "")

    async def _assess(self, system: str, step: TACStep, calls: Calls) -> Assessment:
        # the text of every answered move in the order of legal, the numbers aligned with legal (None for a move left out)
        schema = _ValuedMoves if self.value else _Moves
        answer, reasoning = await calls.reasoned_structured("critic", self.model.config, system, self.prompt(step), schema)
        matched = match_moves(step.legal, list(answer.moves))
        text = "\n".join(f"{move}: {item.assessment.strip()}" for move, item in zip(step.legal, matched) if item is not None)
        values = tuple(item.value if isinstance(item, _ValuedMove) else None for item in matched) if self.value else None
        return Assessment(None, text, values, self.context(step), reasoning or None, moves=True)

    def assessed(self) -> str:
        return f"each legal move of each visited state (notes per {self.notes.value})"
