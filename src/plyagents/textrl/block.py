from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any

from plybench.core.game import TurnBasedGame
from plybench.core.prompt_adapter import PromptAdapter

from plyagents.textrl.apply.base import Apply, lesson_share
from plyagents.textrl.backward.base import Backward
from plyagents.textrl.backward.identity import IdentityBackward
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.feedback import BlockKind, StepFeedback
from plyagents.textrl.model.flow import JudgedGame, Tape
from plyagents.textrl.model.memory import KeyedMemory, MemoryUpdate
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.model.table import TableUpdate
from plyagents.textrl.model.update import ComponentUpdate
from plyagents.textrl.schedulers.base import Scheduler
from plyagents.textrl.utils.calls import Calls


class Block[U: ComponentUpdate](ABC):
    # a learning block of the TextRL (the actor, the critic): a forward pass that reads its learned state (Actor.act, Critic.assess), and in propose a
    # backward pass that turns the feedback of its loss into an update U of that state, which commit applies. Its inputs and outputs are kept in the
    # records, never in the object, so a game replayed on a resume is proposed from them. A block that does not learn keeps the defaults: no update
    def prepare(self, game: TurnBasedGame) -> None:
        # once, when the player is built: what propose needs from the game itself, since a game replayed on a resume is not played again and so
        # never initializes
        pass

    # before every game it plays
    def initialize(self, game: TurnBasedGame, adapter_template: PromptAdapter) -> None:
        pass

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError

    # the backward pass and the write of one game; assessments: the critic's forward pass of the game's steps
    async def propose(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, feedback: list[StepFeedback], calls: Calls) -> U | None:
        return None

    # one update from the games of a batch (batch=epoch, batch=every)
    async def propose_batch(self, games: list[JudgedGame], calls: Calls) -> U | None:
        return None

    def commit(self, update: U) -> None:
        pass

    def parse_update(self, data: dict[str, Any]) -> U:
        raise NotImplementedError(f"{type(self).__name__} does not learn, so it has no update to read")

    def state(self) -> dict[str, Any] | None:
        return None

    def load_state(self, state: dict[str, Any] | None) -> None:
        pass


class MemoryBlock(Block[MemoryUpdate]):
    # a block that keeps a textual memory M, one entry per key (the LLM actor, the LLM critics). Its backward pass turns the verdict of every step into
    # the text written to the entry the step read (Backward, from the block's tape of the step), and Apply writes those gradients into the memory: one
    # write per key, with the step alpha of the scheduler within the budget B. kind: which text of a verdict is its gradient (StepFeedback.gradient);
    # the roles name its calls
    kind: BlockKind
    backward_role: str
    apply_role: str

    def __init__(self, budget: int | None, scheduler: Scheduler, apply: Apply, backward: Backward | None = None) -> None:
        self.budget = budget
        self.scheduler = scheduler
        self.apply = apply
        self.backward = backward if backward is not None else IdentityBackward()
        self.memory = KeyedMemory()

    # what the block stored of a step in its forward pass; limit: the length its gradient may take (share)
    @abstractmethod
    def tape(self, step: TextRLStep, assessment: Assessment, limit: int | None) -> Tape:
        raise NotImplementedError

    def entry_key(self, step: TextRLStep) -> str | None:
        # the entry a step's gradient is written to: the key it read; None: the step has none, so its gradient is dropped
        return step.key

    def share(self, key: str) -> int | None:
        return lesson_share(self.memory.get(key), self.scheduler, self.budget)

    async def propose(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, feedback: list[StepFeedback], calls: Calls) -> MemoryUpdate | None:
        keys = [self.entry_key(step) for step in steps]
        # a step without an entry keeps its key for the visit count; its gradient is dropped, so nothing is written for it
        written = [replace(step, key=key) if key is not None else step for step, key in zip(steps, keys)]
        feedback = [verdict if key is not None else verdict.without_gradient(self.kind) for verdict, key in zip(feedback, keys)]
        tapes = [self.tape(step, assessment, self.share(entry.key)) for step, entry, assessment in zip(steps, written, assessments)]
        gradients = await self.backward.local(tapes, feedback, calls, self.backward_role)
        update = await self.apply.update(self.memory, written, feedback, gradients, self.scheduler, self.budget, calls, self.apply_role)
        return replace(update, gradients=tuple(gradients), limits=tuple(tape.limit for tape in tapes))

    async def propose_batch(self, games: list[JudgedGame], calls: Calls) -> MemoryUpdate | None:
        # the games' steps in order as one: each key gets one write for all its gradients in the batch (the outcome is not read)
        steps = [step for game in games for step in game.steps]
        assessments = [assessment for game in games for assessment in game.assessments]
        feedback = [verdict for game in games for verdict in game.feedback]
        return await self.propose(steps, assessments, games[0].outcome if games else 0.0, feedback, calls)

    def commit(self, update: MemoryUpdate) -> None:
        update.apply_to(self.memory)

    def parse_update(self, data: dict[str, Any]) -> MemoryUpdate:
        return MemoryUpdate.from_dict(data)

    def state(self) -> dict[str, Any] | None:
        return self.memory.to_dict()

    def load_state(self, state: dict[str, Any] | None) -> None:
        self.memory = KeyedMemory.from_dict(state or {})


class TableBlock(Block[TableUpdate]):
    # a block that keeps running estimates in tables (the tabular actor and critic): one game's update is its samples toward their targets, applied in
    # order with the scheduler's step; a batch's update is the samples of its games in game order
    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler

    async def propose_batch(self, games: list[JudgedGame], calls: Calls) -> TableUpdate | None:
        updates = [await self.propose(game.steps, game.assessments, game.outcome, game.feedback, calls) for game in games]
        return TableUpdate(tuple(sample for update in updates if update is not None for sample in update.samples))

    def parse_update(self, data: dict[str, Any]) -> TableUpdate:
        return TableUpdate.from_dict(data)
