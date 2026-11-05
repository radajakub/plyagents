from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from plybench.utils.text import to_bool
from pydantic import BaseModel, Field

from plyagents.tac.apply.base import Apply
from plyagents.tac.apply.build import APPLIES
from plyagents.tac.configs.components import ComponentConfig, ComponentParams
from plyagents.tac.configs.model import ModelSpec
from plyagents.tac.critics.base import Critic
from plyagents.tac.enums import ApplyType, SchedulerType
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


class _Valued(BaseModel):
    assessment: str
    value: float = Field(ge=-1.0, le=1.0)


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
    # also a number in [-1, 1], kept for diagnostics (in the recorded call); the feedback model is shown the text only
    value: bool = False

    @classmethod
    def from_string(cls, params_string: str) -> LLMCriticParams:
        options = read_options(params_string, {"model", "budget", "scheduler", "apply", "assessment_chars", "value"}, "llm critic")
        return cls(
            ModelSpec.from_string(options["model"]) if "model" in options else None,
            int(options["budget"]) if "budget" in options else None,
            SCHEDULERS.config(options["scheduler"]) if "scheduler" in options else cls.scheduler,
            APPLIES.config(options["apply"]) if "apply" in options else cls.apply,
            int(options["assessment_chars"]) if "assessment_chars" in options else None,
            to_bool(options.get("value", False)),
        )


class LLMCritic(Critic[MemoryUpdate]):
    # the critic of tac.tex: the LLM assesses each visited state from its observation and the entry of the critic memory M^V alone (not the trajectory,
    # the outcome or the actor's memory), one call per key, M^V from before the game's update; propose writes the feedback's critic notes into M^V
    def __init__(self, model: ModelSpec, budget: int | None, scheduler: Scheduler, apply: Apply, assessment_chars: int | None, value: bool) -> None:
        self.model = model
        self.budget = budget
        self.scheduler = scheduler
        self.apply = apply
        self.assessment_chars = assessment_chars
        self.value = value
        self.memory = KeyedMemory()

    @staticmethod
    def system(steps: list[TACStep]) -> str:
        # the rules of the game as the agent read them, from the record (the player keeps them with its first step of a game)
        rules = next((step.rules for step in steps if step.rules), None)
        return f"{SYSTEM}\n\nThe rules, as the agent was given them:\n{rules}" if rules else SYSTEM

    def prompt(self, key: str) -> str:
        entry = self.memory.get(key).text
        notes = f"\n\n{LABEL}\n{entry}" if entry else ""
        limit = f", at most {self.assessment_chars} characters" if self.assessment_chars is not None else ""
        return f"The situation:\n{key}{notes}\n\n{REQUEST.format(limit=limit)}" + (VALUE if self.value else "")

    async def _assess(self, system: str, key: str, calls: Calls) -> Assessment:
        if not self.value:
            return Assessment(None, (await calls.text("critic", self.model.config, system, self.prompt(key))).strip())
        answer = await calls.structured("critic", self.model.config, system, self.prompt(key), _Valued)
        return Assessment(answer.value, answer.assessment.strip())

    async def assess(self, steps: list[TACStep], calls: Calls) -> list[Assessment]:
        # one call per key, in the order of first visit; the calls start in that order, so each takes its seed by it
        system, keys = self.system(steps), list(dict.fromkeys(step.key for step in steps))
        assessed = dict(zip(keys, await asyncio.gather(*(self._assess(system, key, calls) for key in keys))))
        return [assessed[step.key] for step in steps]

    async def propose(self, steps: list[TACStep], outcome: float, feedback: list[StepFeedback], calls: Calls) -> MemoryUpdate | None:
        return await self.apply.update(self.memory, steps, feedback, [verdict.critic_note for verdict in feedback], self.scheduler, self.budget, calls, "apply_critic")

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
            f"{self.model.to_string()} assesses each visited state from its notes{value}; notes written by {self.apply.describe()}; alpha_V: {self.scheduler.describe()}; {budget}"
        )
